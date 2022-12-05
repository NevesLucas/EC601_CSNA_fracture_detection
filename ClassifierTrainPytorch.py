import kaggleDataLoader
import json
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
from monai.data import decollate_batch, DataLoader,Dataset,ImageDataset
from monai.networks.nets import DenseNet201
from sklearn.metrics import classification_report
import torch.cuda.amp as amp
import torchio as tio

with open('config.json', 'r') as f:
    paths = json.load(f)

segWeights = paths["seg_weights"]
cachedir = paths["CACHE_DIR"]
memory = Memory(cachedir, verbose=0, compress=True)

segModel = torch.load(segWeights, map_location="cpu") # need 2 gpus for this workflow
segModel.eval()
segResize = tio.Resize((128, 128, 200)) #resize for segmentation
classResize = tio.Resize((256,256,256))

def boundingVolume(pred,original_dims):
    #acquires the 3d bounding rectangular prism of the segmentation mask
    indices = torch.nonzero(pred)
    min_indices, min_val = indices.min(dim=0)
    max_indices, max_val = indices.max(dim=0)
    return (min_indices[1].item(), original_dims[0]-max_indices[1].item(),
            min_indices[2].item(), original_dims[1]-max_indices[2].item(),
            min_indices[3].item(), original_dims[2]-max_indices[3].item())

def cropData(dataElement):
    downsampled = segResize(dataElement)
    originalSize = dataElement[0].size()
    rescale = tio.Resize(originalSize)
    mask = segModel(downsampled.unsqueeze(0))
    mask = torch.argmax(mask, dim=1)
    mask = rescale(mask)
    bounding_prism = boundingVolume(mask,originalSize)
    crop = tio.Crop(bounding_prism)
    cropped = crop(dataElement)
    return classResize(cropped)

smartCrop = tio.Lambda(cropData,types_to_apply=[tio.INTENSITY])

def cacheFunc(data, indexes):

    return data[indexes]

cacheFunc = memory.cache(cacheFunc)

flip = tio.RandomFlip(axes=('LR'))
aniso = tio.RandomAnisotropy()
noise = tio.RandomNoise()
augmentations = tio.Compose([flip, aniso, noise])

class cachingDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return augmentations(cacheFunc(self.dataset,idx))


# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

root_dir="./"
if torch.cuda.is_available():
     print("GPU enabled")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

target_cols = ['C1', 'C2', 'C3',
               'C4', 'C5', 'C6', 'C7',
               'patient_overall', 'none']

# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
competition_weights = {
    '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7, 1], dtype=torch.float, device=device),
    '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14, 1], dtype=torch.float, device=device),
}

# y_hat.shape = (batch_size, num_classes)
# y.shape = (batch_size, num_classes)

# with row-wise weights normalization (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344565)
def competiton_loss_row_norm(y_hat, y):
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1)
    w_sum = weights.sum(axis=1)
    loss = torch.div(loss, w_sum)
    return loss.mean()

dataset = kaggleDataLoader.KaggleDataLoader()

train, val = dataset.loadDatasetAsSegmentor(train_aug=smartCrop)

train = cachingDataset(train)
val = cachingDataset(val)
# train_loader = DataLoader(
#     train, batch_size=1, shuffle=True, prefetch_factor=8, persistent_workers=True, drop_last=True, num_workers=16)
# val_loader = DataLoader(
#     val, batch_size=1, num_workers=16)
#
train_loader = DataLoader(
    train, batch_size=1, shuffle=True, num_workers=0)
val_loader = DataLoader(
    val, batch_size=1, num_workers=0)

N_EPOCHS = 200
model = DenseNet201(spatial_dims=3, in_channels=1, out_channels=9).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler = amp.GradScaler()

val_interval = 1
loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = np.inf
writer = SummaryWriter()
#https://www.kaggle.com/code/samuelcortinhas/rnsa-3d-model-train-pytorch
#Loop over epochs
for epoch in tqdm(range(N_EPOCHS)):
    loss_acc = 0
    val_loss_acc = 0
    train_count = 0
    valid_count = 0

    # Loop over batches
    for batch in train_loader:
        # Send to device
        imgs = batch['ct']['data']

        labels = torch.FloatTensor([[batch[target_col][line] for target_col in target_cols] for line in range(0,len(batch['C1']))])
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        # Forward pass
        with amp.autocast(dtype=torch.float16):
            preds = model(imgs)
            L = competiton_loss_row_norm(preds, labels)

        # Backprop
        scaler.scale(L).backward()
        scaler.step(optimizer)
        scaler.update()

        # # Backprop
        # L.backward()
        # # Update parameters
        # optimizer.step()
        # #
        # # Zero gradients
        # optimizer.zero_grad()

        #Track loss
        loss_acc += L.detach().item()
        train_count += 1
        print("finished batch")
    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        actual = []
        pred = []
        for batch in val_loader:
            # Reshape
            val_imgs = batch['ct']['data']
            val_labels = torch.FloatTensor([[batch[target_col][line] for target_col in target_cols] for line in range(0,len(batch['C1']))])

            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            val_preds = model(val_imgs)
            val_L = competiton_loss_row_norm(val_preds, val_labels)
            pred.append(torch.argmax(val_preds, dim=1).item())
            actual.append(torch.argmax(val_labels, dim=1).item())
            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1
            print("finished validation batch")

        output_valid = classification_report(actual, pred, output_dict=True, target_names=target_cols)
        print(output_valid)
        # Save loss history
        loss_hist.append(loss_acc / train_count)
        val_loss_hist.append(val_loss_acc / valid_count)
        for key in output_valid:
            if isinstance(output_valid[key], dict):
                for key1 in output_valid[key]:
                    if key1 != "support":
                        scaler_tag = {"valid": output_valid[key][key1]}
                        writer.add_scalars(f"{key}/{key1}", scaler_tag, epoch + 1)
            else:
                scaler_tag = {"valid": output_valid[key]}
                writer.add_scalars(key, scaler_tag, epoch + 1)
        writer.add_scalar("train_loss", loss_acc / train_count,epoch + 1)
        writer.add_scalar("val_loss", val_loss_acc / valid_count, epoch + 1)

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {val_loss_acc / valid_count:.5f}')

    # Save model (& early stopping)
    torch.save(model, str("classifier_DenseNet201_" + str(epoch)+".pt"))

writer.close()
print('')
print('Training complete!')
# log loss
data = {'val_loss':val_loss_hist,'loss':loss_hist}
df = pd.DataFrame(data=data)
df.to_csv("results.csv", sep='\t')

# Plot loss
plt.figure(figsize=(10, 5))
plt.plot(loss_hist, c='C0', label='loss')
plt.plot(val_loss_hist, c='C1', label='val_loss')
plt.title('Competition metric')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("train_result.png")
plt.show()
