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

def cacheFunc(data, indexes):

    return data[indexes]

cacheFunc = memory.cache(cacheFunc)

flip = tio.RandomFlip()
affine = tio.RandomAffine()
gamma = tio.RandomGamma(0.5)
aniso = tio.RandomAnisotropy(p=0.25)
noise = tio.RandomNoise(p=0.25)
augmentations = tio.Compose([flip, affine, aniso, noise, gamma])

class cachingDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        batch = cacheFunc(self.dataset, idx)
        return augmentations(batch)


# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

root_dir="./"
if torch.cuda.is_available():
     print("GPU enabled")
device = torch.device('cuda:0,1' if torch.cuda.is_available() else 'cpu')

target_cols = ['C1', 'C2', 'C3',
               'C4', 'C5', 'C6', 'C7',
               'patient_overall']

# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
competition_weights = {
    '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
    '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
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

train, val = dataset.loadDatasetAsClassifier()

train = cachingDataset(train)
val = cachingDataset(val)
train_loader = DataLoader(
    train, batch_size=8, shuffle=True, prefetch_factor=16, persistent_workers=True, drop_last=True, num_workers=24)
val_loader = DataLoader(
    val, batch_size=4, num_workers=24)

# train_loader = DataLoader(
#     train, batch_size=1, shuffle=True, num_workers=0)
# val_loader = DataLoader(
#     val, batch_size=1, num_workers=0)

N_EPOCHS = 200
model = DenseNet201(spatial_dims=3, in_channels=1, out_channels=8).to(device)
model = nn.DataParallel(model)
model.to(device)

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
        optimizer.zero_grad()

        #Track loss
        loss_acc += L.detach().item()
        train_count += 1
        print("finished batch " + str(train_count))
    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        for batch in val_loader:
            # Reshape
            val_imgs = batch['ct']['data']
            val_labels = torch.FloatTensor([[batch[target_col][line] for target_col in target_cols] for line in range(0,len(batch['C1']))])

            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            with amp.autocast(dtype=torch.float16):
                val_preds = model(val_imgs)
                val_L = competiton_loss_row_norm(val_preds, val_labels)

            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1
            print("finished validation batch")

        # Save loss history
        loss_hist.append(loss_acc / train_count)
        val_loss_hist.append(val_loss_acc / valid_count)

        writer.add_scalar("train_loss", loss_acc / train_count,epoch + 1)
        writer.add_scalar("val_loss", val_loss_acc / valid_count, epoch + 1)

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {val_loss_acc / valid_count:.5f}')

    # Save model (& early stopping)
    torch.save(model, str("classifier__dist_DenseNet201_" + str(epoch)+".pt"))

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
