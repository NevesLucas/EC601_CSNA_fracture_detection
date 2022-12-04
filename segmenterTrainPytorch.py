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
from monai.metrics import DiceMetric
from monai.losses.dice import DiceLoss
from monai.networks.nets import UNet, BasicUNet
from monai.networks.layers import Norm
from monai.visualize import plot_2d_or_3d_image
from monai.transforms import AsDiscrete
import torch.cuda.amp as amp
import torchio as tio

with open('config.json', 'r') as f:
    paths = json.load(f)

cachedir = paths["CACHE_DIR"]
memory = Memory(cachedir, verbose=0, compress=True)
resize = tio.Resize((128, 128, 200))
def cacheFunc(data, indexes):
    return resize(data[indexes])

cacheFunc = memory.cache(cacheFunc)

oneHot = tio.OneHot()
flip = tio.RandomFlip(axes=('LR'))
aniso = tio.RandomAnisotropy()
noise = tio.RandomNoise()

augmentations = tio.Compose([flip,aniso,noise,oneHot])
toDiscrete = AsDiscrete(argmax=True, to_onehot=2)

class cachingDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return augmentations(cacheFunc(self.dataset,idx))


root_dir="./"
if torch.cuda.is_available():
     print("GPU enabled")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = kaggleDataLoader.KaggleDataLoader()
train, val = dataset.loadDatasetAsSegmentor(trainPercentage=0.80)

train = cachingDataset(train)
val = cachingDataset(val)
train_loader = DataLoader(
    train, batch_size=1, shuffle=True, prefetch_factor=4, persistent_workers=True, drop_last=True, num_workers=16)

val_loader = DataLoader(
    val, batch_size=1, num_workers=16)

N_EPOCHS = 300
model = BasicUNet(spatial_dims=3,
                  in_channels=1,
                  features=(32, 64, 128, 256, 512, 32),
                  out_channels=2).to(device)

optimizer = torch.optim.Adam(model.parameters(), 1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler = amp.GradScaler()
loss = DiceLoss(softmax=True)
val_interval = 1
dice_metric = DiceMetric(include_background=False, reduction="mean")
PATIENCE = 10

loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = np.inf
batchCount = 0
#https://www.kaggle.com/code/samuelcortinhas/rnsa-3d-model-train-pytorch
writer = SummaryWriter()
#Loop over epochs
for epoch in tqdm(range(N_EPOCHS)):
    loss_acc = 0
    val_loss_acc = 0
    train_count = 0
    valid_count = 0

    # Loop over batches
    for batch in train_loader:
        # Zero gradients
        optimizer.zero_grad()
        # Send to device
        imgs = batch['ct']['data']

        labels = batch['seg']['data']

        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        with amp.autocast(dtype=torch.float16):
             preds = model(imgs)
             L = loss(preds, labels)

        # Backprop
        scaler.scale(L).backward()
        scaler.step(optimizer)
        scaler.update()
#        L.backward()
        # Update parameters
#        optimizer.step()

        # Track loss
        loss_acc += L.detach().item()
        train_count += 1
        print("finished batch")
    # Update learning rate
    scheduler.step()

    # Don't update weights
    with torch.no_grad():
        # Validate
        for batch in val_loader:
            # Reshape
            val_imgs = batch['ct']['data']
            val_labels = batch['seg']['data']

            val_imgs = val_imgs.to(device)
            val_labels = val_labels.to(device)

            # Forward pass
            val_preds = model(val_imgs)
            va_preds = toDiscrete(val_preds[0])
            dice_metric(y_pred=val_preds, y=val_labels[0])
            # Track loss
            valid_count += 1
            print("finished validation batch")
        metric = dice_metric.aggregate().item()
        # reset the status for next validation round
        dice_metric.reset()
        val_loss_hist.append(metric)
        writer.add_scalar("val_mean_dice", metric, epoch + 1)
    loss_acc = abs(loss_acc)

    # Save loss history
    loss_hist.append(loss_acc / train_count)

    #tensorboard logging
    plot_2d_or_3d_image(val_imgs,epoch+1,writer,index=0,tag='image')
    plot_2d_or_3d_image(val_labels,epoch+1,writer,index=0,tag='GT')
    plot_2d_or_3d_image(val_preds,epoch+1,writer,index=0,tag='output')

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {metric:.5f}')

    # Save model (& early stopping)
    if (metric) < best_val_loss:
        best_val_loss = metric
        patience_counter = 0
        print('Valid loss improved --> saving model')
    torch.save(model, str("Unet3D_resized_128x128x200"+str(epoch)+".pt"))

writer.close()
print('')
print('Training complete!')
# log loss
data = {'val_loss':val_loss_hist,'loss':loss_hist}
df = pd.DataFrame(data=data)
df.to_csv("results.csv", sep='\t')

# Plot loss
plt.figure(figsize=(10,5))
plt.plot(loss_hist, c='C0', label='loss')
plt.plot(val_loss_hist, c='C1', label='val_loss')
plt.title('DiceLoss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig("train_result.png")
plt.show()