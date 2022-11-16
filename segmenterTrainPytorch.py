import kaggleDataLoader
import json
from joblib import Memory
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.cuda.amp as amp
import pandas as pd
from monai.data import decollate_batch, DataLoader,Dataset,ImageDataset
from monai.metrics import ROCAUCMetric
from monai.losses.dice import DiceLoss
from monai.networks.nets import BasicUNetPlusPlus

with open('config.json', 'r') as f:
    paths = json.load(f)

cachedir = paths["CACHE_DIR"]
memory = Memory(cachedir, verbose=0, compress=True)

def cacheFunc(data, indexes):
    return data[indexes]

cacheFunc = memory.cache(cacheFunc)

class cachingDataset(Dataset):

    def __init__(self, data):
        self.dataset = data

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return cacheFunc(self.dataset,idx)


# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

root_dir="./"
if torch.cuda.is_available():
     print("GPU enabled")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = kaggleDataLoader.KaggleDataLoader()
train, val = dataset.loadDatasetAsSegmentor()

train = cachingDataset(train)
val = cachingDataset(val)
train_loader = DataLoader(
    train, batch_size=1, shuffle=True, prefetch_factor=4, persistent_workers=True, drop_last=True, num_workers=16)

val_loader = DataLoader(
    val, batch_size=1, num_workers=8)

N_EPOCHS = 500
model = BasicUNetPlusPlus(spatial_dims=3,
                          in_channels=1,
                          out_channels=1).to(device)

scaler = torch.cuda.amp.GradScaler()
optimizer = torch.optim.Adam(model.parameters(), 1e-3)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
loss = DiceLoss(reduction='none')
val_interval = 1

auc_metric = ROCAUCMetric()

PATIENCE = 5

loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = np.inf
#https://www.kaggle.com/code/samuelcortinhas/rnsa-3d-model-train-pytorch
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
        imgs = batch['ct']['data']

        labels = batch['seg']['data']
        # Send to device
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Forward pass
        with amp.autocast():
            preds = model(imgs)
            L = loss(preds, labels)
        # Backprop
        scaler.scale(L).backward()
        scaler.step(optimizer)
        # Update parameters
        scaler.update()


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
            val_L = loss(val_preds, val_labels)
            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1
            print("finished validation batch")

    loss_acc = abs(loss_acc)
    val_loss_acc = abs(val_loss_acc)
    # Save loss history
    loss_hist.append(loss_acc / train_count)
    val_loss_hist.append(val_loss_acc / valid_count)

    # Print loss
    if (epoch + 1) % 1 == 0:
        print(
            f'Epoch {epoch + 1}/{N_EPOCHS}, loss {loss_acc / train_count:.5f}, val_loss {val_loss_acc / valid_count:.5f}')

    # Save model (& early stopping)
    if (val_loss_acc / valid_count) < best_val_loss:
        best_val_loss = val_loss_acc / valid_count
        patience_counter = 0
        print('Valid loss improved --> saving model')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimiser_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': loss_acc / train_count,
            'val_loss': val_loss_acc / valid_count,
        }, "Unet3D.pt")

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
