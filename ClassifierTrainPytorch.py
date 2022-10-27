import kaggleDataLoader
import math
import time
import matplotlib as plt
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.optim import lr_scheduler

from monai.data import decollate_batch, DataLoader,Dataset,ImageDataset
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121

# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

root_dir="./"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

target_cols = ['C1', 'C2', 'C3',
               'C4', 'C5', 'C6', 'C7',
               'patient_overall']


# Replicate competition metric (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854)
loss_fn = nn.BCEWithLogitsLoss(reduction='none')

competition_weights = {
    '-' : torch.tensor([1, 1, 1, 1, 1, 1, 1, 7], dtype=torch.float, device=device),
    '+' : torch.tensor([2, 2, 2, 2, 2, 2, 2, 14], dtype=torch.float, device=device),
}

# y_hat.shape = (batch_size, num_classes)
# y.shape = (batch_size, num_classes)

# with row-wise weights normalization (https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/344565)
def competiton_loss_row_norm(y_hat, y):
    loss = loss_fn(y_hat, y.to(y_hat.dtype))
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    loss = (loss * weights).sum(axis=1)
    w_sum = weights.sum(axis=1)
    loss = torch.div(loss, w_sum)
    return loss.mean()

dataset = kaggleDataLoader.KaggleDataLoader()
train, val = dataset.loadDatasetAsClassifier()

sample = train[0]
train_loader = DataLoader(
    train, batch_size=2, shuffle=True, num_workers=8)

val_loader = DataLoader(
    val, batch_size=1, num_workers=0)

n_epochs = 10
model = DenseNet121(spatial_dims=3, in_channels=1, out_channels=8).to(device)
optimizer = torch.optim.Adam(model.parameters(), 1e-5)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
val_interval = 1
auc_metric = ROCAUCMetric()

N_EPOCHS = 20
PATIENCE = 3

loss_hist = []
val_loss_hist = []
patience_counter = 0
best_val_loss = np.inf
#https://www.kaggle.com/code/samuelcortinhas/rnsa-3d-model-train-pytorch
# Loop over epochs
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
        preds = model(imgs)
        L = competiton_loss_row_norm(preds, labels)

        del imgs
        # Backprop
        L.backward()

        # Update parameters
        optimizer.step()

        # Zero gradients
        optimizer.zero_grad(set_to_none=True)

        # Track loss
        loss_acc += L.detach().item()
        train_count += 1

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
            val_preds = model(val_imgs)
            val_L = competiton_loss_row_norm(val_preds, val_labels)

            # Track loss
            val_loss_acc += val_L.item()
            valid_count += 1

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
        }, "Conv3DNet.pt")
    else:
        patience_counter += 1

        if patience_counter == PATIENCE:
            break

print('')
print('Training complete!')

# Plot loss
plt.figure(figsize=(10,5))
plt.plot(loss_hist, c='C0', label='loss')
plt.plot(val_loss_hist, c='C1', label='val_loss')
plt.title('Competition metric')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
