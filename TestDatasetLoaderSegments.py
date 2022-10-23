#unit test the dataset loader to make sure its working properly

import kaggleDataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

dataLoader = kaggleDataLoader.KaggleDataLoader()

train, val = dataLoader.loadDatasetAsSegmentor()

subject1 = train[0]

subject1.plot()

fig, ax = plt.subplots()
ims = []
for sagittal_slice_tensor in subject1.seg.data[0]:
    im = ax.imshow(sagittal_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

fig, ax = plt.subplots()
ims = []
for coronal_slice_tensor in subject1.seg.data[0].permute(1,2,0):
    im = ax.imshow(coronal_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

fig, ax = plt.subplots()
ims = []
for axial_slice_tensor in subject1.seg.data[0].permute(2,1,0):
    im = ax.imshow(axial_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

