#unit test the dataset loader to make sure its working properly

import kaggleDataLoader

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation

dataLoader = kaggleDataLoader.KaggleDataLoader()

testID = dataLoader.listTrainPatientID()[0]
breaks = dataLoader.fracturedBones(testID)
print(breaks)

# cv, imgSlice = dataLoader.loadSliceImageFromId(testID, 1)
#
# imSlicePxArr = imgSlice.pixel_array
# fig, ax = plt.subplots(1,1, figsize=(10, 10))
#
# ax.imshow(imSlicePxArr)
# plt.show()
#
# id1 = '1.2.826.0.1.3680043.780'
#
# segment = dataLoader.loadSegmentationsForPatient(id1)
# temp = segment[0,:,:]
# for i in range(segment.shape[0]):
#     temp = temp + segment[i,:,:]
#
# fig, ax = plt.subplots(1,1, figsize=(10, 10))
# ax.imshow(temp)
# plt.show()
#
# id1 = '1.2.826.0.1.3680043.10051'
# slice_number = 136
#
# cv, imgSlice = dataLoader.loadSliceImageFromId(id1, slice_number)
# imSlicePxArr = imgSlice.pixel_array
# bboxes = dataLoader.bboxFromIndex(id1, slice_number)
# rect = patches.Rectangle((bboxes[1], bboxes[2]), bboxes[3], bboxes[4], linewidth=2, edgecolor='r', facecolor="none")
# fig, ax = plt.subplots(1,1, figsize=(10, 10))
# ax.imshow(imSlicePxArr)
# ax.add_patch(rect)
# plt.show()

train, val = dataLoader.loadDatasetAsClassifier()

subject1 = train[0]

fig, ax = plt.subplots()
ims = []
for sagittal_slice_tensor in subject1.ct.data[0]:
    im = ax.imshow(sagittal_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

fig, ax = plt.subplots()
ims = []
for coronal_slice_tensor in subject1.ct.data[0].permute(1,2,0):
    im = ax.imshow(coronal_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()

fig, ax = plt.subplots()
ims = []
for axial_slice_tensor in subject1.ct.data[0].permute(2,1,0):
    im = ax.imshow(axial_slice_tensor.numpy(), cmap=plt.cm.bone, animated=True)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
plt.show()
