

from google.colab import drive
drive.mount('/content/drive')

import os
import glob
import numpy as np
import numpy as np
import nibabel as nib
import glob
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tifffile import imsave

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

#t1_list = sorted(glob.glob('BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/*/*t1.nii'))
t2_list = sorted(glob.glob('/content/drive/MyDrive/BraTS-Dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/*t2.nii'))
t1ce_list = sorted(glob.glob('/content/drive/MyDrive/BraTS-Dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/*t1ce.nii'))
flair_list = sorted(glob.glob('/content/drive/MyDrive/BraTS-Dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/*flair.nii'))
# mask_list = sorted(glob.glob('/content/drive/MyDrive/BraTS-Dataset/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData/*/*seg.nii'))

print(t2_list[0])
print(t1ce_list[0])
print(flair_list[0])

for img in range(len(t2_list)):   #Using t1_list as all lists are of same size
    print("Now preparing image and masks number: ", img)

    temp_image_t2=nib.load(t2_list[img]).get_fdata()
    temp_image_t2=scaler.fit_transform(temp_image_t2.reshape(-1, temp_image_t2.shape[-1])).reshape(temp_image_t2.shape)

    temp_image_t1ce=nib.load(t1ce_list[img]).get_fdata()
    temp_image_t1ce=scaler.fit_transform(temp_image_t1ce.reshape(-1, temp_image_t1ce.shape[-1])).reshape(temp_image_t1ce.shape)

    temp_image_flair=nib.load(flair_list[img]).get_fdata()
    temp_image_flair=scaler.fit_transform(temp_image_flair.reshape(-1, temp_image_flair.shape[-1])).reshape(temp_image_flair.shape)


    temp_combined_images = np.stack([temp_image_flair, temp_image_t1ce, temp_image_t2], axis=3)

    #Crop to a size to be divisible by 64 so we can later extract 64x64x64 patches.
    #cropping x, y, and z
    temp_combined_images=temp_combined_images[56:184, 56:184, 13:141]


    np.save('/content/drive/MyDrive/BraTS-Dataset/BraTS2020_ValidationData/input_data_3channels/images/image_'+str(img)+'.npy', temp_combined_images)

