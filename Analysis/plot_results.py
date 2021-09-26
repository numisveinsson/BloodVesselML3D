import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import os

file = '/Users/numisveinsson/Downloads/ct_test.csv'

# Obtain dice scores from test
np_from_csv = pd.read_csv(file).values
dice_scores = np_from_csv[:,0]

# Open the tested images and save size data
dir = '/Users/numisveinsson/Downloads/test_images/ct_val/'

size = []
min_res = []
for file in [f for f in os.listdir(dir) if f.endswith('.nii.gz')]:
    ref_im =  sitk.ReadImage(dir+file)
    size.append( ref_im.GetSize()[0] * ref_im.GetSpacing()[0] )
    min_res.append(np.min(ref_im.GetSize()))

fig, axis = plt.subplots(2,1)
scatter = axis[0].scatter(size, dice_scores, linewidth=0.1)
axis[0].grid()
axis[0].set_ylabel('Dice score')
axis[0].set_xlabel('Volume size [cm]')
axis[0].set_title('Test 1 - Dice against Volume Size')

scatter = axis[1].scatter(min_res, dice_scores, linewidth=0.1)
axis[1].grid()
axis[1].set_ylabel('Dice score')
axis[1].set_xlabel('Min Dimension Resolution')
axis[1].set_title('Test 1 - Dice against Min Resolution')

fig, axis = plt.subplots(2,1)
scatter = axis[0].hist(size, 20)
axis[0].grid()
axis[0].set_ylabel('Count')
axis[0].set_xlabel('Volume size [cm]')
axis[0].set_title('Test 1 - Volume Size')

scatter = axis[1].hist(min_res, 20)
axis[1].grid()
axis[1].set_ylabel('Count')
axis[1].set_xlabel('Min Dimension Resolution')
axis[1].set_title('Test 1 - Min Resolution')
plt.show()
import pdb; pdb.set_trace()
