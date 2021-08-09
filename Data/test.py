import numpy as np
import os

from modules import vtk_functions as vf
from modules import sitk_functions as sf

import SimpleITK as sitk

file_dir_image = '/Users/numisveinsson/Desktop/IMG_1058.jpg'

reader_im = sf.read_image(file_dir_image)

size_im = list(reader_im.GetSize())
spacing_im = list(reader_im.GetSpacing())

new_img = sf.create_new(reader_im)

reader_im.SetExtractIndex([0,0])
reader_im.SetExtractSize([1500,1500])
new_img1 = reader_im.Execute()

sitk.Show(new_img1, title="Hello World: Python", debugOn=True)



#outputImageDir = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/'
#sitk.WriteImage(new_img, outputImageDir+'0002_0001_01_im.mha')

import pdb; pdb.set_trace()
