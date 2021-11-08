import time
start_time = time.time()

import sys
import os
sys.path.append("/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D")

import argparse
import numpy


from Data.modules import sitk_functions as sf
from Data.modules import vtk_functions as vf
from prediction import Prediction
from src.model import UNet3DIsensee



def import_initial(file_dir):

    return


def trace_centerline(image_file, output_folder, model_folder, modality, img_shape, threshold, stepsize, case, point, radius):

    try:
        os.mkdir(output_folder+'volumes')
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'predictions')
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'centerlines')
    except Exception as e: print(e)
    try:
        os.mkdir(output_folder+'surfaces')
    except Exception as e: print(e)

    reader_im, origin_im, size_im, spacing_im = sf.import_image(image_file)

    model = UNet3DIsensee((img_shape[0], img_shape[1], img_shape[2], 1), num_class=1)
    unet = model.build()
    model_name = os.path.realpath(model_folder) + '/weights_unet.hdf5'
    unet.load_weights(model_name)

    for i in range(0,1):

        size_extract, index_extract = sf.map_to_image(point, radius, size_volume, origin_im, spacing_im)

        cropped_volume = sf.extract_volume(reader_im, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())

        volume_fn = output_folder + 'volumes/' + 'volume_'  +case+ '_' +str(i)+ '.vtk'
        sitk.WriteImage(cropped_volume, volume_fn)

        predict = Prediction(unet, model_name, modality, volume_fn ,None, img_shape, out_fn=output_folder+'predictions', d_weights=None, write_intermediate=False)
        predict.volume_prediction(1, threshold)
        predict.resample_prediction()
        predict.write_prediction()



if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--image',  help='Name of the folder containing the image data')
    parser.add_argument('--output',  help='Name of the output folder')
    parser.add_argument('--model',  help='Name of the folders containing the trained model weights')
    parser.add_argument('--modality', nargs='+', help='Name of the modality, mr, ct, split by space')
    parser.add_argument('--size', nargs='+', type=int, help='Size of the image to rescale to, split by space')
    parser.add_argument('--threshold', type=float, help='Threshold for prediction')
    parser.add_argument('--stepsize',  help='Step size along cropped centerlines, prop to radius')
    parser.add_argument('--case', nargs='+', help='Name of the case, eg 0146_1001')
    parser.add_argument('--seed',  help='Inital seed point')
    parser.add_argument('--radius',  help='Inital radius of vessel at seed point')

    args = parser.parse_args()
    print('Finished parsing...')

    import pdb; pdb.set_trace()
    cent_fn = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/centerlines/'+args.case[0]+'.vtp'
    initial_seed, initial_radius = vf.get_seed(cent_fn, 1, 100)

    import pdb; pdb.set_trace()

    trace_centerline(args.image, args.output, args.model, args.modality, args.size, args.threshold, args.stepsize, args.case, initial_seed, initial_radius)
