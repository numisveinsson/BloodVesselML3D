## Python script to go through samples, image data and segs and review their accuracy

import SimpleITK as sitk
from modules import sitk_functions
import os
import matplotlib.pyplot as plt
import numpy as np

def myshow(img, seg, pred=None, title=None, margin=0.05):
    nda = sitk.GetArrayFromImage(img)
    sda = sitk.GetArrayFromImage(seg)
    if pred:
        pda = sitk.GetArrayFromImage(pred)

    spacing = img.GetSpacing()

    if nda.ndim == 3:
        # fastest dim, either component or x
        c = nda.shape[-1]

        # the the number of components is 3 or 4 consider it an RGB image
        if not c in (3,4):
            nda = nda[nda.shape[0]//2,:,:]

    elif nda.ndim == 4:
        c = nda.shape[-1]

        if not c in (3,4):
            raise Runtime("Unable to show 3D-vector Image")

        # take a z-slice
        nda = nda[nda.shape[0]//2,:,:,:]

    ysize = nda.shape[0]
    xsize = nda.shape[1]

    if pred:
        f, axarr = plt.subplots(3,1)
        axarr[0].imshow(nda)
        axarr[1].imshow(sda)
        axarr[2].imshow(pda)
    else:
        f, axarr = plt.subplots(2,1)
        axarr[0].imshow(nda)
        axarr[1].imshow(sda)

    #if nda.ndim == 2:
        #f.set_cmap("gray")

    if(title):
        plt.title(title)



def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None, margin=0.05):
    size = img.GetSize()
    img_xslices = [img[s,:,:] for s in xslices]
    img_yslices = [img[:,s,:] for s in yslices]
    img_zslices = [img[:,:,s] for s in zslices]

    maxlen = max(len(img_xslices), len(img_yslices), len(img_zslices))


    img_null = sitk.Image([0,0], img.GetPixelIDValue(), img.GetNumberOfComponentsPerPixel())

    img_slices = []
    d = 0

    if len(img_xslices):
        img_slices += img_xslices + [img_null]*(maxlen-len(img_xslices))
        d += 1

    if len(img_yslices):
        img_slices += img_yslices + [img_null]*(maxlen-len(img_yslices))
        d += 1

    if len(img_zslices):
        img_slices += img_zslices + [img_null]*(maxlen-len(img_zslices))
        d +=1

    if maxlen != 0:
        if img.GetNumberOfComponentsPerPixel() == 1:
            img = sitk.Tile(img_slices, [maxlen,d])
        #TODO check in code to get Tile Filter working with VectorImages
        else:
            img_comps = []
            for i in range(0,img.GetNumberOfComponentsPerPixel()):
                img_slices_c = [sitk.VectorIndexSelectionCast(s, i) for s in img_slices]
                img_comps.append(sitk.Tile(img_slices_c, [maxlen,d]))
            img = sitk.Compose(img_comps)

    return img

def calc_plot_stats(metric, files, directory, directory_mask):

    stat_im = []
    stat_seg = []
    for file in files:
        img = sitk.ReadImage(directory+file)
        seg = sitk.ReadImage(directory_mask+file)

        im_array = sitk.GetArrayFromImage(img)
        seg_array = sitk.GetArrayFromImage(seg)

        if metric == 'mean':
            stat_im.append(np.mean(im_array))
            stat_seg.append(np.mean(seg_array))
        elif metric == 'std':
            stat_im.append(np.std(im_array))
            stat_seg.append(np.std(seg_array))
        elif metric == 'max':
            stat_im.append(np.max(im_array))
            stat_seg.append(np.max(seg_array))
        elif metric == 'min':
            stat_im.append(np.min(im_array))
            stat_seg.append(np.min(seg_array))
        else:
            print('Invalid metric input')


    fig1, ax1 = plt.subplots()
    ax1.set_title('Image data - ' +metric)
    ax1.boxplot(np.array(stat_im))
    plt.show()
    import pdb; pdb.set_trace()
    fig2, ax2 = plt.subplots()
    ax2.set_title('Segmentation data - ' +metric)
    ax2.boxplot(np.array(stat_seg))
    plt.show()

def calc_stats(files, directory, directory_mask):
    for file in files:
        #print("\n"+file+"\n")
        img = sitk.ReadImage(directory+file)
        seg = sitk.ReadImage(directory_mask+file)

        im_array = sitk.GetArrayFromImage(img)
        seg_array = sitk.GetArrayFromImage(seg)

        if np.mean(seg_array) > 0.6:
            print('BINGO')
            print(file)
            print('the mean of seg is {}'.format(np.mean(seg_array)))
            #sitk.Show(seg)

            size = img.GetSize()
            print("Size is "+str(size))

            num_im = min(size)
            size_array = np.array(size)
            size_prop = size_array//num_im
            size = size_array//size_prop
            try:
                try:
                    img = myshow3d(img, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))
                    seg = myshow3d(seg, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))

                except:
                    img = myshow3d(img, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                    seg = myshow3d(seg, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
            except:
                print("ERROR")
            myshow(img, seg)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()
        #print('the mean of image is {}'.format(np.mean(im_array)))
        #print('the mean of seg is {}'.format(np.mean(seg_array)))
        #
        # print('the max of image is {}'.format(np.max(im_array)))
        # print('the max of seg is {}'.format(np.max(seg_array)))
        #
        # print('the min of image is {}'.format(np.min(im_array)))
        # print('the min of seg is {}'.format(np.min(seg_array)))

def view_volumes_compare(files, index, directory, directory_mask, directory_pred = None):

    troublesome = []
    for file in files[index:]:

        try:
            print("\n"+file+"\n")
            img = sitk.ReadImage(directory+file)
            seg = sitk.ReadImage(directory_mask+file)
            if directory_pred:
                pred = sitk.ReadImage(directory_pred+file)

            size = img.GetSize()
            print("Size is "+str(size))

            num_im = min(size)
            size_array = np.array(size)
            size_prop = size_array//num_im
            size = size_array//size_prop

            #sitk.Show(img, title="Image before "+file, debugOn=True)
            try:
                img = myshow3d(img, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))
                seg = myshow3d(seg, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))
                if directory_pred:
                    pred = myshow3d(pred, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))

            except:
                img = myshow3d(img, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                seg = myshow3d(seg, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                if directory_pred:
                    pred = myshow3d(pred, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
            try:
                myshow(img, seg, pred)
            except:
                myshow(img, seg)
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            value = input("Keep this one? default is yes; no, end, pdb \n")
            v1 = str(value)

            if v1 == "no":
                troublesome.append(directory_mask+file)
                print("Moved to trouble")
            elif v1 == "end":
                print("\nYou are ending the session \n")
                print("Trouble is now:\n")
                for i in troublesome:
                    print(i)
                break

            elif v1 == "pdb":
                import pdb; pdb.set_trace()
            else:
                print("Kept in\n")
                continue
        except Exception as e:

            print("\nError: Not possible to read " + file + "\n")
            print(e)
    return troublesome


if __name__=='__main__':
    #set directories

    image_directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val/'
    output_directory = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output/test7/pred/'
    truth_directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_masks/'
    files = [f for f in os.listdir(output_directory) if f.endswith('.nii.gz')]
    files = ['0092_0001_59.nii.gz',
            '0148_1001_153.nii.gz',
            '0185_0001_65.nii.gz',
            '0172_0001_164.nii.gz',
            '0186_0002_137.nii.gz',
            '0140_2001_45.nii.gz',
            '0189_0001_129.nii.gz',
            '0188_0001_288.nii.gz',
            '0183_1002_531.nii.gz',
            '0140_2001_148.nii.gz',
            '0065_0001_201.nii.gz',
            '0189_0001_386.nii.gz',
            '0186_0002_135.nii.gz',
            '0183_1002_533.nii.gz',
            '0140_2001_61.nii.gz',
            '0065_0001_139.nii.gz',
            '0184_0001_304.nii.gz',
            '0183_1002_733.nii.gz',
            '0139_1001_199.nii.gz',
            '0183_1002_867.nii.gz',
            '0065_0001_138.nii.gz',
            '0188_0001_533.nii.gz']
    view_volumes_compare(files, 0, image_directory, truth_directory, output_directory)
    #
    import pdb; pdb.set_trace()

    directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train/'
    directory_mask = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks/'

    val_directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val/'
    val_directory_mask = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_masks/'
    val_directory_pred = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_pred/pred6/'


    directory = '/Users/numisveinsson/Downloads/test_images/ct_train/'
    directory_mask = '/Users/numisveinsson/Downloads/test_images/ct_train_masks/'


    files = [f for f in os.listdir(val_directory_pred) if f.endswith('.nii.gz')]
    calc_plot_stats('min', files, val_directory_pred, val_directory_mask)
    import pdb; pdb.set_trace()

    # calc_stats(files, directory, directory_mask)

    # index = files.index("0001_0001_21.nii.gz")
    # print("Index is: " + str(index))

    index = 0
    trouble = view_volumes_compare(files, index, val_directory, val_directory_mask, val_directory_pred)

    # calc_plot_stats('mean', files, directory, directory_mask)
    # calc_plot_stats('max', files, directory, directory_mask)
    # calc_plot_stats('min', files, directory, directory_mask)
    # calc_plot_stats('std', files, directory, directory_mask)
