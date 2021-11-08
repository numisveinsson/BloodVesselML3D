## Python script to go through samples, image data and segs and review their accuracy

import SimpleITK as sitk
from modules import sitk_functions
import os
import matplotlib.pyplot as plt
import numpy as np

def myshow(img, seg, pred=None, title=None):
    """
    Function to display 2D sitk images together
    args:
        img: the 2D slices from image data
        seg: the 2D slices from segmentation data (ground truth)
        pred: the 2D slices from predicted segmentation
    return:
        img ready to display
    """
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

    if(title):
        plt.title(title)

def myshow3d(img, xslices=[], yslices=[], zslices=[], title=None):
    """
    Function to create sitk 2D image for viewing from a 3D image
    args:
        img: 3D Volume
        xslices, yslices, zslices: which individual slices to display
    return:
        img ready to display
    """
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
    """
    Function to calculate and plot metrics for samples, image and segmentation
    args:
        metric: 'mean', 'std', 'max', 'min', 'size'
    return:
        -
    """
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
        elif metric == 'size':
            stat_im.append(np.array(img.GetSpacing())[0] * np.array(img.GetSize())[0])
            stat_seg.append(np.array(img.GetSpacing())[0] * np.array(img.GetSize())[0])
        else:
            print('Invalid metric input')

    fig, ax = plt.subplots(1,2)
    ax[0].set_title('Image data - ' +metric)
    ax[0].boxplot(np.array(stat_im))
    ax[1].set_title('Segmentation data - ' +metric)
    ax[1].boxplot(np.array(stat_seg))
    plt.show()

def view_if_mean(files, over_under, mean_threshold, directory, directory_mask):
    """
    Function to view samples that have a mean over or under a threshold
    args:
        over_under = 'over' or 'under'
        mean_threshold = double value for threshold
    return:
        -
    """
    for file in files:
        #print("\n"+file+"\n")
        img = sitk.ReadImage(directory+file)
        seg = sitk.ReadImage(directory_mask+file)

        im_array = sitk.GetArrayFromImage(img)
        seg_array = sitk.GetArrayFromImage(seg)

        if over_under == 'over':
            condition = np.mean(seg_array) > mean_threshold
        elif over_under == 'under':
            condition = np.mean(seg_array) < mean_threshold

        if condition:
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
    """
    Function to go through a list of samples and compare image data, ground truth
    and (possibly) a predicted segmentation and
    Interactive classifying bad samples
    args:
        files: list of sample names
        index = where in the list to start
        directory_ = the directories to samples
    return:
        -
    """
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

            try: # First we try to display every 10th slice
                img = myshow3d(img, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))
                seg = myshow3d(seg, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))
                if directory_pred:
                    pred = myshow3d(pred, xslices=range(1,size[0],size[0]//10), yslices=range(1,size[1],size[1]//10), zslices=range(1,size[2],size[2]//10))

            except: # If total slices < 10, an error will occur and we display all slices instead
                img = myshow3d(img, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                seg = myshow3d(seg, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                if directory_pred:
                    pred = myshow3d(pred, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
            try:
                myshow(img, seg, pred)
            except:
                myshow(img, seg)

            # Data is displayed, mouse click to close
            plt.draw()
            plt.waitforbuttonpress(0)
            plt.close()

            # Interactive portion
            value = input("Keep this one? default is yes; write no to store as faulty sample; write end to end the session; write pdb to examine data \n")
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

    # image_directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val/'
    # output_directory = '/Users/numisveinsson/Documents/Berkeley/Research/BloodVessel_UNet3D/output/test7/pred/'
    # truth_directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_masks/'
    # files = [f for f in os.listdir(output_directory) if f.endswith('.nii.gz')]
    # files = ['0092_0001_59.nii.gz',
    #         '0148_1001_153.nii.gz',
    #         '0185_0001_65.nii.gz',
    #         '0172_0001_164.nii.gz',
    #         '0186_0002_137.nii.gz',
    #         '0140_2001_45.nii.gz',
    #         '0189_0001_129.nii.gz',
    #         '0188_0001_288.nii.gz',
    #         '0183_1002_531.nii.gz',
    #         '0140_2001_148.nii.gz',
    #         '0065_0001_201.nii.gz',
    #         '0189_0001_386.nii.gz',
    #         '0186_0002_135.nii.gz',
    #         '0183_1002_533.nii.gz',
    #         '0140_2001_61.nii.gz',
    #         '0065_0001_139.nii.gz',
    #         '0184_0001_304.nii.gz',
    #         '0183_1002_733.nii.gz',
    #         '0139_1001_199.nii.gz',
    #         '0183_1002_867.nii.gz',
    #         '0065_0001_138.nii.gz',
    #         '0188_0001_533.nii.gz']
    # view_volumes_compare(files, 0, image_directory, truth_directory, output_directory)
    #

    directory = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/test2/ct_train/'
    directory_mask = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/test2/ct_train_masks/'

    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    calc_plot_stats('size', files, directory, directory_mask)
    import pdb; pdb.set_trace()
    calc_plot_stats('max', files, val_directory_pred, val_directory_mask)


    #calc_stats(files, directory, directory_mask)

    # index = files.index("0001_0001_21.nii.gz")
    # print("Index is: " + str(index))

    index = 0
    trouble = view_volumes_compare(files, index, val_directory, val_directory_mask, val_directory_pred)


    # calc_plot_stats('max', files, directory, directory_mask)
    # calc_plot_stats('min', files, directory, directory_mask)
    # calc_plot_stats('std', files, directory, directory_mask)
