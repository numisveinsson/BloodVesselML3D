## Python script to go through samples, image data and segs and review their accuracy

import SimpleITK as sitk
from modules import sitk_functions
import os
import matplotlib.pyplot as plt
import numpy as np

def myshow(img, seg, title=None, margin=0.05):
    nda = sitk.GetArrayFromImage(img)
    sda = sitk.GetArrayFromImage(seg)

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

def view_volumes_compare(files, index, directory, directory_mask):

    troublesome = []
    for file in files[index-1:]:

        try:
            print("\n"+file+"\n")
            img = sitk.ReadImage(directory+file)
            seg = sitk.ReadImage(directory_mask+file)

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

            except:
                img = myshow3d(img, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))
                seg = myshow3d(seg, xslices=range(0,size[0],1), yslices=range(0,size[1],1), zslices=range(0,size[2],1))

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
        except:
            print("\nError: Not possible to read " + file + "\n")

    return troublesome


if __name__=='__main__':
    #set directories

    # output_directory = '/Users/numisveinsson/Downloads/output_test1/'
    # files = [f for f in os.listdir(output_directory) if f.endswith('.nii.gz')]
    # view_volumes_compare(files, 0, output_directory, output_directory)
    #
    # import pdb; pdb.set_trace()

    directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train/')
    val_directory = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val/')

    directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_train_masks/')
    val_directory_mask = str('/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/ct_val_masks/')

    directory = str('/Users/numisveinsson/Downloads/test_images/ct_train/')
    directory_mask = str('/Users/numisveinsson/Downloads/test_images/ct_train_masks/')
    files = [f for f in os.listdir(directory) if f.endswith('.nii.gz')]

    # calc_stats(files, directory, directory_mask)
    #
    # import pdb; pdb.set_trace()

    # index = files.index("0001_0001_21.nii.gz")
    # print("Index is: " + str(index))

    calc_plot_stats('mean', files, directory, directory_mask)
    calc_plot_stats('max', files, directory, directory_mask)
    calc_plot_stats('min', files, directory, directory_mask)
    calc_plot_stats('std', files, directory, directory_mask)
    import pdb; pdb.set_trace()
    index = 0
    trouble = view_volumes_compare(files, index, directory, directory_mask)
