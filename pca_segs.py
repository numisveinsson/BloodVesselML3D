import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from modules.sitk_functions import read_image, read_image_numpy, create_new, create_new_from_numpy

def write_transformed_segmentations(segs, data_dir, img_ext, size = [20,20,20]):
    """
    Write transformed segmentations to file

    Args:
    segs: np array
        The transformed segmentations
    """
    # create a list of all the segmentations names
    segs_names = os.listdir(data_dir)
    segs_names = [f for f in segs_names if f.endswith(img_ext)]

    # create new folder for transformed segmentations
    transformed_dir = data_dir + 'transformed/'
    if not os.path.exists(transformed_dir):
        os.makedirs(transformed_dir)

    # create sitk images from the transformed segmentations
    for i in range(len(segs)):
        seg_np = segs[i].reshape(size[0], size[1], size[2])
        segs_img = sitk.GetImageFromArray(seg_np)
        sitk.WriteImage(segs_img, transformed_dir + segs_names[i].replace(img_ext, '_transformed' + '.mha'))

    return None

def create_array_of_flat_segmentations(data_dir, img_ext, new_size = [10,10,10]):
    """
    Create an array of flat segmentations
    """
    # create new folder for reampled segmentations
    resampled_dir = data_dir + 'resampled/'
    if not os.path.exists(resampled_dir):
        os.makedirs(resampled_dir)

    # create a list of all the segmentations
    segs = os.listdir(data_dir)
    segs = [f for f in segs if f.endswith(img_ext)]

    # only keep 100
    # segs = segs[:1000]

    # read in the segmentations
    segs_img = [sitk.ReadImage(data_dir+f) for f in segs]

    # resample to 20x20x20
    for i in range(len(segs)):
        segs_img[i] = resample_to_size(segs_img[i], new_size = new_size)

    # save the resampled segmentations
    for i in range(len(segs_img)):
        sitk.WriteImage(segs_img[i], resampled_dir + segs[i].replace(img_ext, '_resampled' + '.mha'))

    # convert the segmentations to numpy arrays
    segs = [sitk.GetArrayFromImage(f) for f in segs_img]

    # flatten the segmentations and create numpy array
    segs = [f.flatten() for f in segs]
    segs = np.array(segs)
    print(f"Shape of segs: {segs.shape}")

    return segs

def resample_to_size(img, new_size = [10,10,10]):
    """
    Function to resample an sitk image to new size
    """
    # Calculate new spacing
    new_spacing = [sz*spc/nsz for sz,spc,nsz in zip(img.GetSize(), img.GetSpacing(), new_size)]
    new_spacing = np.array(new_spacing)
    new_spacing = new_spacing.tolist()
    # print(f"Old size: {img.GetSize()}, New size: {new_size}")
    # print(f"Old spacing: {img.GetSpacing()}, New spacing: {new_spacing}")

    # Resample image
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    # Set Linear
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_img = resampler.Execute(img)

    # Make binary
    resampled_img = sitk.GetArrayFromImage(resampled_img)
    resampled_img[resampled_img>0.5] = 1
    resampled_img[resampled_img<=0.5] = 0
    resampled_img = sitk.GetImageFromArray(resampled_img)

    return resampled_img

def tsne_folder(data_dir, img_ext, n_components=2):
    """
    Perform t-SNE on a folder of segmentations
    """
    
    # create an array of flat segmentations
    segs = create_array_of_flat_segmentations(data_dir, img_ext)

    # create a t-SNE object
    tsne = TSNE(n_components=n_components)

    # fit the t-SNE to the segmentations
    embedding = tsne.fit_transform(segs)

    # Plot the data in the t-SNE space
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding[:, 0], embedding[:, 1], alpha=0.5)
    plt.title('t-SNE Visualization')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.show()

    # print information about the t-SNE
    print(f"Kullback-Leibler divergence: {tsne.kl_divergence_}")
    print(f"Number of iterations: {tsne.n_iter_}")
    print(f"Perplexity: {tsne.perplexity}")
    print(f"Random state: {tsne.random_state}")

    return segs, tsne


def pca_folder(data_dir, img_ext, n_components=2, new_size = [10,10,10]):
    """
    Perform PCA on a folder of segmentations

    Args:
    data_dir: str
        The directory containing the segmentations
    img_ext: str
        The file extension of the segmentations
    n_components: int
        The number of principal components to keep

    Returns:
    segs: numpy array
        The transformed segmentations
    pca: PCA object
        The PCA object
    """

    # create an array of flat segmentations
    segs = create_array_of_flat_segmentations(data_dir, img_ext, new_size = new_size)

    # create a PCA object
    pca = PCA(n_components=n_components)

    # fit the PCA to the segmentations
    pca.fit(segs)

    # print the explained variance
    print(f"Explained variance: {pca.explained_variance_ratio_}")
    print(f"Explained variance sum: {np.sum(pca.explained_variance_ratio_)}")
    print(f"Singular values: {pca.singular_values_}")
    print(f"Mean: {pca.mean_}")
    print(f"Noise variance: {pca.noise_variance_}")
    
    # plot singular values
    plt.plot(pca.singular_values_)
    plt.title('Singular Values')
    plt.xlabel('Component')
    plt.ylabel('Singular Value')
    plt.show()

    # plot explained variance
    plt.plot(pca.explained_variance_ratio_)
    plt.title('Explained Variance')
    plt.xlabel('Component')
    plt.ylabel('Explained Variance')
    plt.show()

    # transform the segmentations using the PCA
    values = pca.transform(segs)

    # transform the segmentations back to the original space
    segs = pca.inverse_transform(values)

    # write the transformed segmentations
    write_transformed_segmentations(segs, data_dir, img_ext, size = new_size)

    return segs, pca

def display_pca_component(pca, component_num, size = [10,10,10]):
    """
    Display a PCA component

    Args:
    pca: PCA object
        The PCA object
    component_num: int
    """

    # get the component
    component = pca.components_[component_num]

    # convert to image
    component = component.reshape(size[0], size[1], size[2])

    # create a sitk image
    component = sitk.GetImageFromArray(component)

    # write the image
    sitk.WriteImage(component, f'pca_component_{component_num}.mha')

    return component


if __name__ == "__main__":
    """
    This script takes in a dataset of segmentations and performs a PCA on the segmentations to find the principal axes of the segmentations.
    """

    data_folder = '/Users/numisveins/Documents/Automatic_Tracing_Data/asoca_rotated_centered_volumes/ct_train_masks/'
    img_ext = '.nii.gz'
    size = [10,10,10]

    # perform PCA on the segmentations
    segs, pca = pca_folder(data_folder, img_ext, n_components=100, new_size = size)

    # import pdb; pdb.set_trace()

    # display the first two PCA components, that is, the first two principal axes of the segmentations
    for i in range(20):
        display_pca_component(pca, i, size = size)

    # perform t-SNE on the segmentations
    # segs, tsne = tsne_folder(data_folder, img_ext, n_components=2)
    
