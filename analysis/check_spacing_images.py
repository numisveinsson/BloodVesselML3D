import SimpleITK as sitk
import os
import numpy as np


def check_spacing_images(data_folder, input_format='.mha'):
    """
    Check the spacing of images in a folder and print the spacing and size of each image.
    :param data_folder: Folder containing the images
    :param input_format: Format of the images (default: .mha)
    """

    spacings = []
    sizes = []

    imgs = os.listdir(data_folder)
    imgs = [f for f in imgs if f.endswith(input_format)]

    # sort the files
    imgs = sorted(imgs)

    print(f'Found {len(imgs)} images to check')
    print(f'Images to check: {imgs}')

    for img in imgs:
        img_path = os.path.join(data_folder, img)

        # read the image
        img_sitk = sitk.ReadImage(img_path)

        # print(f'Image {img} read')
        # print(f"Image {img} shape: {img_sitk.GetSize()}")
        print(f"Image {img} spacing: {img_sitk.GetSpacing()}")
        # print(f"Image {img} origin: {img_sitk.GetOrigin()}")
        # print(f"Image {img} direction: {img_sitk.GetDirection()}")

        spacings.append(img_sitk.GetSpacing())
        sizes.append(img_sitk.GetSize())

    spacings = np.array(spacings)
    max_spacings = np.max(spacings, axis=1)
    ratio_between_max_and_min = max_spacings / np.min(spacings, axis=1)
    sizes = np.array(sizes)
    print(f'Average spacing: {np.mean(spacings, axis=0)}')
    print(f'Minimum spacing: {np.min(spacings, axis=0)}')
    print(f'Maximum spacing: {np.max(spacings, axis=0)}')
    print(f'Standard deviation of spacing: {np.std(spacings, axis=0)}')

    print(f'Average size: {np.mean(sizes, axis=0)}')
    print(f'Minimum size: {np.min(sizes, axis=0)}')
    print(f'Maximum size: {np.max(sizes, axis=0)}')

    import matplotlib.pyplot as plt
    # import seaborn as sns
    # sns.set(style="whitegrid")
    # plt.figure(figsize=(10, 6))
    # plt.title('Spacing of images')
    # plt.xlabel('Image index')
    # plt.ylabel('Spacing [cm]')
    # plt.plot(spacings)
    # plt.legend(['x', 'y', 'z'])
    # plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.title('Size of images')
    # plt.xlabel('Image index')
    # plt.ylabel('Size')
    # plt.plot(sizes)
    # plt.legend(['x', 'y', 'z'])
    # plt.show()

    plt.figure(figsize=(10, 6))
    plt.title('Max spacing of images')
    plt.xlabel('Image index')
    plt.ylabel('Spacing [cm]')
    plt.scatter(np.arange(len(max_spacings)), max_spacings)
    plt.grid()
    plt.show()

    # plt.figure(figsize=(10, 6))
    # plt.title('Ratio between max and min spacing of images')
    # plt.xlabel('Image index')
    # plt.ylabel('Ratio')
    # plt.scatter(np.arange(len(ratio_between_max_and_min)), ratio_between_max_and_min)
    # # grid
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    data_folder = '/Users/numisveins/Documents/datasets/vmr/images/'
    input_format = '.mha'  # '.mha' or '.vti'

    check_spacing_images(data_folder, input_format)