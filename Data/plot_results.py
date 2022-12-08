import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt
import pandas as pd
import os
from modules import io

def get_img_sizes(dir):

    size = []
    min_res = []
    for file in [f for f in os.listdir(dir) if f.endswith('.nii.gz')]:
        ref_im =  sitk.ReadImage(dir+file)
        size.append( ref_im.GetSize()[0] * ref_im.GetSpacing()[0] )
        min_res.append(np.min(ref_im.GetSize()))

    return size, min_res

def read_csv_file(file):

    dict = pd.read_csv(file)
    pd_array = dict.values

    return dict, pd_array

#def plot_histograms():


def plot_scatter(data, x_name, y_name, sizes, title, legend):

    fig, axis = plt.subplots(1,1)
    scatter = axis.scatter(data[0], data[1], linewidth=0.1, s = data[2])
    axis.grid()
    axis.set_ylabel(y_name)
    legend1 = axis.legend(legend ,loc = "upper right")
    axis.add_artist(legend1)
    kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="{x:.2f} cm",
          func=lambda s: s/100)
    legend = axis.legend(*scatter.legend_elements(**kw),
                    loc="upper left", title=x_name)
    plt.show()

def plot_histograms(data, x_name, y_name, title):

    # the histogram of the data
    n, bins, patches = plt.hist(data, 50, density=False, facecolor='g', alpha=0.75)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    #plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    #plt.xlim(40, 160)
    #plt.ylim(0, 0.03)
    plt.grid(True)
    plt.show()

def plot_3d_vector(data, x_name, y_name, title):

    from mpl_toolkits.mplot3d import axes3d

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)
    shape_ = data[0].shape
    ax.quiver(np.zeros(shape_), np.zeros(shape_), np.zeros(shape_),data[0], data[1], data[2], length=0.1)
    plt.show()

def organize_vectors(vectors):
    "Function to combine similar vectors, all of length 1 to begin with"

    keep_running = True
    while keep_running:
        ind_to_delete, mult = [], 1
        ind = arr=np.random.random(1)*(vectors.shape[0]-1)
        ind = ind[0].astype('int')
        vector = vectors[ind]
        if np.linalg.norm(vector)< 2:
            print("ind is: ", ind)
            for j in range(vectors.shape[0]):
                if j == ind:
                    continue
                if np.dot(vector, vectors[j]) > 0.99 and np.linalg.norm(vectors[j]<2):
                    #print('dot is ', np.dot(vector, vectors[j]))
                    mult += 1
                    ind_to_delete.append(j)
            vectors[ind] *= mult
            vectors = np.delete(vectors, ind_to_delete, axis = 0)
            print("now: ", vectors.shape[0])
        if vectors.shape[0] < 1000:
            break
    norms = 0
    for p in vectors:
        norm = np.linalg.norm(p)
        print()
    import pdb; pdb.set_trace()

    return vectors
if __name__=='__main__':

    global_config_file = "./config/global.yaml"
    global_config = io.load_yaml(global_config_file)
    out_dir = global_config['OUT_DIR']
    modality = global_config['MODALITY']
    csv_file = "_Sample_stats.csv"
    file = out_dir+modality+csv_file

    # Obtain dice scores from test
    dict_from_csv, np_from_csv = read_csv_file(file)

    sizes = dict_from_csv['SIZE']
    tangentx, tangenty, tangentz = dict_from_csv['TANGENTX'].values, dict_from_csv['TANGENTY'].values, dict_from_csv['TANGENTZ'].values
    radii = dict_from_csv['RADIUS']
    bifurc = dict_from_csv['BIFURCATION']
    num_bif = bifurc.values.sum()

    tangents = np.array([tangentx, tangenty, tangentz]).T

    tang_org = organize_vectors(tangents)

    #plot_scatter([sizes.values, tangenty.values, radii.values],'sizes', 'tangent_x', 'radii', 'Comparison', 'diff')

    import pdb; pdb.set_trace()

    plot_3d_vector([tangentx, tangenty, tangentz], x_name, y_name, title)

    plot_histograms(sizes.values, 'Subvolume Sidelength [cm]', 'Frequency', 'Histogram of Subvolume Sizes')

    plot_histograms(tangentx.values, 'Tangent - X', 'Frequency', 'Histogram of X component of Vessel Tangent')

    plot_histograms(tangenty.values, 'Tangent - Y', 'Frequency', 'Histogram of Y component of Vessel Tangent')

    plot_histograms(tangentz.values, 'Tangent - Z', 'Frequency', 'Histogram of Z component of Vessel Tangent')

    plot_histograms(radii.values, 'Vessel Radius [cm]', 'Frequency', 'Histogram of Vessel Radii in Samples')

    import pdb; pdb.set_trace()
    # Open the tested images and save size data
    dir = out_dir+modality+'_train/'
    size, min_res = get_img_sizes(dir)

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

    fig, axis = plt.subplots(2,1)
    import pdb; pdb.set_trace()
