import os
import pdb
import numpy as np
import vtk
import SimpleITK as sitk
from modules import vtk_functions as vf
from modules import sampling_functions as sf

def discr_points_ref(points, resolution = 10):
    """
    Function to discretize the points into a grid of resolution x resolution x resolution
    We bucket the points into the closest grid point
    Eg for a resolution of 10, the grid is 0, 0.1, 0.2, ..., 1
    and a point of 0.14 would be bucketed to 0.1

    Args:
        points: np.array, N x 3 array of points, range 0 to 1
        resolution: int, resolution of the grid from 0 to 1

    Returns:
        closest_points: np.array, N x 3 array of discretized points
    """
    # get the grid points
    grid_points = np.linspace(0, 1, resolution)
    # find the closest grid point for each point
    closest_points = np.zeros_like(points)
    for i in range(3):
        closest_points[:, i] = np.round(points[:, i] * resolution) / resolution
    return closest_points

def discr_points_bounds(points, bounds, resolution = 10):
    """
    Function to discretize the points into a grid of resolution x resolution x resolution
    We bucket the points into the closest grid point
    Eg for one dimension for a resolution of 10 and bounds of 2, 4, the grid is 2, 2.2, 2.4, ..., 4
    and a point of 2.14 would be bucketed to 2.2 in that dimension
    For 3D, we do this for each dimension
    For resolution of 10, we get 10x10x10 number of grid points that points are rounded to
    We return the grid points as well as a list of the resolution^3 points
    Args:
        points: np.array, N x 3 array of points, range 0 to 1
        bounds: np.array, 2 x 3 array of min and max bounds
        resolution: int, resolution of the grid

    Returns:
        closest_points: np.array, N x 3 array of discretized points, in bounds space
        grid_points: np.array, (resolution^3) x 3 array of grid points (in bounds space)
    """
    # get the grid points
    grid_points = np.zeros((resolution**3, 3))
    grid_points[:, 0] = np.tile(np.linspace(bounds[0, 0], bounds[1, 0], resolution), resolution**2)
    grid_points[:, 1] = np.tile(np.repeat(np.linspace(bounds[0, 1], bounds[1, 1], resolution), resolution), resolution)
    grid_points[:, 2] = np.repeat(np.linspace(bounds[0, 2], bounds[1, 2], resolution**2), resolution)
    # round the points to the grid points
    closest_points = np.zeros_like(points)
    for i in range(3):
        closest_points[:, i] = np.round((points[:, i] - bounds[0, i]) / (bounds[1, i] - bounds[0, i]) * resolution) / resolution # round to the grid
        closest_points[:, i] = closest_points[:, i] * (bounds[1, i] - bounds[0, i]) + bounds[0, i] # transform back to bounds space
    return closest_points, grid_points

def process_centerline(centerline):
    """
    Function to post process the centerline using vtk functionalities.
    
    1. Remove duplicate points.
    2. Smooth centerline.

    Parameters
    ----------
    centerline : vtkPolyData
        Centerline of the vessel.

    Returns
    -------
    centerline : vtkPolyData
    """
    # print(f"Number of points before post processing: {centerline.GetNumberOfPoints()}")
    # Remove duplicate points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(centerline)
    cleaner.SetTolerance(0.01)
    cleaner.Update()
    centerline = cleaner.GetOutput()

    # Smooth centerline
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(centerline)
    smoother.SetNumberOfIterations(15)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff()
    smoother.SetFeatureAngle(120.0)
    smoother.SetPassBand(0.001)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    centerline = smoother.GetOutput()

    # print(f"Number of points after post processing: {centerline.GetNumberOfPoints()}")

    return centerline

def get_all_pts_centerlines(folder, ref = False):
    """
    Function to get all the points in the centerlines in a folder

    Args:
        folder: str, path to the folder containing the centerlines

    Returns:
        all_points: list, list of np.arrays of points in the centerlines
    """
    all_points = []
    for file in os.listdir(folder):
        if file.endswith('.vtp'):
            print(f"Processing {file}")
            try:
                centerline = vf.read_geo(folder + file).GetOutput()
                centerline = process_centerline(centerline)
                num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sf.sort_centerline(centerline)
                if ref:
                    name = os.path.basename(file).replace('.vtp', '')
                    model = name[:9]
                    name_file = name[10:]
                    img = sitk.ReadImage(f'/Users/numisveins/Documents/Automatic_Tracing_Data/train_version_5_aortas_centerline/vtk_data/vtk_{model}/{name_file}.vtk')
                    bounds = sf.get_bounds(img)
                    c_loc = sf.transform_to_ref(c_loc, bounds)
                all_points.append(c_loc)
            except Exception as e:
                print(e)
                print(f"Error processing {file}")
    return all_points
def from_list_to_array(list_points):
    """
    Function to convert a list of np.arrays to a single np.array

    Args:
        list_points: list, list of np.arrays of points

    Returns:
        all_points: np.array, N x 3 array of points
    """
    all_points = np.concatenate(list_points, axis = 0)
    return all_points

def plot_histogram(points, dimension = 0):
    """
    Function to plot a histogram of the points in a dimension

    Args:
        points: np.array, N x 3 array of points
        dimension: int, dimension to plot the histogram

    Returns:
        None
    """
    import matplotlib.pyplot as plt
    plt.hist(points[:, dimension], bins = 20)
    plt.show()

def create_polydata(points):
    """
    Function to create a vtkPolyData object from a list of points

    Args:
        points: list, list of np.arrays of points

    Returns:
        polydata: vtkPolyData, polydata object of the points
    """
    list_all_points = []
    for point in points:
        list_all_points.append(point)
    all_points = np.concatenate(list_all_points, axis = 0)
    polydata = vf.points2polydata(all_points)

    return polydata

def cluster_points(points, num_clusters = 1000):
    """
    Function to cluster the points into num_clusters clusters

    Args:
        points: np.array, N x 3 array of points
        num_clusters: int, number of clusters to cluster the points into

    Returns:
        cluster_points: np.array, N x 3 array of points with cluster labels
        cluster_centers: np.array, num_clusters x 3 array of cluster centers
    """
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters = num_clusters, random_state = 0).fit(points)
    cluster_points = np.zeros_like(points)
    for i in range(num_clusters):
        cluster_points[kmeans.labels_ == i] = kmeans.cluster_centers_[i]
    cluster_centers = kmeans.cluster_centers_

    return cluster_points, cluster_centers

def find_closest_cluster(points, cluster_centers):
    """
    Function to find the closest cluster center for each point

    Args:
        points: np.array, N x 3 array of points
        cluster_centers: np.array, num_clusters x 3 array of cluster centers

    Returns:
        closest_clusters: np.array, N x 3 array of cluster centers
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(cluster_centers)
    _, closest_clusters = tree.query(points)

    return closest_clusters

def get_c_loc(centerline):
    """
    Function to get the centerline points from a vtkPolyData object

    Args:
        centerline: vtkPolyData, centerline of the vessel

    Returns:
        c_loc: np.array, N x 3 array of centerline points
    """
    _, c_loc, _, _, _, _ = sf.sort_centerline(centerline)

    return c_loc

def discr_centerline_vectors(centerline , num_vectors = 100, vector_size = 0.05):
    """
    Function to discretize the centerline into a sequence of vectors
    The centerline has multiple branches
    And each branch is its own sequence of vectors

    Args:
        centerline: vtkPolyData, centerline of the vessel
        num_vectors: int, number of vectors to discretize a unit sphere into (size of vocabulary)
        vector_size: float, magnitude of the vectors
    Returns:
        vectors: np.array, N x 3 array of vectors, where N is the number of vectors
    """
    # create the list of vectors defining a unit sphere
    vectors = discr_sphere(num_vectors) # the vocabulary we are going to use
    # write the vectors for debuggings
    vectors_pd = vf.vectors2polydata(vectors) #vf.points2polydata(vectors)
    vf.write_geo('/Users/numisveins/Documents/Transformer_Tracing/debug_output/vectors.vtp', vectors_pd)
    
    # discretize the centerline based on vector_size
    centerline = process_centerline_length(centerline, vector_size)

def process_centerline_length(centerline, length = 0.05):
    """
    Function to process the centerline based on the vector size
    Goal is to return a centerline with points spaced by vector_size
    To do that we sort the centerline
    For each branch, we get the length of the branch
    We get the number of vectors needed to cover the branch
    We interpolate the points on the branch to get the desired number of points

    Args:
        centerline: vtkPolyData, centerline of the vessel
        length: float, length between points along the centerline

    Returns:
        centerline: vtkPolyData, processed centerline
    """
    # sort the centerline
    num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sf.sort_centerline(centerline)
    # get the length of each branch
    branch_lengths = get_branch_lengths(c_loc, cent_id, num_cent)
    # get the number of vectors needed for each branch
    num_vectors = np.ceil(branch_lengths / length).astype(int)
    # get the interpolated points for each branch
    new_c_loc = np.zeros((np.sum(num_vectors), 3))
    new_cent_id = []
    count = 0
    for i in range(num_cent):
        new_c_loc[count:count + num_vectors[i]] = interpolate_points(c_loc[cent_id[i]], num_vectors[i])
        new_cent_id.append(np.arange(count, count + num_vectors[i]))
        count += num_vectors[i]
    # create the new centerline
    new_centerline = vf.points2polydata(new_c_loc)
    # write the new centerline
    vf.write_geo('/Users/numisveins/Documents/Transformer_Tracing/debug_output/new_centerline_vectors.vtp', new_centerline)
    return new_centerline

def interpolate_points(c_locs, num_vectors):
    """
    Function to interpolate points along a branch to get num_vectors points

    Args:
        c_locs: np.array, N x 3 array of points along the branch
        num_vectors: int, number of vectors to interpolate

    Returns:
        new_c_loc: np.array, num_vectors x 3 array of interpolated points
    """
    from scipy.interpolate import interp1d
    # get the cumulative distance along the branch
    dist = np.cumsum(np.linalg.norm(np.diff(c_locs, axis = 0), axis = 1))
    # add 0 at the beginning
    dist = np.insert(dist, 0, 0)
    # interpolate the points
    f = interp1d(dist, c_locs, axis = 0)
    new_dist = np.linspace(0, dist[-1], num_vectors)
    new_c_loc = f(new_dist)
    return new_c_loc

def get_branch_lengths(c_locs, cent_id, num_cent):
    """
    Function to get the length of each branch in the centerline

    We add the distances between the points together

    Args:
        centerline: vtkPolyData, centerline of the vessel
        c_locs: Nx3 array of point locations
        cent_id: list of lists, each list contains the id of points in a branch
        num_cent: int, number of centerlines

    Returns:
        branch_lengths: np.array, num_cent array of branch lengths
    """
    branch_lengths = np.zeros(num_cent)
    for i in range(num_cent):
        branch_lengths[i] = np.sum(np.linalg.norm(np.diff(c_locs[cent_id[i]], axis = 0), axis=1))
        lengths = np.cumsum(np.insert(np.linalg.norm(np.diff(c_locs[cent_id[i]], axis=0), axis=1), 0, 0))
    return branch_lengths
    

def discr_sphere(num_vectors = 100):
    """
    Function to discretize a unit sphere into num_vectors vectors
    They are uniformly distributed on the sphere, all originating from the center
    They are unit vectors
    They are in the form of a N x 3 array

    Args:
        num_vectors: int, number of vectors to discretize the unit sphere into

    Returns:
        vectors: np.array, N x 3 array of vectors, where N is the number of vectors
    """
    vectors = fibonacci_sphere(num_vectors)
    vectors = np.array(vectors)
    return vectors

def fibonacci_sphere(samples=1000):
    """
    Function to create a fibonacci sphere with samples number of points
    The points are uniformly distributed on the sphere
    They are in the form of a list of tuples

    Args:
        samples: int, number of points on the sphere

    Returns:
        points: list, list of tuples of points on the sphere
    """
    import math

    points = []
    phi = math.pi * (math.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = math.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = math.cos(theta) * radius
        z = math.sin(theta) * radius

        points.append((x, y, z))

    return points

if __name__=='__main__':

    out_dir = '/Users/numisveins/Documents/Transformer_Tracing/debug_output/'

    file_path = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_version_5_aortas_centerline/ct_train_masks_centerlines/0150_0001_6_2.vtp'
    name = os.path.basename(file_path).replace('.vtp', '')
    img_path = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_version_5_aortas_centerline/vtk_data/vtk_0150_0001/6_2.vtk'
    # read in geometry
    centerline = vf.read_geo(file_path).GetOutput()
    centerline = process_centerline(centerline)

    discr_centerline_vectors(centerline, vector_size=0.5)
    import pdb; pdb.set_trace()

    img = sitk.ReadImage(img_path)
    # write both
    vf.write_geo(out_dir + name + '_centerline.vtp', centerline)
    sitk.WriteImage(img, out_dir + name + '_segmentation.mha')
    # get centerline points
    bounds = sf.get_bounds(img)
    num_points, c_loc, radii, cent_id, bifurc_id, num_cent = sf.sort_centerline(centerline)
    c_loc_ref = sf.transform_to_ref(c_loc, bounds) # N x 3 of centerline points, in ref 0-1 space
    # load in cluster centers
    cluster_centers = np.load(out_dir + 'cluster_centers.npy')
    # find the closest cluster center for each point
    closest_clusters = find_closest_cluster(c_loc_ref, cluster_centers)
    # get the closest cluster centers
    cluster_centers = cluster_centers[closest_clusters]
    # back to bounds space
    closest_clusters = sf.transform_from_ref(cluster_centers, bounds)
    # create polydata
    locs_pd = vf.points2polydata(c_loc_ref)
    vf.write_geo(out_dir + name + '_centerline_ref.vtp', locs_pd)
    locs_pd = vf.points2polydata(cluster_centers)
    vf.write_geo(out_dir + 'cluster_centers.vtp', locs_pd)
    # also for closest clusters
    locs_pd = vf.points2polydata(closest_clusters)
    vf.write_geo(out_dir + name + '_centerline_clustered.vtp', locs_pd)



    # now bucket the centerline points
    # dim_res_list = [10, 20, 30, 40, 50]
    # for dim_res in dim_res_list:
    #     closest_points_ref = discr_points_ref(c_loc, dim_res)
    #     closest_points_bounds, grid_points = discr_points_bounds(c_loc, bounds, dim_res)
    #     # create polydata
    #     # locs_pd = vf.points2polydata(closest_points_ref)
    #     # vf.write_geo(out_dir + name + '_centerline_discrete_ref' + str(dim_res) + '.vtp', locs_pd)
    #     locs_pd = vf.points2polydata(closest_points_bounds)
    #     vf.write_geo(out_dir + name + '_centerline_discrete_bounds' + str(dim_res) + '.vtp', locs_pd)
    #     # also for grid points
    #     grid_pd = vf.points2polydata(grid_points)
    #     vf.write_geo(out_dir + name + '_grid_points' + str(dim_res) + '.vtp', grid_pd)

    # get all points in the centerlines
    # folder = '/Users/numisveins/Documents/Automatic_Tracing_Data/train_version_5_aortas_centerline/ct_train_masks_centerlines/'
    # all_points = get_all_pts_centerlines(folder, ref = True)
    # pts_np = from_list_to_array(all_points)
    # plot_histogram(pts_np, 0)
    # plot_histogram(pts_np, 1)
    # plot_histogram(pts_np, 2)
    # # Cluster the points
    # pts_np_cluster, cluster_centers = cluster_points(pts_np, num_clusters = 1000)
    # # save the cluster centers
    # vf.write_geo(out_dir + 'cluster_centers.vtp', vf.points2polydata(cluster_centers))
    # # also as numpy
    # np.save(out_dir + 'cluster_centers.npy', cluster_centers)
    # # create polydata
    # polydata = create_polydata(all_points)
    # vf.write_geo(out_dir + 'all_centerlines.vtp', polydata)
    # polydata_cluster = create_polydata([pts_np_cluster])
    # vf.write_geo(out_dir + 'all_centerlines_clustered.vtp', polydata_cluster)