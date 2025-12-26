import vtk
import numpy as np
from scipy.spatial.distance import pdist, squareform

def load_vtp(file_path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(file_path)
    reader.Update()
    polydata = reader.GetOutput()
    return polydata

def compute_distances(polydata):
    points = polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    print(f"Number of points: {num_points}")
    distances = np.zeros((num_points, num_points))

    for i in range(num_points):
        # Print progress every 10% of points
        if i % (num_points // 10) == 0:
            print(f"Computing distances for point {i} of {num_points}")
        for j in range(i + 1, num_points):
            distance = np.linalg.norm(np.array(points.GetPoint(i)) - np.array(points.GetPoint(j)))
            distances[i, j] = distance
            distances[j, i] = distance

    return distances

def remove_close_points(input_polydata, threshold):
    points = input_polydata.GetPoints()
    num_points = points.GetNumberOfPoints()
    print(f"Number of points before removing: {num_points}")

    # Create a list of points to keep
    points_to_keep = [0]

    # Iterate over all points and keep only those that are far enough from all other points
    for i in range(1, num_points):
        point = np.array(points.GetPoint(i))
        keep_point = True
        for j in points_to_keep:
            distance = np.linalg.norm(point - np.array(points.GetPoint(j)))
            if distance < threshold:
                keep_point = False
                break
        if keep_point:
            points_to_keep.append(i)

    # Create a new polydata with only the points to keep
    output_polydata = vtk.vtkPolyData()
    new_points = vtk.vtkPoints()
    for i in points_to_keep:
        new_points.InsertNextPoint(points.GetPoint(i))
    output_polydata.SetPoints(new_points)

    print(f"Number of points after removing: {new_points.GetNumberOfPoints()}")

    return output_polydata

def compute_mst(distances):
    from scipy.sparse.csgraph import minimum_spanning_tree
    mst = minimum_spanning_tree(distances)
    return mst

def create_polydata_from_edges(edges, points):
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)

    edges_array = np.array(edges)

    lines = vtk.vtkCellArray()
    for edge in edges_array:
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, edge[0])
        line.GetPointIds().SetId(1, edge[1])
        lines.InsertNextCell(line)

    polydata.SetLines(lines)

    return polydata

def save_vtp(polydata, file_path):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(file_path)
    writer.SetInputData(polydata)
    writer.Write()

if __name__ == "__main__":
    # input_file_path = "/Users/numisveins/Downloads/mega_subvolume/final_01_3d_fullres_0_1001_centerlines.vtp"
    input_file_path = "/Users/numisveins/Downloads/output_v1_no_append/3d_fullres_10/final_10_3d_fullres_0_2001_points.vtp"
    output_file_path = "/Users/numisveins/Downloads/output_v1_no_append/3d_fullres_10/final_10_3d_fullres_0_2001_points_min_span.vtp"

    remove_close_points = False
    threshold = 3.0

    # Load VTP file
    input_polydata = load_vtp(input_file_path)

    if remove_close_points:
        print(f"Removing points that are closer than {threshold} units")
        # Remove points that are too close to each other
        input_polydata = remove_close_points(input_polydata, threshold)

    # Compute pairwise distances between points
    distances = compute_distances(input_polydata)

    # Compute minimum spanning tree
    mst = compute_mst(distances)

    # Extract edges from minimum spanning tree
    edges = np.argwhere(mst.toarray())

    # Create polydata from edges
    output_polydata = create_polydata_from_edges(edges, input_polydata.GetPoints())

    # Save polydata to VTP file
    save_vtp(output_polydata, output_file_path)

    print("Output file saved successfully.")
