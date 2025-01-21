# Built on top of code from Martin Pfaller

# !/usr/bin/env python

import os
import vtk

import numpy as np
from collections import defaultdict

from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n
from vtk.util.numpy_support import get_vtk_array_type


class Integration:
    """
    Class to perform integration on slices
    """

    def __init__(self, inp):
        try:
            self.integrator = vtk.vtkIntegrateAttributes()
        except AttributeError:
            raise Exception('vtkIntegrateAttributes is currently only supported by pvpython')

        if not inp.GetOutput().GetNumberOfPoints():
            raise Exception('Empty slice')

        self.integrator.SetInputData(inp.GetOutput())
        self.integrator.Update()

    def evaluate(self, res_name):
        """
        Evaluate integral.
        Distinguishes between scalar integration (e.g. pressure) and normal projection (velocity)
        Optionally divides integral by integrated area
        Args:
            field: pressure, velocity, ...
            res_name: name of array

        Returns:
            Scalar integral
        """
        # type of result
        field = res_name.split('_')[0]

        if field == 'velocity':
            int_name = 'normal_' + res_name
        else:
            int_name = res_name

        # evaluate integral
        integral = v2n(self.integrator.GetOutput().GetPointData().GetArray(int_name))[0]

        # choose if integral should be divided by area
        if field == 'velocity':
            return integral
        else:
            return integral / self.area()

    def area(self):
        """
        Evaluate integrated surface area
        Returns:
        Area
        """
        return v2n(self.integrator.GetOutput().GetCellData().GetArray('Area'))[0]


class ClosestPoints:
    """
    Find closest points within a geometry
    """
    def __init__(self, inp):
        if isinstance(inp, str):
            geo = read_geo(inp)
            inp = geo.GetOutput()
        dataset = vtk.vtkPolyData()
        dataset.SetPoints(inp.GetPoints())

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        self.locator = locator

    def search(self, points, radius=None):
        """
        Get ids of points in geometry closest to input points
        Args:
            points: list of points to be searched
            radius: optional, search radius
        Returns:
            Id list
        """
        ids = []
        for p in points:
            if radius is not None:
                result = vtk.vtkIdList()
                self.locator.FindPointsWithinRadius(radius, p, result)
                ids += [result.GetId(k) for k in range(result.GetNumberOfIds())]
            else:
                ids += [self.locator.FindClosestPoint(p)]
        return ids


def collect_arrays(output):
    res = {}
    for i in range(output.GetNumberOfArrays()):
        name = output.GetArrayName(i)
        data = output.GetArray(i)
        res[name] = v2n(data)
    return res


def get_all_arrays(geo):
    # collect all arrays
    cell_data = collect_arrays(geo.GetCellData())
    point_data = collect_arrays(geo.GetPointData())

    return point_data, cell_data


def read_geo(fname):
    """
    Read geometry from file, chose corresponding vtk reader
    Args:
        fname: vtp surface or vtu volume mesh

    Returns:
        vtk reader, point data, cell data
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == '.vtu':
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_geo(fname, input):
    """
    Write geometry to file
    Args:
        fname: file name
        input: vtk object
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vtp':
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == '.vtu':
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def read_img(fname):
    """
    Read image from file, chose corresponding vtk reader
    Args:
        fname: vti image

    Returns:
        vtk reader
    """
    _, ext = os.path.splitext(fname)
    if ext == '.vti':
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    reader.SetFileName(fname)
    reader.Update()

    return reader


def write_img(fname, input):
    """
    Write image to file
    Args:
        fname: file name
        input: vtk object
    """
    _, ext = os.path.splitext(fname)
    if ext == '.mha':
        writer = vtk.vtkXMLPolyDataWriter()
        # if input is vtkImageData, convert to vtkPolyData
        if isinstance(input, vtk.vtkImageData):
            input = geo(input)
    elif ext == '.vti':
        writer = vtk.vtkXMLImageDataWriter()
    elif ext == '.vtk':
        writer = vtk.vtkDataSetWriter()
    else:
        raise ValueError('File extension ' + ext + ' unknown.')
    writer.SetFileName(fname)
    writer.SetInputData(input)
    writer.Update()
    writer.Write()


def change_vti_vtk(fname):
    """
    Change image file from vti to vtk
    Args:
        fname: file name
    """
    # Read in the VTI file
    reader = vtk.vtkXMLImageDataReader()
    reader.SetFileName(fname)
    reader.Update()

    # Write out the VTK file
    writer = vtk.vtkDataSetWriter()
    writer.SetFileName(fname.replace('.vti','.vtk'))
    writer.SetInputConnection(reader.GetOutputPort())
    writer.Write()


def threshold(inp, t, name):
    """
    Threshold according to cell array
    Args:
        inp: InputConnection
        t: BC_FaceID
        name: name in cell data used for thresholding
    Returns:
        reader, point data
    """
    thresh = vtk.vtkThreshold()
    thresh.SetInputData(inp)
    thresh.SetInputArrayToProcess(0, 0, 0, 1, name)
    thresh.ThresholdBetween(t, t)
    thresh.Update()
    return thresh


def calculator(inp, function, inp_arrays, out_array):
    """
    Function to add vtk calculator
    Args:
        inp: InputConnection
        function: string with function expression
        inp_arrays: list of input point data arrays
        out_array: name of output array
    Returns:
        calc: calculator object
    """
    calc = vtk.vtkArrayCalculator()
    for a in inp_arrays:
        calc.AddVectorArrayName(a)
    calc.SetInputData(inp.GetOutput())
    if hasattr(calc, 'SetAttributeModeToUsePointData'):
        calc.SetAttributeModeToUsePointData()
    else:
        calc.SetAttributeTypeToPointData()
    calc.SetFunction(function)
    calc.SetResultArrayName(out_array)
    calc.Update()
    return calc


def cut_plane(inp, origin, normal):
    """
    Cuts geometry at a plane
    Args:
        inp: InputConnection
        origin: cutting plane origin
        normal: cutting plane normal
    Returns:
        cut: cutter object
    """
    # define cutting plane
    plane = vtk.vtkPlane()
    plane.SetOrigin(origin[0], origin[1], origin[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    # define cutter
    cut = vtk.vtkCutter()
    cut.SetInputData(inp)
    cut.SetCutFunction(plane)
    cut.Update()
    return cut


def get_points_cells_pd(polydata):
    cells = []
    for i in range(polydata.GetNumberOfCells()):
        cell_points = []
        for j in range(polydata.GetCell(i).GetNumberOfPoints()):
            cell_points += [polydata.GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(polydata.GetPoints().GetData()), np.array(cells)


def get_points_cells(inp):
    cells = []
    for i in range(inp.GetOutput().GetNumberOfCells()):
        cell_points = []
        for j in range(inp.GetOutput().GetCell(i).GetNumberOfPoints()):
            cell_points += [inp.GetOutput().GetCell(i).GetPointId(j)]
        cells += [cell_points]
    return v2n(inp.GetOutput().GetPoints().GetData()), np.array(cells)


def connectivity(inp, origin):
    """
    If there are more than one unconnected geometries, extract the closest one
    Args:
        inp: InputConnection
        origin: region closest to this point will be extracted
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp) #.GetOutput())
    con.SetExtractionModeToClosestPointRegion()
    con.SetClosestPoint(origin[0], origin[1], origin[2])
    con.Update()
    return con


def connectivity_all(inp):
    """
    Color regions according to connectivity
    Args:
        inp: InputConnection
    Returns:
        con: connectivity object
    """
    con = vtk.vtkConnectivityFilter()
    con.SetInputData(inp)
    con.SetExtractionModeToAllRegions()
    con.ColorRegionsOn()
    con.Update()
    assert con.GetNumberOfExtractedRegions() > 0, 'empty geometry'
    return con


def extract_surface(inp):
    """
    Extract surface from 3D geometry
    Args:
        inp: InputConnection
    Returns:
        extr: vtkExtractSurface object
    """
    extr = vtk.vtkDataSetSurfaceFilter()
    extr.SetInputData(inp)
    extr.Update()
    return extr.GetOutput()


def clean(inp):
    """
    Merge duplicate Points
    """
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(inp)
    # cleaner.SetTolerance(1.0e-3)
    cleaner.PointMergingOn()
    cleaner.Update()
    return cleaner.GetOutput()


def scalar_array(length, name, fill):
    """
    Create vtkIdTypeArray array with given name and constant value
    """
    ids = vtk.vtkIdTypeArray()
    ids.SetNumberOfValues(length)
    ids.SetName(name)
    ids.Fill(fill)
    return ids


def add_scalars(inp, name, fill):
    """
    Add constant value array to point and cell data
    """
    inp.GetOutput().GetCellData().AddArray(scalar_array(inp.GetOutput().GetNumberOfCells(), name, fill))
    inp.GetOutput().GetPointData().AddArray(scalar_array(inp.GetOutput().GetNumberOfPoints(), name, fill))


def rename(inp, old, new):
    if inp.GetOutput().GetCellData().HasArray(new):
        inp.GetOutput().GetCellData().RemoveArray(new)
    if inp.GetOutput().GetPointData().HasArray(new):
        inp.GetOutput().GetPointData().RemoveArray(new)
    inp.GetOutput().GetCellData().GetArray(old).SetName(new)
    inp.GetOutput().GetPointData().GetArray(old).SetName(new)


def replace(inp, name, array):
    arr = n2v(array)
    arr.SetName(name)
    inp.GetOutput().GetCellData().RemoveArray(name)
    inp.GetOutput().GetCellData().AddArray(arr)


def geo(inp):
    poly = vtk.vtkGeometryFilter()
    poly.SetInputData(inp)
    poly.Update()
    return poly.GetOutput()


def region_grow(geo, seed_points, seed_ids, n_max=99):
    # initialize output arrays
    array_dist = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids = -1 * np.ones(geo.GetNumberOfPoints(), dtype=int)
    array_ids[seed_points] = seed_ids

    # initialize ids
    cids_all = set()
    pids_all = set(seed_points.tolist())
    pids_new = set(seed_points.tolist())

    # surf = extract_surface(geo)
    # pids_surf = set(v2n(surf.GetPointData().GetArray('GlobalNodeID')).tolist())

    # loop until region stops growing or reaches maximum number of iterations
    i = 0
    while len(pids_new) > 0 and i < n_max:
        # count grow iterations
        i += 1

        # update
        pids_old = pids_new

        # print progress
        print_str = 'Iteration ' + str(i)
        print_str += '\tNew points ' + str(len(pids_old)) + '     '
        print_str += '\tTotal points ' + str(len(pids_all))
        print(print_str)

        # grow region one step
        pids_new = grow(geo, array_ids, pids_old, pids_all, cids_all)

        # convert to array
        pids_old_arr = list(pids_old)

        # create point locator with old wave front
        points = vtk.vtkPoints()
        points.Initialize()
        for i_old in pids_old:
            points.InsertNextPoint(geo.GetPoint(i_old))

        dataset = vtk.vtkPolyData()
        dataset.SetPoints(points)

        locator = vtk.vtkPointLocator()
        locator.Initialize()
        locator.SetDataSet(dataset)
        locator.BuildLocator()

        # find closest point in new wave front
        for i_new in pids_new:
            array_ids[i_new] = array_ids[pids_old_arr[locator.FindClosestPoint(geo.GetPoint(i_new))]]
            array_dist[i_new] = i

    return array_ids, array_dist + 1


def grow(geo, array, pids_in, pids_all, cids_all):
    # ids of propagating wave-front
    pids_out = set()

    # loop all points in wave-front
    for pi_old in pids_in:
        cids = vtk.vtkIdList()
        geo.GetPointCells(pi_old, cids)

        # get all connected cells in wave-front
        for j in range(cids.GetNumberOfIds()):
            # get cell id
            ci = cids.GetId(j)

            # skip cells that are already in region
            if ci in cids_all:
                continue
            else:
                cids_all.add(ci)

            pids = vtk.vtkIdList()
            geo.GetCellPoints(ci, pids)

            # loop all points in cell
            for k in range(pids.GetNumberOfIds()):
                # get point id
                pi_new = pids.GetId(k)

                # add point only if it's new and doesn't fullfill stopping criterion
                if array[pi_new] == -1 and pi_new not in pids_in:
                    pids_out.add(pi_new)
                    pids_all.add(pi_new)

    return pids_out


def cell_connectivity(geo):
    """
    Extract the point connectivity from vtk and return a dictionary that can be used in meshio
    """
    vtk_to_meshio = {3: 'line', 5: 'triangle', 10: 'tetra'}

    cells = defaultdict(list)
    for i in range(geo.GetNumberOfCells()):
        cell_type_vtk = geo.GetCellType(i)
        if cell_type_vtk in vtk_to_meshio:
            cell_type = vtk_to_meshio[cell_type_vtk]
        else:
            raise ValueError('vtkCellType ' + str(cell_type_vtk) + ' not supported')

        points = geo.GetCell(i).GetPointIds()
        point_ids = []
        for j in range(points.GetNumberOfIds()):
            point_ids += [points.GetId(j)]
        cells[cell_type] += [point_ids]

    for t, c in cells.items():
        cells[t] = np.array(c)

    return cells


def get_location_cells(surface):
    """
    Compute centers of cells and return their surface_locations
    Args:
        vtk polydata, e.g. surface
    Returns:
        np.array with centroid surface_locations
    """
    ecCentroidFilter = vtk.vtkCellCenters()
    ecCentroidFilter.VertexCellsOn()
    ecCentroidFilter.SetInputData(surface)
    ecCentroidFilter.Update()
    ecCentroids = ecCentroidFilter.GetOutput()

    surface_locations = v2n(ecCentroids.GetPoints().GetData())
    return surface_locations


def voi_contain_caps(voi_min, voi_max, caps_locations):
    """
    See if model caps are enclosed in volume
    Args:
        voi_min: min bounding values of volume
        voi_max: max bounding values of volume
    Returns:
        contain: boolean if a cap point was found within volume
    """
    larger = caps_locations > voi_min
    smaller = caps_locations < voi_max

    contain = np.any(np.logical_and(smaller.all(axis=1), larger.all(axis=1)))
    return contain


def calc_caps(polyData):

    # Now extract feature edges
    boundaryEdges = vtk.vtkFeatureEdges()
    boundaryEdges.SetInputData(polyData)
    boundaryEdges.BoundaryEdgesOn()
    boundaryEdges.FeatureEdgesOff()
    boundaryEdges.NonManifoldEdgesOff()
    boundaryEdges.ManifoldEdgesOff()
    boundaryEdges.Update()
    output = boundaryEdges.GetOutput()

    # get info on points and cells along the cap boundary
    conn = connectivity_all(output)
    data = get_points_cells(conn)#.GetOutput())

    try:
        connects = v2n(conn.GetOutput().GetPointData().GetArray(2))
    except:
        connects = v2n(conn.GetOutput().GetPointData().GetArray(1))

    caps_locs = []
    caps_areas = []
    for i in range(connects.max()+1):

        # get the points that belong to the same cap
        locs = data[0][connects == i]
        # calculate the center of the cap
        center = np.mean(locs, axis=0)
        caps_locs.append(center)

        # calculate the area of the cap
        cells = data[1][connects == i]
        area = 0
        for cell in cells:
            p0 = locs[cell[0]]
            p1 = locs[cell[1]]
            p2 = center
            area += np.linalg.norm(np.cross(p1-p0, p2-p0))/2
        caps_areas.append(area)

    return caps_locs, caps_areas


def get_largest_connected_polydata(poly):

    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToLargestRegion()
    connectivity.Update()
    poly = connectivity.GetOutput()

    return poly


def get_seed(cent_fn, centerline_num, point_on_cent):
    """
    Get a location and radius at a point along centerline
    Args:
        cent_fn: file directory for centerline
        centerline_num: starting from 0, which sub centerline do you wish to sample from
        point_on_cent: starting from 0, how far along the sub centerline you wish to sample
    Returns:
        location coords, radius at the specific point
    """

    ## Centerline
    cent = read_geo(cent_fn).GetOutput()
    num_points = cent.GetNumberOfPoints()               # number of points in centerline
    cent_data = collect_arrays(cent.GetPointData())
    c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
    radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
    cent_id = cent_data['CenterlineId']

    try:
        num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
    except:
        num_cent = 1 # in the case of only one centerline

    ip = centerline_num
    count = point_on_cent

    try:
        ids = [i for i in range(num_points) if cent_id[i,ip]==1] # ids of points belonging to centerline ip
    except:
        ids = [i for i in range(num_points)]
    locs = c_loc[ids]
    rads = radii[ids]

    return locs[count], rads[count]


def calc_normal_vectors(vec0):
    """
    Function to calculate two orthonormal vectors
    to a particular direction vec
    """
    vec0 = vec0/np.linalg.norm(vec0)
    vec1 = np.random.randn(3)       # take a random vector
    vec1 -= vec1.dot(vec0) * vec0   # make it orthogonal to k
    vec1 /= np.linalg.norm(vec1)    # normalize it
    vec2 = np.cross(vec0, vec1)     # calculate third vector

    return vec1, vec2


def clean_boundaries(resampled_image_array):
    """
    Function to see which pixels are inside mesh.
    If they are: set as 1, otherwise 0.
    Input: a binary seg array that has been resampled.
    """
    import pdb; pdb.set_trace()
    # for pixel in resampled_image:

    return new_image


def bound_polydata_by_image(image, poly, threshold):
    """
    Function to cut polydata to be bounded
    by image volume
    """
    bound = vtk.vtkBox()
    image.ComputeBounds()
    b_bound = image.GetBounds()
    b_bound = [b+threshold if (i % 2) ==0 else b-threshold for i, b in enumerate(b_bound)]
    #print("Bounding box: ", b_bound)
    bound.SetBounds(b_bound)
    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(bound)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def bound_polydata_by_sphere(poly, center, radius):

    sphereSource = vtk.vtkSphere()
    sphereSource.SetCenter(center[0], center[1], center[2])
    sphereSource.SetRadius(radius)

    clipper = vtk.vtkClipPolyData()
    clipper.SetClipFunction(sphereSource)
    clipper.SetInputData(poly)
    clipper.InsideOutOn()
    clipper.Update()
    return clipper.GetOutput()


def exportSitk2VTK(sitkIm, spacing=None):
    """
    This function creates a vtk image from a simple itk image
    Args:
        sitkIm: simple itk image
    Returns:
        imageData: vtk image
import SimpleITK as sitk
    """
    if not spacing:
        spacing = sitkIm.GetSpacing()
    import SimpleITK as sitk
    img = sitk.GetArrayFromImage(sitkIm).transpose(2,1,0)
    vtkArray = exportPython2VTK(img)
    imageData = vtk.vtkImageData()
    imageData.SetDimensions(sitkIm.GetSize())
    imageData.GetPointData().SetScalars(vtkArray)
    imageData.SetOrigin([0.,0.,0.])
    imageData.SetSpacing(spacing)
    matrix = build_transform_matrix(sitkIm)
    space_matrix = np.diag(list(spacing)+[1.])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtkmatrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtkmatrix.SetElement(i, j, matrix[i,j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(imageData)
    reslice.SetResliceAxes(vtkmatrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    imageData = reslice.GetOutput()
    # imageData.SetDirectionMatrix(sitkIm.GetDirection())

    return imageData, vtkmatrix


def exportVTK2Sitk(vtkIm):
    """
    This function creates a simple itk image from a vtk image
    Args:
        vtkIm: vtk image
    Returns:
        sitkIm: simple itk image
    """
    import SimpleITK as sitk
    vtkIm = vtkIm.GetOutput()
    vtkIm.GetPointData().GetScalars().SetName('Scalars_')
    vtkArray = v2n(vtkIm.GetPointData().GetScalars())
    vtkArray = np.reshape(vtkArray, vtkIm.GetDimensions(), order='F')
    vtkArray = np.transpose(vtkArray, (2, 1, 0))
    sitkIm = sitk.GetImageFromArray(vtkArray)
    sitkIm.SetSpacing(vtkIm.GetSpacing())
    sitkIm.SetOrigin(vtkIm.GetOrigin())
    return sitkIm


def build_transform_matrix(image):
    matrix = np.eye(4)
    matrix[:-1,:-1] = np.matmul(np.reshape(image.GetDirection(), (3,3)), np.diag(image.GetSpacing()))
    matrix[:-1,-1] = np.array(image.GetOrigin())
    return matrix


def exportPython2VTK(img):
    """
    This function creates a vtk image from a python array
    Args:
        img: python ndarray of the image
    Returns:
        imageData: vtk image
    """
    vtkArray = n2v(num_array=img.flatten('F'), deep=True, array_type=get_vtk_array_type(img.dtype))
    # vtkArray = n2v(img.flatten())
    return vtkArray


def points2polydata(xyz):
    """
    Function to convert list of points to polydata
    """
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Add points
    for i in range(0, len(xyz)):
        try:
            p = xyz.loc[i].values.tolist()
        except:
            p = xyz[i]

        point_id = points.InsertNextPoint(p)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)
    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.Modified()

    return polydata


def remove_duplicate_points(centerline):
    """
    Function to remove duplicate points from centerline
    input: centerline as polydata
    output: centerline as polydata
    """
    # Create the tree
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(centerline)
    locator.BuildLocator()

    # Get the points
    points = centerline.GetPoints()
    num_points = points.GetNumberOfPoints()

    # Create new points
    new_points = vtk.vtkPoints()
    new_points.SetNumberOfPoints(num_points)

    # Create new polydata with all arr
    new_centerline = vtk.vtkPolyData()
    new_centerline.SetPoints(new_points)
    new_centerline.GetPointData().ShallowCopy(centerline.GetPointData())
    new_centerline.GetCellData().ShallowCopy(centerline.GetCellData())

    # Loop through points
    for i in range(num_points):
        # Get the point
        point = points.GetPoint(i)

        # Find the closest point
        closest_point_id = locator.FindClosestPoint(point)

        # Get the closest point
        closest_point = points.GetPoint(closest_point_id)

        # Check if the points are the same
        if np.array_equal(point, closest_point):
            # If they are the same, add the point to the new polydata
            new_points.InsertPoint(i, point)
        else:
            # If they are not the same, add the closest point to the new polydata
            new_points.InsertPoint(i, closest_point)

    # Return the new polydata
    return new_centerline


def vtk_marching_cube(vtkLabel, bg_id, seg_id):
    """
    Use the VTK marching cube to create isosrufaces for all classes excluding the background
    Args:
        labels: vtk image contraining the label map
        bg_id: id number of background class
    Returns:
        mesh: vtk PolyData of the surface mesh
    """
    contour = vtk.vtkMarchingCubes()
    contour.SetInputData(vtkLabel)
    contour.SetValue(0, seg_id)
    contour.Update()
    mesh = contour.GetOutput()

    return mesh


def vectors2polydata(vectors):
    """
    Function to convert list of vectors to polydata
    If the vectors don't have start points, they are assumed to start at origin
    The vectors are assumed to be in 3D
    The function uses vtkPolyData and GetPointData and SetVectors to store the vectors
    and cell type vtkVertex
    
    Args:
        vectors: list or np.array of vectors, shape (n, 3) or (n, 6)
    Returns:
        polydata: vtk polydata object
    """
    # Create the points
    points = vtk.vtkPoints()
    # Create the topology of the point (a vertex)
    vertices = vtk.vtkCellArray()
    # Add points
    for i in range(0, len(vectors)):
        # Get the vector
        try:
            v = vectors.loc[i].values.tolist()
        except:
            v = vectors[i]

        # Check if the vector has start and end points
        if len(v) == 3:
            # If the vector doesn't have start and end points, assume it starts at origin
            start = [0, 0, 0]
            end = v
        elif len(v) == 6:
            # If the vector has start and end points, get the start and end points
            start = v[:3]
            end = v[3:]
        else:
            raise ValueError("The vectors should have either 3 or 6 elements")

        # Add the start point
        point_id = points.InsertNextPoint(start)
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(point_id)

        # Add the end point
        # point_id = points.InsertNextPoint(end)
        # vertices.InsertNextCell(1)
        # vertices.InsertCellPoint(point_id)

    # Create a poly data object
    polydata = vtk.vtkPolyData()
    # Set the points and vertices we created as the geometry and topology of the polydata
    polydata.SetPoints(points)
    polydata.SetVerts(vertices)
    polydata.Modified()

    # Create the vectors
    vectors_vtk = vtk.vtkDoubleArray()
    vectors_vtk.SetNumberOfComponents(3)
    vectors_vtk.SetName("Vectors")
    for i in range(0, len(vectors)):
        # Get the vector
        try:
            v = vectors.loc[i].values.tolist()
        except:
            v = vectors[i]

        # Check if the vector has start and end points
        if len(v) == 3:
            # If the vector doesn't have start and end points, assume it starts at origin
            start = [0, 0, 0]
            end = v
        elif len(v) == 6:
            # If the vector has start and end points, get the start and end points
            start = v[:3]
            end = v[3:]
        else:
            raise ValueError("The vectors should have either 3 or 6 elements")

        # Calculate the vector
        vector = np.array(end) - np.array(start)

        # Change data to list
        vector = vector.tolist()

        # Add the vector to the polydata
        vectors_vtk.InsertNextTuple(vector)

    # Make sure we have same number of vectors as points
    import pdb; pdb.set_trace() 
    assert vectors_vtk.GetNumberOfTuples() == polydata.GetNumberOfPoints(), "Number of vectors should be the same as the number of points"

    # Set the vectors to the polydata
    polydata.GetPointData().SetVectors(vectors_vtk)

    # Return the polydata
    return polydata
