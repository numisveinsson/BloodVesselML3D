import numpy as np
import os

from modules import vtk_functions as vf
from modules import sitk_functions as sf
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n

import SimpleITK as sitk


def points2polydata(xyz):
    import vtk
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


model = '0141_1001.vtp'

file_dir = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/'
fd_cent = file_dir + 'centerlines/' + model
fd_im = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/vascular_data_3d/images/OSMSC0141/OSMSC0141-cm.mha'
fd_sur = file_dir + 'surfaces/' + model

od = '/Users/numisveinsson/Downloads/'

## Centerline data
cent = vf.read_geo(fd_cent).GetOutput()         # read in geometry
num_points = cent.GetNumberOfPoints()               # number of points in centerline
cent_data = vf.collect_arrays(cent.GetPointData())
c_loc = v2n(cent.GetPoints().GetData())             # point locations as numpy array
radii = cent_data['MaximumInscribedSphereRadius']   # Max Inscribed Sphere Radius as numpy array
cent_id = cent_data['CenterlineId']

num_cent = len(cent_id[0]) # number of centerlines (one is assembled of multiple)
ip = 2

ids = ids = [i for i in range(num_points) if cent_id[i,ip]==1]
locs = c_loc[ids]                                           # locations of those points
rads = radii[ids]

## Image data

# Size of extraction
size_r = 5

count = 1000

centerline_point = locs[count]

reader_im = sf.read_image(fd_im)

origin_im = np.array(list(reader_im.GetOrigin()))
size_im = np.array(list(reader_im.GetSize()))
spacing_im = np.array(list(reader_im.GetSpacing()))

size_extract = np.ceil(size_r*rads[count]/spacing_im)
index_extract = np.rint((locs[count]-origin_im - (size_r/2)*rads[count])/spacing_im)
opposite_index = np.rint((locs[count]-origin_im)/spacing_im + (size_r/2)*rads[count]/spacing_im)
center_volume = 1/2*(index_extract + opposite_index)*spacing_im + origin_im

index_point = ((locs[count]-origin_im)//spacing_im - (size_r/2)*rads[count]//spacing_im)*spacing_im + origin_im

voi_min = locs[count] - (size_r/2)*rads[count]
voi_max = locs[count] + (size_r/2)*rads[count]

seed = (np.array(size_extract)//2).tolist()

from extract_3d_data import extract_volume
new_img = extract_volume(fd_im, index_extract.astype(int).tolist(), size_extract.astype(int).tolist())

sitk.WriteImage(new_img, od + 'extraction.vtk')
vf.write_geo(od + 'point_centerline.vtp', points2polydata([centerline_point.tolist()]))
vf.write_geo(od + 'point_center_voi.vtp', points2polydata([center_volume.tolist()]))
vf.write_geo(od + 'index_point.vtp', points2polydata([index_point.tolist()]))

import pdb; pdb.set_trace()
sitk.Show(new_img, title="Hello World: Python", debugOn=True)



#outputImageDir = '/Users/numisveinsson/Documents/Side_SV_projects/SV_ML_Training/3d_ml_data/'
#sitk.WriteImage(new_img, outputImageDir+'0002_0001_01_im.mha')
