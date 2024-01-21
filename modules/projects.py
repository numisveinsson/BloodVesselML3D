# Classes for each project
# Input: Local image volume, local image mask, local image labels (radius, size, origin, spacing)
# Output: Varies based on project


class SeqSeg:
    """
    Class for sequential segmentation project
    For training this project needed:
        - image volume
        - image mask
    This also needs to be able to:
        - vary size of volume
        - shift the volume
    Output:
        - image subvolume
        - image subvolume mask
    """
    def __init__(self, params):

        self.dir = params['dir']
        self.img = params['img']
        
    def execute(self):

        return self.img
    

class TransTracing:
    """
    Class for tracing using transfomers
    For training this project needed:
        - image volume
        - image mask
        - vessel radius
        - vessel centerline points
            xyz coordinates
            bifurcation points
            end points
    This also needs to be able to:
        - vary size of volume
        - shift the volume
        - rotate the volume + centerline
        - not include caps in them
    """

    def __init__(self, params):

        self.directory = params['directory']
        self.centerline = params['centerline']
        self.img = params['img']

        self.n_points = params['n_points']

    def execute(self):
    
        return self.centerline
