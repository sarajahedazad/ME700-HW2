import numpy as np
import geometry as geom

class OverDefinedError( Exception ):
    pass

class UnderDefinedError( Exception ):
    pass


def find_point_idx( frame_obj, point ):
    match = np.where((frame_obj.points == point).all(axis=1))[0].item()
    return match

def find_points_indices( frame_obj, points_arr ):
    points_indices_lst = []
    for point in points_arr:
        match = find_point_idx( frame_obj, point )
        points_indices_lst.append( match )
    return np.array( points_indices_lst )

class BoundaryConditions:
    def __init__(self, frame_obj):
        self.frame = frame_obj
        self.n_points = self.frame.points.shape[0]
        self.n_connectivities = self.frame.connectivity.shape[0]
        self.n_DoFs = self.n_points * 6
        
        # disp bounds
        self.BCs_disp_points = None
        self.BCs_disp_points_idx = None
        self.BCs_disp_values = None

        # rotation bounds
        self.BCs_rot_points = None
        self.BCs_rot_points_idx = None
        self.BCs_rot_values = None

        # force bounds
        self.BCs_force_points = None
        self.BCs_force_points_idx = None
        self.BCs_force_values = None

        # momentum bounds
        self.BCs_momentum_points = None
        self.BCs_momentum_points_idx = None
        self.BCs_momentum_values = None

        self.BCs_X_indices = None
        self.BCs_X_values = None

        self.BCs_F_indices = None
        self.BCs_F_values = None

    def get_disp_bounds( self, bound_points, bound_values ):

        bound_points_idx = find_points_indices( self.frame, bound_points )

        self.BCs_disp_points = np.copy( bound_points )
        self.BCs_disp_points_idx = np.copy( bound_points_idx )
        self.BCs_disp_values = np.copy( bound_values )

    def get_rot_bounds( self, bound_points, bound_values ):

        bound_points_idx = find_points_indices( self.frame, bound_points )

        self.BCs_rot_points = np.copy( bound_points )
        self.BCs_rot_points_idx = np.copy( bound_points_idx )
        self.BCs_rot_values = np.copy( bound_values )

    def get_force_bounds( self, bound_points, bound_values ):

        bound_points_idx = find_points_indices( self.frame, bound_points )

        self.BCs_force_points = np.copy( bound_points )
        self.BCs_force_points_idx = np.copy( bound_points_idx )
        self.BCs_force_values = np.copy( bound_values )

    def get_momentum_bounds( self, bound_points, bound_values ):

        bound_points_idx = find_points_indices( self.frame, bound_points )

        self.BCs_momentum_points = np.copy( bound_points )
        self.BCs_momentum_points_idx = np.copy( bound_points_idx )
        self.BCs_momentum_values = np.copy( bound_values )

    def set_up_bounds( self ):
        '''known X indices'''
        X_indices = np.array( range( self.n_DoFs ) )

        BCs_disp_X_indices_start = 6 * self.BCs_disp_points_idx 
        BCs_disp_X_indices_end = BCs_disp_X_indices_start + 3
        BCs_disp_X_indices = np.array( [ X_indices[BCs_disp_X_indices_start[i]:BCs_disp_X_indices_end[i]] for i in range(len(self.BCs_disp_points_idx ))] ).reshape( -1 )

        BCs_rot_X_indices_start = 6 * self.BCs_rot_points_idx + 3
        BCs_rot_X_indices_end = BCs_rot_X_indices_start + 3
        BCs_rot_X_indices = np.array( [ X_indices[BCs_rot_X_indices_start[i]:BCs_rot_X_indices_end[i]] for i in range(len(self.BCs_rot_points_idx ))] ).reshape( -1 )

        BCs_X_indices = np.concatenate( ( BCs_disp_X_indices,  BCs_rot_X_indices ) )
        BCs_X_values = np.concatenate( ( self.BCs_disp_values.reshape(-1), self.BCs_rot_values.reshape(-1) ) )

        # Get the order of indices that would sort array a
        order = np.argsort( BCs_X_indices )

        self.BCs_X_indices = BCs_X_indices[ order ]
        self.BCs_X_values = BCs_X_values[ order ]

        '''known F indices'''
        F_indices = np.array( range( self.n_DoFs ) )

        BCs_force_F_indices_start = 6 * self.BCs_force_points_idx 
        BCs_force_F_indices_end = BCs_force_F_indices_start + 3
        BCs_force_F_indices = np.array( [ F_indices[BCs_force_F_indices_start[i]:BCs_force_F_indices_end[i]] for i in range(len(self.BCs_force_points_idx ))] ).reshape( -1 )

        BCs_momentum_F_indices_start = 6 * self.BCs_momentum_points_idx + 3
        BCs_momentum_F_indices_end = BCs_momentum_F_indices_start + 3
        BCs_momentum_F_indices = np.array( [ F_indices[BCs_momentum_F_indices_start[i]:BCs_momentum_F_indices_end[i]] for i in range(len(self.BCs_momentum_points_idx ))] ).reshape( -1 )

        BCs_F_indices = np.concatenate( ( BCs_force_F_indices,  BCs_momentum_F_indices ) )
        BCs_F_values = np.concatenate( ( self.BCs_force_values.reshape(-1), self.BCs_momentum_values.reshape(-1) ) )

        # Get the order of indices that would sort array a
        order = np.argsort( BCs_F_indices )

        self.BCs_F_indices = BCs_F_indices[ order ]
        self.BCs_F_values = BCs_F_values[ order ]

        self.check_valid_bounds()

    def check_valid_bounds( self  ):
        intersect_len = len( np.intersect1d( self.BCs_X_indices, self.BCs_F_indices ) )
        union_len = len( np.union1d( self.BCs_X_indices, self.BCs_F_indices ) )
        if intersect_len > 0:
            raise OverDefinedError("The problem is overdefined. ")
        elif union_len < self.n_DoFs:
            raise UnderDefinedError("The problem is underdefined. ")
        else:
            print( "Bounds are good to go!" )

    

        
        
