import numpy as np
import geometry as geom

class OverDefinedError( Exception ):
    pass

class UnderDefinedError( Exception ):
    pass


class BoundaryConditions:
    def __init__(self, frame_obj):
        self.frame = frame_obj
        self.n_points = self.frame.points.shape[0]
        self.n_connectivities = self.frame.connectivities.shape[0]
        self.n_DoFs = self.n_points * 6

        # disp bounds
        self.BCs_disp_indices = []
        self.BCs_disp_values = []

        # rotation bounds
        self.BCs_rot_indices = []
        self.BCs_rot_values = []

        # force bounds
        self.BCs_force_indices = []
        self.BCs_force_values = []

        # momentum bounds
        self.BCs_momentum_indices = []
        self.BCs_momentum_values = []

        # wrapping up bounds
        self.BCs_X_indices = []
        self.BCs_X_values = []

        self.BCs_F_indices = []
        self.BCs_F_values = []

    def add_disp_bound_xyz(self, point_coords, value_x, value_y, value_z):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_disp_indices = self.BCs_disp_indices + [ idx * 6, idx * 6 + 1, idx * 6 + 2 ]
        self.BCs_disp_values = self.BCs_disp_values + [ value_x, value_y, value_z ]

    def add_disp_bound_x(self, point_coords, value: float ):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_disp_indices = self.BCs_disp_indices + [ idx * 6 ]
        self.BCs_disp_values = self.BCs_disp_values + [ value ]

    def add_disp_bound_y(self, point_coords, value: float):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_disp_indices = self.BCs_disp_indices + [ idx * 6 + 1 ]
        self.BCs_disp_values = self.BCs_disp_values + [ value ]

    def add_disp_bound_z(self, point_coords, value: float):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_disp_indices = self.BCs_disp_indices + [ idx * 6 + 2 ]
        self.BCs_disp_values = self.BCs_disp_values + [ value ]

    def add_rot_bound_xyz(self, point_coords, value_x, value_y, value_z):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_rot_indices = self.BCs_rot_indices + [ idx * 6 + 3, idx * 6 + 4, idx * 6 + 5 ]
        self.BCs_rot_values = self.BCs_rot_values + [ value_x, value_y, value_z ]

    def add_rot_bound_x(self, point_coords, value: float ):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_rot_indices = self.BCs_rot_indices + [ idx * 6 + 3]
        self.BCs_rot_values = self.BCs_rot_values + [ value ]

    def add_rot_bound_y(self, point_coords, value: float):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_rot_indices = self.BCs_rot_indices + [ idx * 6 + 4 ]
        self.BCs_rot_values = self.BCs_rot_values + [ value ]

    def add_rot_bound_z(self, point_coords, value: float):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_rot_indices = self.BCs_rot_indices + [ idx * 6 + 5 ]
        self.BCs_rot_values = self.BCs_rot_values + [ value ]

    def add_force_bound_xyz(self, point_coords, value_x, value_y, value_z):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_force_indices = self.BCs_force_indices + [ idx * 6, idx * 6 + 1, idx * 6 + 2 ]
        self.BCs_force_values = self.BCs_force_values + [ value_x, value_y, value_z ]

    def add_force_bound_x(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_force_indices = self.BCs_force_indices + [ idx * 6 ]
        self.BCs_force_values = self.BCs_force_values + [ value ]

    def add_force_bound_y(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_force_indices = self.BCs_force_indices + [ idx * 6 + 1 ]
        self.BCs_force_values = self.BCs_force_values + [ value ]

    def add_force_bound_z(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_force_indices = self.BCs_force_indices + [ idx * 6 + 2 ]
        self.BCs_force_values = self.BCs_force_values + [ value ]

    def add_momentum_bound_xyz(self, point_coords, value_x, value_y, value_z):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_momentum_indices = self.BCs_momentum_indices + [ idx * 6 + 3, idx * 6 + 4, idx * 6 + 5 ]
        self.BCs_momentum_values = self.BCs_momentum_values + [ value_x, value_y, value_z ]

    def add_momentum_bound_x(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_momentum_indices = self.BCs_momentum_indices + [ idx * 6 + 3 ]
        self.BCs_momentum_values = self.BCs_momentum_values + [ value ]

    def add_momentum_bound_y(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_momentum_indices = self.BCs_momentum_indices + [ idx * 6 + 4 ]
        self.BCs_momentum_values = self.BCs_momentum_values + [ value ]

    def add_momentum_bound_z(self, point_coords, value):
        point_coords_key = tuple( point_coords )
        idx = self.frame.points_unique[ point_coords_key ] 

        self.BCs_momentum_indices = self.BCs_momentum_indices + [ idx * 6 + 5 ]
        self.BCs_momentum_values = self.BCs_momentum_values + [ value ]

    def validate_bounds( self  ):
        intersect_len = len( np.intersect1d( self.BCs_X_indices, self.BCs_F_indices ) )
        union_len = len( np.union1d( self.BCs_X_indices, self.BCs_F_indices ) )
        if intersect_len > 0:
            raise OverDefinedError("The problem is overdefined. ")
        elif union_len < self.n_DoFs:
            
            raise UnderDefinedError("The problem is underdefined. ")
        else:
            print( "Bounds are good to go!" )

    def set_up_bounds( self ):
        '''known X indices'''
        BCs_X_indices = np.concatenate( (  self.BCs_disp_indices,  self.BCs_rot_indices ) ) 
        BCs_X_values = np.concatenate( ( self.BCs_disp_values, self.BCs_rot_values ) )

        # Get the order of indices that would sort array a
        order = np.argsort( BCs_X_indices )

        self.BCs_X_indices = BCs_X_indices[ order ]
        self.BCs_X_values = BCs_X_values[ order ]

        '''known F indices'''
        BCs_F_indices = np.concatenate( (  self.BCs_force_indices,  self.BCs_momentum_indices ) ) 
        BCs_F_values = np.concatenate( ( self.BCs_force_values, self.BCs_momentum_values ) )

        # Get the order of indices that would sort array a
        order = np.argsort( BCs_F_indices )

        self.BCs_F_indices = BCs_F_indices[ order ]
        self.BCs_F_values = BCs_F_values[ order ]

        self.validate_bounds()
