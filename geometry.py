import numpy as np
# Things to consider in the future: an alternative method to build the frame:
# Like defining point class, line class, etc
# Is the points and connectivity privided are valid? (like, a single graph)
# what if there is a duplication? 
class Frame: #
    def __init__(self):
        self.points = None
        self.connectivity = None
        self.E_array = None
        self.nu_array = None
        self.A_array = None
        self.L_array = None
        self.Iy_array = None
        self.Iz_array = None
        self.J_array = None
        self.I_z_array = None

    def calc_single_connection_length( self, p0_coords, p1_coords ):
        length = np.linalg.norm( p0_coords - p1_coords)
        return length

    def calc_all_connections_lengths( self ):
        length_lst = []
        for connection in self.connectivity:
            p0 = self.points[ connection[0] ]
            p1 = self.points[ connection[1] ]
            length = self.calc_single_connection_length( p0, p1 )
            length_lst.append( length )
        return np.array( length_lst )

    def generate_frame_directly( self, points, connectivity, E_array, nu_array, A_array, Iy_array, Iz_array, J_array, I_z_array = None ):
        self.points = points
        self.connectivity = connectivity
        self.E_array = E_array
        self.nu_array = nu_array
        self.A_array = A_array
        self.Iy_array = Iy_array
        self.Iz_array = Iz_array
        self.J_array = J_array
        self.I_z_array = I_z_array

        self.L_array = self.calc_all_connections_lengths()

        

