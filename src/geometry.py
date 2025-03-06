import numpy as np
# Things to consider in the future: 
# Is the points and connectivity privided are valid? (like, a single graph)
# what if there is a duplication? 

class DuplicationError(Exception):
    pass

class Point:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.coords = np.array([self.x, self.y, self.z])

class Element:
    def __init__(self, p0, p1, E, nu, A, Iy, Iz, I_rho, J, v_temp = None):
        self.p0 = p0
        self.p1 = p1
        self.E = E
        self.nu = nu
        self.A = A
        self.L = self.calc_connection_length( )
        self.Iy = Iy
        self.Iz = Iz
        self.I_rho = I_rho
        self.J = J
        self.v_temp = v_temp
    def calc_connection_length(self):
        length = np.sqrt( ( self.p0.x - self.p1.x )**2 + ( self.p0.y - self.p1.y )**2 + ( self.p0.z - self.p1.z )**2  )
        return length

class Frame:
    def __init__(self):
        self.points = None
        self.connectivities = None
        self.E_array = None
        self.nu_array = None
        self.A_array = None
        self.L_array = None
        self.Iy_array = None
        self.Iz_array = None
        self.I_rho_array = None
        self.J_array = None
        self.v_temp_array = None
         # a dictionary that gets the coordinations of a point as input and the key is the index
        self.points_unique = {}

    def add_point( self, x, y, z ):
        point = Point( x, y, z )
        return point

    def add_element( self, p0, p1, E, nu, A, Iy, Iz, I_rho, J, v_temp = None ):
        element = Element( p0, p1, E, nu, A, Iy, Iz, I_rho, J, v_temp )
        return element

    def build_frame( self, element_lst ):
        self.points_unique = {}
        point_lst = []
        connectivity_lst = []
        E_lst, nu_lst, A_lst, L_lst, Iy_lst, Iz_lst, I_rho_lst, J_lst, v_temp_lst = [], [], [], [], [], [], [], [], []
        for element in element_lst:
            p0_obj = element.p0
            p1_obj = element.p1
            p0 =  ( p0_obj.x, p0_obj.y, p0_obj.z )
            p1 = ( p1_obj.x, p1_obj.y, p1_obj.z )
            if p0 in self.points_unique.keys():
                p0_idx = self.points_unique[ p0 ]
            else:
                p0_idx = len( self.points_unique )
                self.points_unique[ p0 ] = p0_idx
                point_lst.append( p0 )
            if p1 in self.points_unique.keys():
                p1_idx = self.points_unique[ p1 ]
            else:
                p1_idx = len( self.points_unique )
                self.points_unique[ p1 ] = p1_idx
                point_lst.append( p1 )
            if ([p0_idx, p1_idx] in connectivity_lst) or ([p1_idx, p0_idx] in connectivity_lst):
                raise DuplicationError( 'Duplication in defining the elements! Be careful!' )
            connectivity_lst.append( [p0_idx, p1_idx] )
            E_lst.append( element.E )
            nu_lst.append( element.nu )
            A_lst.append( element.A )
            L_lst.append( element.L )
            Iy_lst.append( element.Iy )
            Iz_lst.append( element.Iz )
            I_rho_lst.append( element.I_rho )
            J_lst.append( element.J )
            v_temp_lst.append( element.v_temp )
        self.points = np.array( point_lst )
        self.connectivities = np.sort( np.array( connectivity_lst ), axis = 1 )
        self.E_array = np.array( E_lst )
        self.nu_array = np.array( nu_lst )
        self.A_array = np.array( A_lst )
        self.L_array = np.array( L_lst )
        self.Iy_array = np.array( Iy_lst )
        self.Iz_array = np.array( Iz_lst )
        self.I_rho_array = np.array( I_rho_lst )
        self.J_array = np.array( J_lst )
        self.v_temp_array = np.array( v_temp_lst )

        print( 'Your frame is good to go!' )   

    def calc_all_connections_lengths(self):
        lengths = []
        for connection in self.connectivities:
            p0_idx, p1_idx = connection
            p0 = self.points[p0_idx]
            p1 = self.points[p1_idx]
            length = np.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2 + (p0[2] - p1[2])**2)
            lengths.append(length)
        return np.array(lengths)

    def generate_frame_directly(self, points, connectivities, E_array, nu_array, A_array, Iy_array, Iz_array, I_rho_array, J_array, v_temp_array=None):
        self.points = points
        self.connectivities = np.sort(connectivities, axis=1)
        self.E_array = E_array
        self.nu_array = nu_array
        self.A_array = A_array
        self.Iy_array = Iy_array
        self.Iz_array = Iz_array
        self.I_rho_array = I_rho_array
        self.J_array = J_array
        self.v_temp_array = v_temp_array
        self.L_array = self

    
