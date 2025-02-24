import numpy as np
import geometry as geom
from math_utils import *
from boundary_conditions import *
from direct_stiffness_method import *
from solver import *

#--------defining the structure----------
points = np.array([ [0, 0, 0], [0, 20, 0], [20, 15, 0] ])
connectivity = np.array([[0, 1], [1, 2]])
E_array = np.array( [1000, 1000] )
nu_array = np.array( [0.33, 0.33] )
A_array = np.array( [ 0.78, 0.78 ] )
Iy_array = np.array( [ 0.05, 0.05 ] )
Iz_array = np.array( [ 0.05, 0.05 ] )
J_array = np.array( [376, 376] )

frame = geom.Frame()
frame.generate_frame_directly( points, connectivity, E_array, nu_array, A_array, Iy_array, Iz_array, J_array )
#--------------Defining the Bounds-------------------
BCs_disp_points = np.array( [ [0, 0, 0], [20, 15, 0] ] )
BCs_disp_values = np.array( [[0, 0, 0], [0, 0, 0]] )

BCs_rot_points = np.array( [[0, 0, 0]] )
BCs_rot_values = np.array( [[0, 0, 0]] )

BCs_force_points = np.array( [[0, 20, 0]] )
BCs_force_values = np.array( [[0, -1, 0]] )

BCs_momentum_points = np.array( [[0, 20, 0], [20, 15, 0]] )
BCs_momentum_values = np.array( [[0, 0, 1], [0, 0, 0]] )

bcs = BoundaryConditions( frame )
bcs.get_disp_bounds( BCs_disp_points, BCs_disp_values )
bcs.get_rot_bounds( BCs_rot_points, BCs_rot_values )
bcs.get_force_bounds( BCs_force_points, BCs_force_values )
bcs.get_momentum_bounds( BCs_momentum_points, BCs_momentum_values )

bcs.set_up_bounds()

#------------Build the Global Stiffness Matrix-----

smat = StiffnessMat( frame )
K = smat.get_stiffmat_global()

#-----------Solving for unknowns----------
X, F = solve_stiffness_system( K, bcs )


