import numpy as np
from geometry import *
from math_utils import *
from boundary_conditions import *
from direct_stiffness_method import *
from solver import *

'''-------------------------------------------------------------'''
'''------------------Defining The Structure---------------------'''
'''-------------------------------------------------------------'''
#------Parameters-------
r = 1
E = 500
nu = 0.3
A = np.pi * r **2
Iy = np.pi * r ** 4 / 4
Iz = np.pi * r ** 4 / 4
J = np.pi * r ** 4 / 2

#-----Building the Frame-----
frame = Frame()
p0 = frame.add_point(0, 0, 0)
p1 = frame.add_point(-5, 1, 10)
p2 = frame.add_point(-1, 5, 13)
p3 = frame.add_point(-3, 7, 11)
p4 = frame.add_point(6, 9, 5)

el0 = frame.add_element( p0, p1, E, nu, A, Iy, Iz, J )
el1 = frame.add_element( p1, p2, E, nu, A, Iy, Iz, J )
el2 = frame.add_element( p2, p3, E, nu, A, Iy, Iz, J )
el3 = frame.add_element( p2, p4, E, nu, A, Iy, Iz, J )
element_lst = [el0, el1, el2, el3]
frame.build_frame( element_lst )
'''-------------------------------------------------------------'''
'''-------------------Boundary Conditions-----------------------'''
'''-------------------------------------------------------------'''
bcs = BoundaryConditions( frame )
# displacement bounds
bcs.add_disp_bound_z( [0, 0, 0], 0 )
bcs.add_disp_bound_xyz( [-3, 7, 11], 0, 0, 0 )
bcs.add_disp_bound_xyz( [6, 9, 5], 0, 0, 0 )

# rotation bounds
bcs.add_rot_bound_xyz( [-3, 7, 11], 0, 0, 0 )

# force bounds
bcs.add_force_bound_x( [0, 0, 0], 0 )
bcs.add_force_bound_y( [0, 0, 0], 0 )
bcs.add_force_bound_xyz( [-5, 1, 10], 0.1, -0.05, -0.075 )
bcs.add_force_bound_xyz( [-1, 5, 13], 0, 0, 0 )

# momentum bounds
bcs.add_momentum_bound_xyz( [0, 0, 0], 0, 0, 0 )
bcs.add_momentum_bound_xyz( [-5, 1, 10], 0, 0, 0 )
bcs.add_momentum_bound_xyz( [-1, 5, 13], 0.5, -0.1, 0.3 )
bcs.add_momentum_bound_xyz( [6, 9, 5], 0, 0, 0 )


# we have to set up the bounds in the end
bcs.set_up_bounds()

'''-------------------------------------------------------------'''
'''-----------Build the Global Stiffness Matrix-----------------'''
'''-------------------------------------------------------------'''
smat = StiffnessMat( frame )
K = smat.get_stiffmat_global()

'''-------------------------------------------------------------'''
'''-------------------Solving for unknowns----------------------'''
'''-------------------------------------------------------------'''
X, F = solve_stiffness_system( K, bcs )

node_idx = 0
print( f'Disp/rotations at Node { node_idx }:', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )
