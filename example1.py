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
b, h = 0.5, 1
E = 1000
nu = 0.3
A = b * h
Iy = h * b**3 / 12
Iz = b * h**3 / 12
J = 0.02861
v_temp_0 = np.array( [0, 0, 1] )
v_temp_1 = np.array( [1, 0, 0] )

#-----Building the Frame-----
frame = Frame()
p0 = frame.add_point(0, 0, 10)
p1 = frame.add_point(15, 0, 10)
p2 = frame.add_point(15, 0, 0)

el0 = frame.add_element( p0, p1, E, nu, A, Iy, Iz, J, v_temp_0 )
el1 = frame.add_element( p1, p2, E, nu, A, Iy, Iz, J, v_temp_1 )
element_lst = [el0, el1]
frame.build_frame( element_lst )
'''-------------------------------------------------------------'''
'''-------------------Boundary Conditions-----------------------'''
'''-------------------------------------------------------------'''
bcs = BoundaryConditions( frame )
# displacement bounds
bcs.add_disp_bound_xyz( [0, 0, 10], 0, 0, 0 )
bcs.add_disp_bound_xyz( [15, 0, 0], 0, 0, 0 )

# rotation bounds
bcs.add_rot_bound_xyz( [0, 0, 10], 0, 0, 0 )

# force bounds
bcs.add_force_bound_xyz( [15, 0, 10], 0.1, 0.05, -0.07 )

# momentum bounds
bcs.add_momentum_bound_xyz( [15, 0, 10], 0.05, -0.1, 0.25 )
bcs.add_momentum_bound_xyz( [15, 0, 0], 0, 0, 0 )

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
print( f'Disp/rotations at Node { node_idx }: ', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }: ', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 1
print( f'Disp/rotations at Node { node_idx }: ', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }: ', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 2
print( f'Disp/rotations at Node { node_idx }: ', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }: ', F[node_idx * 6: node_idx * 6 + 6] )
