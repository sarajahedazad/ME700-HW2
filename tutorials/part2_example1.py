import numpy as np
from geometry import *
from boundary_conditions import *
from stiffness_matrices import *
from solver import *
from shape_functions import * 
'''-------------------------------------------------------------'''
'''------------------Defining The Structure---------------------'''
'''-------------------------------------------------------------'''
#------Parameters-------
r = 1
E = 1000
nu = 0.3
A = np.pi * r **2
Iy = np.pi * r ** 4 / 4
Iz = np.pi * r ** 4 / 4
I_rho = np.pi * r ** 4 / 2
J = np.pi * r ** 4 / 2
L = 10 
P_applied = 1

#-----Building the Frame-----
frame = Frame()
p0 = frame.add_point(0, 0, 0)
p1 = frame.add_point(30, 40, 0)

el0 = frame.add_element( p0, p1, E, nu, A, Iy, Iz, I_rho, J )
element_lst = [el0]
frame.build_frame( element_lst )
'''-------------------------------------------------------------'''
'''-------------------Boundary Conditions-----------------------'''
'''-------------------------------------------------------------'''
bcs = BoundaryConditions( frame )
# displacement bounds
bcs.add_disp_bound_xyz( [0, 0, 0], 0, 0, 0 )

# rotation bounds
bcs.add_rot_bound_xyz( [0, 0, 0], 0, 0, 0 )

# force bounds
bcs.add_force_bound_xyz( [30, 40, 0], - 3 /5 , -4/5, 0 )

# momentum bounds
bcs.add_momentum_bound_xyz( [30, 40, 0], 0, 0, 0 )

# we have to set up the bounds in the end
bcs.set_up_bounds()

'''-------------------------------------------------------------'''
'''-----------Build the Global Stiffness Matrix-----------------'''
'''-------------------------------------------------------------'''
stiffmat = StiffnessMatrices( frame )
K = stiffmat.get_global_elastic_stiffmatrix()


'''-------------------------------------------------------------'''
'''-------------------Solving for unknowns----------------------'''
'''-------------------------------------------------------------'''
Delta, F = solve_stiffness_system( K, bcs )

node_idx = 0
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 1
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

'''-------------------------------------------------------------'''
'''-------------------Critical Loads----------------------'''
'''-------------------------------------------------------------'''

K_g = stiffmat.get_global_geometric_stiffmatrix( Delta )

P_cr, eigenvectors_allstructure = compute_critical_load(K, K_g, bcs)
print( 'critical', P_cr )

'''-------------------------------------------------------------'''
'''-------------------Interpolation----------------------'''
'''-------------------------------------------------------------'''
n , scale = 20, 10
shapefunctions = ShapeFunctions(eigenvectors_allstructure, frame, n=n, scale=scale)
saving_dir_with_name = 'Original Configuration vs Interpolated.png'
shapefunctions.plot_element_interpolation( saving_dir_with_name )