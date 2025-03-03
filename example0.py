import numpy as np
from geometry import *
from math_utils import *
from boundary_conditions import *
from stiffness_matrices import *
from solver import *
from plotting import *
'''-------------------------------------------------------------'''
'''------------------Defining The Structure---------------------'''
'''-------------------------------------------------------------'''
#------Parameters-------
r = 0.25
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
p1 = frame.add_point(0, L, 0)

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
bcs.add_force_bound_xyz( [0, L, 0], 0, -P_applied, 0 )

# momentum bounds
bcs.add_momentum_bound_xyz( [0, L, 0], 0, 0, 0 )

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
X, F = solve_stiffness_system( K, bcs )

node_idx = 0
print( f'Disp/rotations at Node { node_idx }:', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 1
print( f'Disp/rotations at Node { node_idx }:', X[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

'''-------------------------------------------------------------'''
'''----------------Getting Geometric Stiffness------------------'''
'''-------------------------------------------------------------'''
K_geom = stiffmat.get_global_geometric_stiffmat( F )
lambda_critical = compute_critical_load(K, K_geom)
print( 'lambda_critical', lambda_critical )

'''-------------------------------------------------------------'''
'''----------------Theoritcal Critical Force--------------------'''
'''-------------------------------------------------------------'''
Le = 2 * L
P_cr_theo = np.pi ** 2 * E * Iz / Le**2
print( "P cr theory is ", P_cr_theo)

'''------------Plot interpolated shape-----------'''
saving_dir_with_name = 'original_conf.png'
plot_original_configuration( frame, saving_dir_with_name )
saving_dir_with_name = 'orig_vs_deformed_conf.png'
plot_original_vs_deformed_configurations( frame, X, saving_dir_with_name)
# saving_dir_with_name = 'new_stuff.png'
# plot_deformed_frame(frame, X, saving_dir_with_name)
