import numpy as np
from geometry import *
from boundary_conditions import *
from stiffness_matrices import *
from solver import *
import matplotlib.pyplot as plt
from shape_functions import *
'''-------------------------------------------------------------'''
'''------------------Defining The Structure---------------------'''
'''-------------------------------------------------------------'''
#------Parameters-------
r = 0.5
E = 500
nu = 0.3
A = np.pi * r **2
Iy = np.pi * r ** 4 / 4
Iz = np.pi * r ** 4 / 4
I_rho = np.pi * r ** 4 / 2
J = np.pi * r ** 4 / 2

P_applied = 1

#-----Building the Frame-----
frame = Frame()
p0 = frame.add_point(0, 0, 0)
p1 = frame.add_point(10, 0, 0)
p2 = frame.add_point(10, 20, 0)
p3 = frame.add_point(0, 20, 0)
p4 = frame.add_point(0, 0, 25)
p5 = frame.add_point(10, 0, 25)
p6 = frame.add_point(10, 20, 25)
p7 = frame.add_point(0, 20, 25)

el0 = frame.add_element( p0, p4, E, nu, A, Iy, Iz, I_rho, J )
el1 = frame.add_element( p1, p5, E, nu, A, Iy, Iz, I_rho, J )
el2 = frame.add_element( p2, p6, E, nu, A, Iy, Iz, I_rho, J )
el3 = frame.add_element( p3, p7, E, nu, A, Iy, Iz, I_rho, J )
el4 = frame.add_element( p4, p5, E, nu, A, Iy, Iz, I_rho, J )
el5 = frame.add_element( p5, p6, E, nu, A, Iy, Iz, I_rho, J )
el6 = frame.add_element( p6, p7, E, nu, A, Iy, Iz, I_rho, J )
el7 = frame.add_element( p7, p4, E, nu, A, Iy, Iz, I_rho, J )

element_lst = [el0, el1, el2, el3, el4, el5, el6, el7]
frame.build_frame( element_lst )
'''-------------------------------------------------------------'''
'''-------------------Boundary Conditions-----------------------'''
'''-------------------------------------------------------------'''
bcs = BoundaryConditions( frame )
# displacement bounds
bcs.add_disp_bound_xyz( p0.coords, 0, 0, 0 )
bcs.add_disp_bound_xyz( p1.coords, 0, 0, 0 )
bcs.add_disp_bound_xyz( p2.coords, 0, 0, 0 )
bcs.add_disp_bound_xyz( p3.coords, 0, 0, 0 )

# rotation bounds
bcs.add_rot_bound_xyz( p0.coords, 0, 0, 0 )
bcs.add_rot_bound_xyz( p1.coords, 0, 0, 0 )
bcs.add_rot_bound_xyz( p2.coords, 0, 0, 0 )
bcs.add_rot_bound_xyz( p3.coords, 0, 0, 0 )

# force bounds
bcs.add_force_bound_xyz( p4.coords, 0, 0, -P_applied )
bcs.add_force_bound_xyz( p5.coords, 0, 0, -P_applied )
bcs.add_force_bound_xyz( p6.coords, 0, 0, -P_applied )
bcs.add_force_bound_xyz( p7.coords, 0, 0, -P_applied )

# momentum bounds
bcs.add_momentum_bound_xyz( p4.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p5.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p6.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p7.coords, 0, 0, 0 )

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

node_idx = 2
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 3
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 4
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 5
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 6
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 7
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
n , scale = 20, 5
shapefunctions = ShapeFunctions(eigenvectors_allstructure, frame, n=n, scale=scale)
saving_dir_with_name = 'Original Configuration vs Interpolated.png'
shapefunctions.plot_element_interpolation( saving_dir_with_name )