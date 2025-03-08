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
# for a
r = 1
E_a = 10000
nu_a = 0.3
A_a = np.pi * r **2
Iy_a = np.pi * r ** 4 / 4
Iz_a = np.pi * r ** 4 / 4
I_rho_a = np.pi * r ** 4 / 2
J_a = np.pi * r ** 4 / 2
local_z_a = None

# for b
b, h = 0.5, 1
E_b = 50000
nu_b = 0.3
A_b = b * h
Iy_b = h * b ** 3.0 / 12.0
Iz_b = b * h ** 3.0 / 12.0
I_rho_b = b * h / 12.0 * (b ** 2.0 + h ** 2.0)
J_b = 0.028610026041666667
local_z_b = np.asarray([0, 0, 1])

#-----Building the Frame-----
x0, y0, z0 = 0, 0, 0

L1, L2, L3, L4 = 15, 30, 14, 16
#L1, L2, L3, L4 = 11, 23, 15, 13

frame = Frame()
p0 = frame.add_point(x0, y0, z0)
p1 = frame.add_point(x0 + L1, y0, z0)
p2 = frame.add_point(x0 + L1, y0 + L2, z0)
p3 = frame.add_point(x0, y0 + L2, z0)
p4 = frame.add_point(x0, y0, z0 + L3)
p5 = frame.add_point(x0 + L1, y0, z0 + L3)
p6 = frame.add_point(x0 + L1, y0 + L2, z0 + L3)
p7 = frame.add_point(x0, y0 + L2, z0 + L3)
p8 = frame.add_point(x0, y0, z0 + L3 + L4)
p9 = frame.add_point(x0 + L1, y0 , z0 + L3 + L4)
p10 = frame.add_point(x0 + L1, y0 + L2, z0 + L3 + L4)
p11 = frame.add_point(x0, y0 + L2, z0 + L3 + L4)

el0 = frame.add_element( p0, p4, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el1 = frame.add_element( p1, p5, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el2 = frame.add_element( p2, p6, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el3 = frame.add_element( p3, p7, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el4 = frame.add_element( p4, p8, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el5 = frame.add_element( p5, p9, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el6 = frame.add_element( p6, p10, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el7 = frame.add_element( p7, p11, E_a, nu_a, A_a, Iy_a, Iz_a, I_rho_a, J_a )
el8 = frame.add_element( p4, p5, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el9 = frame.add_element( p5, p6, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el10 = frame.add_element( p6, p7, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el11 = frame.add_element( p7, p4, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el12 = frame.add_element( p8, p9, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el13 = frame.add_element( p9, p10, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el14 = frame.add_element( p10, p11, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
el15 = frame.add_element( p11, p8, E_b, nu_b, A_b, Iy_b, Iz_b, I_rho_b, J_b )
element_lst = [el0, el1, el2, el3, el4, el5, el6, el7, el8, el9, el10, el11, el12, el13, el14, el15]
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
bcs.add_force_bound_xyz( p8.coords, 0, 0, -1 )
bcs.add_force_bound_xyz( p9.coords, 0, 0, -1 )
bcs.add_force_bound_xyz( p10.coords, 0, 0, -1 )
bcs.add_force_bound_xyz( p11.coords, 0, 0, -1 )

bcs.add_force_bound_xyz( p4.coords, 0, 0, 0)
bcs.add_force_bound_xyz( p5.coords, 0, 0, 0)
bcs.add_force_bound_xyz( p6.coords, 0, 0, 0)
bcs.add_force_bound_xyz( p7.coords, 0, 0, 0)

# momentum bounds
bcs.add_momentum_bound_xyz( p4.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p5.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p6.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p7.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p8.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p9.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p10.coords, 0, 0, 0 )
bcs.add_momentum_bound_xyz( p11.coords, 0, 0, 0 )

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

node_idx = 3
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 6
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
n, scale = 20, 10
shapefunctions = ShapeFunctions(eigenvectors_allstructure, frame, n=n, scale=scale)
saving_dir_with_name = 'Original Configuration vs Interpolated.png'
shapefunctions.plot_element_interpolation( saving_dir_with_name )