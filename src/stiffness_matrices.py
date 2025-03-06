import numpy as np
from geometry import *
from boundary_conditions import *
'''-------------------------------------------------------------'''
'''--------------from Dr. Lejeune's course material---------------'''
'''-------------------------------------------------------------'''
def local_elastic_stiffness_matrix_3D_beam(E: float, nu: float, A: float, L: float, Iy: float, Iz: float, J: float) -> np.ndarray:
    k_e = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    k_e[0, 0] = axial_stiffness
    k_e[0, 6] = -axial_stiffness
    k_e[6, 0] = -axial_stiffness
    k_e[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    k_e[3, 3] = torsional_stiffness
    k_e[3, 9] = -torsional_stiffness
    k_e[9, 3] = -torsional_stiffness
    k_e[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 7] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 1] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
    k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 5] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 11] = E * -6.0 * Iz / L ** 2.0
    k_e[11, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[5, 5] = E * 4.0 * Iz / L
    k_e[11, 11] = E * 4.0 * Iz / L
    k_e[5, 11] = E * 2.0 * Iz / L
    k_e[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 8] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 2] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 4] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[2, 10] = E * -6.0 * Iy / L ** 2.0
    k_e[10, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
    k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[4, 4] = E * 4.0 * Iy / L
    k_e[10, 10] = E * 4.0 * Iy / L
    k_e[4, 10] = E * 2.0 * Iy / L
    k_e[10, 4] = E * 2.0 * Iy / L
    return k_e

# from Dr. Lejeune's course material
def check_unit_vector(vec: np.ndarray):
    if np.isclose(np.linalg.norm(vec), 1.0):
        return
    else:
        raise ValueError("Expected a unit vector for reference vector.")

# from Dr. Lejeune's course material
def check_parallel(vec_1: np.ndarray, vec_2: np.ndarray):
    if np.isclose(np.linalg.norm(np.cross(vec_1, vec_2)), 0.0):
        raise ValueError("Reference vector is parallel to beam axis.")
    else:
        return


def rotation_matrix_3D(x1: float, y1: float, z1: float, x2: float, y2: float, z2: float, v_temp: np.ndarray = None):
    # from Dr. Lejeune's course material
    L = np.sqrt((x2 - x1) ** 2.0 + (y2 - y1) ** 2.0 + (z2 - z1) ** 2.0)
    lxp = (x2 - x1) / L
    mxp = (y2 - y1) / L
    nxp = (z2 - z1) / L
    local_x = np.asarray([lxp, mxp, nxp])

    # choose a vector to orthonormalize the y axis if one is not given
    if v_temp is None:
        # if the beam is oriented vertically, switch to the global y axis
        if np.isclose(lxp, 0.0) and np.isclose(mxp, 0.0):
            v_temp = np.array([0, 1.0, 0.0])
        else:
            # otherwise use the global z axis
            v_temp = np.array([0, 0, 1.0])
    else:
        # check to make sure that given v_temp is a unit vector
        check_unit_vector(v_temp)
        # check to make sure that given v_temp is not parallel to the local x axis
        check_parallel(local_x, v_temp)
    
    # compute the local y axis
    local_y = np.cross(v_temp, local_x)
    local_y = local_y / np.linalg.norm(local_y)

    # compute the local z axis
    local_z = np.cross(local_x, local_y)
    local_z = local_z / np.linalg.norm(local_z)

    # assemble R
    gamma = np.vstack((local_x, local_y, local_z))
    
    return gamma

# from Dr. Lejeune's course material
def transformation_matrix_3D(gamma: np.ndarray) -> np.ndarray:
    Gamma = np.zeros((12, 12))
    Gamma[0:3, 0:3] = gamma
    Gamma[3:6, 3:6] = gamma
    Gamma[6:9, 6:9] = gamma
    Gamma[9:12, 9:12] = gamma
    return Gamma

# from Dr. Lejeune's course material
def local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
    k_g = np.zeros((12, 12))
    # upper triangle off diagonal terms
    k_g[0, 6] = -Fx2 / L
    k_g[1, 3] = My1 / L
    k_g[1, 4] = Mx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 9] = My2 / L
    k_g[1, 10] = -Mx2 / L
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 3] = Mz1 / L
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 5] = Mx2 / L
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 9] = Mz2 / L
    k_g[2, 10] = -Fx2 / 10.0
    k_g[2, 11] = -Mx2 / L
    k_g[3, 4] = -1.0 * (2.0 * Mz1 - Mz2) / 6.0
    k_g[3, 5] = (2.0 * My1 - My2) / 6.0
    k_g[3, 7] = -My1 / L
    k_g[3, 8] = -Mz1 / L
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[3, 10] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[3, 11] = (My1 + My2) / 6.0
    k_g[4, 7] = -Mx2 / L
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 9] = -1.0 * (Mz1 + Mz2) / 6.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[4, 11] = Mx2 / 2.0
    k_g[5, 7] = -Fx2 / 10.0
    k_g[5, 8] = -Mx2 / L
    k_g[5, 9] = (My1 + My2) / 6.0
    k_g[5, 10] = -Mx2 / 2.0
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 9] = -My2 / L
    k_g[7, 10] = Mx2 / L
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 9] = -Mz2 / L
    k_g[8, 10] = Fx2 / 10.0
    k_g[8, 11] = Mx2 / L
    k_g[9, 10] = (Mz1 - 2.0 * Mz2) / 6.0
    k_g[9, 11] = -1.0 * (My1 - 2.0 * My2) / 6.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g

# from Dr. Lejeune's course material
def local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2):
    k_g = np.zeros((12, 12))
    k_g[0, 6] = -Fx2 / L
    k_g[1, 5] = Fx2 / 10.0
    k_g[1, 7] = -6.0 * Fx2 / (5.0 * L)
    k_g[1, 11] = Fx2 / 10.0
    k_g[2, 4] = -Fx2 / 10.0
    k_g[2, 8] = -6.0 * Fx2 / (5.0 * L)
    k_g[2, 10] = -Fx2 / 10.0
    k_g[3, 9] = -Fx2 * I_rho / (A * L)
    k_g[4, 8] = Fx2 / 10.0
    k_g[4, 10] = -Fx2 * L / 30.0
    k_g[5, 7] = -Fx2 / 10
    k_g[5, 11] = -Fx2 * L / 30.0
    k_g[7, 11] = -Fx2 / 10.0
    k_g[8, 10] = Fx2 / 10.0
    # add in the symmetric lower triangle
    k_g = k_g + k_g.transpose()
    # add diagonal terms
    k_g[0, 0] = Fx2 / L
    k_g[1, 1] = 6.0 * Fx2 / (5.0 * L)
    k_g[2, 2] = 6.0 * Fx2 / (5.0 * L)
    k_g[3, 3] = Fx2 * I_rho / (A * L)
    k_g[4, 4] = 2.0 * Fx2 * L / 15.0
    k_g[5, 5] = 2.0 * Fx2 * L / 15.0
    k_g[6, 6] = Fx2 / L
    k_g[7, 7] = 6.0 * Fx2 / (5.0 * L)
    k_g[8, 8] = 6.0 * Fx2 / (5.0 * L)
    k_g[9, 9] = Fx2 * I_rho / (A * L)
    k_g[10, 10] = 2.0 * Fx2 * L / 15.0
    k_g[11, 11] = 2.0 * Fx2 * L / 15.0
    return k_g
'''-------------------------------------------------------------'''
'''------End of the codes from Dr. Lejeune's course material-----'''
'''-------------------------------------------------------------'''
class StiffnessMatrices:
    def __init__(self, frame_obj):
        self.frame = frame_obj
        self.n_points = self.frame.points.shape[0]
        self.n_connectivities = self.frame.connectivities.shape[0]
        self.n_DoFs = self.n_points * 6

    def get_element_parameters( self, element_idx ):
        connection = self.frame.connectivities[ element_idx ]
        E = self.frame.E_array[ element_idx ]
        nu = self.frame.nu_array[ element_idx ]
        A = self.frame.A_array[ element_idx ]
        L = self.frame.L_array[ element_idx ]
        Iy = self.frame.Iy_array[ element_idx ]
        Iz = self.frame.Iz_array[ element_idx ]
        I_rho = self.frame.I_rho_array[ element_idx ]
        J = self.frame.J_array[ element_idx ]
        v_temp = self.frame.v_temp_array[ element_idx ]
        return connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp

    def get_element_points( self, connection ):
        p0_idx = connection[0]
        p0 = self.frame.points[ connection[ 0 ] ]
        p1_idx = connection[1]
        p1 = self.frame.points[ connection[ 1 ] ]
        return p0_idx, p0, p1_idx, p1

    def get_transformation_matrix_3D( self, p0, p1, v_temp ):
        gamma = rotation_matrix_3D(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], v_temp)
        Gamma = transformation_matrix_3D(gamma)
        return Gamma

    def get_global_elastic_stiffmatrix( self ):
        K = np.zeros(( self.n_DoFs, self.n_DoFs ) )
        for element_idx in range( self.n_connectivities ):
            connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
            p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
            
            k_element_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J )
            Gamma = self.get_transformation_matrix_3D( p0, p1, v_temp)
            k_element_global = Gamma.T @ k_element_local @ Gamma
            k_element_p00 = k_element_global[0:6, 0:6]
            k_element_p01 = k_element_global[0:6, 6:12]
            k_element_p10 = k_element_global[6:12, 0:6]
            k_element_p11 = k_element_global[6:12, 6:12]
            
            p0_DoF_idx = p0_idx * 6
            p1_DoF_idx = p1_idx * 6
            K[ p0_DoF_idx : p0_DoF_idx + 6, p0_DoF_idx:  p0_DoF_idx + 6  ] += k_element_p00
            K[ p0_DoF_idx : p0_DoF_idx + 6, p1_DoF_idx:  p1_DoF_idx + 6  ] += k_element_p01
            K[ p1_DoF_idx : p1_DoF_idx + 6, p0_DoF_idx:  p0_DoF_idx + 6  ] += k_element_p10
            K[ p1_DoF_idx : p1_DoF_idx + 6, p1_DoF_idx:  p1_DoF_idx + 6  ] += k_element_p11
        return K

    def get_element_global_Delta(self, connection, Delta):
        p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
        return np.concatenate( (Delta[ p0_idx * 6 : p0_idx * 6 + 6] , Delta[ p1_idx * 6 : p1_idx * 6 + 6 ] ) )
        
    def get_element_local_internal_F(self, element_idx, Delta ):
        connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
        p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
        Delta_el_global = np.concatenate( (Delta[ p0_idx * 6 : p0_idx * 6 + 6] , Delta[ p1_idx * 6 : p1_idx * 6 + 6 ] ) )
        Gamma = self.get_transformation_matrix_3D( p0, p1, v_temp)
        Delta_el_local = Gamma @ Delta_el_global
        K_el_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J)
        F_el_local = K_el_local @ Delta_el_local
        return F_el_local
    # use the local F to calculate the local k_g
    def get_element_local_geometric_stiffness_matrix( self, element_idx, Delta):
        connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
        F_el_local = self.get_element_local_internal_F(element_idx, Delta )
        My0 = F_el_local[4]
        Mz0 = F_el_local[5]
        Fx1 = F_el_local[6]
        Mx1 = F_el_local[9]
        My1 = F_el_local[10]
        Mz1 = F_el_local[11]
        k_geom_element_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx1, Mx1, My0, Mz0, My1, Mz1)
        return k_geom_element_local

    def get_global_geometric_stiffmatrix(self, Delta):
        K_g = np.zeros((self.n_DoFs, self.n_DoFs))
        for element_idx in range(self.n_connectivities):
            connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters(element_idx)
            p0_idx, p0, p1_idx, p1 = self.get_element_points(connection)
            
            k_g_element_local = self.get_element_local_geometric_stiffness_matrix(element_idx, Delta)
            Gamma = self.get_transformation_matrix_3D(p0, p1, v_temp)
            k_g_element_global = Gamma.T @ k_g_element_local @ Gamma
            k_g_element_p00 = k_g_element_global[0:6, 0:6]
            k_g_element_p01 = k_g_element_global[0:6, 6:12]
            k_g_element_p10 = k_g_element_global[6:12, 0:6]
            k_g_element_p11 = k_g_element_global[6:12, 6:12]
            
            p0_DoF_idx = p0_idx * 6
            p1_DoF_idx = p1_idx * 6
            K_g[p0_DoF_idx: p0_DoF_idx + 6, p0_DoF_idx: p0_DoF_idx + 6] += k_g_element_p00
            K_g[p0_DoF_idx: p0_DoF_idx + 6, p1_DoF_idx: p1_DoF_idx + 6] += k_g_element_p01
            K_g[p1_DoF_idx: p1_DoF_idx + 6, p0_DoF_idx: p0_DoF_idx + 6] += k_g_element_p10
            K_g[p1_DoF_idx: p1_DoF_idx + 6, p1_DoF_idx: p1_DoF_idx + 6] += k_g_element_p11
        return K_g

