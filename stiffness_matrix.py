import numpy as np
import geometry as geom
from math_utils import *
from boundary_conditions import *

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

    def get_transformation_matrix_3D( self, p0, p1, v_temp):
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

    # def get_F_el_global( self, p0_idx, p1_idx, F):
    #     F_el_global = np.concatenate( ( F[p0_idx*6: p0_idx*6 + 6], F[p1_idx*6: p1_idx*6+6] ))
    #     return F_el_global

    def get_F_el_local( self, element_idx, F ):
        connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
        p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
        Gamma = self.get_transformation_matrix_3D( p0, p1, v_temp)
        F_el_global = self.get_F_el_global( p0_idx, p1_idx, F)
        F_el_local = Gamma.T @ F_el_global
        return F_el_local

    def get_F_el_local( self, element_idx, F ):
        connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
        p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
        return self.get_F_el_global( p0_idx, p1_idx, F)

    def get_local_geometric_stiffness_matrix_3D_beam(self, element_idx, F):
        connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
        F_el_local = self.get_F_el_local( element_idx, F )
        My0 = F_el_local[4]
        Mz0 = F_el_local[5]
        Fx1 = F_el_local[6]
        Mx1 = F_el_local[9]
        My1 = F_el_local[10]
        Mz1 = F_el_local[11]
        k_geom_element_local = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx1, Mx1, My0, Mz0, My1, Mz1)
        return k_geom_element_local

    def get_global_geometric_stiffmat( self, F ):
        K = np.zeros(( self.n_DoFs, self.n_DoFs ) )
        for element_idx in range( self.n_connectivities ):
            connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = self.get_element_parameters( element_idx )
            p0_idx, p0, p1_idx, p1 = self.get_element_points( connection )
            
            k_element_local = self.get_local_geometric_stiffness_matrix_3D_beam( element_idx, F)
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

        

