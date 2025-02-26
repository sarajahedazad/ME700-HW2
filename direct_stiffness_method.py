import numpy as np
import geometry as geom
from math_utils import *
from boundary_conditions import *

class StiffnessMat:
    def __init__(self, frame_obj):
        self.frame = frame_obj
        self.n_points = self.frame.points.shape[0]
        self.n_connectivities = self.frame.connectivities.shape[0]
        self.n_DoFs = self.n_points * 6

    def get_stiffmat_global( self ):
        K = np.zeros(( self.n_DoFs, self.n_DoFs ) )
        for i in range( self.n_connectivities ):
            connection = self.frame.connectivities[i]
            p0_idx = connection[0]
            p1_idx = connection[1]

            p0 = self.frame.points[ p0_idx ]
            p1 = self.frame.points[ p1_idx ]

            E = self.frame.E_array[ i ]
            nu = self.frame.nu_array[ i ]
            A = self.frame.A_array[ i ]
            L = self.frame.L_array[ i ]
            Iy = self.frame.Iy_array[ i ]
            Iz = self.frame.Iz_array[ i ]
            J = self.frame.J_array[ i ]
            # change None stuff for this
            I_z = self.frame.v_temp_array[i ]

            k_e_local = local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, Iy, Iz, J )
            gamma = rotation_matrix_3D(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], I_z)
            Gamma = transformation_matrix_3D(gamma)
            k_e_global = Gamma.T @ k_e_local @ Gamma
            k_e_p00 = k_e_global[0:6, 0:6]
            k_e_p01 = k_e_global[0:6, 6:12]
            k_e_p10 = k_e_global[6:12, 0:6]
            k_e_p11 = k_e_global[6:12, 6:12]
            
            p0_DoF_idx = p0_idx * 6
            p1_DoF_idx = p1_idx * 6
            K[ p0_DoF_idx : p0_DoF_idx + 6, p0_DoF_idx:  p0_DoF_idx + 6  ] += k_e_p00
            K[ p0_DoF_idx : p0_DoF_idx + 6, p1_DoF_idx:  p1_DoF_idx + 6  ] += k_e_p01
            K[ p1_DoF_idx : p1_DoF_idx + 6, p0_DoF_idx:  p0_DoF_idx + 6  ] += k_e_p10
            K[ p1_DoF_idx : p1_DoF_idx + 6, p1_DoF_idx:  p1_DoF_idx + 6  ] += k_e_p11
        return K
