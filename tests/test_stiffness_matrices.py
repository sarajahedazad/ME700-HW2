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
