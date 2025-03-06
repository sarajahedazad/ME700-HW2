import numpy as np
import pytest
from stiffness_matrices import StiffnessMatrices, local_elastic_stiffness_matrix_3D_beam, check_unit_vector, check_parallel, rotation_matrix_3D

class MockFrame:
    def __init__(self):
        # Create mock data for testing
        self.points = np.array([[0, 0, 0], [1, 0, 0]])
        self.connectivities = np.array([[0, 1]])
        self.E_array = np.array([200e9])
        self.nu_array = np.array([0.3])
        self.A_array = np.array([0.01])
        self.L_array = np.array([1.0])
        self.Iy_array = np.array([1e-6])
        self.Iz_array = np.array([1e-6])
        self.I_rho_array = np.array([1e-6])
        self.J_array = np.array([1e-6])
        self.v_temp_array = np.array([[0, 0, 1]])

@pytest.fixture
def stiffness_matrices():
    frame = MockFrame()
    return StiffnessMatrices(frame)

def test_get_element_parameters(stiffness_matrices):
    connection, E, nu, A, L, Iy, Iz, I_rho, J, v_temp = stiffness_matrices.get_element_parameters(0)
    assert connection.shape == (2,)
    assert E == 200e9
    assert nu == 0.3
    assert A == 0.01
    assert L == 1.0
    assert Iy == 1e-6
    assert Iz == 1e-6
    assert I_rho == 1e-6
    assert J == 1e-6
    assert v_temp.shape == (3,)

def test_get_element_points(stiffness_matrices):
    p0_idx, p0, p1_idx, p1 = stiffness_matrices.get_element_points([0, 1])
    assert p0_idx == 0
    assert np.allclose(p0, [0, 0, 0])
    assert p1_idx == 1
    assert np.allclose(p1, [1, 0, 0])

def test_get_transformation_matrix_3D(stiffness_matrices):
    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 0, 0])
    v_temp = np.array([0, 0, 1])
    Gamma = stiffness_matrices.get_transformation_matrix_3D(p0, p1, v_temp)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma[0:3, 0:3], np.eye(3))
    assert np.allclose(Gamma[3:6, 3:6], np.eye(3))

def test_check_unit_vector():
    vec = np.array([1, 0, 0])
    check_unit_vector(vec)
    with pytest.raises(ValueError, match="Expected a unit vector for reference vector."):
        check_unit_vector(np.array([1, 1, 0]))

def test_check_parallel():
    vec_1 = np.array([1, 0, 0])
    vec_2 = np.array([0, 1, 0])
    check_parallel(vec_1, vec_2)
    with pytest.raises(ValueError, match="Reference vector is parallel to beam axis."):
        check_parallel(np.array([1, 0, 0]), np.array([1, 0, 0]))

def test_rotation_matrix_3D():
    # Test case where beam is horizontal, v_temp should default to [0, 0, 1.0]
    gamma = rotation_matrix_3D(0, 0, 0, 1, 0, 0)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma, np.eye(3))

    # Test case where beam is vertical, v_temp should default to [0, 1.0, 0.0]
    gamma = rotation_matrix_3D(0, 0, 0, 0, 0, 1)
    assert gamma.shape == (3, 3)
    expected_local_y = np.array([1, 0, 0])
    computed_local_y = np.cross(gamma[2], np.array([0, 0, 1]))
    assert np.allclose(expected_local_y, computed_local_y / np.linalg.norm(computed_local_y))

def test_get_global_elastic_stiffmatrix(stiffness_matrices):
    K = stiffness_matrices.get_global_elastic_stiffmatrix()
    assert K.shape == (12, 12)
    assert np.allclose(K[0, 0], 200e9 * 0.01 / 1.0)

def test_local_geometric_stiffness_matrix_3D_beam():
    L = 1.0
    A = 0.01
    I_rho = 1e-6
    Fx2 = 1000.0
    Mx2 = 200.0
    My1 = 300.0
    Mz1 = 400.0
    My2 = 500.0
    Mz2 = 600.0
    
    k_g = local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2)
    
    assert k_g.shape == (12, 12)
    assert np.isclose(k_g[0, 6], -Fx2 / L)
    assert np.isclose(k_g[1, 3], My1 / L)
    assert np.isclose(k_g[1, 4], Mx2 / L)
    assert np.isclose(k_g[1, 5], Fx2 / 10.0)
    assert np.isclose(k_g[1, 7], -6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[1, 9], My2 / L)
    assert np.isclose(k_g[1, 10], -Mx2 / L)
    assert np.isclose(k_g[1, 11], Fx2 / 10.0)
    assert np.isclose(k_g[2, 3], Mz1 / L)
    assert np.isclose(k_g[2, 4], -Fx2 / 10.0)
    assert np.isclose(k_g[2, 5], Mx2 / L)
    assert np.isclose(k_g[2, 8], -6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[2, 9], Mz2 / L)
    assert np.isclose(k_g[2, 10], -Fx2 / 10.0)
    assert np.isclose(k_g[2, 11], -Mx2 / L)
    assert np.isclose(k_g[3, 4], -1.0 * (2.0 * Mz1 - Mz2) / 6.0)
    assert np.isclose(k_g[3, 5], (2.0 * My1 - My2) / 6.0)
    assert np.isclose(k_g[3, 7], -My1 / L)
    assert np.isclose(k_g[3, 8], -Mz1 / L)
    assert np.isclose(k_g[3, 9], -Fx2 * I_rho / (A * L))
    assert np.isclose(k_g[3, 10], -1.0 * (Mz1 + Mz2) / 6.0)
    assert np.isclose(k_g[3, 11], (My1 + My2) / 6.0)
    assert np.isclose(k_g[4, 7], -Mx2 / L)
    assert np.isclose(k_g[4, 8], Fx2 / 10.0)
    assert np.isclose(k_g[4, 9], -1.0 * (Mz1 + Mz2) / 6.0)
    assert np.isclose(k_g[4, 10], -Fx2 * L / 30.0)
    assert np.isclose(k_g[4, 11], Mx2 / 2.0)
    assert np.isclose(k_g[5, 7], -Fx2 / 10.0)
    assert np.isclose(k_g[5, 8], -Mx2 / L)
    assert np.isclose(k_g[5, 9], (My1 + My2) / 6.0)
    assert np.isclose(k_g[5, 10], -Mx2 / 2.0)
    assert np.isclose(k_g[5, 11], -Fx2 * L / 30.0)
    assert np.isclose(k_g[7, 9], -My2 / L)
    assert np.isclose(k_g[7, 10], Mx2 / L)
    assert np.isclose(k_g[7, 11], -Fx2 / 10.0)
    assert np.isclose(k_g[8, 9], -Mz2 / L)
    assert np.isclose(k_g[8, 10], Fx2 / 10.0)
    assert np.isclose(k_g[8, 11], Mx2 / L)
    assert np.isclose(k_g[9, 10], (Mz1 - 2.0 * Mz2) / 6.0)
    assert np.isclose(k_g[9, 11], -1.0 * (My1 - 2.0 * My2) / 6.0)
    assert np.isclose(k_g[0, 0], Fx2 / L)
    assert np.isclose(k_g[1, 1], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[2, 2], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[3, 3], Fx2 * I_rho / (A * L))
    assert np.isclose(k_g[4, 4], 2.0 * Fx2 * L / 15.0)
    assert np.isclose(k_g[5, 5], 2.0 * Fx2 * L / 15.0)
    assert np.isclose(k_g[6, 6], Fx2 / L)
    assert np.isclose(k_g[7, 7], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[8, 8], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[9, 9], Fx2 * I_rho / (A * L))
    assert np.isclose(k_g[10, 10], 2.0 * Fx2 * L / 15.0)
    assert np.isclose(k_g[11, 11], 2.0 * Fx2 * L / 15.0)

def test_local_geometric_stiffness_matrix_3D_beam_without_interaction_terms():
    L = 1.0
    A = 0.01
    I_rho = 1e-6
    Fx2 = 1000.0

    k_g = local_geometric_stiffness_matrix_3D_beam_without_interaction_terms(L, A, I_rho, Fx2)

    assert k_g.shape == (12, 12)
    assert np.isclose(k_g[0, 6], -Fx2 / L)
    assert np.isclose(k_g[1, 5], Fx2 / 10.0)
    assert np.isclose(k_g[1, 7], -6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[1, 11], Fx2 / 10.0)
    assert np.isclose(k_g[2, 4], -Fx2 / 10.0)
    assert np.isclose(k_g[2, 8], -6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[2, 10], -Fx2 / 10.0)
    assert np.isclose(k_g[3, 9], -Fx2 * I_rho / (A * L))
    assert np.isclose(k_g[4, 8], Fx2 / 10.0)
    assert np.isclose(k_g[4, 10], -Fx2 * L / 30.0)
    assert np.isclose(k_g[5, 7], -Fx2 / 10)
    assert np.isclose(k_g[5, 11], -Fx2 * L / 30.0)
    assert np.isclose(k_g[7, 11], -Fx2 / 10.0)
    assert np.isclose(k_g[8, 10], Fx2 / 10.0)
    assert np.isclose(k_g[0, 0], Fx2 / L)
    assert np.isclose(k_g[1, 1], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[2, 2], 6.0 * Fx2 / (5.0 * L))
    assert np.isclose(k_g[3, 3], Fx2 * I_rho / (A * L))
    assert np.isclose(k_g[4, 4], 2.0 * Fx2 * L / 15.0)
    assert np.isclose(k_g[5, 5], 2.0 * Fx2 * L / 15.0)

def test_get_element_global_Delta(stiffness_matrices):
    Delta = np.zeros(12)
    connection = [0, 1]
    result = stiffness_matrices.get_element_global_Delta(connection, Delta)
    assert result.shape == (12,)
    assert np.allclose(result, np.zeros(12))

def test_get_element_local_internal_F(stiffness_matrices):
    Delta = np.zeros(12)
    result = stiffness_matrices.get_element_local_internal_F(0, Delta)
    assert result.shape == (12,)
    assert np.allclose(result, np.zeros(12))

def test_get_element_local_geometric_stiffness_matrix(stiffness_matrices):
    Delta = np.zeros(12)
    result = stiffness_matrices.get_element_local_geometric_stiffness_matrix(0, Delta)
    assert result.shape == (12, 12)
    assert np.allclose(result, np.zeros((12, 12)))

def test_get_global_geometric_stiffmatrix(stiffness_matrices):
    Delta = np.zeros(12)
    result = stiffness_matrices.get_global_geometric_stiffmatrix(Delta)
    assert result.shape == (12, 12)
    assert np.allclose(result, np.zeros((12, 12)))
