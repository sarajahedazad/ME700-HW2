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
    gamma = rotation_matrix_3D(0, 0, 0, 1, 0, 0)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma, np.eye(3))

def test_get_global_elastic_stiffmatrix(stiffness_matrices):
    K = stiffness_matrices.get_global_elastic_stiffmatrix()
    assert K.shape == (12, 12)
    assert np.allclose(K[0, 0], 200e9 * 0.01 / 1.0)
