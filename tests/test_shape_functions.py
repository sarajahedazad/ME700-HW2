import numpy as np
import pytest
import sympy
from shape_functions import *

class MockFrame:
    def __init__(self):
        # Create mock data for testing
        self.points = np.array([[0, 0, 0], [1, 0, 0]])
        self.connectivities = np.array([[0, 1]])
        self.L_array = np.array([1.0])
        self.v_temp_array = np.array([[0, 0, 1]])
        self.eigenvector = np.zeros(12)

@pytest.fixture
def shape_functions():
    frame = MockFrame()
    eigenvector = np.zeros(12)
    return ShapeFunctions(eigenvector, frame)

def test_evaluate():
    x = sympy.symbols('x')
    expr = x ** 2
    result = evaluate(expr, x, [2])
    expected = np.array([4])
    assert np.allclose(result, expected)

def test_interpolate_two_points():
    p0 = np.array([0, 0, 0])
    p1 = np.array([1, 1, 1])
    n = 4
    points = interpolate_two_points(p0, p1, n)
    expected = np.array([[0, 0, 0], [0.25, 0.25, 0.25], [0.5, 0.5, 0.5], [0.75, 0.75, 0.75], [1, 1, 1]])
    assert np.allclose(points, expected)

def test_rotation_matrix_3D(shape_functions):
    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 1, 0, 0)
    expected_gamma = np.eye(3)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma, expected_gamma)

    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 0, 0, 1)
    expected_local_y = np.array([1, 0, 0])
    computed_local_y = np.cross(gamma[2], np.array([0, 0, 1]))
    assert np.allclose(expected_local_y, computed_local_y / np.linalg.norm(computed_local_y))

def test_transformation_1212_matrix_3D(shape_functions):
    gamma = np.eye(3)
    Gamma = shape_functions.transformation_1212_matrix_3D(gamma)
    expected_Gamma = np.eye(12)
    assert Gamma.shape == (12, 12)
    assert np.allclose(Gamma, expected_Gamma)

def test_transformation_nn_matrix_3D(shape_functions):
    gamma = np.eye(3)
    Gamma = shape_functions.transformation_nn_matrix_3D(gamma)
    expected_Gamma = np.eye(3 * shape_functions.n)
    assert Gamma.shape == (3 * shape_functions.n, 3 * shape_functions.n)
    assert np.allclose(Gamma, expected_Gamma)

def test_linear_N1(shape_functions):
    length = 1.0
    expr = shape_functions.linear_N1(length)
    x_local_val = np.linspace(0, length, num=shape_functions.n)
    result = shape_functions.evaluate(expr, x_local_val)
    expected = 1 - x_local_val / length
    assert np.allclose(result, expected)

def test_linear_N2(shape_functions):
    length = 1.0
    expr = shape_functions.linear_N2(length)
    x_local_val = np.linspace(0, length, num=shape_functions.n)
    result = shape_functions.evaluate(expr, x_local_val)
    expected = x_local_val / length
    assert np.allclose(result, expected)

# Add similar tests for hermite_N1, hermite_N2, hermite_N3, and hermite_N4

def test_get_element_info(shape_functions):
    connection, p0_idx, p0, p1_idx, p1, length, v_temp = shape_functions.get_element_info(0)
    assert np.allclose(connection, [0, 1])
    assert p0_idx == 0
    assert p1_idx == 1
    assert np.allclose(p0, [0, 0, 0])
    assert np.allclose(p1, [1, 0, 0])
    assert length == 1.0
    assert np.allclose(v_temp, [0, 0, 1])

def test_calc_eigenvector_element_local(shape_functions):
    result = shape_functions.calc_eigenvector_element_local(0)
    expected = np.zeros(12)
    assert np.allclose(result, expected)

def test_calc_element_interpolation(shape_functions):
    result = shape_functions.calc_element_interpolation(0)
    expected = np.zeros((shape_functions.n, 3))
    assert result.shape == expected.shape

