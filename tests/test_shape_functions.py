import pytest
import numpy as np
import sympy
from geometry import *
from shape_functions import *

class MockFrame:
    def __init__(self):
        self.connectivities = [(0, 1)]
        self.points = [np.array([0, 0, 0]), np.array([1, 1, 1])]
        self.L_array = [np.sqrt(3)]
        self.v_temp_array = [np.array([0, 0, 1])]

@pytest.fixture
def shape_functions():
    eigenvector = np.zeros(12)
    frame_obj = MockFrame()
    return ShapeFunctions(eigenvector, frame_obj)

def test_evaluate():
    x0 = np.array([1, 2, 3])
    expr = sympy.symbols('a') * x0[0] + sympy.symbols('b') * x0[1]
    symb = [sympy.symbols('a'), sympy.symbols('b')]
    result = evaluate(expr, symb, [1, 1])
    expected = np.array([3])
    np.testing.assert_array_almost_equal(result, expected)

def test_interpolate_two_points():
    p0 = np.array([0, 0])
    p1 = np.array([10, 10])
    n = 5
    result = interpolate_two_points(p0, p1, n)
    expected = np.array([[0.,  0.],
                         [2.,  2.],
                         [4.,  4.],
                         [6.,  6.],
                         [8.,  8.],
                         [10., 10.]])
    np.testing.assert_array_almost_equal(result, expected)

def test_rotation_matrix_3D(shape_functions):
    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 1, 1, 1)
    assert gamma.shape == (3, 3)

def test_transformation_1212_matrix_3D(shape_functions):
    gamma = np.identity(3)
    Gamma = shape_functions.transformation_1212_matrix_3D(gamma)
    assert Gamma.shape == (12, 12)
    np.testing.assert_array_almost_equal(Gamma[:3, :3], gamma)

def test_transformation_nn_matrix_3D(shape_functions):
    gamma = np.identity(3)
    Gamma = shape_functions.transformation_nn_matrix_3D(gamma)
    assert Gamma.shape == (60, 60)  # Assuming n=20

def test_evaluate_shape_functions(shape_functions):
    x0 = np.linspace(0, 1, 20)
    expr = shape_functions.linear_N1(1)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([1 - xi for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_linear_N1(shape_functions):
    length = 1
    expr = shape_functions.linear_N1(length)
    assert expr.shape == (shape_functions.n, 1)

def test_hermite_N1(shape_functions):
    length = 1
    expr = shape_functions.hermite_N1(length)
    assert expr.shape == (shape_functions.n, 1)


