import pytest
import numpy as np
import sympy
from geometry import *
from shape_functions import *
import os

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

def test_linear_N2(shape_functions):
    length = 1
    expr = shape_functions.linear_N2(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([xi / length for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N1(shape_functions):
    length = 1
    expr = shape_functions.hermite_N1(length)
    assert expr.shape == (shape_functions.n, 1)

def test_hermite_N2(shape_functions):
    length = 1
    expr = shape_functions.hermite_N2(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([3 * (xi / length) ** 2 - 2 * (xi / length) ** 3 for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N3(shape_functions):
    length = 1
    expr = shape_functions.hermite_N3(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([xi * (1 - xi / length) ** 2 for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N4(shape_functions):
    length = 1
    expr = shape_functions.hermite_N4(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([xi * ((xi / length) ** 2 - xi / length) for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_get_element_info(shape_functions):
    connection, p0_idx, p0, p1_idx, p1, length, v_temp = shape_functions.get_element_info(0)
    assert connection == (0, 1)
    assert p0_idx == 0
    np.testing.assert_array_almost_equal(p0, [0, 0, 0])
    assert p1_idx == 1
    np.testing.assert_array_almost_equal(p1, [1, 1, 1])
    assert length == np.sqrt(3)
    np.testing.assert_array_almost_equal(v_temp, [0, 0, 1])

def test_get_eigenvector_element_global(shape_functions):
    eigenvector_element_global = shape_functions.get_eigenvector_element_global(0, 1)
    expected = np.zeros(12)
    np.testing.assert_array_almost_equal(eigenvector_element_global, expected)


def test_calc_element_interpolation(shape_functions):
    interpolated_points = shape_functions.calc_element_interpolation(0)
    assert interpolated_points.shape == (shape_functions.n, 3)
    expected = np.linspace([0, 0, 0], [1, 1, 1], num=shape_functions.n)
    np.testing.assert_array_almost_equal(interpolated_points, expected)
    
def test_plot_element_interpolation(shape_functions, tmp_path):
    # Generate a temporary file path
    saving_dir_with_name = tmp_path / "test_plot.png"
    
    # Call the method to generate the plot and save the file
    shape_functions.plot_element_interpolation(str(saving_dir_with_name))
    
    # Check if the file was created
    assert saving_dir_with_name.exists(), "Plot file was not created."
    
    # Optionally, you can add more checks to verify the contents of the file,
    # such as checking its size, format, etc.

    # Cleanup: Remove the generated file if needed
    os.remove(saving_dir_with_name)
def test_rotation_matrix_v_temp_none(shape_functions):
    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 1, 1, 1, None)
    assert gamma.shape == (3, 3)

def test_rotation_matrix_v_temp_none_case1(shape_functions):
    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 0, 0, 1, None)  # lxp and mxp are close to 0.0
    print("gamma:", gamma)
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma[1], [0, 1, 0])  # Check if local_y was set to [0, 1.0, 0.0]

def test_rotation_matrix_v_temp_provided(shape_functions):
    gamma = shape_functions.rotation_matrix_3D(0, 0, 0, 1, 1, 1, np.array([0, 0, 1]))
    assert gamma.shape == (3, 3)
