import pytest
import numpy as np
import os
import matplotlib.pyplot as plt
import sympy

# Import everything from your module.
from geometry import *
from shape_functions import *

# We assume that rotation_matrix_3D is defined as a free function in your module.
# (It is used in calc_element_interpolation as: gamma = rotation_matrix_3D(...).)

# A minimal mock frame to allow testing of geometry-dependent methods.
class MockFrame:
    def __init__(self):
        self.connectivities = [(0, 1)]
        self.points = [
            np.array([0, 0, 0]),
            np.array([1, 1, 1])
        ]
        self.L_array = [np.sqrt(3)]
        self.v_temp_array = [np.array([0, 0, 1])]  # This value is not used when v_temp is provided at call time

@pytest.fixture
def shape_functions():
    eigenvector = np.zeros(12)  # For simplicity, eigenvector is all zeros.
    frame_obj = MockFrame()
    return ShapeFunctions(eigenvector, frame_obj)

# === TESTS FOR THE SHAPE FUNCTION EXPRESSIONS ===

def test_linear_N1(shape_functions):
    length = 1
    expr = shape_functions.linear_N1(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([1 - xi / length for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

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
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([1 - 3*(xi/length)**2 + 2*(xi/length)**3 for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N2(shape_functions):
    length = 1
    expr = shape_functions.hermite_N2(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([3*(xi/length)**2 - 2*(xi/length)**3 for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N3(shape_functions):
    length = 1
    expr = shape_functions.hermite_N3(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([xi * (1 - xi/length)**2 for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

def test_hermite_N4(shape_functions):
    length = 1
    expr = shape_functions.hermite_N4(length)
    assert expr.shape == (shape_functions.n, 1)
    x0 = np.linspace(0, length, shape_functions.n)
    result = shape_functions.evaluate(expr, x0)
    expected = np.array([xi * ((xi/length)**2 - xi/length) for xi in x0])
    np.testing.assert_array_almost_equal(result, expected)

# === TESTS FOR THE FRAME AND EIGENVECTOR HELPER METHODS ===

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

# === TESTS FOR CALCULATED INTERPOLATION METHODS ===

def test_calc_element_interpolation(shape_functions):
    # When the eigenvector is zero, the interpolation should simply be the nodal line.
    interpolated_points = shape_functions.calc_element_interpolation(0)
    # Our mock frame connects points [0,0,0] and [1,1,1]
    # The standard linear interpolation of these two points into n samples is:
    expected = np.linspace([0, 0, 0], [1, 1, 1], num=shape_functions.n)
    np.testing.assert_array_almost_equal(interpolated_points, expected)

# === TEST FOR THE ROTATION MATRIX FUNCTION (v_temp handling) ===

def test_rotation_matrix_v_temp_none_case1(shape_functions):
    # Call the free function rotation_matrix_3D.
    # With points (0,0,0) to (0,0,1) we have:
    # local_x = [0, 0, 1].
    # Since both lxp and mxp are 0, we expect:
    #   v_temp is set to [0, 1, 0],
    #   local_y = cross([0,1,0], [0,0,1]) = [1, 0, 0],
    #   local_z = cross([0,0,1], [1,0,0]) = [0, 1, 0].
    gamma = rotation_matrix_3D(0, 0, 0, 0, 0, 1, None)
    # For clarity print out gamma (remove or comment out in final version)
    # print("gamma:", gamma)  
    assert gamma.shape == (3, 3)
    assert np.allclose(gamma[0], [0, 0, 1]), f"Expected local_x = [0,0,1], got {gamma[0]}"
    assert np.allclose(gamma[1], [1, 0, 0]), f"Expected local_y = [1,0,0], got {gamma[1]}"
    assert np.allclose(gamma[2], [0, 1, 0]), f"Expected local_z = [0,1,0], got {gamma[2]}"

# === TEST FOR THE PLOTTING FUNCTION ===

def test_plot_element_interpolation(shape_functions, tmp_path):
    # We use pytest's tmp_path fixture to get a temporary file name.
    saving_dir_with_name = tmp_path / "test_plot.png"
    
    # Call the plot method (this will display a plot; in a CI environment, the interactive window is usually suppressed).
    shape_functions.plot_element_interpolation(str(saving_dir_with_name))
    
    # Check that the file was indeed created.
    assert saving_dir_with_name.exists(), "Plot file was not created."
    
    # Optionally: remove the file afterward.
    os.remove(saving_dir_with_name)
