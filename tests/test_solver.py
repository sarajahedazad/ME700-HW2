import numpy as np
import pytest
from scipy.linalg import eig
from stiffness_matrices import *
from solver import * 

class MockBCS:
    def __init__(self):
        self.BCs_supported_indices = [0, 1]
        self.BCs_Delta_supported_values = [0.0, 0.0]
        self.BCs_free_indices = [2, 3]
        self.BCs_F_free_values = [100.0, 200.0]
        self.n_DoFs = 4

def test_partition():
    matrix = np.array([[1, 2, 3, 4],
                       [5, 6, 7, 8],
                       [9, 10, 11, 12],
                       [13, 14, 15, 16]])
    known_DoF_idx = [0, 1]
    matrix_ss, matrix_sf, matrix_fs, matrix_ff = partition(matrix, known_DoF_idx)

    assert np.allclose(matrix_ss, np.array([[1, 2], [5, 6]]))
    assert np.allclose(matrix_sf, np.array([[3, 4], [7, 8]]))
    assert np.allclose(matrix_fs, np.array([[9, 10], [13, 14]]))
    assert np.allclose(matrix_ff, np.array([[11, 12], [15, 16]]))

def test_solve_stiffness_system():
    K = np.array([[10, 2, 3, 4],
                  [2, 10, 7, 8],
                  [3, 7, 10, 12],
                  [4, 8, 12, 10]])
    bcs = MockBCS()
    
    Delta, F = solve_stiffness_system(K, bcs)
    
    expected_Delta = np.array([0.0, 0.0, -1.2673476, 19.946942])
    expected_F = np.array([-266.607, 101.875, 100.0, 200.0])
    
    assert np.allclose(Delta, expected_Delta, rtol=1e-5)
    assert np.allclose(F, expected_F, rtol=1e-5)

def test_compute_critical_load():
    K_elastic = np.array([[10, 2, 3, 4],
                          [2, 10, 7, 8],
                          [3, 7, 10, 12],
                          [4, 8, 12, 10]])
    K_geometric = np.array([[2, 1, 1, 1],
                            [1, 2, 1, 1],
                            [1, 1, 2, 1],
                            [1, 1, 1, 2]])
    bcs = MockBCS()
    
    smallest_positive_eigenvalue, eigenvectors_allstructure = compute_critical_load(K_elastic, K_geometric, bcs)
    
    expected_eigenvalue = 1.95110875
    expected_eigenvector = np.array([0.0, 0.0, 0.35889959, -0.93337589])
    
    assert np.isclose(smallest_positive_eigenvalue, expected_eigenvalue, rtol=1e-5)
    assert np.allclose(eigenvectors_allstructure, expected_eigenvector, rtol=1e-5)

