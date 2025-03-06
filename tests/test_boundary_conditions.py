import numpy as np
from geometry import *
from boundary_conditions import *

import numpy as np
import geometry as geom

def setup_frame():
    frame = geom.Frame()
    p0 = frame.add_point(0, 0, 0)
    p1 = frame.add_point(1, 1, 1)
    element = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    frame.build_frame([element])
    return frame

def test_add_disp_bound_xyz():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.add_disp_bound_xyz([0, 0, 0], 1, 2, 3)
    assert bc.BCs_disp_indices == [0, 1, 2]
    assert bc.BCs_disp_values == [1, 2, 3]

def test_add_disp_bound_x():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.add_disp_bound_x([0, 0, 0], 1)
    assert bc.BCs_disp_indices == [0]
    assert bc.BCs_disp_values == [1]

def test_add_rot_bound_xyz():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.add_rot_bound_xyz([0, 0, 0], 1, 2, 3)
    assert bc.BCs_rot_indices == [3, 4, 5]
    assert bc.BCs_rot_values == [1, 2, 3]

def test_add_force_bound_xyz():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.add_force_bound_xyz([0, 0, 0], 1, 2, 3)
    assert bc.BCs_force_indices == [0, 1, 2]
    assert bc.BCs_force_values == [1, 2, 3]

def test_add_momentum_bound_xyz():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.add_momentum_bound_xyz([0, 0, 0], 1, 2, 3)
    assert bc.BCs_momentum_indices == [3, 4, 5]
    assert bc.BCs_momentum_values == [1, 2, 3]

def test_validate_bounds_overdefined():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.BCs_supported_indices = [0, 1, 2]
    bc.BCs_free_indices = [2, 3, 4]
    try:
        bc.validate_bounds()
    except OverDefinedError:
        print("OverDefinedError successfully raised")

def test_validate_bounds_underdefined():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.BCs_supported_indices = [0, 1, 2]
    bc.BCs_free_indices = [3, 4, 5]
    bc.n_DoFs = 12
    try:
        bc.validate_bounds()
    except UnderDefinedError:
        print("UnderDefinedError successfully raised")

def test_set_up_bounds():
    frame = setup_frame()
    bc = BoundaryConditions(frame)
    bc.BCs_disp_indices = [0, 1, 2]
    bc.BCs_disp_values = [1, 2, 3]
    bc.BCs_rot_indices = [3, 4, 5]
    bc.BCs_rot_values = [4, 5, 6]
    # Ensure no overlap between supported and free indices
    bc.BCs_force_indices = [6, 7, 8]
    bc.BCs_force_values = [1, 2, 3]
    bc.BCs_momentum_indices = [9, 10, 11]
    bc.BCs_momentum_values = [4, 5, 6]
    bc.set_up_bounds()
    assert (bc.BCs_supported_indices == np.array([0, 1, 2, 3, 4, 5])).all()
    assert (bc.BCs_Delta_supported_values == np.array([1, 2, 3, 4, 5, 6])).all()
    assert (bc.BCs_free_indices == np.array([6, 7, 8, 9, 10, 11])).all()
    assert (bc.BCs_F_free_values == np.array([1, 2, 3, 4, 5, 6])).all()







