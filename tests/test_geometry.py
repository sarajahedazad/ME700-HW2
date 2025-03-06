import numpy as np
import pytest
from geometry import *

def test_point_creation():
    point = Point(1, 2, 3)
    assert point.x == 1
    assert point.y == 2
    assert point.z == 3
    assert (point.coords == np.array([1, 2, 3])).all()

def test_element_creation():
    p0 = Point(0, 0, 0)
    p1 = Point(1, 1, 1)
    element = Element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    assert element.p0 == p0
    assert element.p1 == p1
    assert np.isclose(element.L, np.sqrt(3))

def test_add_point():
    frame = Frame()
    point = frame.add_point(1, 2, 3)
    assert point.x == 1
    assert point.y == 2
    assert point.z == 3

def test_add_element():
    frame = Frame()
    p0 = frame.add_point(0, 0, 0)
    p1 = frame.add_point(1, 1, 1)
    element = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    assert element.p0 == p0
    assert element.p1 == p1
    assert np.isclose(element.L, np.sqrt(3))

def test_build_frame():
    frame = Frame()
    p0 = frame.add_point(0, 0, 0)
    p1 = frame.add_point(1, 1, 1)
    element = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    frame.build_frame([element])
    assert (frame.points == np.array([[0, 0, 0], [1, 1, 1]])).all()
    assert (frame.connectivities == np.array([[0, 1]])).all()

def test_generate_frame_directly():
    points = np.array([[0, 0, 0], [1, 1, 1]])
    connectivities = np.array([[0, 1]])
    E_array = np.array([210e9])
    nu_array = np.array([0.3])
    A_array = np.array([0.01])
    Iy_array = np.array([1e-6])
    Iz_array = np.array([1e-6])
    I_rho_array = np.array([1e-6])
    J_array = np.array([1e-6])
    v_temp_array = np.array([None])

    frame = Frame()
    frame.generate_frame_directly(points, connectivities, E_array, nu_array, A_array, Iy_array, Iz_array, I_rho_array, J_array, v_temp_array)
    
    assert (frame.points == points).all()
    assert (frame.connectivities == np.array([[0, 1]])).all()
    assert (frame.E_array == E_array).all()
    assert (frame.nu_array == nu_array).all()
    assert (frame.A_array == A_array).all()
    assert (frame.Iy_array == Iy_array).all()
    assert (frame.Iz_array == Iz_array).all()
    assert (frame.I_rho_array == I_rho_array).all()
    assert (frame.J_array == J_array).all()
    assert (frame.v_temp_array == v_temp_array).all()
    assert np.isclose(frame.L_array, np.sqrt(3)).all()

def test_duplicate_elements():
    frame = Frame()
    p0 = frame.add_point(0, 0, 0)
    p1 = frame.add_point(1, 1, 1)
    element1 = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    element2 = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    try:
        frame.build_frame([element1, element2])
    except DuplicationError:
        print("DuplicationError successfully raised")

def test_existing_point():
    frame = Frame()
    p0 = frame.add_point(0, 0, 0)
    frame.points_unique[(0, 0, 0)] = 0
    p1 = frame.add_point(1, 1, 1)
    element1 = frame.add_element(p0, p1, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    element2 = frame.add_element(p1, p0, 210e9, 0.3, 0.01, 1e-6, 1e-6, 1e-6, 1e-6)
    frame.build_frame([element1, element2])
    assert (frame.points == np.array([[0, 0, 0], [1, 1, 1]])).all()
    assert (frame.connectivities == np.array([[0, 1], [1, 0]])).all()

