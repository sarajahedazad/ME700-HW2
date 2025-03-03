import numpy as np
import matplotlib.pyplot as plt
from geometry import *

def plot_original_configuration( frame_obj, saving_dir_with_name ):
    points = frame_obj.points
    connectivities = frame_obj.connectivities
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points[0], color='black', label='Nodes')
    for point in points[1:]:
        ax.scatter(*point, color='black')
    p0 = points[ connectivities[0][0] ]
    p1 = points[ connectivities[0][1] ]
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', label='Elements')
    for connection in connectivities[ 1: ]:
        p0 = points[ connection[0] ]
        p1 = points[ connection[1] ]
        ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.grid(True)

    plt.show()
    plt.savefig( saving_dir_with_name )

# def plot_deformed_configuration( frame_obj, X, saving_dir_with_name):
#     points = frame_obj.points
#     connectivities = frame_obj.connectivities
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     p0 = points[0]
#     disp_p0 = X[ 0: 3 ]
#     point_displaced = p0 + disp_p0
#     ax.scatter(*point_displaced, color='red', label='Displaced Nodes')
#     for i, point in enumerate( points[1:] ):
#         disp = X[6:][ i * 6: i * 6 + 3 ]
#         point_displaced = point + disp
#         ax.scatter(*point_displaced, color='red')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     ax.grid(True)

#     plt.show()
#     plt.savefig( saving_dir_with_name )

# def plot_original_vs_deformed_configurations( frame_obj, X, saving_dir_with_name):
#     points = frame_obj.points
#     connectivities = frame_obj.connectivities
#     fig = plt.figure()

#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(*points[0], color='black', s = 4, label='Nodes')
#     for point in points[1:]:
#         ax.scatter(*point, color='black', s = 4)
#     p0 = points[ connectivities[0][0] ]
#     p1 = points[ connectivities[0][1] ]
#     ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--', label='Elements')
#     for connection in connectivities[ 1: ]:
#         p0 = points[ connection[0] ]
#         p1 = points[ connection[1] ]
#         ax.plot([p0[0], p1[0]], [p0[1], p1[1]], [p0[2], p1[2]], 'k--')

#     p0 = points[0]
#     disp_p0 = X[ 0: 3 ]
#     point_displaced = p0 + disp_p0
#     ax.scatter(*point_displaced, color='red', label='Displaced Nodes')
#     for i, point in enumerate( points[1:] ):
#         disp = X[6:][ i * 6: i * 6 + 3 ]
#         point_displaced = point + disp
#         ax.scatter(*point_displaced, color='red')
#     ax.set_xlabel('X')
#     ax.set_ylabel('Y')
#     ax.set_zlabel('Z')
#     ax.legend()
#     ax.grid(True)

#     plt.show()
#     plt.savefig( saving_dir_with_name )


def plot_original_vs_deformed_configurations(frame_obj, X, saving_dir_with_name, scale=1.0, num_points=50):
    nodes = frame_obj.points
    elements = frame_obj.connectivities
    
    """
    Plots the undeformed and continuously deformed shape of a planar frame structure.
    
    This version forces the bending direction to be the global y-axis.
    
    Parameters
    ----------
    nodes : ndarray, shape (n_nodes, 3)
        Nodal coordinates in the undeformed configuration.
    elements : ndarray, shape (n_elements, 2)
        Connectivity array where each row holds the indices (into nodes) of the two nodes 
        defining an element.
    X : ndarray, shape (6*n_nodes,)
        Global displacement vector. The assumed DOF order for each node is:
          [axial displacement (u), lateral displacement (v), out-of-plane displacement (w),
           rotation about x (rx), rotation about y (ry), rotation about z (rz)].
        For our purposes we assume bending is out‐of‑plane (global y direction) and:
          u = disp[0]  (axial, along the beam)
          v = disp[1]  (lateral, global y)
          phi = disp[5] (rotation about z, controlling bending in global y)
    scale : float, optional
        Scale factor for visualizing displacements.
    num_points : int, optional
        Number of interpolation points along each element.
    
    Returns
    -------
    None. Displays a 3D plot with equal axis scaling.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the undeformed structure (dashed black lines)
    for elem in elements:
        i, j = elem
        p1, p2 = nodes[i], nodes[j]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'k--', lw=1)
    
    # For each element, define:
    #   - local x-axis along the element (from node i to node j)
    #   - force the local bending direction (local y) to be global [0,1,0]
    #   - compute local z-axis as cross(local x, local y)
    for elem in elements:
        i, j = elem
        p1, p2 = nodes[i], nodes[j]
        L = np.linalg.norm(p2 - p1)
        e_x = (p2 - p1) / L  # local x-axis
        
        # Force local y-axis to be global y for all elements
        e_y = np.array([0, 1, 0])
        # Compute local z-axis to complete a right-handed system
        e_z = np.cross(e_x, e_y)
        if np.linalg.norm(e_z) < 1e-8:
            # If e_x is parallel to global y, choose an alternative fixed e_y (e.g., global z)
            e_y = np.array([0, 0, 1])
            e_z = np.cross(e_x, e_y)
        e_z /= np.linalg.norm(e_z)
        
        # Extract nodal displacements (global DOFs)
        # For consistency we assume:
        #   u = disp[0] (axial, along element),
        #   v = disp[1] (lateral, out-of-plane, global y),
        #   phi = disp[5] (rotation about z)
        disp_i = X[6*i:6*i+6]
        disp_j = X[6*j:6*j+6]
        u_i, v_i, phi_i = disp_i[0], disp_i[1], disp_i[5]
        u_j, v_j, phi_j = disp_j[0], disp_j[1], disp_j[5]
        
        # Normalized coordinate along the element
        xi = np.linspace(0, 1, num_points)
        
        # Hermite cubic shape functions for bending in the (global) y direction:
        N1 = 1 - 3*xi**2 + 2*xi**3
        N2 = L*(xi - 2*xi**2 + xi**3)
        N3 = 3*xi**2 - 2*xi**3
        N4 = L*(-xi**2 + xi**3)
        
        # Interpolate axial (u) displacement linearly and bending (v) displacement via Hermite interpolation
        u_local = np.linspace(u_i, u_j, num_points)
        v_local = N1*v_i + N2*phi_i + N3*v_j + N4*phi_j
        
        # Local coordinate along the beam
        x_local = np.linspace(0, L, num_points)
        # Compute deformed local coordinates:
        x_def = x_local + scale * u_local   # along element
        y_def = scale * v_local               # out-of-plane (global y)
        # Assume no deformation in the local z-direction (in-plane)
        z_def = np.zeros_like(x_def)
        
        # Map the local deformed coordinates to global coordinates.
        # p1 is the origin of the element's local coordinate system.
        # x_def is along e_x, y_def is along the forced global e_y, and z_def along e_z.
        deformed_pts = np.zeros((num_points, 3))
        for k in range(num_points):
            deformed_pts[k, :] = p1 + x_def[k]*e_x + y_def[k]*e_y + z_def[k]*e_z
        
        ax.plot(deformed_pts[:,0], deformed_pts[:,1], deformed_pts[:,2], 'r-', lw=2)
    
    # Set equal aspect ratio for a proper 3D view
    ax.set_box_aspect([1, 1, 1])
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title("Continuously Deformed Frame Structure")
    plt.show()
    plt.savefig( saving_dir_with_name )



