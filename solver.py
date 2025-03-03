import numpy as np
from scipy.linalg import eig
# Note: ChatGPT was used for this function

def partition(matrix, known_DoF_idx):
    known_idx = known_DoF_idx
    n = matrix.shape[0]
    
    # Identify free DOF indices (those not prescribed)
    free_idx = np.setdiff1d(np.arange(n), known_idx)
    
    # Partition the stiffness matrix into submatrices:
    # K_pp: prescribed-prescribed, K_pu: prescribed-free,
    # K_up: free-prescribed, K_uu: free-free.
    matrix_pp = matrix[np.ix_(known_idx, known_idx)]
    matrix_pu = matrix[np.ix_(known_idx, free_idx)]
    matrix_up = matrix[np.ix_(free_idx, known_idx)]
    matrix_uu = matrix[np.ix_(free_idx, free_idx)]

    return matrix_pp, matrix_pu, matrix_up, matrix_uu


def solve_stiffness_system(K, bcs):
    known_disp_idx = bcs.BCs_X_indices 
    X_prescribed = bcs.BCs_X_values

    known_F_idx = bcs.BCs_F_indices 
    F_prescribed = bcs.BCs_F_values
     
    F_full = np.zeros( (bcs.n_DoFs) )
    F_full[known_F_idx] = F_prescribed
    """
    Solve the system F = KX when some displacements are prescribed and forces are applied 
    on the remaining free degrees of freedom.

    Parameters:
        K : np.ndarray
            The global stiffness matrix of size (n x n).
        known_disp_idx : array-like
            The indices of the degrees of freedom (DOFs) with prescribed (known) displacements.
        X_prescribed : array-like
            The prescribed displacement values corresponding to known_disp_idx.
        F_full : np.ndarray
            The global force vector of size n. For free DOFs, F_full should contain the applied forces.
            The entries corresponding to prescribed DOFs are typically set to zero (or can be ignored),
            as they will be computed as reaction forces.

    Returns:
        X : np.ndarray
            The full displacement vector (size n), with computed unknown displacements and prescribed ones.
        F : np.ndarray
            The full force vector (size n), where the reaction forces at the prescribed DOFs have been computed.
    """
    n = K.shape[0]
    K_pp, K_pu, K_up, K_uu = partition(K, known_disp_idx)
    
    # Partition the displacement and force vectors:
    X_p = np.array(X_prescribed)
    F_u = F_full[known_F_idx]
    
    # For the free DOFs, the governing equations are:
    # F_u = K_up * X_p + K_uu * X_u
    # Solve for the unknown displacements X_u:
    X_u = np.linalg.solve(K_uu, F_u - K_up @ X_p)
    
    # Compute reaction forces at the prescribed DOFs:
    F_p = K_pp @ X_p + K_pu @ X_u
    
    # Assemble the complete displacement and force vectors:
    X = np.zeros(n)
    F = np.copy(F_full)
    X[known_disp_idx] = X_p
    X[known_F_idx] = X_u
    F[known_disp_idx] = F_p  # reaction forces
    
    return X, F

def compute_critical_load(K_elastic, K_geometric):
    """
    Solves the buckling eigenvalue problem to find the critical load.

    Parameters:
    K_elastic (ndarray): Elastic stiffness matrix.
    K_geometric (ndarray): Geometric stiffness matrix.
    P_applied (float): Applied force before buckling.

    Returns:
    float: Critical buckling load P_cr
    float: Smallest eigenvalue (buckling factor)
    ndarray: Corresponding eigenvector (buckling mode shape)
    
    """
    # known_DoF_idx = bcs.BCs_X_indices 
    # _, _, _, k_uu = partition(K_elastic, known_DoF_idx) 
    # _, _, _, k_g_uu = partition(K_geometric, known_DoF_idx) 

    # Solve the generalized eigenvalue problem: (K_elastic - lambda * K_geometric) = 0
    eigenvalues, eigenvectors = eig(K_elastic, K_geometric)

    # Extract real positive eigenvalues (buckling factors)
    real_eigenvalues = np.real(eigenvalues)  # Take only the real part
    positive_eigenvalues = real_eigenvalues[real_eigenvalues > 0]  # Keep only positive values

    #print( real_eigenvalues )
    if len(positive_eigenvalues) == 0:
        raise ValueError("No positive eigenvalue found! Check input matrices.")

    # Smallest positive eigenvalue (critical load factor)
    lambda_critical = np.min(positive_eigenvalues)

    return lambda_critical


