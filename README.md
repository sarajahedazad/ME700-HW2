[![codecov](https://codecov.io/gh/sarajahedazad/ME700-HW2/graph/badge.svg?token=bECtu6QvVu)](https://codecov.io/gh/sarajahedazad/ME700-HW2)
[![Run Tests](https://github.com/sarajahedazad/ME700-HW2/actions/workflows/test.yml/badge.svg)](https://github.com/sarajahedazad/ME700-HW2/actions/workflows/test.yml)

# Computational Mechanics: Nonlinear Analysis and Software Development - HW2

This project implements a 3D Frame Analysis Solver using the Direct Stiffness Method. Designed to analyze 3D frame structures, it calculates nodal displacements, rotations, reaction forces, and moments, as well as the critical load of the structure. The solver also includes post-processing capabilities for visualizing the deformed shape of structures. The development of this project adheres to the Test-Driven Development (TDD) approach and emphasizes good coding and documentation practices.

The repository’s primary codes are located in **src**.

---
## Table of Contents

- [Course Information](#course-information)
- [Problem Overview](#problem)
- [Theory](#theory)
- [Conda Environment, Installation, and Testing](#install)
- [How to Use The Codes](#htu)
- [References](#references)

---

## Course Information

**ME700 – Advanced Topics in Mechanical Engineering**  
This course is offered at Boston University and, for Spring 2025, will be taught by Dr. Lejeune under the specific topic:  
**Computational Mechanics: Nonlinear Analysis and Software Development**.  
Each semester, the course may cover different topics, and the assignments are designed to blend theoretical insights with practical coding and testing practices.

---
## Problem Overview  <a name="problem"></a>
This project aims to create a robust matrix structural analysis tool specifically designed for analyzing 3D frames using the Direct Stiffness Method (DSM). The solver offers comprehensive functionality to address complex structural analysis challenges, making it an essential resource for understanding and predicting the behavior of frame structures under various loading conditions. Its design focuses on providing detailed insights into structural mechanics, facilitating effective analysis and decision-making in complex engineering projects.   
**Part 1: Direct Stiffness Method Implementation**  
The first part of the project involves implementing a DSM solver that accurately predicts the structural behavior of 3D frames under various loading conditions. The solver will require inputs such as frame geometry, including node locations and element connectivity, element section properties (modulus of elasticity, Poisson's ratio, cross-sectional area, and moments of inertia), applied nodal loads (forces and moments), and boundary conditions (constraints on displacements and rotations at specific nodes). The output from this part of the solver will include detailed displacements and rotations at each node and the reaction forces and moments at supported nodes. This section also includes the creation of a detailed tutorial guiding users through the process of setting up a frame, analyzing it with the solver, and understanding the results.

**Part 2: Post-processing and Elastic Critical Load Analysis**  
The second part of the project expands on the first by adding capabilities for post-processing and critical load analysis. This includes functions to calculate and visualize internal forces and moments in member-local coordinates and the interpolated deformed shape of the entire structure. Additionally, an Elastic Critical Load solver will be integrated to assess the buckling load of the frame, enhancing the solver's utility in stability analysis. This extension aims to provide a robust tool for detailed examination and optimization of frame structures under critical conditions, supported by a tutorial that extends the one from Part 1 to cover new functionalities.

This solver is being developed with a strong emphasis on test-driven development (TDD) and good coding practices, ensuring reliability and ease of use. This GitHub repository serves as a hub for accessing the solver, associated tutorials, and for community interaction to discuss improvements and share insights.

---
### Theoretical Background

#### Structural Analysis Using the Direct Stiffness Method

The Direct Stiffness Method (DSM) is a fundamental approach in structural analysis, primarily used to determine the response of structures under various loading conditions. It revolves around three key components: the displacement vector $\Delta$, the elastic stiffness matrix $K$, and the force vector $F$.

- **Elastic Stiffness Matrix ($K$)**: This matrix represents the rigidity of the structure and is assembled from individual element stiffness matrices. The transformation matrices are used to align local degrees of freedom with the global system. The general form of the stiffness matrix for an element in the global coordinate system can be expressed as:

  
  $K_{global} = \Gamma^T K_{local} \Gamma$
  

  where $\Gamma$ is the transformation matrix, $K_{local}$ is the element's stiffness matrix in its local coordinates, and $\Gamma^T$ denotes the transpose of $\Gamma$.

- **Displacement Vector ($\Delta$)**: It contains the displacements and rotations at each node of the structure. Nodes are categorized as either supported (where displacements are known and typically zero) or free (where displacements are unknown and need to be calculated).

- **Force Vector ($F$)**: This vector includes all external forces and moments applied to the nodes of the structure. For supported nodes, the corresponding entries in the force vector are the reaction forces and moments.

The relationship between these components is captured by the matrix equation:


$K\Delta = F$


To solve for unknown displacements at the free nodes, the stiffness matrix $K$ is partitioned into submatrices corresponding to the supported and free nodes, leading to a system that can be rearranged to isolate and solve for the unknown displacements.

#### Geometric Stiffness and Stability Analysis

For stability analysis, the focus shifts to the geometric stiffness matrix $K_g$ and its role in determining the buckling load of the structure.

- **Geometric Stiffness Matrix ($K_g$)**: This matrix accounts for the effects of preloading on the structure. It captures the stiffness changes due to the applied loads.
  


- **Elastic Critical Load and Eigenvalue Analysis**: The critical load is determined using eigenvalue analysis of the combined stiffness matrix $K + \lambda K_g$, where $\lambda$ represents a load factor. The smallest eigenvalue from this analysis gives the critical load factor, indicating the load level at which the structure is likely to buckle.

- **Eigenvectors and Interpolated Shapes**: Eigenvectors from the eigenvalue analysis represent potential buckled shapes of the structure. These are used to visualize the deformed configuration under critical loading conditions.

- **Shape Functions**: Hermite and linear shape functions are used to interpolate the displacements along the elements for plotting deformed shapes. Hermite functions consider both displacement and slope continuity across the elements, providing a smooth and accurate visualization, whereas linear functions offer a simpler but less smooth approximation.

This detailed theoretical framework supports the functionalities developed in the matrix structural analysis solver, enabling comprehensive structural analysis and advanced stability assessments of 3D frame structures.


---

### Conda Environment, Installation, and Testing <a name="install"></a>
_This section is copied and pasted from [Lejeune's Lab Graduate Course Materials: Bisection Method](https://github.com/Lejeune-Lab-Graduate-Course-Materials/bisection-method.git)_

To install this package, please begin by setting up a conda environment (mamba also works):
```bash
conda create --name hw2-env python=3.12
```
Once the environment has been created, activate it:

```bash
conda activate hw2-env
```
Double check that python is version 3.12 in the environment:
```bash
python --version
```
Ensure that pip is using the most up to date version of setuptools:
```bash
pip install --upgrade pip setuptools wheel
```
Create an editable install of the codes (note: you must be in the correct directory):
```bash
pip install -e .
```
Test that the code is working with pytest:
```bash
pytest -v --cov=src --cov-report term-missing
```
Code coverage should be 100%. Now you are prepared to write your own code based on this method and/or run the tutorial. 


If you are using VSCode to run this code, don't forget to set VSCode virtual environment to hw2-env.

If you would like the open the tutorials located in the `tutorials` folder ( the `.ipynb` file ) as a Jupyter notebook in the browser, you might need to install Jupyter notebook in your conda environment as well:
```bash
pip install jupyter
```
```bash
cd tutorials/
```
```bash
jupyter notebook tutorials_matrix_structural_analysis.ipynb
```
### An alternative way to test the codes without installing the package <a name="alter"></a>  
Below is an example that demonstrates how to use the codes in src without installation.  
- Step 1: Download the `.py` files from the folder `src`([here](https://github.com/sarajahedazad/ME700-HW2/tree/main/src)). Place them in the same folder as your working directory.
- Step 2: Create a python file in that folder and write your example in that file. You can import the mdules and functions in those files with the following line:
`from boundary_conditions import *`
- Step 3: Run your code and enjoy!
Here is an example that demonstrates how you can use the files in `src` folder (they should be in the same folder as the python file that you intend to run):

## How to Use The Codes <a name="htu"></a>
To see more examples, see `tutorials_matrix_structural_analysis.ipynb` [here](https://github.com/sarajahedazad/ME700-HW2/tree/main/tutorials).
**How to import modules**
```
import numpy as np
from geometry import *
from boundary_conditions import *
from stiffness_matrices import *
from solver import *
from shape_functions import *
```

**How to define a structure**
```
frame = Frame()
p0 = frame.add_point(x0, y0, z0) # Defining a point
p1 = frame.add_point(x1, y1, z1)

el0 = frame.add_element( p0, p1, E, nu, A, Iy, Iz, I_rho, J, v_temp ) # defining an element
# p0: point object 0, p1: point object 0, E: Young's modulus, nu: Poisson's ratio  
# A: area, Iy: moment of inertia about the y-axis, Iz: moment of inertia about the z-axis   
# I_rho: polar moment of inertia, J: , v_temp: local z axis
element_lst = [el0] 
frame.build_frame( element_lst ) # Required for building a frame structure
```
**How to define Boundary Conditions**    
```
bcs = BoundaryConditions( frame )
# displacement bounds
bcs.add_disp_bound_xyz( [x0, y0, z0], disp_bound_x, disp_bound_y, disp_bound_z ) # you can also use bcs.add_disp_bound_xyz( p0.coords, disp_bound_x, disp_bound_y, disp_bound_z ) 

# rotation bounds
bcs.add_rot_bound_xyz( [x0, y0, z0], rot_bound_x, rot_bound_y, rot_bound_z )

# force bounds
bcs.add_force_bound_xyz( [x1, y1, z1], force_bound_x, force_bound_y, force_bound_z )

# momentum bounds
bcs.add_momentum_bound_xyz( [x1, y1, z1], momentum_bound_x, momentum_bound_y, momentum_bound_z )

# we have to set up the bounds in the end. This step is required.
bcs.set_up_bounds()
```  
Note: If you need to bound only one axis, you can do it like this:    
```
bcs.add_force_bound_x( [x0, y0, z0], force_bound_x )
```

**How to build a stiffness matrix** 
```
stiffmat = StiffnessMatrices( frame )
K = stiffmat.get_global_elastic_stiffmatrix()
```

**Solving for unknown displacements/rotations, and forces/momentums**
```
Delta, F = solve_stiffness_system( K, bcs )

node_idx = 0
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )

node_idx = 1
print( f'Disp/rotations at Node { node_idx }:', Delta[node_idx * 6: node_idx * 6 + 6] )
print( f'Reactions at Node { node_idx }:', F[node_idx * 6: node_idx * 6 + 6] )
```

**Building the geometric stiffness matrix and finding the critical load**

```
K_g = stiffmat.get_global_geometric_stiffmatrix( Delta )

P_cr, eigenvectors_allstructure = compute_critical_load(K, K_g, bcs)
print( 'critical', P_cr )
```
**Saving the interpolated configuration of the structure**
```
n , scale = 20, 10
shapefunctions = ShapeFunctions(eigenvectors_allstructure, frame, n=n, scale=scale)
saving_dir_with_name = 'Original Configuration vs Interpolated.png'
shapefunctions.plot_element_interpolation( saving_dir_with_name )
```

## References
* [Lejeune Lab Graduate Course Materials: Bisection-Method](https://github.com/Lejeune-Lab-Graduate-Course-Materials/bisection-method/tree/main)
* ChatGPT: was used for completing the documentation. More details about the AI use is provided in the `assignment_2_genAIuse.txt`.
