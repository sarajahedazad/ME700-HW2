[![codecov](https://codecov.io/gh/sarajahedazad/ME700-HW2/graph/badge.svg?token=bECtu6QvVu)](https://codecov.io/gh/sarajahedazad/ME700-HW2)
[![Run Tests](https://github.com/sarajahedazad/ME700-HW2/actions/workflows/test.yml/badge.svg)](https://github.com/sarajahedazad/ME700-HW2/actions/workflows/test.yml)

# Computational Mechanics: Nonlinear Analysis and Software Development - HW2

This project implements a 3D Frame Analysis Solver using the Direct Stiffness Method. Designed to analyze 3D frame structures, it calculates nodal displacements, rotations, reaction forces, and moments, as well as the critical load of the structure. The solver also includes post-processing capabilities for visualizing the deformed shape of structures. The development of this project adheres to the Test-Driven Development (TDD) approach and emphasizes good coding and documentation practices.

The repository’s primary codes are located in **src**.

---
## Table of Contents

- [Course Information](#course-information)
- [Assignment Overview](#assignment-overview)
- [Theory](#theory)
  - [sth](#sth)
- [Installation and Environment Setup](#installation-and-environment-setup)
- [Tutorials and Testing](#tutorials-and-testing)
- [References](#references)

---

## Course Information

**ME700 – Advanced Topics in Mechanical Engineering**  
This course is offered at Boston University and, for Spring 2025, will be taught by Dr. Lejeune under the specific topic:  
**Computational Mechanics: Nonlinear Analysis and Software Development**.  
Each semester, the course may cover different topics, and the assignments are designed to blend theoretical insights with practical coding and testing practices.

---
## Problem Overview   
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

  
  $K_{global} = T^T K_{local} T$
  

  where $T$ is the transformation matrix, $K_{local}$ is the element's stiffness matrix in its local coordinates, and $T^T$ denotes the transpose of $T$.

- **Displacement Vector ($\Delta$)**: It contains the displacements and rotations at each node of the structure. Nodes are categorized as either supported (where displacements are known and typically zero) or free (where displacements are unknown and need to be calculated).

- **Force Vector ($F$)**: This vector includes all external forces and moments applied to the nodes of the structure. For supported nodes, the corresponding entries in the force vector are the reaction forces and moments.

The relationship between these components is captured by the matrix equation:


$K\Delta = F$


To solve for unknown displacements at the free nodes, the stiffness matrix $K$ is partitioned into submatrices corresponding to the supported and free nodes, leading to a system that can be rearranged to isolate and solve for the unknown displacements.

#### Geometric Stiffness and Stability Analysis

For stability analysis, the focus shifts to the geometric stiffness matrix $K_g$ and its role in determining the buckling load of the structure.

- **Geometric Stiffness Matrix ($K_g$)**: This matrix accounts for the effects of preloading on the structure. It captures the stiffness changes due to the applied loads and is defined as:

  
  $K_g = \sum \frac{N}{L} \begin{bmatrix} 1 & 0 & -1 & 0 \\ 0 & 0 & 0 & 0 \\ -1 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}$
  

  where $N$ is the axial force in the member and $L$ is the length of the member.

- **Elastic Critical Load and Eigenvalue Analysis**: The critical load is determined using eigenvalue analysis of the combined stiffness matrix $K + \lambda K_g$, where $\lambda$ represents a load factor. The smallest eigenvalue from this analysis gives the critical load factor, indicating the load level at which the structure is likely to buckle.

- **Eigenvectors and Interpolated Shapes**: Eigenvectors from the eigenvalue analysis represent potential buckled shapes of the structure. These are used to visualize the deformed configuration under critical loading conditions.

- **Shape Functions**: Hermite and linear shape functions are used to interpolate the displacements along the elements for plotting deformed shapes. Hermite functions consider both displacement and slope continuity across the elements, providing a smooth and accurate visualization, whereas linear functions offer a simpler but less smooth approximation.

This detailed theoretical framework supports the functionalities developed in the matrix structural analysis solver, enabling comprehensive structural analysis and advanced stability assessments of 3D frame structures.


