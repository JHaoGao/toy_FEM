"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-03-31
Description: use 2d truss to test the finite element analysis solver
"""

from Tumu.pyTumu import solution, stresses_2D_truss, form_stiffness_2D_truss, plot_2D_truss_static
import numpy as np

# Example 1
# Constants
E = 30e6
A = 2
EA = E * A

# Node coordinates and element connectivity
number_elements = 3
number_nodes = 4
element_nodes = np.array([[1, 2], [1, 3], [1, 4]]) - 1  # Adjust indices to be zero-based
node_coordinates = np.array([[0, 0], [0, 120], [120, 120], [120, 0]])
forces = np.array([[0.0, -10000.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
displacements = np.zeros(forces.shape)
# Boundary conditions and solution
prescribed_dof = np.arange(2, 8)

# Displacements, force, and stiffness
GDof = 2 * number_nodes
displacements_flatten = displacements.flatten()
forces_flatten = forces.flatten()

# Functions for stiffness matrix, solution, and drawing mesh need to be implemented
stiffness = form_stiffness_2D_truss(GDof, number_elements, element_nodes, node_coordinates, EA)

# Solution
displacements_flatten = solution(GDof, prescribed_dof, stiffness, forces_flatten)
reactions_flatten = stiffness @ displacements_flatten
displacements = displacements_flatten.reshape([len(displacements_flatten) // 2, 2])
reactions = reactions_flatten.reshape([len(displacements_flatten) // 2, 2])
node_coordinates_end = node_coordinates + displacements

stress = stresses_2D_truss(element_nodes, node_coordinates, node_coordinates_end, E)

plot_2D_truss_static(node_coordinates_end, reactions, stress, element_nodes)


# Example 2
# Constants
E = 70000
A = 300
EA = E * A

# Node coordinates and element connectivity

element_nodes = np.array([[1, 2], [1, 3], [2, 3], [2, 4], [1, 4], [3, 4], [3, 6], [4, 5], [4, 6], [3, 5],
                          [5, 6]]) - 1  # Adjust indices to be zero-based
node_coordinates = np.array([[0, 0], [0, 3000], [3000, 0], [3000, 3000], [6000, 0], [6000, 3000]])
forces = np.array([[0.0, 0.0], [0.0, -50000], [0.0, 0.0], [0.0, -100000], [0.0, 0.0], [0.0, -50000]]) * 5
displacements = np.zeros(forces.shape)
# Boundary conditions and solution
prescribed_dof = np.array([1, 2, 10]) - 1

number_elements = len(element_nodes)
number_nodes = len(node_coordinates)

# Displacements, force, and stiffness
GDof = 2 * number_nodes
displacements_flatten = displacements.flatten()
forces_flatten = forces.flatten()

# Functions for stiffness matrix, solution, and drawing mesh need to be implemented
stiffness = form_stiffness_2D_truss(GDof, number_elements, element_nodes, node_coordinates, EA)

# Solution
displacements_flatten = solution(GDof, prescribed_dof, stiffness, forces_flatten)
reactions_flatten = stiffness @ displacements_flatten
displacements = displacements_flatten.reshape([len(displacements_flatten) // 2, 2])
reactions = reactions_flatten.reshape([len(displacements_flatten) // 2, 2])
node_coordinates_end = node_coordinates + displacements

stress = stresses_2D_truss(element_nodes, node_coordinates, node_coordinates_end, E)

plot_2D_truss_static(node_coordinates_end, reactions, stress, element_nodes)