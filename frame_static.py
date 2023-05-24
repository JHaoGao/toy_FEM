"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-01
Description: use 2d frame static problem to test the finite element analysis solver
"""

from Tumu.pyTumu import plot_2D_frame_static, solution, form_stiffness_2D_frame
import matplotlib.pyplot as plt
import numpy as np

# Constants
E = 210000
A = 100
I = 2e8
EA = E * A
EI = E * I

# Generation of coordinates and connectivities
number_elements = 3
p1 = 3000 * (1 + np.cos(np.pi / 4))
node_coordinates = np.array([[0, 3000], [3000, 3000], [p1, 0], [p1 + 3000, 0]])
element_nodes = np.column_stack((np.arange(0, number_elements), np.arange(1, number_elements + 1)))
number_nodes = node_coordinates.shape[0]

# Global number of degrees of freedom
GDof = 3 * number_nodes
U = np.zeros(GDof)
force = np.zeros(GDof)

# Force vector
force[5] = -10000
force[6] = -10000
force[9] = -5e6
force[10] = 5e6

# Boundary conditions and solution
prescribed_dof = np.array([0, 3, 4, 7, 8, 11])

stiffness = form_stiffness_2D_frame(GDof, number_nodes, number_elements, element_nodes, node_coordinates, EI, EA)

# stiffness = form_stiffness_2D_frame(GDof, number_elements, element_nodes, number_nodes, node_coordinates[:, 0], node_coordinates[:, 1], EI, EA)
displacements_flatten = solution(GDof, prescribed_dof, stiffness, force)
displacements = np.array([displacements_flatten[0: len(displacements_flatten) // 3],
                          displacements_flatten[
                          len(displacements_flatten) // 3: len(displacements_flatten) // 3 * 2]]).T
node_coordinates_end = node_coordinates + displacements
reactions = stiffness @ displacements_flatten

plot_2D_frame_static(node_coordinates, node_coordinates + 500 * displacements, reactions, element_nodes)
plt.show()

# Constants
E = 210e9
A = 2e-4
I = 2e-4
EA = E * A
EI = E * I

# Generation of coordinates and connectivities
number_elements = 3
node_coordinates = np.array([[0, 0], [0, 6], [6, 6], [6, 0]])
element_nodes = np.array([[0, 1], [1, 2], [2, 3]])
number_nodes = node_coordinates.shape[0]

# Global number of degrees of freedom
GDof = 3 * number_nodes
U = np.zeros(GDof)
# force = np.zeros(GDof)
#
# # Force vector
# force[1] = 15e3
# force[9] = 10e6

# Boundary conditions and solution
prescribed_dof = np.array([0, 3, 4, 7, 8, 11])

stiffness = form_stiffness_2D_frame(GDof, number_nodes, number_elements, element_nodes, node_coordinates, EI, EA)

# stiffness = form_stiffness_2D_frame(GDof, number_elements, element_nodes, number_nodes, node_coordinates[:, 0], node_coordinates[:, 1], EI, EA)

force = np.zeros(GDof)

# Force vector
force[1] = 15e3
force[9] = 10e6 * 0.3

displacements_flatten = solution(GDof, prescribed_dof, stiffness, force)
displacements = np.array([displacements_flatten[0: len(displacements_flatten) // 3],
                          displacements_flatten[
                          len(displacements_flatten) // 3: len(displacements_flatten) // 3 * 2]]).T
node_coordinates_end = node_coordinates + displacements
reactions = stiffness @ displacements_flatten

plot_2D_frame_static(node_coordinates, node_coordinates_end, reactions, element_nodes)
plt.show()
