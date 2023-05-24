"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-18
Description: use 2d frame dynamic problem to test the finite element analysis solver
To do:
    1. clear the code
    2. the gif time is not correct
"""

from Tumu.pyTumu import plot_2D_frame_static, solution, form_stiffness_2D_frame, Newmark, form_lumped_mass_2D_frame, \
    Newmark_with_prescribed_dof
import matplotlib.pyplot as plt
import numpy as np
import imageio

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

# plot_2D_frame_static(node_coordinates, node_coordinates_end, reactions, element_nodes)

ll = 1
rho = 1
Area = 1

force = np.zeros(GDof)
# Force vector
force[1] = 15e3
force[9] = 10e6 * 0.3
test = np.zeros((len(force), 1000))
for i in range(1000):
    test[:, i] = force * np.sin(0.03 * i)

mass = form_lumped_mass_2D_frame(GDof, number_nodes, number_elements, element_nodes, node_coordinates, rho, Area)

displacements_flatten = Newmark_with_prescribed_dof(mass=mass, stiffness=stiffness, force=test,
                                                    prescribed_dof=prescribed_dof, GDof=GDof, dt=0.01)

images = []
for i in range(displacements_flatten.shape[1]):
    displacements_flatten_sample = displacements_flatten[:, i]
    displacements = np.array([displacements_flatten_sample[0: len(displacements_flatten_sample) // 3],
                              displacements_flatten_sample[
                              len(displacements_flatten_sample) // 3: len(displacements_flatten_sample) // 3 * 2]]).T
    node_coordinates_end = node_coordinates + displacements
    reactions = stiffness @ displacements_flatten_sample

    plot_2D_frame_static(node_coordinates, node_coordinates_end, reactions, element_nodes)
    # Save the current figure to a temporary file and add it to the images list

    temp_file = f'temp/column_{i + 1}.png'
    plt.savefig(temp_file)
    images.append(imageio.v2.imread(temp_file))

    # Clear the current figure
    plt.clf()

# Create a GIF from the images
imageio.mimsave('columns.gif', images, duration=0.1)

# Remove temporary files
import os

for i in range(displacements_flatten.shape[0]):
    os.remove(f'temp/column_{i + 1}.png')
