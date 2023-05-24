"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-7
Description: use euler beam to test the finite element analysis solver
"""

from Tumu.pyTumu import solution, form_stiffness_bernoulli_beam
import matplotlib.pyplot as plt
import numpy as np


# Euler Beam
E, I, EI = 1, 1, 1
number_elements = 80
node_coordinates = np.linspace(0, 1, number_elements + 1)[:, np.newaxis]
element_nodes = np.array([(i, i + 1) for i in range(number_elements)])
P = -1
GDof = 2 * len(node_coordinates)

# set boundary conditions
fixedNodeU = [0, 2 * number_elements]
fixedNodeV = []
prescribedDof = np.array(fixedNodeU + fixedNodeV)

# Form matrix
stiffness, force = form_stiffness_bernoulli_beam(GDof, number_elements, element_nodes, node_coordinates, EI, P)

# Solver
displacements = solution(GDof, prescribedDof, stiffness, force)
reactions = np.dot(stiffness, displacements)

U = displacements[0::2]
plt.plot(node_coordinates, U, '.')
plt.show()


