"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-8
Description: use timoshenko beam to test the finite element analysis solver, and modal analysis.
"""

from Tumu.pyTumu import solution, form_stiffness_mass_timoshenko_beam, shape_function_l2, Newmark, \
    Newmark_with_prescribed_dof
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

# Timoshenko beam in bending

# E: modulus of elasticity
# G: shear modulus
# I: second moments of area
# L: length of beam
# thickness: thickness of beam

rho = 1
E = 10e7
poisson = 0.30
L = 1
thickness = 0.001
A = 1 * thickness
modeNumber = 4
I = thickness ** 3 / 12
EI = E * I
kapa = 5 / 6
P = -1  # uniform pressure

# constitutive matrix
G = E / 2 / (1 + poisson)
C = np.array([[EI, 0], [0, kapa * thickness * G]])

# mesh
number_elements = 100
node_coordinates = np.linspace(0, L, number_elements + 1)
xx = node_coordinates.reshape(-1, 1)

element_nodes = np.zeros((number_elements, 2), dtype=int)
element_nodes[:, 0] = np.arange(number_elements)
element_nodes[:, 1] = np.arange(1, number_elements + 1)

# generation of coordinates and connectivities
number_nodes = xx.shape[0]

# GDof: global number of degrees of freedom
GDof = 2 * number_nodes

# computation of the system stiffness matrix
stiffness, force, mass = form_stiffness_mass_timoshenko_beam(
    GDof, number_elements, element_nodes, number_nodes, xx, C, P, 1, I, thickness
)

# boundary conditions (simply-supported at both bords)
# fixed_node_w = np.array([0, number_nodes - 1])
# fixed_node_tx = np.array([])

# boundary conditions (clamped at both bords)
fixed_node_w = np.array([0, number_nodes - 1])
fixed_node_tx = fixed_node_w + number_nodes

# boundary conditions (cantilever)
# fixed_node_w = np.array([0])
# fixed_node_tx = np.array([0])

prescribed_dof = np.concatenate((fixed_node_w, fixed_node_tx))

# solution
displacements = solution(GDof, prescribed_dof, stiffness, force)

# free vibration problem
activeDof = np.setdiff1d(np.arange(0, GDof), prescribed_dof)

D, V = eigh(stiffness[activeDof, :][:, activeDof], mass[activeDof, :][:, activeDof])

V = V[:number_elements - 1, :4]
V = V * 0.35 / abs(V).max()
plt.figure()
plt.plot(activeDof[:int(len(activeDof) / 2)], V[:number_elements - 1, 0])
plt.plot(activeDof[:int(len(activeDof) / 2)], V[:number_elements - 1, 1])
plt.plot(activeDof[:int(len(activeDof) / 2)], V[:number_elements - 1, 2])
plt.plot(activeDof[:int(len(activeDof) / 2)], V[:number_elements - 1, 3])

plt.scatter(range(number_nodes), displacements[:number_nodes])
plt.show()

plt.figure()
force[:, np.newaxis].shape
test = np.zeros((len(force), 100))
for i in range(100):
    test[:, i] = force * np.sin(i)

Q = Newmark_with_prescribed_dof(mass=mass, stiffness=stiffness, GDof=GDof, prescribed_dof=prescribed_dof, force=test,
                                dt=0.01)
Deflection = Q[:len(Q) // 2, :]
Rotation = Q[len(Q) // 2:, :]
plt.plot(Deflection[:, 20])
plt.show()
