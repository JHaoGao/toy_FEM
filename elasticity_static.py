"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-26
Description: 2d solid elasticity, static problem
"""

import numpy as np
import matplotlib.pyplot as plt
from Tumu.pyTumu import solution, rectangularMesh, formStiffness2D, strainstress2D, Compute_VonMises_2D, drawingField

E = 10e7
poisson = 0.30

# Matrix C
C = E / (1 - poisson ** 2) * np.array([[1, poisson, 0], [poisson, 1, 0], [0, 0, (1 - poisson)]])

# Load
P = 1e5

# Mesh generation
Lx, Ly = 5, 1
numberElementsX, numberElementsY = 20, 10
nodeCoordinates, elementNodes = rectangularMesh(Lx, Ly, numberElementsX, numberElementsY)
elementNodes = elementNodes.astype(int)
numberElements = elementNodes.shape[0]
numberNodes = nodeCoordinates.shape[0]
GDof = 2 * numberNodes

# Calculation of the system stiffness matrix
stiffness, mass = formStiffness2D(GDof = GDof,
                                  numberElements = numberElements,
                                  elementNodes = elementNodes,
                                  numberNodes = numberNodes,
                                  nodeCoordinates = nodeCoordinates,
                                  C = C,
                                  rho = 1,
                                  thickness = 1)

# Define boundary conditions
prescribed_dof = np.hstack((np.where(nodeCoordinates[:, 0] == 0)[0], np.where(nodeCoordinates[:, 0] == 0)[0] + numberNodes))
active_dof = np.setdiff1d(np.arange(numberNodes), prescribed_dof)
force = np.zeros(GDof)
force_position = np.where(nodeCoordinates[:, 0] == Lx)[0] + numberNodes
force[force_position] = P * Ly / numberElementsY

# Linear solver
displacements_flatten = solution(GDof, prescribed_dof, stiffness, force)

displacement = np.array([displacements_flatten[:numberNodes], displacements_flatten[numberNodes:]]).T
strain, stress, InternalForce = strainstress2D(elementNodes, nodeCoordinates, displacements_flatten, C, numberElements, numberNodes)
vonmises = Compute_VonMises_2D(stress)
drawingField(nodeCoordinates + displacement, elementNodes, ScalarField = vonmises, Field_name = 'Von Mises Stress')
plt.show()
assert np.linalg.norm((InternalForce - force)[active_dof]) < 1e-6, 'Internal force not equal to external force'