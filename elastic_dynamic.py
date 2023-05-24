"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-26
Description: 2d solid elasticity,dynamic problem
"""

import numpy as np
import matplotlib.pyplot as plt
from Tumu.pyTumu import rectangularMesh, formStiffness2D, Newmark_with_prescribed_dof, strainstress2D, \
    Compute_VonMises_2D, drawingField
from matplotlib.animation import FuncAnimation

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
stiffness, mass = formStiffness2D(GDof=GDof,
                                  numberElements=numberElements,
                                  elementNodes=elementNodes,
                                  numberNodes=numberNodes,
                                  nodeCoordinates=nodeCoordinates,
                                  C=C,
                                  rho=1,
                                  thickness=1)

# Define boundary conditions
prescribed_dof = np.hstack(
    (np.where(nodeCoordinates[:, 0] == 0)[0], np.where(nodeCoordinates[:, 0] == 0)[0] + numberNodes))
active_dof = np.setdiff1d(np.arange(numberNodes), prescribed_dof)
force_i = np.zeros(GDof)
force_position = np.where(nodeCoordinates[:, 0] == Lx)[0] + numberNodes
force_i[force_position] = P * Ly / numberElementsY

force = np.zeros((len(force_i), 100))
for i in range(100):
    force[:, i] = force_i * np.sin(0.1 * i)

# Compute displacement time history
Q = Newmark_with_prescribed_dof(mass=mass, stiffness=stiffness, GDof=GDof, prescribed_dof=prescribed_dof, force=force,
                                dt=0.1)


def update(frame):
    plt.clf()  # Clear the current figure
    displacement = np.array([Q[:numberNodes, frame], Q[numberNodes:, frame]]).T
    displacements_flatten = np.zeros((GDof))
    displacements_flatten[:numberNodes] = displacement[:, 0]
    displacements_flatten[numberNodes:] = displacement[:, 1]
    strain, stress, InternalForce = strainstress2D(elementNodes, nodeCoordinates, displacements_flatten, C,
                                                   numberElements, numberNodes)
    vonmises = Compute_VonMises_2D(stress)
    drawingField(nodeCoordinates + displacement, elementNodes, ScalarField=stress[:, 0], Field_name='stress_xx')
    # drawingField(nodeCoordinates + displacement, elementNodes, ScalarField=vonmises)
    plt.xlim([-1, 6])
    plt.ylim([-2.5, 3.5])


n_frames = 50
fig = plt.figure()
anim = FuncAnimation(fig, update, frames=n_frames, interval=50, blit=False)
anim.save('Result/vibration.gif', writer='imagemagick', fps=60)
