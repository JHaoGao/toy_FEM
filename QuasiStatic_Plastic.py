'''
Title: QuasiStatic_Plastic.py
Author: Jianhao Gao, jianhaogao2022@gmail.com
Function: 2D J2 plasticity quasi static computing
Initial:
      2023/4/25,
         - 4/27, finished J2 static simulation
Edit: 2023/4/28
         - 4/29, debug for some iteration problem
'''

import numpy as np
import matplotlib.pyplot as plt
from Tumu.pyTumu import solution, rectangularMesh, gaussQuadrature, shape_function_Q4, Jacobian, \
    formStiffness2D, Compute_VonMises_2D, CombHardJ2_PlaneStrain, drawingField, displace2strain2D_gauss, \
    formStiffness2D_plastic

material_properties = {
    'Young': 2e9,
    'nu': 0.30,
    'mu': 1e9 / (2 * (1 + 0.30)),
    'lambda': 0.30 * 1e9 / ((1 + 0.30) * (1 - 2 * 0.30)),
    'beta': 0,
    'H': 2e8,
    'Y0': 1e6 * np.sqrt(3)
}

E = material_properties['Young']
poisson = material_properties['nu']
Y0 = material_properties['Y0']

# Matrix C
C = E / (1 - 2 * poisson) / (1 + poisson) * np.array([[1 - poisson, poisson, 0],
                                                      [poisson, 1 - poisson, 0],
                                                      [0, 0, (1 - 2 * poisson) / 2]])
# Load
P = 1 * 1e5
# Mesh generation
Lx, Ly = 5, 1
numberElementsX, numberElementsY = 50, 10
nodeCoordinates, elementNodes = rectangularMesh(Lx, Ly, numberElementsX, numberElementsY)
elementNodes = elementNodes.astype(int)
numberElements = elementNodes.shape[0]
numberNodes = nodeCoordinates.shape[0]
GDof = 2 * numberNodes

# Define boundary conditions
prescribed_dof = np.hstack(
    (np.where(nodeCoordinates[:, 0] == 0)[0], np.where(nodeCoordinates[:, 0] == 0)[0] + numberNodes))
active_dof = np.setdiff1d(np.arange(GDof), prescribed_dof)
force = np.zeros(GDof)
force_position = np.where(nodeCoordinates[:, 0] == Lx)[0] + numberNodes
force_position_boundary = np.array([force_position[0], force_position[-1]])
force[force_position] = P * Ly / numberElementsY
force[force_position_boundary] = 0.5 * P * Ly / numberElementsY

numberIncremental = 3
numberIncremental_with0 = numberIncremental + 1
maxiteration = 50
numberIteration_with0 = maxiteration + 1

ep = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 4))  # plastic strain norm history at gauss point
strain = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 4, 3))  # strain history at gauss point
d_strain = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 4, 3))  # strain history at gauss point
stress = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 4, 3))  # stress history at gauss point
alpha = np.zeros((numberIncremental_with0, numberIteration_with0, numberElements, 4, 3))  # alpha history at gauss point
InternalForce = np.zeros((numberIncremental_with0, numberIteration_with0, GDof))  # InternalForce history at GDof
C_consistent_gauss = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 4, 3, 3))  # C_consistent history at gauss point
C_consistent = np.zeros(
    (numberIncremental_with0, numberIteration_with0, numberElements, 3, 3))  # C_consistent history at element
C_consistent_assemble = np.zeros(
    (numberIncremental_with0, numberIteration_with0, GDof, GDof))  # Assembel a big C consistent matrix

# Original states
ep[0, -1, :, :] = np.zeros((numberElements, 4))
strain[0, -1, :, :, :] = np.zeros((numberElements, 4, 3))
d_strain[0, -1, :, :, :] = np.zeros((numberElements, 4, 3))
stress[0, -1, :, :, :] = np.zeros((numberElements, 4, 3))
alpha[0, -1, :, :, :] = np.zeros((numberElements, 4, 3))
C_consistent_assemble[0, -1, :, :], mass = formStiffness2D(GDof=GDof,
                                                           numberElements=numberElements,
                                                           elementNodes=elementNodes,
                                                           numberNodes=numberNodes,
                                                           nodeCoordinates=nodeCoordinates,
                                                           C=C,
                                                           rho=1,
                                                           thickness=1)

force_state = np.zeros(GDof)
force_incre = force / numberIncremental

# %% Start main loop
displacements_total = np.zeros(GDof)
displacements_flatten = np.zeros(GDof)
for incre in range(1, 1 + numberIncremental):  # Load seperate to 10 incremenals
    if incre == 1:
        last_maxiter = -1
    else:
        last_maxiter = iteration + 1
    # Linear solver
    d_displacements = solution(GDof=GDof,
                               prescribed_dof=prescribed_dof,
                               stiffness=C_consistent_assemble[0, -1, :, :],
                               force=force_incre)
    displacements_total = displacements_total + d_displacements
    force_state = force_state + force_incre
    for iteration in range(maxiteration):
        # step 1 compute strain
        d_strain[incre, iteration, :, :, :] = displace2strain2D_gauss(elementNodes,
                                                                      nodeCoordinates,
                                                                      displacements_flatten=d_displacements,
                                                                      numberElements=numberElements,
                                                                      numberNodes=numberNodes)

        # step 2, compute stress by return mapping
        deps = d_strain[incre, iteration, :, :, :]
        for ele in range(numberElements):
            for gpt in range(4):
                stress[incre, iteration + 1, ele, gpt, :], alpha[incre, iteration + 1, ele, gpt, :], ep[
                    incre, iteration + 1, ele, gpt], C_consistent_gauss[incre, iteration + 1, ele, gpt, :,
                                                     :], ignore = CombHardJ2_PlaneStrain(
                    material_properties=material_properties,
                    C=C,
                    deps=deps[ele, gpt, :],
                    stressN=stress[incre - 1, last_maxiter, ele, gpt, :],
                    alphaN=alpha[incre - 1, last_maxiter, ele, gpt, :],
                    epN=ep[incre - 1, last_maxiter, ele, gpt])

        C_consistent = C_consistent_gauss.sum(axis=3)

        C_consistent_assemble[incre, iteration + 1, :, :] = formStiffness2D_plastic(GDof=GDof,
                                                                                    numberElements=numberElements,
                                                                                    elementNodes=elementNodes,
                                                                                    numberNodes=numberNodes,
                                                                                    nodeCoordinates=nodeCoordinates,
                                                                                    C=C_consistent[incre, iteration + 1,
                                                                                      :, :, :],
                                                                                    thickness=1)
        # Step 3, compute internal force
        weights, locations = gaussQuadrature('complete')
        for ele in range(numberElements):
            indice = elementNodes[ele, :]  # indice (4, )
            elementDof = np.hstack((indice, indice + numberNodes))  # elementDof (8, )

            for gpt in range(4):
                pt = locations[gpt, :]  # pt (2, )
                wt = weights[gpt]  # wt float
                xi, eta = pt  # xi, eta float

                shapeFunction, naturalDerivatives = shape_function_Q4(xi, eta)
                Jacob, invJacobian, XYderivatives = Jacobian(nodeCoordinates[indice, :], naturalDerivatives)

                B = np.zeros((3, 2 * len(indice)))  # B (3, 8)
                B[0, :len(indice)] = XYderivatives[:, 0]
                B[1, len(indice):] = XYderivatives[:, 1]
                B[2, :len(indice)] = XYderivatives[:, 1]
                B[2, len(indice):] = XYderivatives[:, 0]
                ''' B
                [dN1/dx dN2/dx dN3/dx dN4/dx 0      0      0      0      0      0      0      0     ]
                [0      0      0      0      dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0     ]
                [dN1/dy dN2/dy dN3/dy dN4/dy 0      0      0      0      dN1/dx dN2/dx dN3/dx dN4/dx]
                '''

                stress_vector = stress[incre, iteration + 1, ele, gpt, :]
                int_force_contribution = B.T @ stress_vector * wt * np.linalg.det(Jacob)

                # Assemble the internal force contribution to the global internal force vector
                InternalForce[incre, iteration + 1, elementDof] += int_force_contribution

        force_residual = force_state - InternalForce[incre, iteration + 1, :]
        force_residual[prescribed_dof] = 0
        residual_norm = np.sum(force_residual[active_dof] ** 2)

        print('------------')
        print('Incremental:', incre)
        print('Iteration:', iteration)
        if residual_norm < 1e-6:
            print('Convergence has meet!')
            print('Maximum displacement:', displacements_total.max())
            break
        else:
            print('Norm of force difference:', residual_norm)
            d_d_displacements = solution(GDof,
                                         prescribed_dof,
                                         stiffness=C_consistent_assemble[incre, iteration + 1, :, :],
                                         force=force_residual)
            displacements_total = displacements_total + d_d_displacements
            d_displacements = d_displacements + d_d_displacements
            print('Maximum displacement:', displacements_total.max())

    if residual_norm > 1e-6:
        print('Warning: Maximum iteration has reached, convergence has not meet.')

displacement = np.array([displacements_total[:numberNodes], displacements_total[numberNodes:]]).T
stress2plot = stress[incre, iteration + 1, :, :, :].sum(axis=1)
vonmises = Compute_VonMises_2D(stress2plot)

drawingField(nodeCoordinates + 10 * displacement, elementNodes, ScalarField=vonmises)
plt.title('Vonmises stress (Pa)')
plt.text(0.05, 0.95, 'displacement scalar factor 10:1', transform=plt.gca().transAxes, verticalalignment='top',
         fontsize=10)
plt.show()

drawingField(nodeCoordinates + 10 * displacement, elementNodes, ScalarField=stress2plot[:, 0])
plt.title('Sigma xx (Pa)')
plt.text(0.05, 0.95, 'displacement scalar factor 10:1', transform=plt.gca().transAxes, verticalalignment='top',
         fontsize=10)
plt.show()

drawingField(nodeCoordinates + 10 * displacement, elementNodes, ScalarField=stress2plot[:, 2])
plt.title('Sigma xy (Pa)')
plt.text(0.05, 0.95, 'displacement scalar factor 10:1', transform=plt.gca().transAxes, verticalalignment='top',
         fontsize=10)
plt.show()

drawingField(nodeCoordinates + 10 * displacement, elementNodes, ScalarField=stress2plot[:, 1])
plt.title('Sigma yy (Pa)')
plt.text(0.05, 0.95, 'displacement scalar factor 10:1', transform=plt.gca().transAxes, verticalalignment='top',
         fontsize=10)
plt.show()
