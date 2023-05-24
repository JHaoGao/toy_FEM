"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-04-20
Description: 1d return mapping
"""
from Tumu.pyTumu import combHardJ2_3D
import numpy as np
import matplotlib.pyplot as plt
from Tumu.pyTumu import CombHardJ2_PlaneStrain

# Material properties
material_properties = {
    'Young': 24000,
    'nu': 0.2,
    'mu': 24000 / (2 * (1 + 0.2)),
    'lambda': 0.2 * 24000 / ((1 + 0.2) * (1 - 2 * 0.2)),
    'beta': 0,
    'H': 1000,
    'Y0': 200 * np.sqrt(3)
}

# 3D
Young = material_properties['Young']
nu = material_properties['nu']
mu = material_properties['mu']
lambda_ = material_properties['lambda']  # Unit: MPa
beta = material_properties['beta']
H = material_properties['H']
Y0 = material_properties['Y0']  # Assume linear isotropic hardening

# Identity tensor
Iden = np.array([1, 1, 1, 0, 0, 0])

# Elastic stiffness matrix
# This is for engineering strain
C = np.array([
    [2 * mu + lambda_, lambda_, lambda_, 0, 0, 0],
    [lambda_, 2 * mu + lambda_, lambda_, 0, 0, 0],
    [lambda_, lambda_, 2 * mu + lambda_, 0, 0, 0],
    [0, 0, 0, 2 * mu, 0, 0],
    [0, 0, 0, 0, 2 * mu, 0],
    [0, 0, 0, 0, 0, 2 * mu]])
'''
Use with real strain, not engineering strain
C = D = E/(1 + nu)/(1 - 2*nu) *
[1-nu nu   nu   0     0     0    ]
[nu   1-nu nu   0     0     0    ]
[nu   nu   1-nu 0     0     0    ]
[0    0    0    1-2nu 0     0    ]
[0    0    0    0     1-2nu 0    ]
[0    0    0    0     0     1-2nu]
or
C = D =
[2mu + lambda lambda       lambda       0     0     0    ]
[lambda       2mu + lambda lambda       0     0     0    ]
[lambda       lambda       2mu + lambda 0     0     0    ]
[0            0            0            mu/2  0     0    ]
[0            0            0            0     mu/2  0    ]
[0            0            0            0     0     mu/2 ]
'''

# Initial zero states
stressN = np.zeros(6)
deps = np.zeros(6)
alphaN = np.zeros(6)
epN = 0

# Pre-allocate variables for efficiency
numIncrements = 20
X = np.zeros(numIncrements + 1)
Y = np.zeros(numIncrements + 1)

# Loop for load increments
for i in range(numIncrements):
    deps[3] = 0.003  # Shear strain increment Δγ12 = 0.003 at each step
    stress, alpha, ep, C_consistent, C_continuum = combHardJ2_3D(material_properties, C=C, deps=deps, stressN=stressN,
                                                                 alphaN=alphaN, epN=epN)
    X[i + 1] = (i + 1) * deps[3]
    Y[i + 1] = stress[3]
    stressN = stress
    alphaN = alpha
    epN = ep

# Set print options
np.set_printoptions(precision=2, linewidth=120, suppress=True)

# Plot shear stress τ12 vs. shear strain γ12 curve
print('Consistent tangent modulu:\n', C_consistent)
print('Continuum tangent modulu:\n', C_continuum)
plt.plot(X, Y, '-s', linewidth=2, markersize=8)
plt.xlabel('Shear strain')
plt.ylabel('Shear stress (MPa)')
plt.show()

# Plain strain
E = material_properties['Young']
nu = material_properties['nu']
D = E / (1 - 2 * nu) / (1 + nu) * np.array([[1 - nu, nu, 0],
                                            [nu, 1 - nu, 0],
                                            [0, 0, (1 - 2 * nu)]])
# Initial zero states
stressN = np.zeros(3)
deps = np.zeros(3)
alphaN = np.zeros(3)
epN = 0.0

# Pre-allocate variables for efficiency
numIncrements = 20
X = np.zeros(numIncrements + 1)
Y = np.zeros(numIncrements + 1)

# Loop for load increments
for i in range(numIncrements):
    deps[2] = 0.003  # Shear strain increment Δγ12 = 0.003 at each step
    stress, alpha, ep, C_consistent, C_continuum = CombHardJ2_PlaneStrain(material_properties, D, deps, stressN, alphaN,
                                                                          epN=epN)
    X[i + 1] = (i + 1) * deps[2]
    Y[i + 1] = stress[2]
    stressN = stress
    alphaN = alpha
    epN = float(ep)

# Plot shear stress τ12 vs. shear strain γ12 curve
print('Consistent tangent modulu:\n', C_consistent)
print('Continuum tangent modulu:\n', C_continuum)
plt.plot(X, Y, '-s', linewidth=2, markersize=8)
plt.xlabel('Shear strain')
plt.ylabel('Shear stress (MPa)')
plt.show()
