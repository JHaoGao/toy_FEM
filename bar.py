"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-03-31
Description: use 1 d bar to test the finite element analysis solver
"""

import numpy as np
from Tumu.pyTumu import finite_element_analysis_1D_truss_static

# Example 1
number_nodes = 4
elemnt_stiffness = np.array([1, 1, 1])
element_nodes = np.array([[0, 1], [1, 2], [1, 3]])
prescribed_dof = np.array([0, 2, 3])
force = np.array([0, 10, 0, 0])

result = finite_element_analysis_1D_truss_static(number_nodes, elemnt_stiffness, element_nodes, prescribed_dof, force)
print(result['displacements'])


# Example 2
E = 70 * 10e9
A = 200 * 10e-6
L = 2
k = 2000e3

number_nodes = 4
elemnt_stiffness = np.array([E * A / L, E * A / L, k])
element_nodes = np.array([[0, 1], [1, 2], [2, 3]])
prescribed_dof = np.array([0, 3])
force = np.array([0, 8000, 0, 0])

result = finite_element_analysis_1D_truss_static(number_nodes, elemnt_stiffness, element_nodes, prescribed_dof, force)
print(result['displacements'] * 1000)