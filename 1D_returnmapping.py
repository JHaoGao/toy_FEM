"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-03-30
Description: 1d return mapping
"""
import numpy as np
import matplotlib.pyplot as plt
from Tumu import isotropic_calculate_stress, kinematic_isotropic_calculate_stress, visco_isotropic_calculate_stress

material_properties = {
    'E': 10 ** 5,  # Young's modulus
    'K': 10 ** 5,  # Plastic modulus
    'sigma_y': 2000,  # Yield stress
    'H': 10 ** 5,  # kinematic modulus
    'eta': 0.5 * 10 ** 5,  # viscosity coefficient
}

material_properties['tao'] = material_properties['eta'] / (material_properties['E'] + material_properties['K'])
strains = np.concatenate([np.linspace(0, 0.1, 1000)[:-1],
                          np.linspace(0.1, 0, 1000)[:-1],
                          np.linspace(0, -0.1, 1000)[:-1],
                          np.linspace(-0.1, 0, 1000)])

stresses, plastic_strains = isotropic_calculate_stress(material_properties=material_properties,
                                                       strains=strains)
plt.plot(strains, stresses, linestyle='-', color='blue', marker='', label='Isotropic')

stresses, plastic_strains = kinematic_isotropic_calculate_stress(material_properties=material_properties,
                                                                 strains=strains)
plt.plot(strains, stresses, linestyle='--', color='green', marker='', label='Kinematic-Isotropic')

dt = 0.005
stresses, plastic_strains = visco_isotropic_calculate_stress(material_properties=material_properties,
                                                             strains=strains,
                                                             dt=dt)
plt.plot(strains, stresses, linestyle='-.', color='red', marker='', label=f'Visco-Isotropic, dt = {dt}s')

plt.xlabel("Strain")
plt.ylabel("Stress (Pa)")
plt.title("Strain-Stress Curve, step = 1E-4")
plt.grid(True)
plt.legend(loc='best')
plt.show()
