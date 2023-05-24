"""
Author: Jianhao Gao, jianhaogao2022@gmail.com
Initial Date: 2023-03-20
First Edit Date: 2023-03-30, add isotropic_trial_yieldcondition and isotropic_return_mapping functions
Description: Computational inelasticity library
To do:
    1. The inelastic library should be moved from pyTumu to here
"""

import numpy as np


# Use a strain history to calculate stress directly
def isotropic_calculate_stress(material_properties, strains):
    # Initialize arrays
    stresses = np.zeros(len(strains))
    plastic_strains = np.zeros(len(strains))
    alphas = np.zeros(len(strains))

    trial_stresses = np.zeros(len(strains))
    trial_yieldcondition = np.zeros(len(strains))

    # Isotropic hardening loop
    for i, dstrain in enumerate(np.diff(strains)):
        trial_stresses[i + 1] = material_properties['E'] * (strains[i + 1] - plastic_strains[i])
        trial_yieldcondition[i + 1] = abs(trial_stresses[i + 1]) - (
                    material_properties['sigma_y'] + material_properties['K'] * alphas[i])

        if trial_yieldcondition[i + 1] <= 0:
            stresses[i + 1] = trial_stresses[i + 1]
            plastic_strains[i + 1] = plastic_strains[i]
            alphas[i + 1] = alphas[i]
        else:
            # Return mapping
            dgamma = trial_yieldcondition[i + 1] / (material_properties['E'] + material_properties['K'])
            assert dgamma > 0, f'In iteration {i}, dgamma must be larger than 0 to perform return mapping'
            stresses[i + 1] = (1 - (dgamma * material_properties['E']) / abs(trial_stresses[i + 1])) * trial_stresses[
                i + 1]
            plastic_strains[i + 1] = plastic_strains[i] + dgamma * np.sign(trial_stresses[i + 1])
            alphas[i + 1] = alphas[i] + dgamma
    return stresses, plastic_strains


# isotropic hardening yield condition
def isotropic_trial_yieldcondition(material_properties, strain, plastic_strain, alpha):
    trial_stress = material_properties['E'] * (strain - plastic_strain)
    trial_yieldcondition = abs(trial_stress) - (material_properties['sigma_y'] + material_properties['K'] * alpha)
    return trial_stress, trial_yieldcondition


# istropic hardening return mapping
def isotropic_return_mapping(material_properties, plastic_strain, alpha, trial_stress, trial_yieldcondition):
    # Return mapping
    dgamma = trial_yieldcondition / (material_properties['E'] + material_properties['K'])
    assert dgamma > 0, f'In iteration, dgamma must be larger than 0 to perform return mapping'
    stress = (1 - (dgamma * material_properties['E']) / abs(trial_stress)) * trial_stress
    plastic_strain = plastic_strain + dgamma * np.sign(trial_stress)
    alpha = alpha + dgamma

    return stress, plastic_strain, alpha


# Use a strain history to calculate stress directly
def kinematic_isotropic_calculate_stress(material_properties, strains):
    strain_steps = np.diff(strains)
    # Initialize arrays
    stresses = np.zeros(len(strains))
    plastic_strains = np.zeros(len(strains))
    alphas = np.zeros(len(strains))
    q = np.zeros(len(strains))

    trial_xi = np.zeros(len(strains))
    trial_stresses = np.zeros(len(strains))
    trial_yieldcondition = np.zeros(len(strains))

    # Isotropic hardening loop
    for i, dstrain in enumerate(strain_steps):
        trial_stresses[i + 1] = material_properties['E'] * (strains[i + 1] - plastic_strains[i])
        trial_xi[i + 1] = trial_stresses[i + 1] - q[i]
        trial_yieldcondition[i + 1] = abs(trial_xi[i + 1]) - (
                    material_properties['sigma_y'] + material_properties['K'] * alphas[i])

        if trial_yieldcondition[i + 1] <= 0:
            stresses[i + 1] = trial_stresses[i + 1]
            plastic_strains[i + 1] = plastic_strains[i]
            alphas[i + 1] = alphas[i]
            q[i + 1] = q[i]

        else:
            # Return mapping

            dgamma = trial_yieldcondition[i + 1] / (
                        material_properties['E'] + material_properties['K'] + material_properties['H'])
            assert dgamma > 0, f'In iteration {i}, dgamma must be larger than 0 to perform return mapping'

            stresses[i + 1] = trial_stresses[i + 1] - dgamma * material_properties['E'] * np.sign(trial_xi[i + 1])
            plastic_strains[i + 1] = plastic_strains[i] + dgamma * np.sign(trial_xi[i + 1])
            q[i + 1] = q[i] + dgamma * material_properties['H'] * np.sign(trial_xi[i + 1])
            alphas[i + 1] = alphas[i] + dgamma
    return stresses, plastic_strains


# Use a strain history to calculate stress directly
def visco_isotropic_calculate_stress(material_properties, strains, dt=0.1):
    strain_steps = np.diff(strains)
    # Initialize arrays
    stresses = np.zeros(len(strains))
    plastic_strains = np.zeros(len(strains))
    alphas = np.zeros(len(strains))

    inf_stresses = np.zeros(len(strains))
    inf_alphas = np.zeros(len(strains))

    trial_stresses = np.zeros(len(strains))
    trial_yieldcondition = np.zeros(len(strains))

    # Isotropic hardening loop
    for i, dstrain in enumerate(strain_steps):
        trial_stresses[i + 1] = material_properties['E'] * (strains[i + 1] - plastic_strains[i])
        trial_yieldcondition[i + 1] = abs(trial_stresses[i + 1]) - (
                    material_properties['sigma_y'] + material_properties['K'] * alphas[i])

        if trial_yieldcondition[i + 1] <= 0:
            stresses[i + 1] = trial_stresses[i + 1]
            plastic_strains[i + 1] = plastic_strains[i]
            alphas[i + 1] = alphas[i]

        else:
            # Return mapping without time, to get infinite properties
            dgamma = trial_yieldcondition[i + 1] / (material_properties['E'] + material_properties['K'])
            assert dgamma > 0, f'In iteration {i}, dgamma must be larger than 0 to perform return mapping'

            stresses[i + 1] = (1 - (dgamma * material_properties['E']) / abs(trial_stresses[i + 1])) * trial_stresses[
                i + 1]
            plastic_strains[i + 1] = plastic_strains[i] + dgamma * np.sign(trial_stresses[i + 1])
            alphas[i + 1] = alphas[i] + dgamma

            # Viscoplastic regularization
            stresses[i + 1] = (trial_stresses[i + 1] + dt / material_properties['tao'] * stresses[i + 1]) / (
                        1 + dt / material_properties['tao'])
            alphas[i + 1] = (alphas[i] + dt / material_properties['tao'] * alphas[i + 1]) / (
                        1 + dt / material_properties['tao'])
            plastic_strains[i + 1] = strains[i + 1] - stresses[i + 1] / material_properties['E']
    return stresses, plastic_strains
