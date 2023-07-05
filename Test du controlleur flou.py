import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt
import time

# Definition des grandeurs floue dont on aura besoin dans notre controlleur
#pour l'entrée qui 'est l'erreur
error = ctrl.Antecedent(np.arange(-149, 151, 1), 'error')
#pour la sortie qui'est l'angle de correction
correction = ctrl.Consequent(np.arange(-0.395, 0.391, 0.001), 'correction')




# Définir les fonctions d'appartenance pour les ensembles flous
#fonction triangulaire pour l'erreur
error['sharp_left'] = fuzz.trapmf(error.universe,  [-149, -149, -88.1, -40.2])
error['left'] = fuzz.trimf(error.universe,  [-69.9 ,-38.2, -25.67])
error['tend to left '] = fuzz.trimf(error.universe,  [-32.38, -15, -5])
error['center'] = fuzz.trimf(error.universe, [-5 ,0, 5])
error['tend to right'] = fuzz.trimf(error.universe,  [5, 20.82, 31])
error['right'] = fuzz.trimf(error.universe,  [25 ,41.87 ,59.7])
error['sharp_right'] = fuzz.trapmf(error.universe,  [45.6, 92.1, 150 ,150])




#creation des fonction d'appartenance triangulaire pour la grandeur floue de sortie
#pour l'angle de la correction
correction['gau3'] = fuzz.trapmf(correction.universe,  [-0.395, -0.392 ,-0.36, -0.2461])
correction['gau2'] = fuzz.trimf(correction.universe,  [-0.213 ,-0.1382 ,-0.105])
correction['gau1'] = fuzz.trimf(correction.universe,  [-0.1155 ,-0.0622, 0])
correction['tout_droit'] = fuzz.trapmf(correction.universe,  [-0.01,0.01, 0.01,0.01])
correction['dr1'] = fuzz.trimf(correction.universe,  [0.013 ,0.0842 ,0.165])
correction['dr2'] = fuzz.trimf(correction.universe,  [0.0819, 0.1792, 0.259])
correction['dr3'] = fuzz.trapmf(correction.universe,  [0.232 ,0.3605 ,0.391, 0.391])
# Définition des règles floues
rules = [
    ctrl.Rule(error['sharp_left'], correction['dr3']),
    ctrl.Rule(error['left'], correction['dr2']),
    ctrl.Rule(error['tend to left '], correction['dr1']),
    ctrl.Rule(error['center'], correction['tout_droit']),
    ctrl.Rule(error['tend to right'], correction['gau1']),
    ctrl.Rule(error['right'], correction['gau2']),
    ctrl.Rule(error['sharp_right'], correction['gau3'])
]
# rules = [
#     ctrl.Rule(error['sharp_left'], correction['gau3']),
#     ctrl.Rule(error['left'], correction['gau2']),
#     ctrl.Rule(error['tend to left'], correction['gau1']),
#     ctrl.Rule(error['center'], correction['tout_droit']),
#     ctrl.Rule(error['tend to right'], correction['dr1']),
#     ctrl.Rule(error['right'], correction['dr2']),
#     ctrl.Rule(error['sharp_right'], correction['dr3'])

# ]


# Création du système de contrôle flou
controller = ctrl.ControlSystem(rules)
control_simulation = ctrl.ControlSystemSimulation(controller)

while True:
    # Effacer le graphe précédent

    # Obtention des degrés d'appartenance pour une entrée donnée
    input_error = float(input("Entrez la valeur de l'erreur (entre -149 et 150) : "))

    # Évaluation du système de contrôle flou avec la valeur d'erreur donnée
    control_simulation.input['error'] = input_error
    control_simulation.compute()
    output_correction = control_simulation.output['correction']


    # Calculate the membership value for the input_error in each membership function
    membership_values = {}
    for mf_name in error.terms.keys():
        membership_values[mf_name] = fuzz.interp_membership(error.universe, error[mf_name].mf, input_error)

    # Find the membership function with the maximum membership value

    max_membership_func = max(membership_values, key=membership_values.get)
    max_membership_value = membership_values[max_membership_func]

    # Plotting the membership functions for error and correction
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(error.universe, fuzz.trapmf(error.universe,  [-149, -149, -88.1, -40.2]))
    ax1.plot(error.universe,fuzz.trimf(error.universe,  [-69.9 ,-38.2, -25.67]))
    ax1.plot(error.universe,fuzz.trimf(error.universe,  [-32.38, -15, -5]))
    ax1.plot(error.universe,fuzz.trimf(error.universe, [-5 ,0, 5]))
    ax1.plot(error.universe,fuzz.trimf(error.universe,  [5, 20.82, 31]) )
    ax1.plot(error.universe,fuzz.trimf(error.universe, [25, 41.87, 59.7]))
    ax1.plot(error.universe,fuzz.trapmf(error.universe, [45.6, 92.1, 150, 150]))
    ax1.set_xlabel('Error')
    ax1.set_ylabel('Membership')
    ax1.set_title('Membership functions for Error')
    ax1.legend()

    # Plot lines connecting error to correction
    ax1.plot([input_error, input_error], [0, max_membership_value], 'k--', linewidth=1.5)
    ax1.plot([input_error, 150], [max_membership_value, max_membership_value], 'k--', linewidth=1.5)


    ax2.plot(correction.universe, fuzz.trapmf(correction.universe,  [-0.395, -0.392 ,-0.36, -0.2461]))
    ax2.plot(correction.universe, fuzz.trimf(correction.universe,  [-0.213 ,-0.1382 ,-0.105]))
    ax2.plot(correction.universe, fuzz.trimf(correction.universe,  [-0.1155 ,-0.0622, 0]))
    ax2.plot(correction.universe, fuzz.trapmf(correction.universe,  [-0.01,0.01, 0.01,0.01]))
    ax2.plot(correction.universe, fuzz.trimf(correction.universe,  [0.013 ,0.0842 ,0.165]))
    ax2.plot(correction.universe, fuzz.trimf(correction.universe, [0.0819, 0.1792, 0.259]))
    ax2.plot(correction.universe, fuzz.trapmf(correction.universe, [0.232, 0.3605, 0.391, 0.391]))


    ax2.set_xlabel('Correction')
    ax2.set_ylabel('Membership')
    ax2.set_title('Membership functions for Correction')
    ax2.legend()
    ax2.plot([-0.4, 0.4], [max_membership_value, max_membership_value], 'k--', linewidth=1.5)
    ax2.plot([output_correction, output_correction], [max_membership_value, 0], 'k--', linewidth=1.5)

    print(output_correction)
    #Show the plot
    plt.tight_layout()

    # plt.show()










