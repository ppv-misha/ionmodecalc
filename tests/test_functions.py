# import requi9red module
import sys
  
# append the path of the
# parent directory
sys.path.append("..")

import ionmodecalc as imc
import numpy as np

print(imc.__version__)
print(dir(imc))
## Declaration of ion types used in the simulation
ion_types = [{'mass': 40, 'charge': 1}, {'mass': 44, 'charge': 1}]
## Ions ordering. You should place ion type number from the previous list in a desired order.
ions_order = [1]*10
ions_order.extend([1, 0, 0, 0, 0, 0, 0, 0, 0, 1]*8)
ions_order.extend([1]*11)
ions_order = np.array(ions_order)
ion_number = ions_order.shape[0]
# Initial distance between two neighboring ions
ions_initial_splitting = 1e-5
print('Ion\'s order:', ions_order)

# AQT Pine trap parameters
R0_eff = 0.624e-3  # in meters
Z0 = 2.25e-3  # in meters
kappa = 0.0567
RF_frequency = 30e6  # in Hz

reference_ion_type = 1  # number of ion's type for each frequencies are defined

# If you want to operate with secular frequencies, but not voltages, uncomment block bellow

w_z = 3e4  # axial secular frequency in Hz
w_r = 3.0692e6  # radial secular frequency in Hz

DC_voltage, RF_voltage = imc.frequency_to_voltage(
    w_z, w_r, ion_types[reference_ion_type], RF_frequency, Z0, R0_eff, kappa
)
#print("Endcap voltage:", DC_voltage, "V, blade voltage:", RF_voltage, "V")

# Description of Paul trap parameters
trap = {
    "radius": R0_eff,
    "length": Z0,
    "kappa": kappa,
    "frequency": RF_frequency,
    "voltage": RF_voltage,
    "endcapvoltage": DC_voltage,
    "pseudo": True,
    "anisotropy": 0.999,
}

pinned_ions = np.zeros(ion_number)

# mode 1
#pinned_ions_indexes = [19, 20, 39, 40, 59, 60, 79, 80]
#pinned_ions[pinned_ions_indexes] = 1

# mode 2
#pinned_ions_indexes = [29, 30, 49, 50, 69, 70]
#pinned_ions[pinned_ions_indexes] = 1

print('Pinned ions: ', pinned_ions)
w_tw = 0.45481027634035615*w_r

"""Simulation of ion crystal structure"""

# for current_order in ions_order:
final_z = imc.simulation_run_equations(ion_types, ions_order, ions_initial_splitting, trap)
radial_modes, radial_freqs, axial_modes, axial_freqs = imc.get_modes(
    ion_types,
    ions_order,
    trap,
    final_z=final_z,
    pinned_ions = pinned_ions,
    w_tweezer = w_tw
)

freq_var_coeff = imc.get_freq_variation_coefficient(
    ion_types,
    ions_order,
    trap,
    ion_ref_type = 0
)

#imc.radial_modes_plot(ions_order, radial_modes, axial_modes, radial_freqs, axial_freqs, 'mode_matrix.png')
imc.radial_modes_plot(ions_order, radial_modes, axial_modes, radial_freqs, axial_freqs, 'mode_matrix.png')