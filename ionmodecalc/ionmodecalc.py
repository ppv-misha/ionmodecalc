import numpy as np
from numpy.linalg import eigh
from scipy.optimize import fsolve
import pylion as pl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

__version__ = '0.9.1'

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# Constants declaration
ech = 1.602176634e-19  # electron charge, C
amu = 1.66053906660e-27  # atomic mass unit, kg
eps0 = 8.8541878128e-12  # vacuum electric permittivity

def frequency_to_voltage(
    w_z, w_r, ref_ion, RF_frequency, Z0, R0_eff, kappa
):  # frequencies in Hz
    mass_ref = ref_ion["mass"] * amu
    charge_ref = ref_ion["charge"] * ech
    az = (2 * w_z / RF_frequency) ** 2
    qr = (2 * (2 * w_r / RF_frequency) ** 2 + az) ** 0.5
    DC_voltage = (
        az
        * mass_ref
        * Z0 ** 2
        * 4
        * np.pi ** 2
        * RF_frequency ** 2
        / 8
        / charge_ref
        / kappa
    )
    RF_voltage = (
        qr
        * mass_ref
        * R0_eff ** 2
        * 4
        * np.pi ** 2
        * RF_frequency ** 2
        / 2
        / charge_ref
    )
    return (DC_voltage, RF_voltage)  # voltage in volts

def voltage_to_frequency(
    ref_ion, trap
):  # voltage in volts
    mass_ref = ref_ion["mass"] * amu
    charge_ref = ref_ion["charge"] * ech
    az = (
        8
        * charge_ref
        * trap['kappa']
        * trap['endcapvoltage']
        / (mass_ref * trap['length'] ** 2 * 4 * np.pi ** 2 * trap['frequency'] ** 2)
    )
    qr = (
        2
        * charge_ref
        * trap['voltage']
        / (mass_ref * trap['radius'] ** 2 * 4 * np.pi ** 2 * trap['frequency'] ** 2)
    )
    w_z = trap['frequency'] / 2 * (az) ** 0.5
    w_r = trap['frequency'] / 2 * (0.5 * qr ** 2 - az / 2) ** 0.5
    if (
        np.real(w_z) < 0
        or np.real(w_r) < 0
        or np.imag(w_z) > 1e-10
        or np.imag(w_r) > 1e-10
    ):
        print("Error: Stable confinement wasn't achieved! Change voltages!")
    return (w_z, w_r)  # frequencies in Hz

def axial_hessian_matrix(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    ion_number = ion_positions.shape[0]
    axial_freqs = np.zeros(ion_number)
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(ion_types[ions_order[i]], trap)
        axial_freqs[i] = w_z
    ions_mass = np.array([ion_types[x]['mass']*amu for x in ions_order])
    ions_charge = np.array([ion_types[x]['charge']*ech for x in ions_order])
    interaction_coeff = 1/8/np.pi/eps0
    A_matrix = np.diag(ions_mass*axial_freqs**2*4*np.pi**2/2)
    A_matrix += np.diag(ions_mass*(pinned_ions*w_tweezer)**2*4*np.pi**2/2)
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                A_matrix[i, j] = -2*ions_charge[i]*ions_charge[j]*interaction_coeff/np.abs(ion_positions[i]-ion_positions[j])**3
                S += 2*ions_charge[i]*ions_charge[j]*interaction_coeff/np.abs(ion_positions[i]-ion_positions[j])**3
        A_matrix[i, i] += S
    M_matrix = np.diag(ions_mass**(-0.5))
    return (A_matrix*2, M_matrix)

def radial_hessian_matrix(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    ion_number = ion_positions.shape[0]
    radial_freqs = np.zeros(ion_number)
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(ion_types[ions_order[i]], trap)
        radial_freqs[i] = w_r
    ions_mass = np.array([ion_types[x]['mass']*amu for x in ions_order])
    ions_charge = np.array([ion_types[x]['charge']*ech for x in ions_order])
    interaction_coeff = 1/8/np.pi/eps0
    B_matrix = np.diag(ions_mass*radial_freqs**2*4*np.pi**2/2)
    B_matrix += np.diag(ions_mass*(pinned_ions*w_tweezer)**2*4*np.pi**2/2)
    for i in range(ion_number):
        S = 0
        for j in range(ion_number):
            if j != i:
                B_matrix[i, j] = ions_charge[i]*ions_charge[j]*interaction_coeff/np.abs(ion_positions[i]-ion_positions[j])**3
                S -= ions_charge[i]*ions_charge[j]*interaction_coeff/np.abs(ion_positions[i]-ion_positions[j])**3
        B_matrix[i, i] += S
    M_matrix = np.diag(ions_mass**(-0.5))
    return (B_matrix*2, M_matrix)

def axial_normal_modes(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    axial_hessian, mass_matrix = axial_hessian_matrix(ion_positions, ion_types, ions_order, trap, pinned_ions, w_tweezer)
    D = mass_matrix.dot(axial_hessian.dot(mass_matrix))
    freq, normal_vectors = eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq**0.5/2/np.pi
    return (freq, normal_vectors)


def radial_normal_modes(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    radial_hessian, mass_matrix = radial_hessian_matrix(ion_positions, ion_types, ions_order, trap, pinned_ions, w_tweezer)
    D = mass_matrix.dot(radial_hessian.dot(mass_matrix))
    freq, normal_vectors = eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq**0.5/2/np.pi
    return (freq[::-1], normal_vectors[::-1])


def simulation_run(
    ion_types, ions_order, ions_initial_splitting, trap
):
    
    ion_number = ions_order.shape[0]
    ions_positions_z = (
        np.linspace(-(ion_number-1) / 2, (ion_number-1) / 2, ion_number)
        * ions_initial_splitting
    )

    ions_positions = (
        np.concatenate(
            [
                (np.random.rand(ion_number) - 0.5) * 1e-6,
                (np.random.rand(ion_number) - 0.5) * 1e-6,
                ions_positions_z,
            ]
        )
        .reshape((3, ion_number))
        .T
    )

    simulation_instance = pl.Simulation("normal_modes")

    # Adding ions and trap into simulation
    for i in range(len(ion_types)):
        ion_number_type = np.count_nonzero(ions_order == i)
        ions = pl.placeions(
            ion_types[i],
            ions_positions[np.where(ions_order == i), :]
            .reshape(ion_number_type, 3)
            .tolist(),
        )
        simulation_instance.append(ions)

        pseudotrap = pl.linearpaultrap(trap, ions, all=False)
        simulation_instance.append(pseudotrap)

    # ions = pl.placeions(ion_types, ions_positions)
    # s.append(ions)
    # s.append(pl.linearpaultrap(trap, ions, all=False))

    # Creation of Langevin bath
    langevinbath = pl.langevinbath(0, 3e-6)
    simulation_instance.append(langevinbath)

    # Description of output file
    dump = pl.dump(
        "positions.txt", variables=["id", "mass", "q", "x", "y", "z"], steps=100
    )
    simulation_instance.append(dump)

    # Definition of the evolution time
    simulation_instance.append(pl.evolve(5e4))

    # Start of the simulation

    simulation_instance.execute()

    _, data = pl.readdump("positions.txt")

    # Loading of the simulation results
    final_x = data[-1, :, 3]
    final_y = data[-1, :, 4]
    final_z = data[-1, :, 5]
    final_mass = np.round(data[-1, :, 1] / amu).astype("int32")
    final_charge = np.round(data[-1, :, 2] / ech).astype("int32")

    # Cristal order check
    sorted_inds = final_z.argsort()
    final_x = final_x[sorted_inds]
    final_y = final_y[sorted_inds]
    final_z = final_z[sorted_inds]
    final_mass = final_mass[sorted_inds]
    final_charge = final_charge[sorted_inds]
    initial_mass = np.array([ion_types[x]["mass"] for x in ions_order])
    initial_charge = np.array([ion_types[x]["charge"] for x in ions_order])
    mass_error = np.sum((initial_mass - final_mass) ** 2)
    charge_error = np.sum((initial_charge - final_charge) ** 2)
    if (mass_error + charge_error) == 0:
        print("Ion crysal ordering is correct!")
    else:
        raise Exception(
            "Ion crysal ordering has been changed! Try to set different trap potentials or ion positions!"
        )

    # Check if the crystal structure is linear
    max_radial_variation = (
        (np.max(final_x) - np.min(final_x)) ** 2
        + (np.max(final_y) - np.min(final_y)) ** 2
    ) ** 0.5
    min_axial_distance = np.min(final_z[1:] - final_z[:-1])

    if min_axial_distance * 1e-3 > max_radial_variation:
        print("Ion crystal is linear.")
    else:
        raise Exception("Ion crystal is not linear. Mode structure will be incorrect!")

    timestep = simulation_instance.attrs["timestep"]

    return final_z

def simulation_run_equations(
    ion_types, ions_order, ions_initial_splitting, trap
):
    def equations(z):
        z = np.array(z)
        out = np.empty(ion_number)
        for i in range(ion_number):
            out[i] = A[i]*z[i] - 2*interaction_coeff*sum(ions_charge[i]*ions_charge[j]*(z[i]-z[j])/(np.abs(z[i]-z[j])**3)
                                       for j in range(ion_number) if j != i)
        return out
    
    def print_dump_file(file_name = "positions.txt"):
        ion_types = np.unique(ions_order)
        shift = 1
        data_array = []
        for ion_type in ion_types:
            z_coord = final_z[np.where(ions_order == ion_type)] 
            x_coord = np.zeros(z_coord.shape)
            y_coord = np.zeros(z_coord.shape)
            masses = ions_mass[np.where(ions_order == ion_type)]
            charges = ions_charge[np.where(ions_order == ion_type)]
            id_array = np.arange(z_coord.shape[0])+shift
            shift += z_coord.shape[0]
            arr = np.array([id_array, id_array, masses, charges, x_coord, y_coord, z_coord]).T
            data_array.extend(list(arr))
            
        with open(file_name, 'w') as pos_f:
            pos_f.write('ITEM: TIMESTEP\n')
            pos_f.write('0\n')
            pos_f.write('ITEM: NUMBER OF ATOMS\n')
            pos_f.write(str(ion_number)+'\n')
            pos_f.write('ITEM: BOX BOUNDS mm mm mm\n')
            pos_f.write('-1.0000000000000000e-03 1.0000000000000000e-03\n')
            pos_f.write('-1.0000000000000000e-03 1.0000000000000000e-03\n')
            pos_f.write('-1.0000000000000000e-03 1.0000000000000000e-03\n')
            pos_f.write('ITEM: ATOMS id id mass q x y z\n')
            for line in data_array:
                pos_f.write(str(int(line[0]))+' '+str(int(line[1]))+' '+str(line[2])+' '+str(line[3])+' '+str(line[4])+' '+str(line[5])+' '+str(line[6])+'\n')
    
    ion_number = ions_order.shape[0]
    axial_freqs = []
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(
            ion_types[ions_order[i]],
            trap
        )
        axial_freqs.append(w_z)
    axial_freqs = np.array(axial_freqs)
    ions_mass = np.array([ion_types[x]['mass']*amu for x in ions_order])
    ions_charge = np.array([ion_types[x]['charge']*ech for x in ions_order])
    interaction_coeff = 1/8/np.pi/eps0
    
    A = ions_mass*axial_freqs**2*4*np.pi**2
    ions_positions_z = (
        np.linspace(-(ion_number-1) / 2, (ion_number-1) / 2, ion_number)
        * ions_initial_splitting
    )
    
    final_z = fsolve(equations, ions_positions_z)
    
    print_dump_file()
    return final_z


def comprehensive_plot(
    ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs
):
    fig = plt.figure()
    grid = plt.GridSpec(3, 5, wspace=0.4, hspace=0.3, height_ratios=[0.3, 1, 1])
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    final_x = data[0]
    #final_y = data[1]
    final_z = data[2]

    tmp = np.max(ions_order) + 1
    color_column = np.linspace(0.1, 0.8, tmp).reshape(tmp, 1)
    color_map_unit = np.hstack((np.zeros((tmp, 1)), color_column, color_column))
    color_map = color_map_unit[ions_order]
    fig.add_subplot(grid[0, 0:])
    plt.scatter(final_z, final_x, c=color_map)
    plt.title("Ion's equilibrium positions")
    plt.xlabel("Ion's z coordinates")
    plt.ylabel("Ion's x coordinates")
    plt.ylim(
        [
            -max(1, 1.2 * np.max(np.abs(final_x))),
            max(1, 1.2 * np.max(np.abs(final_x))),
        ]
    )

    ax1 = fig.add_subplot(grid[1:, :2])
    plt.imshow(radial_modes, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Radial mode matrix")
    plt.xlabel("ion number")
    plt.ylabel("mode number")
    #plt.tight_layout()

    fig.add_subplot(grid[1:, 2])
    plt.plot([], [], color="red", label="radial", linewidth=0.5)
    plt.plot([], [], color="blue", label="axial", linewidth=0.5)

    for omega in radial_freqs:
        plt.plot([-1, 0], [omega, omega], color="red", linewidth=0.5)
    for omega in axial_freqs:
        plt.plot([-1, 0], [omega, omega], color="blue", linewidth=0.5)

    plt.ylabel("$\omega/\omega_{\mathrm{com}}^{\mathrm{ax}}$")
    plt.xticks([])
    plt.xlim(-1, 2)
    plt.ylim(bottom=0)
    plt.title("Mode frequency spectrum")
    plt.legend(loc="upper right")
    #plt.tight_layout()

    fig.add_subplot(grid[1:, 3:])
    plt.imshow(axial_modes, cmap="bwr", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Axial mode matrix")
    plt.xlabel("ion number")
    plt.ylabel("mode number")
    #plt.tight_layout()

    plt.savefig("Normal modes structure", dpi=300)
    plt.tight_layout()
    plt.show()

def radial_modes_plot(ions_order, data, radial_modes, axial_modes, radial_freqs, axial_freqs):    
    #fig = plt.figure()
    #grid = plt.GridSpec(1, 2, wspace=1, width_ratios=[1, 0.4])
    #fig.subplots_adjust(hspace=0.4, wspace=0.4)
    
    #ax1 = fig.add_subplot(grid[:, 0])
    plt.imshow(radial_modes, cmap="bwr")#, vmin=-1, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('normal mode vectors', rotation=270, labelpad=15)
    #plt.title("Radial mode matrix")
    plt.xlabel("Ion number")
    plt.ylabel("Radial mode number")
    plt.savefig('modes homogenious', dpi = 600)
    '''fig.add_subplot(grid[:, 1])'''
    '''plt.axes().set_aspect('equal')
    #plt.plot([], [], color="red", label="radial", linewidth=0.5)
    #plt.plot([], [], color="blue", label="axial", linewidth=0.5)
    cmap = cm.get_cmap('viridis')
    
    for omega in radial_freqs:
        plt.plot([-0.05, 0], [omega/1e6, omega/1e6], color = cmap(0.5),  linewidth=0.3)
    #for omega in axial_freqs:
    #    plt.plot([-0.5, 0], [omega/1e6, omega/1e6], color="blue", linewidth=0.5)

    plt.ylabel("$\omega/2\pi, MHz$")
    plt.xticks([])
    plt.xlim(-0.06, 0.01)
    #plt.ylim(bottom=0)
    plt.title("Mode frequency spectrum")
    #plt.legend(loc="right")
    plt.savefig('mode spectrum', dpi = 600)'''
    plt.show()

def plot_crystal(ions_order, final_z, final_x):
    plt.figure()
    plt.axes().set_aspect('equal')
    cmap = cm.get_cmap('viridis')
    #print(final_x, final_z)
    plt.scatter(final_z, final_x, s = 2**2, c = ions_order, cmap=cmap, vmin = 0, vmax = 1.4)
    plt.title('Ion\'s equilibrium positions')
    plt.xlabel('Ion\'s z coordinates, $\mu m$')
    plt.ylabel('Ion\'s x coordinates, $\mu m$')
    plt.ylim([-max(3e-4, 1.2*np.max(np.abs(final_x))), max(3e-4, 1.2*np.max(np.abs(final_x)))])
    plt.tight_layout()
    type1 = Line2D([0], [0], marker='o', color='w', label='${}^{40}Ca^{+}$', markerfacecolor=cmap(0), markersize=4)
    type2 = Line2D([0], [0], marker='o', color='w', label='${}^{44}Ca^{+}$', markerfacecolor=cmap(1/1.4), markersize=4)
    plt.legend(handles=[type1, type2])
    plt.savefig('ion string', dpi = 600)
    plt.show()
    
def read_dump(filename = "positions.txt"):
    _, data = pl.readdump(filename)
    
    ion_number = data.shape[1]
    final_x = data[-1, :, 3]
    final_y = data[-1, :, 4]
    final_z = data[-1, :, 5]
    final_mass = np.round(data[-1, :, 1] / amu).astype("int32")
    final_charge = np.round(data[-1, :, 2] / ech).astype("int32")

    # Cristal order check
    sorted_inds = final_z.argsort()
    final_x = final_x[sorted_inds]
    final_y = final_y[sorted_inds]
    final_z = final_z[sorted_inds]
    final_mass = final_mass[sorted_inds]
    final_charge = final_charge[sorted_inds]
    
    res = np.concatenate([final_x, final_y, final_z, final_mass, final_charge]).reshape(5, ion_number)
    return res

def get_modes(
    ion_types,
    ions_order,
    trap,
    final_z = None,
    pinned_ions = None,
    w_tweezer = None
):
    
    if final_z is None:
        ions_initial_splitting = 5e-6
        final_z = simulation_run(
            ion_types, ions_order, ions_initial_splitting, trap
        )
    
    if w_tweezer is None:
        w_tweezer = 0
    
    if pinned_ions is None:
        pinned_ions = np.zeros_like(ions_order)
    
    axial_freqs, axial_modes = axial_normal_modes(
        final_z,
        ion_types,
        ions_order,
        trap,
        pinned_ions,
        w_tweezer
    )
    
    radial_freqs, radial_modes = radial_normal_modes(
        final_z,
        ion_types,
        ions_order,
        trap,
        pinned_ions,
        w_tweezer
    )
    
    return (radial_modes, radial_freqs, axial_modes, axial_freqs)

def get_freq_variation_coefficient(
    ion_types,
    ions_order,
    trap,
    ion_ref_type = 0
):
    ion_number = ions_order.shape[0]
    radial_freqs = np.zeros(ion_number)
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(ion_types[ions_order[i]], trap)
        radial_freqs[i] = w_r
    ions_mass = np.array([ion_types[x]['mass']*amu for x in ions_order])
    A = 2*radial_freqs/ions_mass**2*(ion_types[ion_ref_type]['mass']*amu)**2
    return A