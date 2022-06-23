import numpy as np
from numpy.linalg import eigh
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.lines import Line2D

__version__ = '0.9.3'

# Font settings for plots
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
    w_z, 
    w_r, 
    ref_ion, 
    RF_frequency, 
    Z0, 
    R0_eff, 
    kappa
):
    """
    Calculation of DC and RF voltages required for the creation of confinement
    for an ion of a given type for a given axial and radial secular frequencies 
    and fixed trap geometry
    
    Parameters
    ----------
    w_z : float
        Desired axial secular frequency of a trap in Hz.
    w_r : float
        Desired radial secular frequency of a trap in Hz.
    ref_ion : dict
        Dictionary describing the ion for which voltages are calculated with 
        the following shape {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu
            charge: int
                Ion's charge in elementary charge units
    RF_frequency : float
        RF frequency in Hz.
    Z0 : float 
        The distance between endcap electrodes in m.
    R0_eff : float
        The effective distance between blade electrodes in m. It is equal to 
        the distance between hyperbolic-shaped electrodes, which create the 
        same RF confinement as the trap under consideration. 
    kappa : float
        A dimensionless unit that describes a finite efficiency of the endcap 
        electrodes.

    Returns
    -------
    DC_voltage : float
        Required DC voltage on the endcap electrodes.
    RF_voltage : float
        Required RF voltage on the blade-shaped electrodes.
        
    """
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
    return (DC_voltage, RF_voltage)

def voltage_to_frequency(
    ref_ion, 
    trap
):
    """
    Calculation of axial and radial secular frequencies which are created
    for ion of a given type with a trap of fixed geometry for given DC and RF
    voltages
    
    Parameters
    ----------
    ref_ion : dict
        Dictionary describing the ion for which voltages are calculated with 
        the following shape {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu
            charge: int
                Ion's charge in elementary charge units
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    Returns
    -------
    w_z : float
        Axial secular frequency of a trap in Hz.
    w_r : float
        Radial secular frequency of a trap in Hz.
        
    """
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
    return (w_z, w_r)

def axial_hessian_matrix(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    """
    Calculates axial hessian matrix for a linear multispecies ion string with 
    tweezers applied to some ions.

    Parameters
    ----------
    ion_positions : np.array with dtype of float
        An array with equilibrium positions (in m) of ions in the sting along 
        the z direction. 
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu
            charge: int
                Ion's charge in elementary charge units
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    pinned_ions : np.array with dtype of float
        An array which describes the strength with which ion is pinned by a 
        tweezer. For an element equals to 0, ion is not pinned. For an element
        equals to 1, ion is pinned at full strength.
    w_tweezer : float
        A maximum trapping frequency induced by tweezers.

    Returns
    -------
    A_matrix : np.array with square shape dtype float
        Hessian matrix for axial oscillations of an ion crystal.
    M_matrix : np.array with square shape dtype float
        Diagonal matrix with square roots of ion masses (in kg).

    """
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
    A_matrix = A_matrix*2
    M_matrix = np.diag(ions_mass**(-0.5))
    return (A_matrix, M_matrix)

def radial_hessian_matrix(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    """
    Calculates radial hessian matrix for a linear multispecies ion string with 
    tweezers applied to some ions.

    Parameters
    ----------
    ion_positions : np.array with dtype of float
        An array with equilibrium positions (in m) of ions in the sting along 
        the z direction. 
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu
            charge: int
                Ion's charge in elementary charge units
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    pinned_ions : np.array with dtype of float
        An array which describes the strength with which ion is pinned by a 
        tweezer. For an element equals to 0, ion is not pinned. For an element
        equals to 1, ion is pinned at full strength.
    w_tweezer : float
        A maximum trapping frequency induced by tweezers.

    Returns
    -------
    B_matrix : np.array with square shape dtype float
        Hessian matrix for radial oscillations of an ion crystal.
    M_matrix : np.array with square shape dtype float
        Diagonal matrix with square roots of ion masses (in kg).

    """
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
    B_matrix = B_matrix*2
    M_matrix = np.diag(ions_mass**(-0.5))
    return (B_matrix, M_matrix)

def axial_normal_modes(
    ion_positions,
    ion_types,
    ions_order,
    trap,
    pinned_ions,
    w_tweezer
):
    """
    Calculates axial normal mode frequencies and matrix with nomal mode vectors
    for a linear multispecies ion string with tweezers applied to some ions.

    Parameters
    ----------
    ion_positions : np.array with dtype of float
        An array with equilibrium positions (in m) of ions in the sting along 
        the z direction. 
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu
            charge: int
                Ion's charge in elementary charge units
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    pinned_ions : np.array with dtype of float
        An array which describes the strength with which ion is pinned by a 
        tweezer. For an element equals to 0, ion is not pinned. For an element
        equals to 1, ion is pinned at full strength.
    w_tweezer : float
        A maximum trapping frequency induced by tweezers.

    Returns
    -------
    freq : np.array with dtype float
        An array with axial normal mode frequencies in Hz.
    normal_vectors : np.array with dtype float
        Square array with components of axial normal mode vectors. First index 
        is the mode number, second index is ion number.

    """
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
    """
    Calculates radial normal mode frequencies and matrix with nomal mode 
    vectors for a linear multispecies ion string with tweezers applied to some 
    ions.

    Parameters
    ----------
    ion_positions : np.array with dtype of float
        An array with equilibrium positions (in m) of ions in the sting along 
        the z direction. 
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu.
            charge: int
                Ion's charge in elementary charge units.
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    pinned_ions : np.array with dtype of float
        An array which describes the strength with which ion is pinned by a 
        tweezer. For an element equals to 0, ion is not pinned. For an element
        equals to 1, ion is pinned at full strength.
    w_tweezer : float
        A maximum trapping frequency induced by tweezers.

    Returns
    -------
    freq : np.array with dtype float
        An array with radial normal mode frequencies in Hz.
    normal_vectors : np.array with dtype float
        Square array with components of radial normal mode vectors. First index
        is the mode number, second index is ion number.
        
    """
    radial_hessian, mass_matrix = radial_hessian_matrix(ion_positions, ion_types, ions_order, trap, pinned_ions, w_tweezer)
    D = mass_matrix.dot(radial_hessian.dot(mass_matrix))
    freq, normal_vectors = eigh(D)
    normal_vectors = -normal_vectors.T
    freq = freq**0.5/2/np.pi
    return (freq[::-1], normal_vectors[::-1])

def simulation_run_equations(
    ion_types, 
    ions_order, 
    ions_initial_splitting, 
    trap
):
    """
    Calculates equilibrium positions of ion's inside a multispecies crystal in 
    a given configuration by minimization of a potential energy along the axial
    direction. It is a preferable method for calculation of an ion positions in a 
    crystal of a reasonable length (up to 200 ion).

    Parameters
    ----------
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu.
            charge: int
                Ion's charge in elementary charge units.
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    ions_initial_splitting : float
        Initialy ions are created in with splitting equals to 
        ions_initial_splitting along the z direction in the middle of the trap.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.

    Returns
    -------
    final_z : np.array of dtype float
        Equilibrium positions of ions inside the trap.

    """
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
    ions_order, 
    data, 
    radial_modes, 
    axial_modes, 
    radial_freqs, 
    axial_freqs
):
    """
    Makes a plot of ion crystal structure, axial and radial mode matrices and
    normal frequency spectrum.

    Parameters
    ----------
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    data : TYPE
        DESCRIPTION.
    radial_modes : np.array with dtype float
        Square array with components of radial normal mode vectors. First index 
        is the mode number, second index is ion number.
    radial_freqs : np.array with dtype float
        An array with radial normal mode frequencies in Hz.
    axial_modes : np.array with dtype float
        Square array with components of axial normal mode vectors. First index 
        is the mode number, second index is ion number.
    axial_freqs : np.array with dtype float
        An array with axial normal mode frequencies in Hz.

    Returns
    -------
    None.

    """
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

    #ax1 = fig.add_subplot(grid[1:, :2])
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

def radial_modes_plot(
    ions_order,
    radial_modes, 
    axial_modes, 
    radial_freqs, 
    axial_freqs,
    save_path = None,
):
    """
    Makes a plot of radial mode matrices.

    Parameters
    ----------
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    radial_modes : np.array with dtype float
        Square array with components of radial normal mode vectors. First index 
        is the mode number, second index is ion number.
    radial_freqs : np.array with dtype float
        An array with radial normal mode frequencies in Hz.
    axial_modes : np.array with dtype float
        Square array with components of axial normal mode vectors. First index 
        is the mode number, second index is ion number.
    axial_freqs : np.array with dtype float
        An array with axial normal mode frequencies in Hz.
    save_path : string
        If given, the plot will be saved to the file which is located on this 
        path. The default is None.
    
    Returns
    -------
    None.

    """
    plt.figure()
    plt.imshow(radial_modes, cmap="bwr")#, vmin=-1, vmax=1)
    cbar = plt.colorbar()
    cbar.set_label('normal mode vectors', rotation=270, labelpad=15)
    plt.title("Radial mode matrix")
    plt.xlabel("Ion number")
    plt.ylabel("Radial mode number")
    if save_path is not None and isinstance(save_path, str):
        plt.tight_layout()
        plt.savefig(save_path, dpi = 600)
    plt.show()

def plot_crystal(
    ions_order, 
    final_z, 
    final_x,
    save_path = None,
    hide_axis = False,
):
    """
    Function makes a plot of an ion crystal configuration.

    Parameters
    ----------
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    final_z : np.array of dtype float
        Equilibrium positions of ions inside the trap along trap axis.
    final_z : np.array of dtype float, optional
        Equilibrium positions of ions inside the trap along one of the radial
        directions.
    save_path : string
        If given, the plot will be saved to the file which is located on this 
        path. The default is None.
    
    Returns
    -------
    None.

    """
    plt.figure()
    plt.axes().set_aspect('equal')
    cmap = cm.get_cmap('viridis')
    plt.scatter(final_z, final_x, s = 2**2, c = ions_order, cmap=cmap, vmin = 0, vmax = 1.4)
    plt.title('Ion\'s equilibrium positions')
    plt.xlabel('Ion\'s z coordinates, $\mu m$')
    plt.ylabel('Ion\'s x coordinates, $\mu m$')
    plt.ylim([-max(3e-4, 1.2*np.max(np.abs(final_x))), max(3e-4, 1.2*np.max(np.abs(final_x)))])
    type1 = Line2D([0], [0], marker='o', color='w', label='${}^{40}Ca^{+}$', markerfacecolor=cmap(0), markersize=4)
    type2 = Line2D([0], [0], marker='o', color='w', label='${}^{44}Ca^{+}$', markerfacecolor=cmap(1/1.4), markersize=4)
    plt.legend(handles=[type1, type2])
    if save_path is not None and isinstance(save_path, str):
        plt.tight_layout()
        plt.savefig(save_path, dpi = 600)
    plt.show()

def get_modes(
    ion_types,
    ions_order,
    trap,
    final_z = None,
    pinned_ions = None,
    w_tweezer = None
):
    """
    Calculates normal mode frequencies and matrix with nomal mode vectors for 
    axial and radial modes of a linear multispecies ion string with tweezers 
    applied to some ions.

    Parameters
    ----------
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu.
            charge: int
                Ion's charge in elementary charge units.
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    final_z : np.array of dtype float, optional
        Equilibrium positions of ions inside the trap.
    pinned_ions : TYPE, optional
        DESCRIPTION. The default is None.
    w_tweezer : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    radial_modes : np.array with dtype float
        Square array with components of radial normal mode vectors. First index 
        is the mode number, second index is ion number.
    radial_freqs : np.array with dtype float
        An array with radial normal mode frequencies in Hz.
    axial_modes : np.array with dtype float
        Square array with components of axial normal mode vectors. First index 
        is the mode number, second index is ion number.
    axial_freqs : np.array with dtype float
        An array with axial normal mode frequencies in Hz.

    """
    
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
    """
    Calculates derivatives of a radial hessian matrix over frequency for each 
    mode for a given ion type. It is used for estimation of MS gate robustness
    to radial potential fluctuations.

    Parameters
    ----------
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu.
            charge: int
                Ion's charge in elementary charge units.
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.
    ion_ref_type : int, optional
        The number of ion's type. The default is 0.

    Returns
    -------
    A : np.array of dtype float
        Array consists of derivatives of a radial hessian matrix over frequency
        for each mode for a given ion type.

    """
    ion_number = ions_order.shape[0]
    radial_freqs = np.zeros(ion_number)
    for i in range(ion_number):
        w_z, w_r = voltage_to_frequency(ion_types[ions_order[i]], trap)
        radial_freqs[i] = w_r
    ions_mass = np.array([ion_types[x]['mass']*amu for x in ions_order])
    A = 2*radial_freqs/ions_mass**2*(ion_types[ion_ref_type]['mass']*amu)**2
    return A

##---------------------------------------------------------------------------##
# (Py)Lion based functions

def read_dump(
    filename = "positions.txt"
):
    """
    Read file that was created by (Py)Lion library. Extracts the information
    about the final coordinates of ions and their masses and charges.

    Parameters
    ----------
    filename : string, optional
        Path to the file that need to be read. The default is "positions.txt".

    Returns
    -------
    res : np.array of dtype float
        The array with data. Has a shape of (5, N), where N is ion number.

    """
    import pylion as pl
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

def simulation_run(
    ion_types, 
    ions_order, 
    ions_initial_splitting, 
    trap
):
    """
    Calculates equilibrium positions of ion's inside a multispecies crystal in 
    a given configuration using (Py)Lion. To use this function you will need to
    install (Py)Lion package with the instructions from 
    https://bitbucket.org/dtrypogeorgos/pylion/src/master/
    This function can't be used in an IPython notebook, because (Py)Lion works
    unreliable in such environment. For calculation of an ion positions in a 
    crystal of a reasonable length (up to 200 ion) is better to use 
    simulation_run_equations method of this package.

    Parameters
    ----------
    ion_types : list
        A list which describes ion types which are present in the ion string.
        Each element of a list is a dict of the following shape
        {'mass': mass, 'charge': charge}.
            mass: int
                Ion's mass in amu.
            charge: int
                Ion's charge in elementary charge units.
    ions_order : np.array with dtype of int
        An ordered array that desribes the configuration of an ion string. 
        Indexes of the array enumerate ions in the sting. Ion with index i+1 
        will have bigger z coordinate in comparison to ion with index i. Each
        array element is an integer number that describes an ion type.
    ions_initial_splitting : float
        Initialy ions are created in with splitting equals to 
        ions_initial_splitting along the z direction in the middle of the trap.
    trap : dict
        Dictionary which fully describes an ion trap with the following shape
        (the same as in the (Py)Lion package)
        {
            'radius': R0_eff,
            'length': Z0,
            'kappa': kappa,
            'frequency': RF_frequency,
            'voltage': RF_voltage,
            'endcapvoltage': DC_voltage,
            'pseudo': pseudo,
            'anisotropy': anisotropy,
        }
            R0_eff : float
                The effective distance between blade electrodes in m. It is equal to 
                the distance between hyperbolic-shaped electrodes, which create the 
                same RF confinement as the trap under consideration.
            Z0 : float 
                The distance between endcap electrodes in m.
            kappa : float
                A dimensionless unit that describes a finite efficiency of the endcap 
                electrodes.
            RF_frequency : float
                RF frequency in Hz.
            RF_voltage : float
                RF voltage on the blade-shaped electrodes in volts.
            DC_voltage : float
                DC voltage on the endcap electrodes in volts.
            pseudo : boolean
                If True, pseudopotential approximation is used. Else, the full
                dynamical simulation is performed. In this function do nothing.
            anisotropy : float
                The ratio between radial secular frequencies in two directions.

    Raises
    ------
    Exception
        "Ion crysal ordering has been changed! Try to set different trap 
        potentials or ion positions!".
            Raised if the ions order after simulations differs from initial 
            ion's order. Usualy indicates incorrect initial ion splitting or
            unstable crystal configuration.
        "Ion crystal is not linear. Mode structure will be incorrect!"
            Ratio of axial secualar frequency to radial secualar frequency is
            too high for an ion crystal with a given ion number to be linear.
            Normal mode frequencies and normal mode vectors can't be calculated
            properly in this case. Try to adjust secular frequencies.
    Returns
    -------
    final_z : np.array of dtype float
        Equilibrium positions of ions inside the trap.

    """
    import pylion as pl
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