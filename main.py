import pandas as pd
from EasyAtomisticTD_functions.TD_plotting import analyze_reaction_with_plotting

# Load the CSV file
file_path = 'AITD_data.csv'
df = pd.read_csv(file_path)

# Extracting information from the DataFrame
compounds = df['Compound'].tolist() #info about compounds
DFT_energies = df['TOTEN'].tolist() #energies
Center_atoms_per_unit = df['Central_atoms_perUC'].tolist() #Central atoms per unit, essential for consistent analysis
full_formulas = df['Full_formula'].tolist() #Formulas per unit cell



mu_ranges = (-2.2, 0.0)


mu_values_2 = {'CH4': -2, 'N2': -2.46, 'H2O': -0.85,'CO': -3.5}  # Fixed chemical potentials for gas-phase reactants

# Example usage of the function
reaction_objects_CH4_2, pressure_ranges_CH4_2 = analyze_reaction_with_plotting(
    solid_reference='UO2(OH)2',  # Reference solid compound
    gas_phase_reactants=['CH4', 'N2', 'H2O', 'CO'],  # Gas-phase reactants
    data_filter='General_model_2',  # Data filter column
    dataframe=df,  # Input dataframe
    temperatures=[600, 800, 1000],  # Temperatures for pressure range
    mu_values=mu_values_2,  # Fixed \mu values for reactants
    mu_ranges=mu_ranges,
    file_name='Reduction_UO2OH2_2'
)