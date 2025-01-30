import pandas as pd

def load_gas_phase_energies(csv_file_path):
    """
    Reads a CSV file of gas-phase DFT energies and returns a dict {compound: energy_in_eV, ...}.
    """
    df = pd.read_csv(csv_file_path)
    # Check that the format is correct
    if not {'Compound', 'Energy'}.issubset(df.columns):
        raise ValueError("CSV must contain 'Compound' and 'Energy' columns.")
    # Build dictionary from the dataframe
    gas_energies = dict(zip(df['Compound'], df['Energy']))
    return gas_energies
