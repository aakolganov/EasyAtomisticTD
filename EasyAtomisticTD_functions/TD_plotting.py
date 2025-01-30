import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
from GPhase_DFT_parse import load_gas_phase_energies
from TD_data import p_range

def analyze_reaction_with_plotting(
    solid_reference,
    gas_phase_reactants,
    dataframe,
    data_filter=None,
    temperatures=None,
    mu_values=None,
    mu_ranges=(-5, 5),
    gas_phase_energies=None,
    gas_phase_energies_csv=None,
    element_of_interest=None,
    file_name=None
):
    """
    Analyze formation reactions from a single solid reference + multiple gas-phase species,
    calculate reaction energies (per chosen element_of_interest), compute Gibbs free energy,
    and generate plots of ΔG vs. chemical potential.

    Parameters
    ----------
    solid_reference : str
        The reference solid compound (e.g., 'Re2O7'), must appear in `dataframe['Comp']`.
    gas_phase_reactants : list of str
        Names of gas-phase compounds involved (e.g. ['H2', 'N2']).
        These should match the keys in `gas_phase_energies` or appear in the CSV used.
    dataframe : pd.DataFrame
        Must have columns: 'Comp' (compound name) and 'Energy' (DFT energy).
    data_filter : str, optional
        Column name in `dataframe` used to filter rows (e.g., 'Include'). Only rows where
        dataframe[data_filter] == 'yes' are analyzed. If None, no filter is applied.
    temperatures : list of float, optional
        Temperatures (in Celsius) at which to compute log(p) ranges for plotting.
    mu_values : dict, optional
        Dictionary of baseline chemical potentials for the gas-phase reactants, e.g.:
        {'N2': -6.0, 'H2': -6.0}.
    mu_ranges : tuple of float, optional
        Range of μ values (min_mu, max_mu) for plotting. Default is (-5, 5).
    gas_phase_energies : dict, optional
        A dictionary {compound: energy_in_eV, ...} for gas-phase species. If None, you must
        provide `gas_phase_energies_csv`.
    gas_phase_energies_csv : str, optional
        Path to a CSV containing columns "Compound" and "Energy" for gas-phase species.
        If provided, we load these energies into a dict.
    element_of_interest : str, optional
        Element for normalizing reaction energies, default 'U'.
    file_name : str, optional
        If provided, the final plot is saved to this file. Otherwise it is shown interactively.

    Returns
    -------
    reaction_objects : dict of {str: GeneralReaction}
        Reaction objects keyed by compound name (i.e. each 'Comp' in the filtered dataframe).
    pressure_ranges : dict
        Dictionary of pressure ranges for each gas and temperature, used for the top x-axes in plots.
    """

    # -------------------------------------------------------------------------
    # Load gas-phase energies if needed
    # -------------------------------------------------------------------------
    if gas_phase_energies is None and gas_phase_energies_csv is not None:
        gas_phase_energies = load_gas_phase_energies(gas_phase_energies_csv)

    if gas_phase_energies is None:
        raise ValueError(
            "You must provide either `gas_phase_energies_csv`."
        )

    # -------------------------------------------------------------------------
    # Filter the main dataframe (if requested)
    # -------------------------------------------------------------------------
    if data_filter is not None:
        filtered_df = dataframe[dataframe[data_filter] == 'yes']
    else:
        filtered_df = dataframe

    # -------------------------------------------------------------------------
    # Retrieve reference solid's energy
    # -------------------------------------------------------------------------
    if solid_reference not in filtered_df['Comp'].values:
        # if the reference solid is not in filtered_df,
        # maybe it's in the broader dataframe:
        if solid_reference not in dataframe['Comp'].values:
            raise ValueError(f"Solid reference {solid_reference} not found in dataframe['Comp']!")
        else:
            ref_energy = dataframe.loc[dataframe['Comp'] == solid_reference, 'Energy'].values[0]
    else:
        ref_energy = filtered_df.loc[filtered_df['Comp'] == solid_reference, 'Energy'].values[0]

    # -------------------------------------------------------------------------
    # Define helper functions
    # -------------------------------------------------------------------------
    def get_full_formula(compound):
        """Return the 'Full_formula' from the dataframe, or just compound if not found."""
        if compound in dataframe['Comp'].values:
            rows = dataframe.loc[dataframe['Comp'] == compound]
            if 'Full_formula' in rows.columns:
                return rows['Full_formula'].values[0]
            else:
                return compound
        return compound

    def parse_compound(formula_str):
        """Parse a formula string like 'SO2' into {'S':1, 'O':2}."""
        pattern = r'([A-Z][a-z]?)(\d*)'
        matches = re.findall(pattern, formula_str)
        elements = {}
        for elem, num_str in matches:
            num = int(num_str) if num_str else 1
            elements[elem] = elements.get(elem, 0) + num
        return elements

    def get_elements(list_of_compound_dicts):
        """Collect all elements present across multiple compound dicts."""
        s = set()
        for dct in list_of_compound_dicts:
            s.update(dct.keys())
        return s



    def calculate_pressure_ranges(temp_list, gas_list):
        """
        For each temperature (°C) and gas in `gas_list`, compute the
        (lower_bound, upper_bound) in log(p) for the mu_ranges.
        """
        pranges = {}
        for T in temp_list:
            T_k = T + 273.15 #convert to K
            pranges[T] = {}
            for g in gas_list:
                lower = p_range(T_k, g, mu_ranges[0]) / np.log(10)
                upper = p_range(T_k, g, mu_ranges[1]) / np.log(10)
                pranges[T][g] = (lower, upper)
        return pranges

    # -------------------------------------------------------------------------
    # Class for Reaction Calculation
    # -------------------------------------------------------------------------
    class GeneralReaction:
        def __init__(self, compound, energy, reference_energy):
            """
            Represents the reaction: solid_reference + gas_phase_reactants -> compound
            with stoichiometry balanced. We'll normalize the product's coefficient to 1.
            """
            self.compound = compound
            self.full_formula = get_full_formula(compound)
            self.reactants = [solid_reference] + gas_phase_reactants
            self.product_energy = energy
            self.reference_energy = reference_energy

            # Count how many atoms of `element_of_interest` are in the product formula
            product_dict = parse_compound(self.full_formula)
            self.elem_count_in_product = product_dict.get(element_of_interest, 0)

            self.coefficients = None  # will hold stoichiometric coefficients once balanced

        def balance_reaction(self):
            """Balance the reaction by setting coefficient of solid_reference = 1, solving for others."""
            if self.coefficients is not None:
                return  # already done

            # parse all reactant formulas
            reactant_dicts = [parse_compound(get_full_formula(r)) for r in self.reactants]
            product_dicts = [parse_compound(self.full_formula)]
            all_dicts = reactant_dicts + product_dicts

            elems = get_elements(all_dicts)
            num_reactants = len(reactant_dicts)
            coeffs = symbols(' '.join([f'x{i}' for i in range(num_reactants + 1)]))  # +1 for product

            eqs = []
            # Each element must balance
            for e in elems:
                expr = 0
                for i, rdct in enumerate(reactant_dicts):
                    expr += rdct.get(e, 0) * coeffs[i] #reactants with positive coefficients
                for i, pdct in enumerate(product_dicts):
                    expr -= pdct.get(e, 0) * coeffs[num_reactants + i]
                eqs.append(Eq(expr, 0)) #products with negative coefficient

            # Force coefficient of the first reactant (solid_reference) to 1
            eqs.append(Eq(coeffs[0], 1))

            solution = solve(eqs, coeffs, dict=True)
            if not solution:
                raise ValueError(f"Could not balance reaction for {self.compound}.")
            sol = solution[0]

            # The last coefficient is for the product
            product_coeff = sol[coeffs[-1]]
            if abs(product_coeff) < 1e-12:
                raise ValueError(f"Product coefficient is zero for {self.compound}.")

            # Normalize so product coefficient = 1
            self.coefficients = [float(sol[c]) / float(product_coeff) for c in coeffs]

        def get_coefficient(self, name):
            """Return stoichiometric coefficient for a given compound in the reaction."""
            if self.coefficients is None:
                self.balance_reaction()
            full_list = self.reactants + [self.compound]
            if name not in full_list:
                return 0.0
            return self.coefficients[full_list.index(name)]

        def reaction_energy_per_element(self):
            """
            Reaction energy normalized per `element_of_interest` atom in the *product*.
            ΔE = Σ(product_coeff * product_energy) - Σ(reactant_coeff * reactant_energy).
            """
            if self.coefficients is None:
                self.balance_reaction()

            # sum up energies: note that we have 1 product, but multiple reactants
            total_E = 0.0
            for i, coeff in enumerate(self.coefficients):
                if i < len(self.reactants):
                    # reactant
                    rname = self.reactants[i]
                    if rname == solid_reference:
                        total_E -= coeff * self.reference_energy
                    else:
                        # assume it's a gas
                        gas_e = gas_phase_energies.get(rname, 0.0)
                        total_E -= coeff * gas_e
                else:
                    # product
                    total_E += coeff * self.product_energy

            if self.elem_count_in_product == 0:
                return None  # can't normalize if the product has zero of that element

            return total_E / self.elem_count_in_product

        def gibbs_free_energy(self, local_mu_values):
            """
            ΔG per element_of_interest = ΔE_per_elem - Σ(ν_gas / #elem_in_prod)*mu_gas
            """
            dE = self.reaction_energy_per_element()
            if dE is None:
                return None

            dG = dE
            for g in gas_phase_reactants:
                nu_g = self.get_coefficient(g)
                # because we normalized product to 1, the stoich factor is exactly 'nu_g'
                # but we must also account for how many element_of_interest are in the product:
                if self.elem_count_in_product != 0:
                    dG -= (local_mu_values.get(g, 0.0) * nu_g / self.elem_count_in_product)
            return dG

        def display_reaction(self):
            """Return a string describing the balanced reaction and its energy."""
            if self.coefficients is None:
                self.balance_reaction()
            nr = len(self.reactants)
            react_strs = []
            for coeff, rname in zip(self.coefficients[:nr], self.reactants):
                cval = float(coeff)
                if abs(cval) > 1e-8:
                    c_str = f"{cval:.2f}" if abs(cval - 1.0) > 1e-8 else ""
                    react_strs.append(f"{c_str}{get_full_formula(rname)}")
            prod_str = [get_full_formula(self.compound)]  # product is normalized to 1

            eq_str = " + ".join(react_strs) + " -> " + " + ".join(prod_str)
            e_val = self.reaction_energy_per_element()
            if e_val is None:
                e_info = f"No {element_of_interest} in product; cannot normalize."
            else:
                e_info = f"Reaction energy per {element_of_interest}: {e_val:.4f} eV"
            return f"{self.compound}:\n{eq_str}\n{e_info}\n"

    # -------------------------------------------------------------------------
    # Create Reaction objects
    # -------------------------------------------------------------------------
    reaction_objects = {}
    for _, row in filtered_df.iterrows():
        comp = row['Comp']
        # skip if it matches the solid_reference itself? up to you
        # if comp == solid_reference:
        #    continue
        c_energy = row['Energy']
        reaction_objects[comp] = GeneralReaction(comp, c_energy, ref_energy)

    # -------------------------------------------------------------------------
    # Calculate pressure ranges
    # -------------------------------------------------------------------------
    pressure_ranges = calculate_pressure_ranges(temperatures, gas_phase_reactants)

    # -------------------------------------------------------------------------
    # Plot ΔG vs μ for each gas
    # -------------------------------------------------------------------------
    mu_vals = np.linspace(mu_ranges[0], mu_ranges[1], 100)
    fig, axs = plt.subplots(1, len(gas_phase_reactants), figsize=(7 * len(gas_phase_reactants), 6))

    if len(gas_phase_reactants) == 1:
        axs = [axs]  # ensure we can iterate if there's only one subplot

    # color and line style for clarity
    colors = plt.cm.gnuplot2(np.linspace(0, 0.7, len(reaction_objects)))
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 10))]

    line_list = []
    label_list = []

    def create_multiple_parasite_axes(main_ax, pranges, gas_label):
        """
        Add multiple top x-axes to show log(p) for each temperature.
        """
        for i, T in enumerate(temperatures):
            top_ax = main_ax.twiny()
            top_ax.set_xlim(main_ax.get_xlim())
            # We'll place 5 ticks across the range
            xtick_positions = np.linspace(mu_ranges[0], mu_ranges[1], 5)
            top_ax.set_xticks(xtick_positions)

            lower_logp, upper_logp = pranges[T][gas_label]
            logp_values = np.linspace(lower_logp, upper_logp, 5)
            top_ax.set_xticklabels([f"{lp:.1f}" for lp in logp_values])

            top_ax.spines['top'].set_position(('outward', 20 + 30 * i))
            xlabel = top_ax.set_xlabel(f"{T}°C", fontsize=12, labelpad=-15)
            xlabel.set_x(-0.06)
            top_ax.tick_params(axis='x', which='both', labelsize=10)
            top_ax.tick_params(axis='y', which='both', labelsize=10)

    for ax, gas in zip(axs, gas_phase_reactants):
        for i, (cmp_name, rxn_obj) in enumerate(reaction_objects.items()):
            color = colors[i]
            style = line_styles[i % len(line_styles)]
            dG_values = [
                rxn_obj.gibbs_free_energy({**mu_values, gas: mu}) for mu in mu_vals
            ]
            (line,) = ax.plot(mu_vals, dG_values, linestyle=style, color=color, label=cmp_name)
            # collect legend only once
            if gas == gas_phase_reactants[0]:
                line_list.append(line)
                label_list.append(cmp_name)

        create_multiple_parasite_axes(ax, pressure_ranges, gas)
        ax.set_title(f"log(p) for {gas}", fontsize=14)
        ax.set_xlabel(f"$\\mu_{{{gas}}}$ (eV)", fontsize=14)
        ax.set_ylabel(f"$\\Delta G$ per {element_of_interest} (eV)", fontsize=14)
        ax.tick_params(axis='x', labelsize=12)
        ax.tick_params(axis='y', labelsize=12)

    plt.tight_layout()

    # Save or show
    if file_name:
        plt.savefig(file_name, dpi=300)
        print(f"Plot saved to {file_name}")
    else:
        plt.show()

    # Optional: separate figure for the legend
    fig_legend = plt.figure(figsize=(6, 4))
    ax_legend = fig_legend.add_subplot(111)
    ax_legend.axis('off')
    legend = ax_legend.legend(line_list, label_list, loc='center', frameon=False)
    fig_legend.savefig('legend.png')
    print("Legend saved as legend.png")

    return reaction_objects, pressure_ranges