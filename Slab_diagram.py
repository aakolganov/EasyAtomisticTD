from EasyAtomisticTD_functions.TwoD_slab_plots import plot_diagram_for_two_gases, plot_diagram_for_specified_two, plot_diagrams_for_three_gases

def plot_stability_diagrams(
    reaction_objects,
    gas_phase_reactants,
    mu_ranges,
    fixed_mu_values=None,
    pressure_ranges=None,
    output_file_prefix="stability"
):
    """
    Main function that dispatches to different plotting routines depending
    on the number of gas-phase reagents.

    Parameters
    ----------
    reaction_objects : dict
        Dictionary of reaction objects keyed by compound names.
    gas_phase_reactants : list of str
        List of the gas-phase reactants (e.g., ['H2', 'O2']).
    mu_ranges : dict
        Dictionary mapping each gas to an array or tuple of µ-values to consider.
        e.g.: {'H2': np.linspace(-5, 0, 50), 'O2': np.linspace(-7, -2, 50)}.
    fixed_mu_values : dict, optional
        Baseline or fixed chemical potentials for each gas
        (e.g., {'H2': -4.0, 'O2': -6.0}).
    pressure_ranges : dict, optional
        For adding parasite axes to show log(p).
        Typically a dictionary of the form:
        {
          temperature_in_C : {
            'H2': (low_logp, high_logp),
            'O2': (low_logp, high_logp),
            ...
          }
        }
    output_file_prefix : str, optional
        Prefix for saved figures.

    Returns
    -------
    None
    """

    n_gases = len(gas_phase_reactants)
    if n_gases == 2:
        plot_diagram_for_two_gases(
            reaction_objects,
            gas_phase_reactants,
            mu_ranges,
            fixed_mu_values,
            pressure_ranges,
            output_file_prefix
        )

    elif n_gases == 3:
        plot_diagrams_for_three_gases(
            reaction_objects,
            gas_phase_reactants,
            mu_ranges,
            fixed_mu_values,
            pressure_ranges,
            output_file_prefix
        )

    elif n_gases > 4:
        # For >= 5 gases, user must specify which two to plot
        # (Alternatively, you could also handle exactly 4 in a similar manner.)
        # We'll assume the user passes in something like:
        #   fixed_mu_values = {
        #     'H2': -5.0, 'O2': -6.0, 'N2': -4.5, ...
        #   }
        #   or some discrete steps for certain ones, etc.
        # The key is we must know which 2 are on the axes:
        gas_x = ...  # user-chosen
        gas_y = ...  # user-chosen
        # Possibly the user also provides a list or array of values to step
        # for a 3rd gas, etc. So we might do:
        plot_diagram_for_specified_two(
            reaction_objects,
            gas_phase_reactants,
            mu_ranges,
            gas_x,
            gas_y,
            fixed_mu_values,
            pressure_ranges,
            output_file_prefix
        )

    else:
        # Covers cases n_gases < 2 or =4, which you might also want to handle.
        raise ValueError("Unhandled number of gas-phase reactants.")