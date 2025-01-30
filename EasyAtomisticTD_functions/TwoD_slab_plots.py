import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import center_of_mass  # for labeling

def plot_diagram_for_two_gases(
    reaction_objects,
    gas_phase_reactants,
    mu_ranges,
    fixed_mu_values,
    pressure_ranges,
    output_file_prefix
):
    """Plot one 2D stability diagram for exactly 2 gas-phase reactants."""
    gas1, gas2 = gas_phase_reactants  # we know length is 2

    # Ranges of µ for each gas
    mu1_vals = mu_ranges[gas1]
    mu2_vals = mu_ranges[gas2]

    # Construct a meshgrid
    MU1, MU2 = np.meshgrid(mu1_vals, mu2_vals)

    compounds = list(reaction_objects.keys())
    colors = plt.cm.viridis(np.linspace(0, 1, len(compounds)))

    # We'll compute ΔG for each (mu1, mu2) point and find the most stable compound
    min_energy = np.full(MU1.shape, np.inf)
    stability_map = np.zeros(MU1.shape, dtype=int)

    for idx, cmpd in enumerate(compounds):
        dG = np.zeros(MU1.shape)
        for i in range(MU1.shape[0]):
            for j in range(MU1.shape[1]):
                local_mu = {
                    gas1: MU1[i, j],
                    gas2: MU2[i, j]
                }
                # Merge with any fixed values if needed
                if fixed_mu_values is not None:
                    local_mu.update(fixed_mu_values)

                dG[i, j] = reaction_objects[cmpd].gibbs_free_energy(local_mu)

        mask = dG < min_energy
        stability_map[mask] = idx
        min_energy[mask] = dG[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    # Use imshow to display the stability region
    cax = ax.imshow(
        stability_map,
        origin="lower",
        extent=[mu1_vals[0], mu1_vals[-1], mu2_vals[0], mu2_vals[-1]],
        cmap=plt.cm.viridis,
        aspect="auto",
        vmin=0,
        vmax=len(compounds) - 1,
    )

    # Labeling axes
    ax.set_xlabel(f"$\\mu_{{{gas1}}}$ (eV)", fontsize=14)
    ax.set_ylabel(f"$\\mu_{{{gas2}}}$ (eV)", fontsize=14)
    ax.set_title("Stability Diagram (2 Gases)", fontsize=16)

    # (Optional) Add parasite axes to show log(p) at various T if pressure_ranges is given
    # Similar approach as in your original code...

    # Mark the stable regions with the compound name
    for idx, cmpd in enumerate(compounds):
        region_mask = (stability_map == idx)
        if region_mask.any():
            cy, cx = center_of_mass(region_mask)
            # Convert from index to actual µ-values
            x_coord = mu1_vals[int(cx)]
            y_coord = mu2_vals[int(cy)]
            ax.text(x_coord, y_coord, cmpd, color="white", ha="center", va="center")

    plt.tight_layout()
    file_name = f"{output_file_prefix}_2gases.png"
    plt.savefig(file_name, dpi=300)
    print(f"Saved 2-gas diagram as: {file_name}")
    plt.show()


from itertools import combinations


def plot_diagrams_for_three_gases(
        reaction_objects,
        gas_phase_reactants,
        mu_ranges,
        fixed_mu_values,
        pressure_ranges,
        output_file_prefix
):
    """Plot 2D stability diagrams for each pair of 3 gas-phase reactants,
    stepping the third one in mu_ranges as well."""

    # Suppose gas_phase_reactants = [gasA, gasB, gasC]
    # We'll form pairs: (gasA, gasB), (gasB, gasC), (gasA, gasC)
    pairs = list(combinations(gas_phase_reactants, 2))

    # For each pair, the third gas is "the one not in the pair"
    # We can step that third gas in mu_ranges to produce multiple diagrams
    for pair in pairs:
        gas1, gas2 = pair
        # Identify the third one
        gas3 = [g for g in gas_phase_reactants if g not in pair][0]

        mu1_vals = mu_ranges[gas1]
        mu2_vals = mu_ranges[gas2]
        mu3_vals = mu_ranges[gas3]

        # Construct meshgrid for gas1 vs gas2
        MU1, MU2 = np.meshgrid(mu1_vals, mu2_vals)

        for mu3 in mu3_vals:  # step through the third gas's range
            # We do the same stability calculation as before
            compounds = list(reaction_objects.keys())
            min_energy = np.full(MU1.shape, np.inf)
            stability_map = np.zeros(MU1.shape, dtype=int)

            for idx, cmpd in enumerate(compounds):
                dG = np.zeros(MU1.shape)
                for i in range(MU1.shape[0]):
                    for j in range(MU1.shape[1]):
                        local_mu = {
                            gas1: MU1[i, j],
                            gas2: MU2[i, j],
                            gas3: mu3
                        }
                        if fixed_mu_values is not None:
                            local_mu.update(fixed_mu_values)
                        dG[i, j] = reaction_objects[cmpd].gibbs_free_energy(local_mu)

                mask = dG < min_energy
                stability_map[mask] = idx
                min_energy[mask] = dG[mask]

            # Plot the map
            fig, ax = plt.subplots(figsize=(8, 6))
            cax = ax.imshow(
                stability_map,
                origin="lower",
                extent=[mu1_vals[0], mu1_vals[-1], mu2_vals[0], mu2_vals[-1]],
                cmap=plt.cm.viridis,
                aspect="auto",
                vmin=0,
                vmax=len(compounds) - 1
            )
            ax.set_xlabel(f"$\\mu_{{{gas1}}}$ (eV)", fontsize=14)
            ax.set_ylabel(f"$\\mu_{{{gas2}}}$ (eV)", fontsize=14)
            ax.set_title(f"Stability Diagram\n({gas1}, {gas2}) with {gas3} = {mu3:.2f} eV", fontsize=14)

            # Label stable regions
            for idx, cmpd in enumerate(compounds):
                region_mask = (stability_map == idx)
                if region_mask.any():
                    cy, cx = center_of_mass(region_mask)
                    x_coord = mu1_vals[int(cx)]
                    y_coord = mu2_vals[int(cy)]
                    ax.text(x_coord, y_coord, cmpd, color="white", ha="center", va="center")

            plt.tight_layout()
            fname = f"{output_file_prefix}_{gas1}_{gas2}_{gas3}_{mu3:.2f}.png"
            plt.savefig(fname, dpi=300)
            print(f"Saved 3-gas diagram: {fname}")
            plt.show()

def plot_diagram_for_specified_two(
    reaction_objects,
    gas_phase_reactants,
    mu_ranges,
    gas_x,
    gas_y,
    fixed_mu_values,
    pressure_ranges,
    output_file_prefix
):
    """
    For >=5 gas-phase reactants, user must specify which two to put on x/y axes.
    We then fix or discretely vary the others as they choose.
    """

    # Pull out x and y ranges
    mu_x_vals = mu_ranges[gas_x]
    mu_y_vals = mu_ranges[gas_y]

    MUx, MUy = np.meshgrid(mu_x_vals, mu_y_vals)

    # If user wants to step some other gas, they'd have to pass that info somehow.
    # Here's a minimal version that just fixes all other gases:
    compounds = list(reaction_objects.keys())
    min_energy = np.full(MUx.shape, np.inf)
    stability_map = np.zeros(MUx.shape, dtype=int)

    for idx, cmpd in enumerate(compounds):
        dG = np.zeros(MUx.shape)
        for i in range(MUx.shape[0]):
            for j in range(MUx.shape[1]):
                local_mu = {
                    gas_x: MUx[i, j],
                    gas_y: MUy[i, j]
                }
                if fixed_mu_values is not None:
                    local_mu.update(fixed_mu_values)

                dG[i, j] = reaction_objects[cmpd].gibbs_free_energy(local_mu)

        mask = dG < min_energy
        stability_map[mask] = idx
        min_energy[mask] = dG[mask]

    # Plot as before
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(
        stability_map,
        origin="lower",
        extent=[mu_x_vals[0], mu_x_vals[-1], mu_y_vals[0], mu_y_vals[-1]],
        cmap=plt.cm.viridis,
        aspect="auto",
        vmin=0,
        vmax=len(compounds) - 1
    )
    ax.set_xlabel(f"$\\mu_{{{gas_x}}}$ (eV)", fontsize=14)
    ax.set_ylabel(f"$\\mu_{{{gas_y}}}$ (eV)", fontsize=14)
    ax.set_title(f"Stability Diagram: {gas_x} vs. {gas_y}", fontsize=14)

    # Label stable regions
    from scipy.ndimage import center_of_mass
    for idx, cmpd in enumerate(compounds):
        region_mask = (stability_map == idx)
        if region_mask.any():
            cy, cx = center_of_mass(region_mask)
            x_coord = mu_x_vals[int(cx)]
            y_coord = mu_y_vals[int(cy)]
            ax.text(x_coord, y_coord, cmpd, color="white", ha="center", va="center")

    plt.tight_layout()
    fname = f"{output_file_prefix}_{gas_x}_{gas_y}_fixedOthers.png"
    plt.savefig(fname, dpi=300)
    print(f"Saved multi-gas (specified-two) diagram: {fname}")
    plt.show()