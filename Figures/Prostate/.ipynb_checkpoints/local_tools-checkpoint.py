import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

short_mapping = {
        'T00': 'T0',
        'T01_Cast_Day1': 'C1',
        'T02_Cast_Day7': 'C7',
        'T03_Cast_Day14': 'C14',
        'T04_Cast_Day28': 'C28',
        'T05_Regen_Day1': 'R1',
        'T06_Regen_Day2': 'R2',
        'T07_Regen_Day3': 'R3',
        'T08_Regen_Day7': 'R7',
        'T09_Regen_Day14': 'R14',
        'T10_Regen_Day28': 'R28'
    }

full_mapping = {
        'T00': 'T0',
        'T01_Cast_Day1': 'Castration Day 1',
        'T02_Cast_Day7': 'Castration Day 7',
        'T03_Cast_Day14': 'Castration Day 14',
        'T04_Cast_Day28': 'Castration Day 28',
        'T05_Regen_Day1': 'Regeneration Day 1',
        'T06_Regen_Day2': 'Regeneration Day 2',
        'T07_Regen_Day3': 'Regeneration Day 3',
        'T08_Regen_Day7': 'Regeneration Day 7',
        'T09_Regen_Day14': 'Regeneration Day 14',
        'T10_Regen_Day28': 'Regeneration Day 28'
    }

def rename_time_points(
    adata, 
    obs_name='time', 
    obs_name_renamed=None,
    mapping_type='short'
):

    if mapping_type=='short':
        mapping = short_mapping
    elif mapping_type=='full':
        mapping = full_mapping
    else:
        raise ValueError('mapping_type should be either short or full')
    
    if obs_name_renamed is None:
        obs_name_renamed = obs_name + '_renamed'
    adata.obs[obs_name_renamed] = adata.obs[obs_name].map(mapping)

    return adata

##
def plot_gene_expression_box(
    time_labels,
    expression_vectors,
    gene_name: str = "",
    show_points: bool = True,
    save_path: str | None = None,
    color_mapping: dict | None = None,
    gap_after_first: bool = True,
    gap_size: float = 0.8,
    y_max: float | None = None,
):
    """
    Box-and-dot plot of single-cell expression over discrete time points,
    formatted for Nature-style figures (font size 12 pt everywhere).
    
    Parameters
    ----------
    time_labels : list[str | int]
        Labels for each time point (x-axis).
    expression_vectors : list[np.ndarray]
        One array per time point with expression values.
    gene_name : str, optional
        Y-axis title (typically the gene symbol).
    show_points : bool, default True
        Overlay individual observations on top of the box.
    save_path : str | None
        If provided, saves PNG and PDF without extension.
    color_mapping : dict | None
        Dictionary mapping time labels to colors. If None, uses default palette.
    gap_after_first : bool, default True
        Whether to add extra gap after the first timepoint.
    gap_size : float, default 0.8
        Size of the gap after first timepoint (larger = bigger gap).
    y_max : float, optional
        Maximum value for y-axis. If None, uses automatic scaling.
    """
    # --- Styling ----------------------------------------------------------------
    sns.set_style(
        "white",
        {
            "axes.edgecolor": "0.25",
            "axes.linewidth": 0.7,
            "grid.color": ".9",
        },
    )
    plt.figure(figsize=(7.0, 5.8), dpi=400)
    ax = plt.gca()
    
    # --- Create custom x-positions with gap ------------------------------------
    if gap_after_first and len(time_labels) > 1:
        # Create custom positions: 0, then 1+gap, 2+gap, etc.
        x_positions = [0] + [i + gap_size for i in range(1, len(time_labels))]
    else:
        # Standard evenly spaced positions
        x_positions = list(range(len(time_labels)))
    
    # --- Prepare long-form dataframe with custom positions --------------------
    time_position_map = dict(zip(time_labels, x_positions))
    long_df = pd.DataFrame(
        {
            "Time": np.repeat(time_labels, [len(v) for v in expression_vectors]),
            "Expression": np.concatenate(expression_vectors),
            "x_pos": np.repeat(x_positions, [len(v) for v in expression_vectors]),
        }
    )
    
    # --- Prepare colors ---------------------------------------------------------
    if color_mapping is not None:
        # Create ordered list of colors based on time_labels order
        colors = [color_mapping.get(str(label), "#1f77b4") for label in time_labels]
        palette = colors
    else:
        # Use default seaborn palette
        palette = "deep"
        colors = sns.color_palette(palette, len(time_labels))
    
    # --- Plot boxplots manually at custom positions ---------------------------
    box_parts = ax.boxplot(
        expression_vectors,
        positions=x_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        showcaps=True,
        medianprops={"color": "0.3", "linewidth": 1.2},
        whiskerprops={"linewidth": 0.8, "color": "0.3"},
        boxprops={"linewidth": 0.8, "edgecolor": "0.3"},
        capprops={"linewidth": 0.8, "color": "0.3"},
    )
    
    # Color the boxes
    for patch, color in zip(box_parts['boxes'], colors):
        if color_mapping is not None:
            import matplotlib.colors as mcolors
            # Make box fill color lighter
            rgb_color = mcolors.to_rgb(color)
            light_color = tuple(min(1.0, c + 0.3) for c in rgb_color)
            patch.set_facecolor(light_color)
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # --- Overlay individual cells at custom positions -------------------------
    if show_points:
        for i, (x_pos, expr_values, color) in enumerate(zip(x_positions, expression_vectors, colors)):
            # Add jitter to x positions
            n_points = len(expr_values)
            jitter_strength = 0.25 * 0.55  # scaled by box width
            np.random.seed(42)  # For reproducible jitter
            x_jittered = np.random.uniform(
                x_pos - jitter_strength, 
                x_pos + jitter_strength, 
                n_points
            )
            
            ax.scatter(
                x_jittered,
                expr_values,
                s=3.2**2,  # size squared for scatter
                alpha=0.55,
                c=color,
                edgecolors="white",
                linewidths=0.3,
            )
    
    # --- Labels, ticks, grids ----------------------------------------------------
    ax.set_xlabel("Time point", fontsize=12, labelpad=8)
    ax.set_ylabel(f"{gene_name} log normalised expression", fontsize=12, labelpad=8)
    ax.set_title(gene_name, loc="left", pad=12, fontsize=12)
    
    # Set custom tick positions and labels
    ax.set_xticks(x_positions)
    ax.set_xticklabels(time_labels)
    ax.tick_params(axis="both", labelsize=12)
    
    ax.grid(True, axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.set_axisbelow(True)
    
    # Fine-tune spines
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("0.25")
        spine.set_linewidth(0.7)
    
    # Adjust x-axis limits to accommodate the gap
    if gap_after_first and len(time_labels) > 1:
        ax.set_xlim(-0.5, x_positions[-1] + 0.5)
    
    # Set y-axis maximum if specified
    if y_max is not None:
        ax.set_ylim(top=y_max)
    
    plt.tight_layout()
    
    # --- Save if requested -------------------------------------------------------
    if save_path:
        plt.savefig(f"{save_path}.png", dpi=400, bbox_inches="tight", facecolor="white")
        plt.savefig(f"{save_path}.pdf", dpi=400, bbox_inches="tight", facecolor="white")
    
    return plt