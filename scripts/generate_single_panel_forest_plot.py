import os
import matplotlib.pyplot as plt
import numpy as np

def generate_publication_forest_plot():
    # Set publication styling
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10

    # Width: ~516 pt = 7.167 inches. Height: ~4.1 inches.
    fig_width = 516 / 72.0  # 7.1667 in
    fig_height = 4.1        # in
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Data specified by user (sorted by point estimate ΔC descending from top to bottom)
    # Order in list: bottom row (y=0) to top row (y=7.4)
    # Fields: (label, mean, lower, upper, margin, is_auc, exceeds_margin)
    items = [
        # Bottom row: ΔAUC (margin ±0.04)
        ("Full semantic block (33, 270-day AUC)", -0.004, -0.036, 0.028, 0.04, True, False),
        # ΔC rows (margin ±0.03), ordered by ΔC from lowest to highest so highest is at top
        ("Conflict (3)†",                         -0.011, -0.035, 0.003, 0.03, False, True),  # Exceeds -0.03
        ("Dispersion (1)†",                       -0.007, -0.021, 0.003, 0.03, False, False),
        ("Dominance/concentration (1)",            0.000, -0.014, 0.014, 0.03, False, False),
        ("Temporal/streak (8)†",                   0.003,  0.000, 0.006, 0.03, False, False),
        ("Aspect-specific counts (16)",            0.004, -0.014, 0.021, 0.03, False, False),
        ("Sentiment (4)",                          0.010, -0.002, 0.025, 0.03, False, False),
        ("Full semantic block (33), ΔC",           0.004, -0.021, 0.028, 0.03, False, False),
    ]

    # Assign y-positions with a clear gap separating ΔC and ΔAUC
    y_positions = [0.0, 1.4, 2.4, 3.4, 4.4, 5.4, 6.4, 7.4]

    # Draw row-specific equivalence margin bands
    band_color = '#E5F5E0'  # light green
    for (label, mean, lower, upper, margin, is_auc, exceeds_margin), y in zip(items, y_positions):
        band_height = 0.76
        ax.barh(y, width=2 * margin, left=-margin, height=band_height,
                color=band_color, edgecolor='none', alpha=0.85, zorder=0)

    # Reference line at difference = 0
    ax.axvline(0, color='#666666', linestyle='-', linewidth=0.8, alpha=0.75, zorder=1)

    # Horizontal separator line between ΔAUC (row 0) and ΔC rows (rows 1-7)
    separator_y = 0.70
    ax.axhline(separator_y, color='#B0BEC5', linestyle='--', linewidth=0.85, zorder=1)

    # Plot error bars and point estimates
    for (label, mean, lower, upper, margin, is_auc, exceeds_margin), y in zip(items, y_positions):
        if is_auc:
            pt_color = '#C0392B'     # Rust / dark red-orange for ΔAUC
            err_color = '#D9534F'
            m_face = pt_color
            m_edge = pt_color
        elif exceeds_margin:
            # Hollow / open marker for row that EXCEEDS equivalence margin (Conflict)
            pt_color = '#1F4E79'
            err_color = '#1F4E79'
            m_face = 'white'         # Hollow marker
            m_edge = '#1F4E79'
        else:
            # Solid blue marker for ΔC inside margin
            pt_color = '#1F4E79'
            err_color = '#4682B4'
            m_face = pt_color
            m_edge = pt_color

        ax.errorbar(
            mean, y,
            xerr=[[mean - lower], [upper - mean]],
            fmt='o',
            color=pt_color,
            ecolor=err_color,
            elinewidth=2.2,
            capsize=0,
            markersize=6.8,
            markerfacecolor=m_face,
            markeredgecolor=m_edge,
            markeredgewidth=2.2,
            zorder=3
        )

    # Equivalence margin annotations on the plot margins
    ax.text(0.031, 7.40, r'margin $\pm$0.03', fontsize=9, color='#2E7D32', va='center', ha='left', fontstyle='italic')
    ax.text(0.041, 0.00, r'margin $\pm$0.04', fontsize=9, color='#2E7D32', va='center', ha='left', fontstyle='italic')

    # Y-axis ticks and labels
    ax.set_yticks(y_positions)
    ax.set_yticklabels([item[0] for item in items], fontsize=10)
    
    # Unified X-axis label
    ax.set_xlabel("Difference in metric: C-index / AUC (Model C − Model A)", fontsize=10, fontweight='bold')
    
    # Range of X
    ax.set_xlim(-0.052, 0.055)
    ax.set_ylim(-0.6, 8.1)
    
    # Clean spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(axis='both', which='both', labelsize=10, color='#888888')

    plt.tight_layout()

    # Create directories and save
    os.makedirs('images', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    os.makedirs('scratch', exist_ok=True)

    # Save PDF
    fig.savefig('images/forest_plot.pdf', format='pdf', bbox_inches='tight')
    fig.savefig('figures/forest_plot.pdf', format='pdf', bbox_inches='tight')
    fig.savefig('forest_plot.pdf', format='pdf', bbox_inches='tight')

    # Save PNG for inspection
    fig.savefig('figures/forest_plot.png', dpi=300, bbox_inches='tight')
    fig.savefig('scratch/forest_plot_single_panel.png', dpi=300, bbox_inches='tight')

    print("[SUCCESS] Exported refined single-panel forest_plot.pdf to images/, figures/, and root.")

if __name__ == '__main__':
    generate_publication_forest_plot()
