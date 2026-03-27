'''
# THIS PART OF THE CODE WAS NOT INCLUDING BOOTSTRAPPING FOR SIGNIFICANCE TESTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

# --- CONFIGURATION ---
mode = "_Objects" # "" or "_Objects"
data_types = ["ERA5", "HCLIM", "MetUM", "RACMO2"]
regions = ["EAN", "WAN", "SOO"]
features = ['AR', 'CY', 'ACY', 'FR', 'JET']

region_titles = {
    "EAN": "East Antarctica (EAN)",
    "WAN": "West Antarctica (WAN)",
    "SOO": "Southern Ocean (SOO)"
}

#colors = {
#    "ERA5":  "#E63946",   # Red
#    "HCLIM": "#457B9D",   # Blue
#    "MetUM": "#2A9D8F",   # Teal
#    "RACMO2": "#F4A261"   # Orange
#}

colors = {
    "ERA5":   "#E63946",   # Red
    "HCLIM":  "#457B9D",   # Steel blue
    "MetUM":  "#2D6A4F",   # Forest green  ← changed
    "RACMO2": "#F4A261"    # Orange
}

display_names = {
    "ERA5": "ERA5",
    "HCLIM": "HCLIM",
    "MetUM": "MetUM",
    "RACMO2": "RACMO"  # Display as RACMO
}

# --- LOAD DATA ---
def load_frequencies(data_type, region, features):
    """Load the with_features CSV and return percentage frequency per feature."""
    csv_path = f"top100_{region}_with_features_{data_type}{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found: {csv_path} — filling with zeros.")
        return [0.0] * len(features)
    df = pd.read_csv(csv_path)
    return [df[feat].sum() / len(df) * 100 for feat in features]

# Build a dict: freq_data[region][data_type] = [val1, val2, ...]
freq_data = {
    region: {
        dt: load_frequencies(dt, region, features)
        for dt in data_types
    }
    for region in regions
}

# --- RADAR PLOT HELPER ---
def make_radar_ax(ax, values_dict, feature_labels, colors_dict, title):
    """
    Draw a radar chart on a given polar axis.
    values_dict: {label: [v1, v2, ...]} where values are 0-100
    """
    N = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    # Close the loop
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Draw feature labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=13, fontweight='bold')

    # Radial axis: 0 to 100%
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=7.5, color='grey')
    ax.yaxis.set_tick_params(labelcolor='grey')
    # ax.set_rlabel_position(108)  

    # Subtle grid
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['polar'].set_visible(False)

    # Plot each dataset
    for label, vals in values_dict.items():
        vals_closed = vals + vals[:1]
        color = colors_dict[label]
        ax.plot(angles, vals_closed, color=color, linewidth=2, linestyle='solid', zorder=3)
        ax.fill(angles, vals_closed, color=color, alpha=0.08)
        # Dot markers at each vertex
        ax.scatter(angles[:-1], vals, color=color, s=40, zorder=4)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


# --- MAIN PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
#fig.suptitle("Atmospheric Feature Association — Top 100 Extreme Precipitation Events\n(2001–2020)",
#             fontsize=15, fontweight='bold', y=1.02)
fig.suptitle("",
             fontsize=15, fontweight='bold', y=1.02)

for ax, region in zip(axes, regions):
    make_radar_ax(
        ax=ax,
        values_dict=freq_data[region],
        feature_labels=features,
        colors_dict=colors,
        title=region_titles[region]
    )

# Shared legend
legend_handles = [
    mpatches.Patch(color=colors[dt], label=display_names[dt], alpha=0.75)
    for dt in data_types
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=4,          
    fontsize=12,
    frameon=True,
    framealpha=0.9,
    edgecolor='grey',
    bbox_to_anchor=(0.5, -0.05)
)

plt.tight_layout()
output_fig = f"Antarctica_Extremes_Radar_AllModels{mode}.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Radar plot saved to: {output_fig}")
'''

# This version includes bootstrapping for significance testing against ERA5.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

np.random.seed(42) # for reproducibility

# --- CONFIGURATION ---
mode         = "_Objects"
data_types   = ["ERA5", "HCLIM", "MetUM", "RACMO2"]
regions      = ["EAN", "WAN", "SOO"]
features     = ['AR', 'CY', 'ACY', 'FR', 'JET']
region_titles = {
    "EAN": "East Antarctica (EAN)",
    "WAN": "West Antarctica (WAN)",
    "SOO": "Southern Ocean (SOO)"
}
colors = {
    "ERA5":   "#E63946",
    "HCLIM":  "#457B9D",
    "MetUM":  "#2D6A4F",
    "RACMO2": "#F4A261"
}
display_names = {
    "ERA5":   "ERA5",
    "HCLIM":  "HCLIM",
    "MetUM":  "MetUM",
    "RACMO2": "RACMO"
}

N_BOOTSTRAP = 1000 # number of bootstrap samples for CI and significance testing
CI_LOWER    = 2.5 # for 95% confidence interval
CI_UPPER    = 97.5

# --- LOAD DATA ---
def load_dataframe(data_type, region):
    csv_path = f"top100_{region}_with_features_{data_type}{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"⚠️  File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

# --- BOOTSTRAP FREQUENCIES ---
def bootstrap_frequencies(df, features, n_bootstrap=1000):
    n = len(df)
    boot_freqs = np.zeros((n_bootstrap, len(features)))
    for i in range(n_bootstrap):
        sample = df.sample(n=n, replace=True)
        boot_freqs[i] = [sample[feat].sum() / n * 100 for feat in features]
    mean_vals = boot_freqs.mean(axis=0).tolist()
    ci_lower  = np.percentile(boot_freqs, CI_LOWER,  axis=0).tolist()
    ci_upper  = np.percentile(boot_freqs, CI_UPPER, axis=0).tolist()
    return mean_vals, ci_lower, ci_upper

# --- BOOTSTRAP DIFFERENCE (for significance vs ERA5) ---
def bootstrap_significance(df_model, df_era5, features, n_bootstrap=1000):

    n_model = len(df_model)
    n_era5  = len(df_era5)
    boot_diffs = np.zeros((n_bootstrap, len(features)))

    for i in range(n_bootstrap):
        sample_model = df_model.sample(n=n_model, replace=True)
        sample_era5  = df_era5.sample(n=n_era5,  replace=True)
        freq_model   = np.array([sample_model[f].sum() / n_model * 100 for f in features])
        freq_era5    = np.array([sample_era5[f].sum()  / n_era5  * 100 for f in features])
        boot_diffs[i] = freq_model - freq_era5

    ci_lo = np.percentile(boot_diffs, CI_LOWER, axis=0)
    ci_hi = np.percentile(boot_diffs, CI_UPPER, axis=0)

    # Significant where the CI does NOT straddle zero
    significant = ((ci_lo > 0) | (ci_hi < 0)).tolist()
    return significant

# --- PRE-COMPUTE ---
freq_data = {}   # freq_data[region][dt] = (mean, lower, upper)
sig_data  = {}   # sig_data[region][dt]  = [bool per feature]

for region in regions:
    freq_data[region] = {}
    sig_data[region]  = {}
    df_era5 = load_dataframe("ERA5", region)

    for dt in data_types:
        df = load_dataframe(dt, region)
        if df is None:
            freq_data[region][dt] = ([0]*len(features),)*3
            sig_data[region][dt]  = [False]*len(features)
        else:
            freq_data[region][dt] = bootstrap_frequencies(df, features, N_BOOTSTRAP)
            if dt == "ERA5" or df_era5 is None:
                # ERA5 vs itself: significance not applicable
                sig_data[region][dt] = [False]*len(features)
            else:
                sig_data[region][dt] = bootstrap_significance(df, df_era5, features, N_BOOTSTRAP)

# --- RADAR PLOT HELPER ---
def make_radar_ax(ax, freq_dict, sig_dict, feature_labels, colors_dict, title):
    N      = len(feature_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(feature_labels, fontsize=13, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=7.5, color='grey')
    ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.6)
    ax.spines['polar'].set_visible(False)

    for label, (mean_vals, ci_lower, ci_upper) in freq_dict.items():
        color  = colors_dict[label]
        sig    = sig_dict[label]   # list of bools, one per feature

        mean_c  = mean_vals + mean_vals[:1]
        lower_c = ci_lower  + ci_lower[:1]
        upper_c = ci_upper  + ci_upper[:1]

        # Main line + CI band
        ax.plot(angles, mean_c, color=color, linewidth=2, linestyle='solid', zorder=3)
        ax.fill_between(angles, lower_c, upper_c, color=color, alpha=0.15, zorder=2)

        # Markers: star if significant vs ERA5, circle otherwise
        for i, (ang, val, is_sig) in enumerate(zip(angles[:-1], mean_vals, sig)):
            marker = '*' if is_sig else 'o'
            size   = 120 if is_sig else 40
            ax.scatter(ang, val, color=color, s=size, marker=marker, zorder=4)

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)


# --- PRINT OTHER FRACTIONS (for additional information) ---
print("\nFraction of events with no feature detected (OTHER):")
print(f"{'Dataset':<10} {'Region':<6} {'OTHER [%]':>10}")
print("-" * 28)
for region in regions:
    for dt in data_types:
        df = load_dataframe(dt, region)
        if df is not None:
            other = ((df[features] == False).all(axis=1)).sum() / len(df) * 100
            print(f"{dt:<10} {region:<6} {other:>9.1f}%")
    print()  # blank line between regions

# --- MAIN PLOT ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
fig.suptitle("", fontsize=15, fontweight='bold', y=1.02)

for ax, region in zip(axes, regions):
    make_radar_ax(
        ax=ax,
        freq_dict=freq_data[region],
        sig_dict=sig_data[region],
        feature_labels=features,
        colors_dict=colors,
        title=region_titles[region]
    )

# Shared legend 
legend_handles = [
    mpatches.Patch(color=colors[dt], label=display_names[dt], alpha=0.75)
    for dt in data_types
]
legend_handles += [
    plt.scatter([], [], marker='*', color='grey', s=120, label='Significant vs ERA5 (95% CI)'),
    plt.scatter([], [], marker='o', color='grey', s=40,  label='Not significant'),
]
fig.legend(
    handles=legend_handles,
    loc='lower center',
    ncol=6,
    fontsize=16,
    frameon=True,
    framealpha=0.9,
    edgecolor='grey',
    bbox_to_anchor=(0.5, -0.07)
)

plt.tight_layout()
output_fig = f"Antarctica_Extremes_Radar_AllModels{mode}_bootstrap_sig.png"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.show()
print(f"Saved: {output_fig}")

