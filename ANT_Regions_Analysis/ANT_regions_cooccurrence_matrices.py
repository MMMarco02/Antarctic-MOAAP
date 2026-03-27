'''
# VERSION 1: Basic matrices with co-occurrence values per each dataset, without significance testing
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- 1. SETTINGS ---
data_types  = ["ERA5", "HCLIM", "MetUM", "RACMO2"]
display_names = {
    "ERA5": "ERA5",
    "HCLIM": "HCLIM",
    "MetUM": "MetUM",
    "RACMO2": "RACMO"  # Display as RACMO
}
regions     = ["EAN", "WAN", "SOO"]
region_titles = ["East Antarctica (EAN)", "West Antarctica (WAN)", "Southern Ocean (SOO)"]
features    = ['AR', 'CY', 'ACY', 'FR', 'JET']
mode = "_Objects" # "" or "_Objects"

# --- 2. CO-OCCURRENCE FUNCTION ---
def get_cooccurrence_matrix(csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    n_feat = len(features)
    co_matrix = np.zeros((n_feat, n_feat))
    df['total_features'] = df[features].sum(axis=1)
    for i in range(n_feat):
        for j in range(n_feat):
            if i == j:
                solo = df[(df[features[i]] == True) & (df['total_features'] == 1)]
                co_matrix[i, j] = (len(solo) / len(df)) * 100
            else:
                both = df[(df[features[i]] == True) & (df[features[j]] == True)]
                co_matrix[i, j] = (len(both) / len(df)) * 100
    return co_matrix

# --- 3. FIGURE SETUP ---
# 3 rows (models) x 3 cols (regions) + 1 narrow col for colorbar
fig, axes = plt.subplots(
    3, 4,
    figsize=(18, 14),
    gridspec_kw={'width_ratios': [1, 1, 1, 0.04]}
)

cbar_ax = fig.add_axes([0.875, 0.12, 0.015, 0.75])  # [left, bottom, width, height]

# Track whether we've drawn the colorbar yet
cbar_drawn = False

# --- 4. PLOTTING ---
for row, dt in enumerate(data_types):
    for col, reg in enumerate(regions):
        ax = axes[row, col]
        file_path = f"top100_{reg}_with_features_{dt}{mode}.csv"
        co_matrix = get_cooccurrence_matrix(file_path)

        if co_matrix is None:
            ax.axis('off')
            continue

        mask = np.triu(np.ones_like(co_matrix, dtype=bool), k=1)

        sns.heatmap(
            co_matrix,
            ax=ax,
            mask=mask,
            annot=True,
            fmt=".0f",
            cmap="YlOrRd",
            vmin=0, vmax=100,
            xticklabels=features,
            yticklabels=features,
            square=True,
            linewidths=0.5,
            annot_kws={"size": 9},
            cbar=not cbar_drawn,
            cbar_ax=cbar_ax if not cbar_drawn else None,
            cbar_kws={'label': 'Occurrence / Co-occurrence [%]'} if not cbar_drawn else None
        )
        cbar_drawn = True

        # Column titles — only on the top row
        if row == 0:
            ax.set_title(region_titles[col], fontsize=13, fontweight='bold', pad=32)

        # Row (model) labels — only on the first column
        if col == 0:
            #ax.set_ylabel(dt, fontsize=13, fontweight='bold', labelpad=32, rotation=90, va='center')
            ax.set_ylabel(display_names[dt], fontsize=13, fontweight='bold', labelpad=32, rotation=90, va='center')
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)

    # Hide the unused 4th axes column (it's handled by cbar_ax manually)
    axes[row, 3].set_visible(False)

# --- 5. TITLE & SPACING ---
#fig.suptitle(
#    "Feature Occurrence & Co-occurrence — Top 100 Extreme Precipitation Events (2001–2020)",
#    fontsize=14, fontweight='bold', y=0.98
#)

plt.subplots_adjust(wspace=0.45, hspace=0.35, left=0.09, right=0.86, top=0.93, bottom=0.07)

# --- 6. SAVE ---
output_name = f"Antarctica_CoOccurrence_Combined_AllModels{mode}.png"
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Saved: {output_name}")
plt.show()
'''

'''
# VERSION 2: Added the RACMO2 model and updated the layout to accommodate 4 rows (ERA5 + 3 models). Also added a second colorbar for the difference matrices.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os

# --- 1. SETTINGS ---
data_types    = ["ERA5", "HCLIM", "MetUM", "RACMO2"]   # ← added RACMO2
regions       = ["EAN", "WAN", "SOO"]
region_titles = ["East Antarctica (EAN)", "West Antarctica (WAN)", "Southern Ocean (SOO)"]
features      = ['AR', 'CY', 'ACY', 'FR', 'JET']
mode          = "_Objects"


ROW_LABELS = ["ERA5", "HCLIM − ERA5", "MetUM − ERA5", "RACMO − ERA5"]  # ← added
DIFF_LIM   = 30
CMAP_ABS   = "YlOrRd"
CMAP_DIFF  = "RdBu_r"

# --- 2. CO-OCCURRENCE FUNCTION ---
def get_cooccurrence_matrix(csv_path):
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    n_feat = len(features)
    co_matrix = np.zeros((n_feat, n_feat))
    df['total_features'] = df[features].sum(axis=1)
    for i in range(n_feat):
        for j in range(n_feat):
            if i == j:
                solo = df[(df[features[i]] == True) & (df['total_features'] == 1)]
                co_matrix[i, j] = (len(solo) / len(df)) * 100
            else:
                both = df[(df[features[i]] == True) & (df[features[j]] == True)]
                co_matrix[i, j] = (len(both) / len(df)) * 100
    return co_matrix

# --- 3. PRE-COMPUTE ALL MATRICES ---
matrices = {}
for dt in data_types:
    matrices[dt] = {}
    for reg in regions:
        file_path = f"top100_{reg}_with_features_{dt}{mode}.csv"
        matrices[dt][reg] = get_cooccurrence_matrix(file_path)


# --- 4. FIGURE SETUP ---
fig, axes = plt.subplots(
    4, 3,                      # ← 4 rows instead of 3
    figsize=(16, 18),          # ← taller to accommodate the extra row
)

# --- 5. PLOTTING (no colorbars — we'll add them manually after) ---
for row, (dt, row_label) in enumerate(zip(data_types, ROW_LABELS)):
    for col, reg in enumerate(regions):
        ax = axes[row, col]

        if row == 0:
            mat = matrices["ERA5"][reg]
            if mat is None:
                ax.axis('off')
                continue
            mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
            sns.heatmap(
                mat, ax=ax, mask=mask,
                annot=True, fmt=".0f",
                cmap=CMAP_ABS, vmin=0, vmax=100,
                xticklabels=features, yticklabels=features,
                square=True, linewidths=0.5,
                annot_kws={"size": 9},
                cbar=False
            )
        else:
            mat_model = matrices[dt][reg]
            mat_era5  = matrices["ERA5"][reg]
            if mat_model is None or mat_era5 is None:
                ax.axis('off')
                continue
            diff = mat_model - mat_era5
            mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
            sns.heatmap(
                diff, ax=ax, mask=mask,
                annot=True, fmt="+.0f",
                cmap=CMAP_DIFF, vmin=-DIFF_LIM, vmax=DIFF_LIM,
                center=0,
                xticklabels=features, yticklabels=features,
                square=True, linewidths=0.5,
                annot_kws={"size": 9},
                cbar=False
            )

        # Column titles — only on the top row
        if row == 0:
            ax.set_title(region_titles[col], fontsize=13, fontweight='bold', pad=32)

        # Row labels — only on the first column
        if col == 0:
            ax.set_ylabel(row_label, fontsize=13, fontweight='bold',
                          labelpad=32, rotation=90, va='center')
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)

# --- 6. SPACING (must come before reading axis positions) ---
plt.subplots_adjust(wspace=0.45, hspace=0.35,
                    left=0.09, right=0.87, top=0.93, bottom=0.07)

# --- 7. COLORBARS ---
fig.canvas.draw()

pos_r0 = axes[0, 2].get_position()
pos_r1 = axes[1, 2].get_position()
pos_r2 = axes[2, 2].get_position()
pos_r3 = axes[3, 2].get_position()   # ← new: RACMO2 row

cbar_left  = pos_r0.x1 + 0.015
cbar_width = 0.015

# ERA5 colorbar: row 0 only — unchanged
cbar_ax_abs = fig.add_axes([
    cbar_left,
    pos_r0.y0,
    cbar_width,
    pos_r0.y1 - pos_r0.y0
])
fig.colorbar(
    cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=100), cmap=CMAP_ABS),
    cax=cbar_ax_abs,
    label='Occurrence / Co-occurrence [%]'
)

# Diff colorbar: now spans rows 1–3  ← updated
cbar_ax_diff = fig.add_axes([
    cbar_left,
    pos_r3.y0,                      # bottom of RACMO2 row
    cbar_width,
    pos_r1.y1 - pos_r3.y0          # top of HCLIM row → bottom of RACMO2 row
])
fig.colorbar(
    cm.ScalarMappable(
        norm=mcolors.TwoSlopeNorm(vmin=-DIFF_LIM, vcenter=0, vmax=DIFF_LIM),
        cmap=CMAP_DIFF
    ),
    cax=cbar_ax_diff,
    label='Difference vs ERA5 [pp]'
)

# --- 8. SAVE ---
output_name = f"Antarctica_CoOccurrence_Combined_AllModels{mode}.png"
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Saved: {output_name}")
plt.show()
'''
# VERSION 3: Added bootstrap significance testing to identify which differences between models and ERA5 are statistically significant at the 5% level.
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import os

np.random.seed(42) # for reproducibility

# --- 1. SETTINGS ---
data_types    = ["ERA5", "HCLIM", "MetUM", "RACMO2"]
regions       = ["EAN", "WAN", "SOO"]
region_titles = ["East Antarctica (EAN)", "West Antarctica (WAN)", "Southern Ocean (SOO)"]
features      = ['AR', 'CY', 'ACY', 'FR', 'JET']
mode          = "_Objects"

ROW_LABELS  = ["ERA5", "HCLIM − ERA5", "MetUM − ERA5", "RACMO − ERA5"] # We use "RACMO" for display instead of "RACMO2"
DIFF_LIM    = 30
CMAP_ABS    = "YlOrRd"
CMAP_DIFF   = "RdBu_r"
N_BOOTSTRAP = 1000 # Number of bootstrap resamples for significance testing
CI_LOWER    = 2.5 # 95% confidence interval lower bound
CI_UPPER    = 97.5 # 95% confidence interval upper bound

# --- 2. LOAD DATAFRAMES ---
def load_dataframe(data_type, region):
    csv_path = f"top100_{region}_with_features_{data_type}{mode}.csv"
    if not os.path.exists(csv_path):
        print(f"Warning: File not found: {csv_path}")
        return None
    return pd.read_csv(csv_path)

# --- 3. CO-OCCURRENCE MATRIX FROM A DATAFRAME ---
def compute_cooccurrence(df):
    n_feat    = len(features)
    co_matrix = np.zeros((n_feat, n_feat))
    df['total_features'] = df[features].sum(axis=1)
    for i in range(n_feat):
        for j in range(n_feat):
            if i == j:
                solo = df[(df[features[i]] == True) & (df['total_features'] == 1)]
                co_matrix[i, j] = (len(solo) / len(df)) * 100
            else:
                both = df[(df[features[i]] == True) & (df[features[j]] == True)]
                co_matrix[i, j] = (len(both) / len(df)) * 100
    return co_matrix

# --- 4. BOOTSTRAP SIGNIFICANCE ---
def bootstrap_significance_cooccurrence(df_model, df_era5, n_bootstrap=1000):
    n_model    = len(df_model)
    n_era5     = len(df_era5)
    n_feat     = len(features)
    boot_diffs = np.zeros((n_bootstrap, n_feat, n_feat))

    for k in range(n_bootstrap):
        sample_model  = df_model.sample(n=n_model, replace=True).reset_index(drop=True)
        sample_era5   = df_era5.sample(n=n_era5,   replace=True).reset_index(drop=True)
        boot_diffs[k] = compute_cooccurrence(sample_model) - compute_cooccurrence(sample_era5)

    ci_lo = np.percentile(boot_diffs, CI_LOWER, axis=0)
    ci_hi = np.percentile(boot_diffs, CI_UPPER, axis=0)
    return (ci_lo > 0) | (ci_hi < 0)  

# --- 5. GREY-OUT NON-SIGNIFICANT CELLS ---
def grey_out_nonsignificant(ax, sig_matrix, mask):

    n = sig_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            if mask[i, j]:
                continue                       
            if not sig_matrix[i, j]:           # not significant --> grey out
                ax.add_patch(mpatches.Rectangle(
                    (j, i),                    
                    1, 1,                      
                    color='grey',
                    alpha=0.6,
                    zorder=5,                  
                    linewidth=0
                ))

# --- 6. PRE-COMPUTE MATRICES AND SIGNIFICANCE ---
matrices = {}
sig_mats = {}

for dt in data_types:
    matrices[dt] = {}
    sig_mats[dt] = {}
    for reg in regions:
        df = load_dataframe(dt, reg)
        matrices[dt][reg] = compute_cooccurrence(df) if df is not None else None
        sig_mats[dt][reg] = None

print("Running bootstrap significance tests...")
for dt in data_types[1:]:
    for reg in regions:
        df_model = load_dataframe(dt, reg)
        df_era5  = load_dataframe("ERA5", reg)
        if df_model is not None and df_era5 is not None:
            print(f"  Bootstrapping {dt} vs ERA5 in {reg}...")
            sig_mats[dt][reg] = bootstrap_significance_cooccurrence(
                df_model, df_era5, N_BOOTSTRAP
            )
print("Done.")

# --- 7. FIGURE SETUP ---
fig, axes = plt.subplots(4, 3, figsize=(16, 18))

# --- 8. PLOTTING ---
for row, (dt, row_label) in enumerate(zip(data_types, ROW_LABELS)):
    for col, reg in enumerate(regions):
        ax = axes[row, col]

        if row == 0:
            mat = matrices["ERA5"][reg]
            if mat is None:
                ax.axis('off')
                continue
            mask = np.triu(np.ones_like(mat, dtype=bool), k=1)
            sns.heatmap(
                mat, ax=ax, mask=mask,
                annot=True, fmt=".0f",
                cmap=CMAP_ABS, vmin=0, vmax=100,
                xticklabels=features, yticklabels=features,
                square=True, linewidths=0.5,
                annot_kws={"size": 9},
                cbar=False
            )
            # No greying for ERA5 row — no significance test applied

        else:
            mat_model = matrices[dt][reg]
            mat_era5  = matrices["ERA5"][reg]
            if mat_model is None or mat_era5 is None:
                ax.axis('off')
                continue
            diff = mat_model - mat_era5
            mask = np.triu(np.ones_like(diff, dtype=bool), k=1)
            sns.heatmap(
                diff, ax=ax, mask=mask,
                annot=True, fmt="+.0f",
                cmap=CMAP_DIFF, vmin=-DIFF_LIM, vmax=DIFF_LIM,
                center=0,
                xticklabels=features, yticklabels=features,
                square=True, linewidths=0.5,
                annot_kws={"size": 9},
                cbar=False
            )
            # Grey out non-significant cells
            if sig_mats[dt][reg] is not None:
                grey_out_nonsignificant(ax, sig_mats[dt][reg], mask)

        if row == 0:
            ax.set_title(region_titles[col], fontsize=14, fontweight='bold', pad=32)
        if col == 0:
            ax.set_ylabel(row_label, fontsize=14, fontweight='bold',
                          labelpad=32, rotation=90, va='center')
        else:
            ax.set_ylabel('')

        ax.tick_params(axis='y', rotation=0)
        ax.tick_params(axis='x', rotation=45)

plt.subplots_adjust(wspace=0.45, hspace=0.35,
                    left=0.09, right=0.87, top=0.93, bottom=0.07)

# --- 9. COLORBARS ---
fig.canvas.draw()

pos_r0 = axes[0, 2].get_position()
pos_r1 = axes[1, 2].get_position()
pos_r3 = axes[3, 2].get_position()

cbar_left  = pos_r0.x1 + 0.015
cbar_width = 0.015

cbar_ax_abs = fig.add_axes([cbar_left, pos_r0.y0, cbar_width, pos_r0.y1 - pos_r0.y0])
cbar_abs = fig.colorbar(                          
    cm.ScalarMappable(norm=mcolors.Normalize(vmin=0, vmax=100), cmap=CMAP_ABS),
    cax=cbar_ax_abs,
)
cbar_abs.set_label('Occurrence / Co-occurrence [%]', fontsize=12)  
cbar_abs.ax.tick_params(labelsize=12)                               

cbar_ax_diff = fig.add_axes([cbar_left, pos_r3.y0, cbar_width, pos_r1.y1 - pos_r3.y0])
cbar_diff = fig.colorbar(                        
    cm.ScalarMappable(
        norm=mcolors.TwoSlopeNorm(vmin=-DIFF_LIM, vcenter=0, vmax=DIFF_LIM),
        cmap=CMAP_DIFF
    ),
    cax=cbar_ax_diff,
)
cbar_diff.set_label('Difference vs ERA5 [pp]', fontsize=12)         
cbar_diff.ax.tick_params(labelsize=12)                               

# --- 10. LEGEND FOR GREY SHADING ---
grey_patch = mpatches.Patch(color='grey', alpha=0.6, label='Not significant (95% CI)')
fig.legend(
    handles=[grey_patch],
    loc='lower center',
    fontsize=16,
    frameon=True,
    framealpha=0.9,
    edgecolor='grey',
    bbox_to_anchor=(0.5, 0.01)
)

# --- 11. SAVE ---
output_name = f"Antarctica_CoOccurrence_Combined_AllModels{mode}_bootstrap_sig.png"
plt.savefig(output_name, dpi=300, bbox_inches='tight')
print(f"Saved: {output_name}")
plt.show()