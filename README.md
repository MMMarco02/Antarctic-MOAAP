# Antarctic-MOAAP version

[![License: MIT](https://img.shields.io/badge/Code%20License-MIT-blue.svg)](LICENSE)
[![License: CC BY 4.0](https://img.shields.io/badge/Data%20License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)


**Author:** Marco Muccioli, MSc student in Environmental Sciences (Atmosphere & Climate) 
**Affiliation**: Institute for Atmosphere & Climate (IAC), ETH Zürich, Zürich 

---

## Overview

This repository contains all scripts and domain grid files associated with the MSc thesis *"Multi-Feature Interactions during Antarctic
Extreme Precipitation Events: A Reanalysis and Polar-CORDEX Intercomparison"* (Muccioli, 2026). The MSc thesis was supervised by: Prof. Andreas Prein (ETH Zürich, Institute for Atmosphere & Climate, ETH Zürich, Zürich) and Research Prof. Priscilla Mooney (NORCE Climate & Environment, Bjerknes Centre for Climate Research, Bergen).

The project develops a **unified multi-feature tracking framework** for the Antarctic region, building on the [MOAAP algorithm](https://github.com/AndreasPrein/MOAAP) of [Prein et al. (2023)](https://doi.org/10.1029/2023EF003534). The framework simultaneously detects and tracks atmospheric rivers (ARs), cyclones (CYs), anticyclones (ACYs), atmospheric fronts (FRs) and upper-level jets (JETs), and links them to extreme precipitation events (EPEs) over Antarctica.

The calibrated algorithm is applied over the 2001-2020 period to:
- **ERA5** reanalysis
- Three **PolarCORDEX high-resolution (0.11°) regional climate models**: HCLIM, MetUM and RACMO

The analysis quantifies both the individual and compound contributions of synoptic-scale features to Antarctic EPEs, while evaluating the added value of high-resolution RCMs in reproducing their atmospheric drivers.

> ⚠️ **Note on reproducibility:** All scripts were originally run on the **Betzy HPC cluster** (SIGMA2, Norwegian Research Infrastructure). Some file paths may still  be cluster-specific and will need to be adapted to your own system and directory structure before running.

---

## Repository Structure

```
.
├── ANT_Regions_Analysis/
├── EPE_Objects/
├── Feature_Climatologies/
├── Features_Visualization/
├── IVT_95th_Percentile/
├── Mean_Precipitation/
├── MOAAP_code/
├── PolarCORDEX_RCM_Embedding/
├── PR_Objects/
├── PR_Seasonal_Percentiles/
├── Domain_Grid_Files/
├── README.md
└── LICENSE
```

---

## Scripts Description

### 1. `ANT_Regions_Analysis/`
Contains all scripts related to the analysis of atmospheric feature occurrence and co-occurrence in top-100 extreme precipitation events, within three IPCC AR6 regions.

| Script | Description |
|--------|-------------|
| `ANT_regions_analysis_objects.py` | Generates CSV files of the top-100 EPEs for the three IPCC regions and the corresponding feature association tables. Requires IPCC reference region shapefiles as input. Works for all datasets. |
| `ANT_regions_cooccurrence_matrices.py` | Generates feature co-occurrence matrices for the top-100 EPEs per region, for ERA5 and RCM–ERA5 differences, with bootstrapping for significance testing. Requires CSV files produced with `ANT_regions_analysis_objects.py`. |
| `ANT_regions_occurrence_radar.py` | Generates radar plots of feature occurrence within the top-100 EPEs for each region, for ERA5 and RCMs, with bootstrapping. Requires CSV files produced with `ANT_regions_analysis_objects.py`. |
| `ANT_regions_plot_100_events_locations.py` | Plots the geographical locations of the top-100 EPEs for all regions and all datasets. |

---

### 2. `EPE_Objects/`
Contains all scripts related to the detection of extreme precipitation objects and their association with atmospheric features.

| Script | Description |
|--------|-------------|
| `EPE_Object_detection_all.py` | Core EPE object detection function. Identifies extreme precipitation objects using seasonal thresholds (99th and 99.9th percentile) of hourly precipitation. Includes a plotting helper for visualisation. Works with ERA5 and PolarCORDEX datasets. |
| `EPE_feature_association_buffer.py` | Implements the buffer-based, non-exclusive association between EPE objects and MOAAP-detected features. Applicable to any dataset. Outputs NPZ files and 6-panel association maps. |
| `EPE_feature_association_buffer_difference_part1.py` | Computes percentage-point differences between a specified RCM and ERA5 in EPE–feature fractional associations. Saves results as NPZ files. |
| `EPE_feature_association_buffer_difference_part2.py` | Follows Part 1. Applies pixel-wise Wilcoxon signed-rank test to the computed differences to produce 6-panel plots. |
| `EPE_feature_association_buffer_composite.py` | Same as 'EPE_feature_association_buffer_difference_part2.py', but for all RCMs simultaneously, producing a single 6×3 composite figure (features × RCMs). Requires Part 1 to have been run for all target RCMs. |
| `EPE_frequency.py` | Computes multi-year frequency of EPE objects (both 99th and 99.9th percentiles) and plots them. Applicable to all datasets individually. |
| `EPE_frequency_composite.py` | Computes multi-year EPE object frequency for both percentiles across all RCMs and produces a composite 3×2 figure. |

---

### 3. `Feature_Climatologies/`
Contains scripts for computation and visualisation of multi-year MOAAP-detected feature climatologies.

| Script | Description |
|--------|-------------|
| `Feature_climatology.py` | Produces annual and seasonal multi-year climatologies of atmospheric features detected by MOAAP. |
| `Feature_climatology_difference.py` | Produces figures showing ERA5 feature climatologies and the relative difference for each RCM, with pixel-wise Wilcoxon significance testing. |

---

### 4. `Features_Visualization/`
Contains scripts for visualisation of case studies and animations.

| Script | Description |
|--------|-------------|
| `plot_case_studies.py` | Generates the case study plots used in the thesis, showing precipitation and feature overlays for selected Antarctic EPEs. Modular and adaptable to additional cases or features. |
| `plot_features_with_1hr_epe_obj.py` | Plots frames or animations of hourly EPE objects and 6-hourly detected features, with the hourly precipitation field as background. Configurable by dataset, percentile, year and timestep range. |

---

### 5. `IVT_95th_Percentile/`
Contains scripts for computation of Integrated Vapour Transport (IVT) and its 95th percentile, used as threshold for AR detection.

| Script | Description |
|--------|-------------|
| `compute_ivte_ivtn.py` | Computes IVT east and north components from wind (ua, va) and specific humidity (hus) using the trapezoidal rule for vertical integration. Memory-efficient: loads one pressure level at a time. Applies a surface mask based on geopotential height. Here written for RACMO2, but logic is transferable to ERA5, HCLIM, and MetUM. Designed to run after `prep_model_data.py`. |
| `ivt_computation.sh` | Computes IVT magnitude from the east/north components for each year using CDO. Represents the first step of the IVT percentile pipeline. |
| `ivt_merge.sh` | Merges annual IVT magnitude files (2001–2020) into a single combined file per dataset, and sets the variable name to `ivt`.Represents the second step of the IVT percentile pipeline. |
| `ivt_percentile.sh` | Computes the 95th percentile of IVT over 2001–2020, using monthly min/max bounds. Final step of the IVT percentile pipeline. |

---

### 6. `Mean_Precipitation/`
Contains scripts for computing and comparing mean annual precipitation across datasets.

| Script | Description |
|--------|-------------|
| `ERA5_mean_annual_pr.py` | Computes mean total annual precipitation for ERA5 on a grid-by-grid basis over the analysis period and generates a geographical plot. |
| `RCM_mean_annual_pr_difference.py` | Computes absolute and relative mean annual precipitation differences between RCMs and ERA5, producing a 6-panel comparison figure. |

---

### 7. `MOAAP_code/`
Core feature detection framework, contains MOAAP adapted for Antarctica.

| Script | Description |
|--------|-------------|
| `Tracking_Functions_ANT.py` | Contains all detection logics, methods, and thresholds for the features tracked by MOAAP (ARs, CYs, ACYs, FRs, JETs). This is the version specifically tuned for Antarctic dynamical and thermodynamical conditions, while retaining the structure of the original global configuration [(Prein et al., 2023)](https://doi.org/10.1029/2023EF003534). |
| `MOAAP_years_all.py` | Imports `Tracking_Functions_ANT` and runs feature detection for specified years and features. Fully configurable: can run all features together or individually. Applicable to all datasets (ERA5 and RCMs). |

---

### 8. `PolarCORDEX_RCM_Embedding/`
Contains all scripts related to ipeline for preprocessing and embedding RCM data within the ERA5 background on the expanded PolarCORDEX domain.

| Script | Description |
|--------|-------------|
| `prep_model_data.py` | Processes geopotential height (zg) files from the specified RCM and years. Converts from metres to geopotential (m² s⁻²) if needed, to align with ERA5 convention. Backs up original files with a `.raw.nc` extension. It is the first step of the embedding pipeline. |
| `prep_model_pr.py` | Processes RCM precipitation (pr) files. Converts units from kg m⁻² s⁻¹ to m (total hourly accumulation) and shifts time labels by +30 min to align with ERA5 convention. Backs up originals. First step of the precipitation embedding pipeline. |
| `remap_model_data.sh` | Remaps RCM output to the common PolarCORDEX grid (`PolarRES_WP3_Antarctic_domain.nc`) using CDO. Iterates over years and variables (including precipitation), applies the chosen remapping method and saves results in a structured directory. It is the second step of the embedding pipeline. |
| `embed_model_data.py` | Embeds RCM data into the ERA5 background for the specified variables and years. Includes a blending mechanism (currently set to 0 points, i.e., no blending) and generates diagnostic plots to check for boundary discontinuities. Represents the final step of the embedding pipeline. |
| `embed_model_pr.py` | Same as `embed_model_data.py`, but optimised for the `pr` variable (1-hourly). Includes a 1-hour temporal trimming step for proper alignment. Generates diagnostic plots for boundary checks. Final step of the precipitation embedding pipeline. |

---

### 9. `PR_Objects/`
Contains scripts for detection of 6-hourly precipitation objects and related computation of feature contributions to mean precipitation.

| Script | Description |
|--------|-------------|
| `PR_Object_6hr_detection.py` | Detects 6-hourly precipitation objects by summing hourly precipitation over 6-hour windows centred on feature detection timesteps (00, 06, 12, 18 UTC). Applicable to all datasets. |
| `PR_features_contribution.py` | Computes the contribution of atmospheric features to mean annual precipitation using a buffer-based approach on PR objects detected by `PR_Object_6hr_detection.py`. Applicable to all datasets. Outputs 6-panel figures. |
| `PR_features_contribution_difference.py` | Computes the percentage-point difference between a specified RCM and ERA5 in feature contributions to mean annual precipitation, with pixel-wise Wilcoxon signed-rank significance testing. Outputs 6-panel figures. |

---

### 10. `PR_Seasonal_Percentiles/`
Contains scripts related to computation of seasonal precipitation percentiles (99th and 99.9th) used for detection of EPE objects.

| Script | Description |
|--------|-------------|
| `pr_percentile_computation_ERA5.sh` | Computes seasonal 99th and 99.9th percentiles of ERA5 hourly precipitation. Processes each year individually, merges into seasonal stacks, and computes the final percentile fields. |
| `pr_percentile_computation_PolarRES.sh` | Same as above, but for PolarCORDEX RCM data. |

---

### Domain Grid Files

| File | Description |
|------|-------------|
| `PolarRES_WP3_Antarctic_Domain.nc` | Standard Antarctic PolarCORDEX domain at 0.11° resolution, centred over Antarctica. |
| `PolarRES_WP3_Antarctic_Domain_expanded.nc` | Expanded version of the PolarCORDEX domain, extended by ~50% in each horizontal direction into the southern mid-latitudes, to capture upstream synoptic features. |
| `grid_expansion.py` | Script used to generate the expanded PolarCORDEX domain from the original grid file. |

---

## Data

The analysis is based on the following datasets:

- **ERA5** reanalysis [(Hersbach et al., 2020)](https://doi.org/10.1002/qj.3803) — available via the [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
- **PolarCORDEX RCMs**: HCLIM, MetUM, RACMO — available via the [PolarRES project](https://polarres.eu/polar-storyline-lens/)
- **IPCC AR6 Reference Regions** shapefiles — available at [IPCC-WG1/Atlas](https://github.com/IPCC-WG1/Atlas)

---

## Requirements
 
The scripts require Python 3.11 and the packages listed in `environment_core.yml`. Reproduce the environment with:
 
```bash
conda env create -f environment_core.yml
conda activate ant_env
```
 
Key dependencies include: `xarray`, `netcdf4`, `cartopy`, `geopandas`, `scipy`, `matplotlib`, `seaborn`, `dask`, `metpy`, `regionmask`, `numba`, and `pandas`.
 
Bash scripts additionally require [CDO (Climate Data Operators)](https://code.mpimet.mpg.de/projects/cdo).

---

## How to Cite

If you use this code or data in your research, please cite:

> Muccioli, M. (2026). *Tracking and Interaction of Atmospheric Phenomena in Antarctic Extreme Precipitation*. MSc Thesis, ETH Zürich. DOI: [10.5281/zenodo.XXXXXXX](https://doi.org/10.5281/zenodo.XXXXXXX)

This framework builds on:

> Prein, A. F., et al. (2023). The Multi-Scale Interactions of Atmospheric Phenomenon in Mean and Extreme Precipitation. *[Earth's Future]*, [11], [e2023EF003534]. DOI: [10.1029/2023EF003534]

---

## License

- **Code:** [MIT License](LICENSE)
- **Data and documentation:** [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/)

---

## Contact

**Marco Muccioli** — mmuccioli@xxx.ethz.ch  
Institute for Atmospheric and Climate Science (IAC), ETH Zürich

Feel free to reach out for questions, feedback, or potential collaborations.
