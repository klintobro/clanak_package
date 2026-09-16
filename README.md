# CLANAK Research Code

Research software accompanying the CLANAK dataset and the associated research on Man-in-the-Middle (MitM) detection under TLS, meteorological, and diurnal conditions.
## DOI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22738123.svg)](https://doi.org/10.5281/zenodo.22738123)

**Software DOI:** https://doi.org/10.5281/zenodo.22738123

## Software components

## How to Cite
Obrorindo, I. C. (2026). *CLANAK Research Code: Dataset Generation, CLANAK Algorithm, and Extended Analysis* (Version v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.22738123

## Associated Dataset
The dataset used with this research code is separately archived on Zenodo:
**Dataset DOI:** https://doi.org/10.5281/zenodo.22151094

## Repository Contents
This repository contains the reproducible research implementation of CLANAK, including:
- CLANAK algorithm implementation
- Dataset-generation code
- Extended analysis code
- Google Colab notebooks
- Supporting documentation
- Original/source materials where applicable

The dataset and software are maintained as separate research outputs to support reproducibility and independent citation.
1. **Dataset generation** — Python script originally developed and executed in PyCharm. It measures TLS connection/handshake timing and records meteorological variables obtained from OpenWeatherMap.
2. **CLANAK algorithm** — Google Colab implementation of the leakage-controlled ELM-LSTM ensemble pipeline.
3. **Extended analysis** — Google Colab master analysis covering descriptive/statistical analysis, conventional machine-learning benchmarks, ELM, LSTM, and the ELM-LSTM unanimous ensemble.

## Repository structure

```text
CLANAK-Research-Code/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── dataset_generation/
│   ├── clanak_dataset_generation.py
│   └── README.md
├── clanak_algorithm/
│   ├── CLANAK_Algorithm_Colab.py
│   ├── CLANAK_Algorithm.ipynb
│   └── README.md
├── extended_analysis/
│   ├── CLANAK_Extended_Analysis_Colab.py
│   ├── CLANAK_Extended_Analysis.ipynb
│   └── README.md
├── data/
│   └── README.md
├── docs/
│   └── reproducibility.md
└── original_sources/
    ├── dataset_code_original.txt
    ├── clanak_algorithm_original.txt
    └── clanak_extended_analysis_original.txt
```

## Data availability

The dataset is archived separately from this software repository.

- **Zenodo dataset:** https://doi.org/10.5281/zenodo.22151093
- **Kaggle dataset:** https://doi.org/10.34740/kaggle/dsv/19509973

The Zenodo record is the archival dataset source; the Kaggle record provides a corresponding distribution for dataset exploration. The GitHub repository is intended for source code and reproducibility materials, not as the authoritative dataset archive.

## Execution environments

| Component | Original development/execution environment |
|---|---|
| Dataset generation | Python / PyCharm |
| CLANAK algorithm | Google Colaboratory (Google Colab) |
| Extended analysis | Google Colaboratory (Google Colab) |

The two Colab components are supplied both as `.ipynb` notebooks and Colab-oriented `.py` exports. The dataset-generation component is supplied as a standard Python script.

## Dataset generation

The dataset-generation program asks for a target in the form `host:port`, creates an SSL/TLS connection, measures elapsed connection time, queries OpenWeatherMap, and writes the observations to `mycsv.csv`.

Before running it, set your own OpenWeatherMap credentials. The publication-ready script uses environment variables:

```text
OPENWEATHERMAP_API_KEY
OPENWEATHERMAP_CITY
```

Example on Windows PowerShell:

```powershell
$env:OPENWEATHERMAP_API_KEY="YOUR_API_KEY"
$env:OPENWEATHERMAP_CITY="YOUR_CITY"
python dataset_generation/clanak_dataset_generation.py
```

**Do not commit an API key to GitHub.** The public source contains no secret credential.

## CLANAK algorithm

Open `clanak_algorithm/CLANAK_Algorithm.ipynb` in Google Colab and run the notebook sequentially. The source pipeline implements:

- D1: TLS only
- D2: TLS + meteorological variables
- D3: TLS + meteorological variables + cyclic time
- train-only imputation
- train-only MinMax scaling
- train-only SMOTE
- train-only information-gain feature selection for ELM
- train-only PCA for LSTM
- stratified 5-fold cross-validation
- ELM and LSTM predictions
- unanimous AND ensemble: attack only when both classifiers predict attack
- CSV, Excel, confusion-matrix and ZIP result generation

Random seed: **42**.

For the authoritative parameter values, use the executable notebook/source included in this release. The separate Extended Analysis component has its own documented parameterization.

## Extended analysis

Open `extended_analysis/CLANAK_Extended_Analysis.ipynb` in Google Colab and run the cells sequentially. The supplied master code identifies the following principal settings:

- ELM hidden neurons: **110**
- LSTM hidden neurons: **128**
- PCA maximum components: **6**
- Maximum LSTM epochs: **30**
- Early stopping patience: **5**
- Stratified 5-fold cross-validation
- Random seed: **42**
- ELM-LSTM ensemble rule: **ELM = 1 AND LSTM = 1**

The extended analysis source also documents a final cleaned dataset of 10,595 observations after removal of 5,193 blank/padded records and 212 exact duplicates from 16,000 raw records. These figures describe the cleaning logic documented in that analysis code and should be checked against the final archived dataset before being quoted as the authoritative dataset size.

## Software and dataset licensing

The code is released under the **MIT License**. This software license is intentionally separate from the dataset license. The dataset is distributed under **CC BY 4.0**.

The two licenses apply to different research outputs and should not be treated as interchangeable.

## Citation

Please cite the software release using the metadata in `CITATION.cff`. After the GitHub repository is created and the release is archived by Zenodo, the repository URL and software DOI should be added to `CITATION.cff` and this README.

## Provenance and original source preservation

The `original_sources/` directory preserves the three supplied source-text files as received for this packaging exercise. The publication-ready dataset-generation script contains only non-scientific safety/portability changes: API credentials are read from environment variables, the optional OpenWeatherMap rain field is handled safely, and an unsuccessful weather request raises an explicit error. The Colab algorithm and extended-analysis source code are preserved without scientific-method changes.

## Reproducibility

The original source files did not record complete package-version information. Therefore, `requirements.txt` records the packages required by the imports rather than falsely claiming exact historical package versions. Before the first public release, the software should be tested in a clean environment and the verified environment should be captured with `pip freeze` (or an equivalent environment manifest).

For the Colab components, users should run the supplied notebooks sequentially. For dataset generation, users must provide their own OpenWeatherMap API credentials.

## Author

**Dr. Immunhierokene Clinton Obrorindo**  
Petroleum Training Institute, Effurun, Delta State, Nigeria

Research interests include cybersecurity, machine learning, network security, anomaly detection, and intelligent cyber-physical systems.
