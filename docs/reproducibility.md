# Reproducibility notes

## Environments

- Dataset generation: Python / PyCharm.
- CLANAK algorithm: Google Colaboratory.
- Extended analysis: Google Colaboratory.

## Randomness

The analysis sources set the random seed to **42** for NumPy, Python's `random`, and TensorFlow.

## Data transformations

The corrected CLANAK algorithm documents train-only fitting of imputation, MinMax scaling, SMOTE, information-gain feature selection, and PCA. This is intended to prevent information from the test fold leaking into training transformations.

## PCA and neural-network settings

The extended-analysis source explicitly states a maximum of 6 PCA components, LSTM size 128, maximum 30 epochs, and early-stopping patience of 5. The separate CLANAK algorithm source should be treated according to its own executable parameters; do not infer that its PCA setting is identical to the extended-analysis source without checking the relevant code section.

## Version capture recommendation

The supplied files do not contain complete historical package-version information. Before creating the first archival release, reproduce the code in a clean environment and capture the installed versions, for example:

```bash
python --version
pip freeze > requirements-lock.txt
```

Add that lock file to the release if the reproduction run is successful.
