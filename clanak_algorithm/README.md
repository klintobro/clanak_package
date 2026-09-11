# CLANAK algorithm

**Original execution environment:** Google Colaboratory (Google Colab).

Open `CLANAK_Algorithm.ipynb` in Google Colab. The notebook expects the four CLANAK CSV files and uses Google Colab's file-upload interface.

The supplied implementation is the corrected leakage-controlled ELM-LSTM pipeline. It applies imputation, scaling, SMOTE, information-gain feature selection, and PCA within the training folds rather than fitting these transformations on the full dataset before cross-validation.

The source code uses D1, D2 and D3 feature configurations and a unanimous ELM/LSTM decision rule.
