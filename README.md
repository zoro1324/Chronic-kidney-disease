# Chronic Kidney Disease Prediction

This project trains machine learning models to predict Chronic Kidney Disease (CKD) from clinical and laboratory measurements. The notebook (`Chronic_kidney_disease.ipynb`) contains the full analysis: data loading, preprocessing, feature selection, model training/evaluation, and a small Gradio demo to interact with the best model.

## Repository structure

- `Chronic_kidney_disease.ipynb` - Main analysis notebook (data cleaning, EDA, modeling, Gradio demo).
- `content/kidney_disease.csv` - The dataset used for training and evaluation.
- `README.md` - This file.

## Dataset

The dataset includes clinical and laboratory features commonly used to diagnose kidney disease, such as specific gravity, albumin, hemoglobin, packed cell volume, white/red blood cell counts, and comorbidities like hypertension and diabetes.

Source: included in `content/kidney_disease.csv`.

## What the notebook does

Key steps implemented in `Chronic_kidney_disease.ipynb`:

- Load dataset with pandas and inspect shape/info.
- Rename columns for readability (for example `bp` -> `Blood_Pressure`, `hemo` -> `Hemoglobin`, etc.).
- Convert some columns to numeric (e.g., `Packed_Cell_Volume`, `White_Blood_Cell_Count`, `Red_Blood_Cell_Count`) and coerce invalid entries to NaN.
- Identify and clean inconsistent string labels (for example trimming stray tabs and leading spaces in `classification` and `Diabetes_Mellitus`).
- Impute missing values separately for numeric and categorical features by using group-level (by `classification`) statistics: numeric columns filled with group mean (fallback to mode), categorical columns filled with group mode.
- Encode categorical/object columns with sklearn's `LabelEncoder`.
- Plot a correlation heatmap to inspect relationships between features.
- Split features (`X`) and target (`y` = `classification`) and use `SelectKBest` (ANOVA f-test) to pick the top 7 features.
- Train several classifiers and evaluate them on a test split:
	- Random Forest (default 100 trees)
	- Decision Tree
	- K-Nearest Neighbors
	- Support Vector Classifier (linear kernel)
	- Gaussian Naive Bayes

- The notebook prints evaluation metrics for each model: accuracy, F1 score, confusion matrix, precision, and recall. The Random Forest model was selected as the best model in the notebook.

## Gradio demo

The notebook includes a Gradio-based UI to collect a small set of input features and run a prediction with the trained Random Forest model. The demo expects numeric inputs for features like Specific Gravity, Albumin, Hemoglobin, Packed Cell Volume, Red Blood Cell Count, and radio buttons for Hypertension and Diabetes Mellitus (yes/no). The prediction function maps those inputs to the model and returns a human-readable message.

Additionally, a standalone Gradio app is provided in `gradio_app.py` (and a notebook version `gradio_app.ipynb`). The app loads saved artifacts from the `models/` directory and runs predictions via a web UI.

Saved artifacts

- `models/best_model.joblib` - the trained Random Forest model as saved by the notebook.
- `models/label_encoders.joblib` - a dict of per-column sklearn LabelEncoders used to reproduce training-time categorical mappings.
- `models/pipeline.joblib` - (optional) a full sklearn Pipeline including preprocessing and model. The Gradio app will prefer the pipeline if present, otherwise it uses the model+encoders flow.

## How to run locally

1. Create a Python environment (recommended: Python 3.8+).

2. Install dependencies. A minimal `requirements.txt` is provided in this repo. Example using PowerShell:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
```

3. Open the notebook `Chronic_kidney_disease.ipynb` and run all cells (or run the notebook in Google Colab using the badge link present in the notebook).

4. To launch the Gradio demo from the notebook, re-run the training cells (or load a pre-trained model if you save one), then run the final Gradio interface cell. Gradio will print a local URL where the UI can be accessed.

### Run the standalone Gradio app

If you'd rather run a standalone app (recommended for quick local testing), run the provided `gradio_app.py` script. It expects the model artifacts to exist in the `models/` directory.

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1; pip install -r requirements.txt
python gradio_app.py
```

After starting, open the URL printed by Gradio (typically http://127.0.0.1:7861) in your browser.

## Notes, assumptions and small fixes

- The notebook performs label encoding in-place on all object columns; that mapping is not persisted. If you plan to export the model, save the LabelEncoder(s) or use one-hot/dedicated mapping so the same encoding can be applied to new data.
- The repository now saves LabelEncoders to `models/label_encoders.joblib`. When you run the Gradio app it will attempt to load these encoders and the saved model. Make sure to keep the encoders and model together so categorical mappings stay consistent.
- You might see scikit-learn version mismatch warnings when loading saved artifacts (e.g., model pickled with scikit-learn 1.6.1 and your environment uses 1.7.x). Two safe options:
	- Install the scikit-learn version the artifacts were created with (e.g., `pip install scikit-learn==1.6.1`) to avoid warnings and potential incompatibilities.
	- Re-run the training cells in your current environment and re-save the artifacts so they match your installed sklearn version.
- The notebook fills missing numeric values using group means per `classification`. If you want to deploy this model, prefer a consistent imputation pipeline (e.g., sklearn's SimpleImputer) fit on training data only.
- Some column renames in the notebook appear to have small typos (for example `Blood_Unicorn`, `Specific_Chromatin`, `Ann_Artery_Disease`) — they don't affect the later pipeline because selected features are chosen after cleaning. You may want to clean the rename mappings if you reuse the dataset.

## Requirements

See `requirements.txt` for the primary packages used to run the notebook and demo.


## Acknowledgements

- Project based on a publicly available CKD dataset and common ML patterns for classification tasks.
