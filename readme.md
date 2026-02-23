# Dermatology Disease — Class Predictor

A small Flask app that predicts dermatology disease class from clinical and
histopathology features. The web UI is in `templates/index.html`; the app logic
is in `main.py`. A sample dataset is provided as
`dataset_35_dermatology (1).csv` and an interactive notebook is available as
`SKIN DISORDER.ipynb`.

**Not medical advice.** This repository is for demo/research only.

**Files**
- `main.py`: Flask application and prediction endpoint (`/predict`).
- `templates/index.html`: form UI for entering features and showing results.
- `dataset_35_dermatology (1).csv`: CSV dataset used for training/analysis.
- `SKIN DISORDER.ipynb`: exploratory notebook.
- `model.pkl` / `svm.pkl`: (optional) place your trained model here.
- `scaler.pkl`, `preprocessor.pkl`, `label_encoder.pkl`: optional preprocessing
	artifacts the app will load if present.

Quick start
1. Install Python 3.8+ and create a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install flask joblib numpy scikit-learn pandas
```

2. Add a trained model file named `model.pkl` (or `svm.pkl`) in the project
	 root. Optionally add `scaler.pkl`, `preprocessor.pkl`, and
	 `label_encoder.pkl` if your training pipeline used them.

3. Run the app:

```powershell
python main.py
```

4. Open http://127.0.0.1:5000 in your browser and fill the form. All fields are
	 required; most features are integer scores (typically 0–3) and `Age` is
	 numeric.

Notes
- The form field order is important — `main.py` expects inputs in the order
	defined by `FEATURE_ORDER`.
- If no model file is present the app will render an explanatory message.
- The dataset contains a `class` column (target). Some rows contain `?`
	placeholders; clean/validate before training.

Want me to:
- run the Flask app here to verify it starts, or
- create a `requirements.txt` and a minimal test script to call `/predict`?

