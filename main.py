# main.py
import os
import traceback
import numpy as np
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# The form posts to /predict and expects "prediction_text" to be rendered in index.html.
# Field order mirrors the HTML form exactly. :contentReference[oaicite:1]{index=1}
FEATURE_ORDER = [
    # Clinical Features
    "erythema",
    "scaling",
    "definite_borders",
    "itching",
    "koebner_phenomenon",
    "polygonal_papules",
    "follicular_papules",
    "oral_mucosal_involvement",
    "knee_and_elbow_involvement",
    "scalp_involvement",
    "family_history",

    # Histopathology Features
    "melanin_incontinence",
    "eosinophils_in_the_infiltrate",
    "PNL_infiltrate",
    "fibrosis_of_the_papillary_dermis",
    "exocytosis",
    "acanthosis",
    "hyperkeratosis",
    "parakeratosis",
    "clubbing_of_the_rete_ridges",
    "elongation_of_the_rete_ridges",
    "thinning_of_the_suprapapillary_epidermis",
    "spongiform_pustule",
    "munro_microabcess",
    "focal_hypergranulosis",
    "disappearance_of_the_granular_layer",
    "vacuolisation_and_damage_of_basal_layer",
    "spongiosis",
    "saw-tooth_appearance_of_retes",     # note the hyphen, matches HTML name exactly
    "follicular_horn_plug",
    "perifollicular_parakeratosis",
    "inflammatory_monoluclear_inflitrate",
    "band-like_infiltrate",              # note the hyphen, matches HTML name exactly

    # Demographics
    "Age",
]

# Try to load commonly used artifacts if present.
MODEL_PATHS = ["model.pkl", "svm.pkl"]
SCALER_PATHS = ["scaler.pkl"]
PREPROCESSOR_PATHS = ["preprocessor.pkl"]
LABEL_ENCODER_PATHS = ["label_encoder.pkl", "target_encoder.pkl"]

def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def load_artifacts():
    model = None
    scaler = None
    preprocessor = None
    target_encoder = None

    mp = _first_existing(MODEL_PATHS)
    if mp:
        model = joblib.load(mp)

    sp = _first_existing(SCALER_PATHS)
    if sp:
        scaler = joblib.load(sp)

    pp = _first_existing(PREPROCESSOR_PATHS)
    if pp:
        preprocessor = joblib.load(pp)

    lp = _first_existing(LABEL_ENCODER_PATHS)
    if lp:
        target_encoder = joblib.load(lp)

    return model, scaler, preprocessor, target_encoder


MODEL, SCALER, PREPROCESSOR, TARGET_ENCODER = load_artifacts()


@app.route("/", methods=["GET"])
def index():
    # Make sure your index.html is placed at: ./templates/index.html
    # It expects an optional `prediction_text` variable. :contentReference[oaicite:2]{index=2}
    return render_template("index.html")


def parse_inputs(form):
    values = []
    missing = []
    for key in FEATURE_ORDER:
        raw = form.get(key, "").strip()
        if raw == "":
            missing.append(key)
            values.append(np.nan)
            continue
        try:
            # Most are integers (0–3), Age is numeric; float covers both. :contentReference[oaicite:3]{index=3}
            values.append(float(raw))
        except Exception:
            missing.append(key)
            values.append(np.nan)
    return np.array(values, dtype=float), missing


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row, missing = parse_inputs(request.form)

        if len(missing) > 0 or np.isnan(row).any():
            return render_template(
                "index.html",
                prediction_text=f"Please fill all fields with valid numbers. Missing/invalid: {', '.join(missing)}"
            )

        X = row.reshape(1, -1)

        # Apply preprocessing if available
        if PREPROCESSOR is not None:
            X_trans = PREPROCESSOR.transform(X)
        else:
            X_trans = X
            if SCALER is not None:
                X_trans = SCALER.transform(X_trans)

        # Ensure a model exists
        if MODEL is None:
            # No model file found: return a clear message instead of crashing.
            return render_template(
                "index.html",
                prediction_text=(
                    "Model file not found (expected one of: "
                    f"{', '.join(MODEL_PATHS)}). Add your trained model to the app folder."
                ),
            )

        # Predict
        y_pred = MODEL.predict(X_trans)
        # Optionally decode the class label
        if TARGET_ENCODER is not None:
            try:
                y_pred = TARGET_ENCODER.inverse_transform(y_pred)
            except Exception:
                # if y_pred is shape (1,), keep as-is
                pass

        # Format output
        pred_label = y_pred[0] if np.ndim(y_pred) > 0 else y_pred
        return render_template("index.html", prediction_text=f"{pred_label}")

    except Exception as e:
        # Show a concise error to the user; log full traceback to console.
        traceback.print_exc()
        return render_template("index.html", prediction_text=f"Error during prediction: {e}")


if __name__ == "__main__":
    # Use host='0.0.0.0' for container platforms; change port if needed.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
