import json
import os

import joblib
import numpy as np
from flask import Flask, jsonify, request

MODEL_DIR = os.getenv("MODEL_DIR", "/models")

app = Flask(__name__)
model = joblib.load(os.path.join(MODEL_DIR, "model.pkl"))
with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
    metrics = json.load(f)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/model-info", methods=["GET"])
def model_info():
    return jsonify(metrics)


@app.route("/predict", methods=["POST"])
def predict():
    body = request.get_json(silent=True)
    if not body or "features" not in body:
        return jsonify({"error": "Send JSON like {\"features\": [30 numbers]}"}), 400

    X = np.array(body["features"], dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.shape[1] != metrics["n_features"]:
        return jsonify({"error": f"Expected {metrics['n_features']} features, got {X.shape[1]}"}), 400

    preds = model.predict(X)
    probs = model.predict_proba(X)
    results = [
        {"prediction": metrics["classes"][int(p)], "probability": round(float(prob[int(p)]), 4)}
        for p, prob in zip(preds, probs)
    ]
    return jsonify({"results": results})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)