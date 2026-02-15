from flask import Flask, render_template, request
import numpy as np
import pickle
import json

app = Flask(__name__)

# ---------------- LOAD MODELS ----------------

with open("KNN_MODEL.pkl", "rb") as f:
    KNN_MODEL = pickle.load(f)

with open("NB_MODEL.pkl", "rb") as f:
    NB_MODEL = pickle.load(f)


# ---------------- LOAD JSON FUNCTION ----------------

def load_json(file_name):
    with open(file_name, "r") as f:
        return json.load(f)


# ---------------- HOME PAGE ----------------

@app.route("/")
def home():
    return render_template(
        "index.html",
        prediction="",
        train_accuracy="",
        train_confusion_matrix="",
        train_classification_report="",
        test_accuracy="",
        test_confusion_matrix="",
        test_classification_report=""
    )


# ---------------- PREDICT ----------------

@app.route("/predict", methods=["POST"])
def predict():

    # -------- INPUT VALUES --------
    sl = float(request.form["sl"])
    sw = float(request.form["sw"])
    pl = float(request.form["pl"])
    pw = float(request.form["pw"])

    model_selected = request.form.get("model")

    features = np.array([[sl, sw, pl, pw]])

    # -------- MODEL SELECTION --------
    if model_selected == "knn":
        model = KNN_MODEL
        train_json = load_json("knn_train.json")
        test_json = load_json("knn_test.json")

    else:
        model = NB_MODEL
        train_json = load_json("NB_train.json")
        test_json = load_json("NB_test.json")

    # -------- PREDICTION --------
    result = model.predict(features)[0]

    if result == 0:
        prediction = "Setosa"
    elif result == 1:
        prediction = "Versicolor"
    else:
        prediction = "Virginica"

    # -------- SEND DATA TO HTML --------
    return render_template(
        "index.html",

        prediction=prediction,

        train_accuracy=train_json.get("train_accuracy"),
        train_confusion_matrix=train_json.get("confusion_matrix"),
        train_classification_report=train_json.get("classfication_report"),

        test_accuracy=test_json.get("test_accuracy"),
        test_confusion_matrix=test_json.get("confusion_matrix"),
        test_classification_report=test_json.get("classification_report")
    )


# ---------------- RUN SERVER ----------------

if __name__ == "__main__":
    app.run(debug=True)