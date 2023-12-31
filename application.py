import pickle
from flask import Flask, request, render_template
import numpy as np

application= Flask(__name__)
app=application
model_rf = pickle.load(open("Random_forest_model.pkl", "rb"))
model_xgb = pickle.load(open("XGBoost_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index_thyroid.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = list(request.form.values())
    features_np = np.array(features).reshape(1, -1)
    pred_1 = model_rf.predict(features_np)
    pred_2 = model_xgb.predict(features_np)

    prediction = (pred_1 + pred_2) / 2
    prediction = np.where(prediction >= 0.49, 1, 0)
    
    if prediction == 0:
        result = "Good News! You are free from thyroidal disease."
    else:
        result = "Our model has predicted that you have thyroidal disease."
        
    return render_template("index_thyroid.html", predict=result)

if __name__ == "__main__":
    app.run(host="0.0.0.0")
