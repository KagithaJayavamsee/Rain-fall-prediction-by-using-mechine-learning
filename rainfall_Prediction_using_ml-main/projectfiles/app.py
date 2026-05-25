from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load pipeline
# Load trained model
pipeline = joblib.load(
    r"D:\rainfall_Prediction_using_ml-main\projectfiles\rainfall_model_Today.pkl"
)

# Features (must match training order)
features = [
    "Rainfall", "WindGustSpeed", "WindSpeed9am", "WindSpeed3pm",
    "Humidity9am", "Humidity3pm", "Pressure9am", "Pressure3pm",
    "Temp9am", "Temp3pm", "Cloud9am"
]

@app.route("/")
def index():
    return render_template("index.html", features=features)
@app.route("/form")
def form():
    return render_template("form.html", features=features)
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect input
        values = [float(request.form[f]) for f in features]
        input_df = pd.DataFrame([values], columns=features)

        # Predict today
        prediction = pipeline.predict(input_df)[0]

        if prediction == 1:
            result = "🌧️ Chances Of Rain Tomorrow"
            image = "rain.jpg"
        else:
            result = "☀️ No Chances Of Rain Tomorrow"
            image = "beach.jpg"

        return render_template("result.html", prediction=result, image=image)
    except Exception as e:
        return f"❌ Error in prediction: {e}"

if __name__ == "__main__":
    app.run(debug=True)
