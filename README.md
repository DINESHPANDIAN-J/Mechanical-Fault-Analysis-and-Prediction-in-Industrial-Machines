##  Machine Condition Prediction using Random Forest

This project uses a trained **Random Forest Classifier** to predict the condition of a machine based on key parameters like temperature, vibration, oil quality, RPM, and more.

---

### Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

### Files to be Used for Prediction

* **`random_forest_model.pkl`** – Trained Random Forest classifier.
* **`scaler.pkl`** – Scikit-learn `StandardScaler` used to normalize input features.
* **`selected_features.pkl`** – List of feature names used during training (to enforce correct input order).

These files must be present in the working directory or correctly referenced in your prediction script.

---

### How Prediction Works

1. **Load the Model & Scaler:**

   * `joblib.load('random_forest_model.pkl')` loads the trained model.
   * `joblib.load('scaler.pkl')` loads the feature scaler.
   * `joblib.load('selected_features.pkl')` ensures correct column ordering.

2. **Prepare Input Data:**

   * Create a `pandas.DataFrame` with a **single row** containing all selected features.
   * Ensure all feature names match exactly and are present.

3. **Preprocess the Input:**

   * Use the loaded `scaler` to transform the input data to match training distribution.

4. **Predict Output:**

   * Call `.predict()` to get the predicted class.
   * Call `.predict_proba()` to get class probability scores.

---

### Running Predictions

Use the following template in your `predict.py`:

```python
import joblib
import pandas as pd

# Load model, scaler, and selected feature list
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Replace this with actual user input
new_data = pd.DataFrame([{
    'Temperature_C': 75,
    'Vibration_mm_s': 2.5,
    'Oil_Quality_Index': 88,
    'RPM': 1500,
    'Pressure_bar': 5.2,
    'Shaft_Misalignment_deg': 0.3,
    'Noise_dB': 70,
    'Load_%': 85,
    'Power_kW': 12.5
}])

# Ensure correct feature order
new_data = new_data[selected_features]

# Scale input data
scaled_data = scaler.transform(new_data)

# Make predictions
prediction = model.predict(scaled_data)
prediction_proba = model.predict_proba(scaled_data)

print("Predicted Class:", prediction[0])
print("Prediction Probabilities:", prediction_proba[0])
```

---

### Notes

* Always make sure your new input contains **exactly the same features** as used in training.
* Feature values should be within the expected range used during model training.
* Feature order matters — do not shuffle columns.

---

### Optional: Model Retraining

To retrain or update the model:

* Use the same preprocessing pipeline.
* Ensure consistent feature engineering and scaling.
* Re-save the updated model, scaler, and selected features using `joblib`.

---

### Example Use-Cases

* Predict if machinery is in **normal** or **faulty** condition.
* Use in manufacturing plants, maintenance diagnostics, or IoT sensor applications.
