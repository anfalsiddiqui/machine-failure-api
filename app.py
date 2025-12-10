import joblib
import gradio as gr
import pandas as pd

# 1️⃣ Load saved model and SHAP explainer
model = joblib.load("model.pkl")
explainer = joblib.load("shap_explainer.pkl")

# 2️⃣ Define features
feature_cols = [
    "Rotational speed [rpm]_scaled",
    "Torque [Nm]_scaled",
    "Tool wear [min]",
    "Torque_x_RotSpeed_scaled",
    "Torque_x_ToolWear_scaled",
    "RotSpeed_x_ToolWear_scaled",
    "Torque_x_RotSpeed_x_ToolWear_scaled"
]

# 3️⃣ Define prediction function
def predict_machine_failure(*inputs):
    # Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs], columns=feature_cols)

    # Model prediction
    prob = model.predict_proba(input_df)[:,1][0]
    label = "Failure" if prob > 0.5 else "No Failure"

    # SHAP values
    shap_values = explainer.shap_values(input_df)
    shap_dict = dict(zip(feature_cols, shap_values[0]))

    return label, prob, shap_dict

# 4️⃣ Build Gradio interface
iface = gr.Interface(
    fn=predict_machine_failure,
    inputs=[gr.Number(label=f) for f in feature_cols],
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Number(label="Failure Probability"),
        gr.JSON(label="SHAP Values")
    ],
    title="Machine Failure Prediction",
    description="Enter sensor readings to predict machine failure and see SHAP explanation."
)

iface.launch()

