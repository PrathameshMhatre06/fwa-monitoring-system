import shap
import pandas as pd


def generate_shap_explanation(calibrated_model, input_df):

    # Extract base XGBoost model
    try:
        base_model = calibrated_model.calibrated_classifiers_[0].estimator
    except:
        base_model = calibrated_model

    # Use modern SHAP API (more stable)
    explainer = shap.Explainer(base_model)

    shap_values = explainer(input_df)

    shap_df = pd.DataFrame({
        "feature": input_df.columns,
        "shap_value": shap_values.values[0]
    })

    shap_df["impact"] = shap_df["shap_value"].abs()
    shap_df = shap_df.sort_values("impact", ascending=False)

    shap_df["direction"] = shap_df["shap_value"].apply(
        lambda x: "Increases Risk" if x > 0 else "Decreases Risk"
    )

    return shap_df
