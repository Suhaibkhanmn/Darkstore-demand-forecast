# Quick import and signature check so Streamlit does not crash on missing names.
import two_stage

required = [
    "prepare_features",
    "train_two_stage",
    "forecast_next_h",
    "order_plan_from_forecast",
]

print("Imported two_stage module")
for name in required:
    print(f"{name}: {'OK' if hasattr(two_stage, name) else 'MISSING'}")
