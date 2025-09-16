import streamlit as st
import pandas as pd
import io, json

from two_stage import (
    prepare_features,
    train_two_stage,
    forecast_next_h,
    order_plan_from_forecast,
    save_artifact,
)

st.set_page_config(page_title="Hyperlocal Product Demand", layout="wide")
st.title("Hyperlocal Product Demand")

# Helper functions
def load_csv(uploaded):
    if uploaded is None:
        return None
    return pd.read_csv(uploaded)

def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

# Sidebar: inputs and guidance
with st.sidebar:
    with st.expander("Upload requirements and data format", expanded=False):
        st.markdown("""
        **Your CSV must include the following columns:**

        - **Required**
          - `date` → daily date in `YYYY-MM-DD` format  
          - `store` → store identifier (e.g., `store_nbr` or `store_id`)  
          - `sku` → product identifier (e.g., `family` or `item_id`)  
          - `sales` → units sold per day (numeric, not revenue)  

        - **Optional (recommended)**
          - `onpromotion`, `transactions`, `oil_price`, `holiday_flag`
        """)

        # Sample CSV download
        example = pd.DataFrame({
            "date": ["2023-01-01","2023-01-01"],
            "store_nbr": ["OUT001","OUT002"],
            "family": ["DRINKS","BREAD"],
            "sales": [12, 5],
            "onpromotion": [0, 1],
        })
        st.download_button(
            label="Download sample CSV",
            data=example.to_csv(index=False).encode("utf-8"),
            file_name="sample_train.csv",
            mime="text/csv"
        )


    st.header("Input files")
    up_train = st.file_uploader("train.csv (single or multi-file workflow)", type=["csv"])
    up_txn   = st.file_uploader("transactions.csv (optional)", type=["csv"])
    up_oil   = st.file_uploader("oil.csv (optional)", type=["csv"])
    up_hol   = st.file_uploader("holidays_events.csv (optional)", type=["csv"])

    st.header("Settings")
    horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=28, value=14, step=1)
    fast = st.checkbox("Fast sampling", value=True)
    max_stores = st.number_input("Max stores (fast)", 5, 1000, 10)
    max_items  = st.number_input("Max items (fast)", 50, 20000, 200)

# Session state defaults
ss = st.session_state
for k, v in {
    "feat": None, "feat_cols": None,
    "artifact": None, "oof": None, "debias": 1.0,
    "forecasts": None
}.items():
    if k not in ss: ss[k] = v

# Step 1: Prepare data
st.header("1) Prepare data")
if st.button("Prepare"):
    if up_train is None:
        st.error("Upload train.csv first.")
    else:
        train_df = load_csv(up_train)
        txn_df   = load_csv(up_txn) if up_txn else None
        oil_df   = load_csv(up_oil) if up_oil else None
        hol_df   = load_csv(up_hol) if up_hol else None

        try:
            feat, feat_cols = prepare_features(
                train_df, txn_df, oil_df, hol_df,
                fast=fast, max_stores=int(max_stores), max_items=int(max_items),
                user_map=None
            )
        except KeyError as ex:
            st.warning(f"Auto-detect failed: {ex}. Map columns and continue.")
            cols = list(train_df.columns)
            with st.form("manual_map", clear_on_submit=True):
                date_col  = st.selectbox("Date column", cols)
                store_col = st.selectbox("Store column", cols)
                sku_col   = st.selectbox("SKU column (family/item)", cols)
                qty_col   = st.selectbox("Quantity/Sales column", cols)
                promo_col = st.selectbox("Promo column (optional)", ["<none>"] + cols, 0)
                submitted = st.form_submit_button("Continue")
            if not submitted:
                st.stop()
            user_map = {"date": date_col, "store": store_col, "sku": sku_col, "qty": qty_col}
            if promo_col != "<none>":
                user_map["promo"] = promo_col
            feat, feat_cols = prepare_features(
                train_df, txn_df, oil_df, hol_df,
                fast=fast, max_stores=int(max_stores), max_items=int(max_items),
                user_map=user_map
            )

        ss["feat"], ss["feat_cols"] = feat, feat_cols
        st.success(f"Prepared rows: {len(feat):,}")
        st.dataframe(feat.head(20))

# Step 2: Train model
st.header("2) Train model")
if st.button("Train"):
    if ss["feat"] is None or ss["feat_cols"] is None:
        st.error("Prepare data first.")
    else:
        artifact, pre, post, oof = train_two_stage(ss["feat"], ss["feat_cols"], horizon=int(horizon))
        ss["artifact"] = artifact
        ss["oof"] = oof
        ss["debias"] = float(post.get("debias_factor_oof", 1.0))
        st.subheader("OOF metrics (pre)")
        st.json(pre)
        st.subheader("OOF metrics (post)")
        st.json(post)

        buf = io.BytesIO()
        save_artifact(artifact, buf)  # joblib accepts a file-like buffer
        st.download_button("Download model artifact", data=buf.getvalue(), file_name="two_stage_model.pkl")

        st.download_button("Download OOF predictions (CSV)", data=to_csv_bytes(oof), file_name="oof_predictions.csv")
        st.download_button("Download debias factor (JSON)",
                           data=json.dumps({"debias_factor_oof": ss["debias"]}, indent=2),
                           file_name="debias_factor_oof.json")

# Step 3: Forecast and plan
st.header("3) Forecast and order plan")

lead_time_days = st.number_input("Lead time (days)", 1, 30, 2, 1)

colA, colB = st.columns(2)
with colA:
    if st.button("Forecast next H days"):
        if ss["artifact"] is None or ss["feat"] is None:
            st.error("Train a model first.")
        else:
            needed = ["store_id","sku_id","date","qty","is_promo","transactions","oil_price","holiday_flag"]
            missing = [c for c in needed if c not in ss["feat"].columns]
            if missing:
                st.error(f"Missing columns for forecasting: {missing}")
            else:
                df_cont = ss["feat"][needed].copy()
                fut = forecast_next_h(df_cont, ss["artifact"], horizon=int(horizon), debias_factor=float(ss["debias"]))
                ss["forecasts"] = fut
                st.success(f"Forecast rows: {len(fut):,}")
                st.dataframe(fut.head(50))
                st.download_button("Download forecasts (CSV)", data=to_csv_bytes(fut),
                                   file_name=f"forecast_next_{horizon}d.csv")

with colB:
    if st.button("Build order plan"):
        if ss["forecasts"] is None:
            st.error("Run forecasting first.")
        elif ss["oof"] is None:
            st.error("Train first to compute residuals.")
        else:
            plan = order_plan_from_forecast(
                forecasts=ss["forecasts"],
                oof=ss["oof"],
                lead_time_days=int(lead_time_days),
                service_z={"A":1.645, "B":1.282, "C":0.842}
            )
            st.dataframe(plan.head(50))
            st.download_button("Download order plan (CSV)", data=to_csv_bytes(plan),
                               file_name=f"order_plan_L{lead_time_days}d.csv")

st.caption("Steps: Prepare → Train → Forecast/Plan.")
