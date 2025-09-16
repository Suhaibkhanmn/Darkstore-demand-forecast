from __future__ import annotations
from typing import Optional, List, Tuple, Dict, Any

import numpy as np
import pandas as pd
import inspect

from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import  StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

_OHE_KW = {"handle_unknown": "ignore"}
if "sparse_output" in inspect.signature(OneHotEncoder).parameters:
    _OHE_KW["sparse_output"] = False    # sklearn >= 1.4
else:
    _OHE_KW["sparse"] = False           # sklearn < 1.4


# =========================
# 0) Utilities
# =========================

def _autodetect_and_standardize(
    train_df: pd.DataFrame,
    user_map: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Normalize input columns to a canonical schema and keep useful extras.

    Produces these columns:
      - date, store_id, sku_id, qty, is_promo

    If the input already includes exogenous variables, they are preserved:
      - transactions, oil_price, holiday_flag
    """
    df = train_df.copy()

    if user_map is None:
        cols = {c.lower(): c for c in df.columns}
        date_col  = cols.get("date")
        store_col = cols.get("store_nbr") or cols.get("store_id") or cols.get("store")
        sku_col   = cols.get("family") or cols.get("item_nbr") or cols.get("sku_id") or cols.get("sku")
        qty_col   = cols.get("sales") or cols.get("qty") or cols.get("quantity")
        promo_col = cols.get("onpromotion") or cols.get("is_promo") or None
        if not all([date_col, store_col, sku_col, qty_col]):
            raise KeyError("Could not autodetect one of: date/store/sku/qty columns.")
    else:
        date_col  = user_map["date"]
        store_col = user_map["store"]
        sku_col   = user_map["sku"]
        qty_col   = user_map["qty"]
        promo_col = user_map.get("promo")

    out = pd.DataFrame({
        "date": pd.to_datetime(df[date_col]),
        "store_id": df[store_col],
        "sku_id": df[sku_col],
        "qty": pd.to_numeric(df[qty_col], errors="coerce").astype(float),
    })
    out["is_promo"] = (
        pd.to_numeric(df[promo_col], errors="coerce").fillna(0).astype(int)
        if (promo_col and promo_col in df.columns)
        else 0
    )

    # Keep optional exogenous columns if already present in the single CSV
    if "transactions" in df.columns:
        out["transactions"] = pd.to_numeric(df["transactions"], errors="coerce")
    if "oil_price" in df.columns:
        out["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce")
    if "holiday_flag" in df.columns:
        out["holiday_flag"] = (pd.to_numeric(df["holiday_flag"], errors="coerce") > 0).astype(int)

    return out


def _merge_side_inputs(
    base: pd.DataFrame,
    txn_df: Optional[pd.DataFrame],
    oil_df: Optional[pd.DataFrame],
    hol_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    df = base.copy()

    # transactions: if already present (single CSV) keep; otherwise merge or default to 0
    if "transactions" not in df.columns:
        if txn_df is not None:
            t = txn_df.copy()
            if "date" not in t.columns:
                raise KeyError("transactions.csv must have a 'date' column.")
            t["date"] = pd.to_datetime(t["date"])
            store_col = "store_nbr" if "store_nbr" in t.columns else ("store_id" if "store_id" in t.columns else None)
            if store_col and "transactions" in t.columns:
                t = t.rename(columns={store_col: "store_id"})
                df = df.merge(t[["date","store_id","transactions"]], on=["date","store_id"], how="left")
            elif "transactions" in t.columns:
                tx = t.groupby("date", as_index=False)["transactions"].sum()
                df = df.merge(tx, on="date", how="left")
            else:
                df["transactions"] = 0
        else:
            df["transactions"] = 0

    # oil_price: if present keep; else merge or fill
    if "oil_price" not in df.columns:
        if oil_df is not None:
            o = oil_df.copy()
            if not {"date","dcoilwtico"}.issubset(o.columns):
                raise KeyError("oil.csv must have columns 'date' and 'dcoilwtico'.")
            o["date"] = pd.to_datetime(o["date"])
            o = o.sort_values("date")
            o["dcoilwtico"] = o["dcoilwtico"].ffill()
            df = df.merge(o.rename(columns={"dcoilwtico": "oil_price"})[["date","oil_price"]], on="date", how="left")
        else:
            df["oil_price"] = np.nan

    # holiday_flag: if present keep; else merge or default 0
    if "holiday_flag" not in df.columns:
        if hol_df is not None:
            h = hol_df.copy()
            if "date" not in h.columns:
                raise KeyError("holidays_events.csv must have a 'date' column.")
            h["date"] = pd.to_datetime(h["date"])
            flag = h.groupby("date").size().rename("holiday_hits").reset_index()
            flag["holiday_flag"] = (flag["holiday_hits"] > 0).astype(int)
            df = df.merge(flag[["date","holiday_flag"]], on="date", how="left")
        else:
            df["holiday_flag"] = 0

    # final clean
    df["transactions"] = pd.to_numeric(df["transactions"], errors="coerce").fillna(0).astype(int)
    df["oil_price"] = pd.to_numeric(df["oil_price"], errors="coerce").ffill().bfill().fillna(0.0)
    df["holiday_flag"] = (pd.to_numeric(df["holiday_flag"], errors="coerce") > 0).astype(int)

    return df


def _make_continuous_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Create a continuous daily series per (store_id, sku_id) and fill gaps.

    Missing sales are set to 0; exogenous features are forward/backward filled.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    pairs = df[["store_id","sku_id"]].drop_duplicates()
    dmin, dmax = df["date"].min(), df["date"].max()
    all_days = pd.DataFrame({"date": pd.date_range(dmin, dmax, freq="D")})
    out_list = []
    for _, row in pairs.iterrows():
        sub = df[(df["store_id"]==row["store_id"]) & (df["sku_id"]==row["sku_id"])]
        sub = all_days.merge(sub, on="date", how="left")
        sub["store_id"] = sub["store_id"].fillna(row["store_id"])
        sub["sku_id"]   = sub["sku_id"].fillna(row["sku_id"])
        sub["qty"] = sub["qty"].fillna(0.0)
        sub["is_promo"] = sub["is_promo"].fillna(0).astype(int)
        sub["transactions"] = pd.to_numeric(sub["transactions"], errors="coerce").ffill().bfill().fillna(0)
        sub["oil_price"]    = pd.to_numeric(sub["oil_price"], errors="coerce").ffill().bfill().fillna(0)
        sub["holiday_flag"] = (pd.to_numeric(sub["holiday_flag"], errors="coerce") > 0).astype(int)
        out_list.append(sub)
    out = pd.concat(out_list, ignore_index=True).sort_values(["store_id","sku_id","date"]).reset_index(drop=True)
    return out


def _add_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Add lagged, rolling, and calendar features.

    Returns a trainable DataFrame and a dictionary of feature column groups.
    """
    d = df.copy()
    d["dow"] = d["date"].dt.dayofweek
    d["dom"] = d["date"].dt.day
    d["week"] = d["date"].dt.isocalendar().week.astype(int)
    d["month"] = d["date"].dt.month
    d["year"] = d["date"].dt.year

    d = d.sort_values(["store_id","sku_id","date"])
    g = d.groupby(["store_id","sku_id"])["qty"]
    d["lag_1"]  = g.shift(1)
    d["lag_7"]  = g.shift(7)
    d["lag_14"] = g.shift(14)
    d["lag_28"] = g.shift(28)
    d["rm7"]  = g.shift(1).rolling(7).mean()
    d["rm28"] = g.shift(1).rolling(28).mean()

    num1 = [
        "is_promo","transactions","oil_price","holiday_flag",
        "dow","dom","week","month","year",
        "lag_1","lag_7","lag_14","lag_28","rm7","rm28"
    ]
    num2 = ["is_promo","dow","month","lag_1","rm7"]
    catc = ["store_id","sku_id"]

    d_trainable = d.dropna(subset=["lag_1","rm7"]).copy()

    feature_cols = {
        "num1": [c for c in num1 if c in d_trainable.columns],
        "num2": [c for c in num2 if c in d_trainable.columns],
        "cat":  catc
    }
    return d_trainable, feature_cols


# =========================
# 1) Public API – PREPARE
# =========================

def prepare_features(
    train_df: pd.DataFrame,
    txn_df: Optional[pd.DataFrame] = None,
    oil_df: Optional[pd.DataFrame] = None,
    hol_df: Optional[pd.DataFrame] = None,
    fast: bool = False,
    max_stores: int = 10,
    max_items: int = 200,
    user_map: Optional[Dict[str, str]] = None
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    base = _autodetect_and_standardize(train_df, user_map=user_map)
    if fast:
        top_stores = base.groupby("store_id")["qty"].sum().sort_values(ascending=False).head(max_stores).index
        base = base[base["store_id"].isin(top_stores)]
        top_items = base.groupby("sku_id")["qty"].sum().sort_values(ascending=False).head(max_items).index
        base = base[base["sku_id"].isin(top_items)]

    df = _merge_side_inputs(base, txn_df, oil_df, hol_df)
    df = _make_continuous_daily(df)
    feat, feat_cols = _add_features(df)
    return feat, feat_cols


# =========================
# 2) Public API – TRAIN
# =========================

def train_two_stage(
    feat: pd.DataFrame,
    feat_cols: Dict[str, List[str]],
    horizon: int = 14
) -> Tuple[Dict[str, Any], Dict[str, float], Dict[str, float], pd.DataFrame]:
    """
    Train a two-stage model:
      - Stage 1: HistGradientBoostingRegressor on the full feature set
      - Stage 2: Ridge regression fallback when some features are missing

    Uses the last 14 days as a validation window.

    Returns an artifact with models and preprocessors, pre/post metrics
    (post includes a debias_factor_oof), and out-of-fold predictions.
    """
    df = feat.copy()
    last_date = df["date"].max()
    val_start = last_date - pd.Timedelta(days=14)
    trn = df[df["date"] < val_start].copy()
    val = df[df["date"] >= val_start].copy()

    num1, num2, catc = feat_cols["num1"], feat_cols["num2"], feat_cols["cat"]

        # --- preprocessing (version-proof one-hot) ---
    pre1 = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num1),
            ("cat", OneHotEncoder(**_OHE_KW), catc),
        ]
    )
    pre2 = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num2),
            ("cat", OneHotEncoder(**_OHE_KW), catc),
        ]
    )


    m1 = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.08, max_iter=300, random_state=42)
    m2 = Ridge(alpha=2.0, random_state=42)

    # stage-1 fit
    t1 = trn.dropna(subset=num1).copy()
    X1 = t1[num1 + catc]; y1 = t1["qty"].values
    pre1.fit(X1)
    m1.fit(pre1.transform(X1), y1)

    # stage-2 fit
    t2 = trn.dropna(subset=num2).copy()
    X2 = t2[num2 + catc]; y2 = t2["qty"].values
    pre2.fit(X2)
    m2.fit(pre2.transform(X2), y2)

    def _choose_stage_row(row) -> int:
        if all(pd.notna(row.get(c)) for c in num1): return 1
        if all(pd.notna(row.get(c)) for c in num2): return 2
        return 2

    def _predict_two_stage(df_):
        stages = df_.apply(_choose_stage_row, axis=1)
        yhat = np.zeros(len(df_))
        msk1 = stages == 1
        if msk1.any():
            X = df_.loc[msk1, num1 + catc]
            yhat[msk1] = m1.predict(pre1.transform(X))
        msk2 = ~msk1
        if msk2.any():
            X = df_.loc[msk2, num2 + catc].copy()
            for c in num2:
                if c in X.columns and X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())
            yhat[msk2] = m2.predict(pre2.transform(X))
        return np.clip(yhat, 0, None), stages

    # OOF on validation window
    yv = val["qty"].values
    pv, _ = _predict_two_stage(val)
    mae = mean_absolute_error(yv, pv)
    rmse = mean_squared_error(yv, pv, squared=False)
    mape = float(np.mean(np.abs((yv - pv) / np.clip(yv, 1e-6, None))) * 100)

    pre_metrics = {"MAE": float(mae), "RMSE": float(rmse), "MAPE%": mape, "rows": int(len(val))}
    debias = float(np.clip((yv.sum() / np.clip(pv.sum(), 1e-6, None)), 0.5, 1.5))
    post_metrics = dict(pre_metrics)
    post_metrics["debias_factor_oof"] = debias

    oof = val[["store_id","sku_id","date","qty"]].copy()
    oof["yhat"] = pv

    artifact = {
        "stage1_model": m1,
        "stage2_model": m2,
        "pre1": pre1,
        "pre2": pre2,
        "num1": num1,
        "num2": num2,
        "cat": catc,
        "horizon": int(horizon),
    }
    return artifact, pre_metrics, post_metrics, oof


# =========================
# 3) Public API – FORECAST
# =========================

def forecast_next_h(
    df_cont: pd.DataFrame,
    artifact: Dict[str, Any],
    horizon: int,
    debias_factor: float = 1.0
) -> pd.DataFrame:
    """
    Roll forecasts forward for H days.

    Input must include: store_id, sku_id, date, qty, is_promo,
    transactions, oil_price, holiday_flag.
    """
    df = df_cont.copy()
    df["date"] = pd.to_datetime(df["date"])
    last_date = df["date"].max()

    needed = ["store_id","sku_id","date","qty","is_promo","transactions","oil_price","holiday_flag"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise KeyError(f"forecast_next_h: missing columns: {miss}")

    m1 = artifact["stage1_model"]; m2 = artifact["stage2_model"]
    pre1 = artifact["pre1"]; pre2 = artifact["pre2"]
    num1 = artifact["num1"]; num2 = artifact["num2"]; catc = artifact["cat"]

    def make_feats(d: pd.DataFrame) -> pd.DataFrame:
        d = d.copy().sort_values(["store_id","sku_id","date"])
        g = d.groupby(["store_id","sku_id"])["qty"]
        d["dow"] = d["date"].dt.dayofweek
        d["dom"] = d["date"].dt.day
        d["week"] = d["date"].dt.isocalendar().week.astype(int)
        d["month"] = d["date"].dt.month
        d["year"] = d["date"].dt.year
        d["lag_1"]  = g.shift(1)
        d["lag_7"]  = g.shift(7)
        d["lag_14"] = g.shift(14)
        d["lag_28"] = g.shift(28)
        d["rm7"]  = g.shift(1).rolling(7).mean()
        d["rm28"] = g.shift(1).rolling(28).mean()
        return d

    out_rows = []
    cur = make_feats(df)

    for step in range(1, int(horizon)+1):
        target_day = last_date + pd.Timedelta(days=step)

        last_rows = cur.groupby(["store_id","sku_id"]).tail(1).copy()
        last_rows["date"] = target_day
        # Carry forward exogenous values from the last known day.
        # Replace with actual future values if you have them.
        last_rows["is_promo"] = last_rows["is_promo"].fillna(0)
        last_rows["transactions"] = last_rows["transactions"].fillna(0)
        last_rows["oil_price"] = last_rows["oil_price"].ffill().bfill().fillna(0)
        last_rows["holiday_flag"] = 0

        tmp = pd.concat([cur, last_rows], ignore_index=True)
        tmp = make_feats(tmp)

        to_score = tmp[tmp["date"] == target_day].copy()

        def choose_stage_row(row):
            if all(pd.notna(row.get(c)) for c in num1): return 1
            if all(pd.notna(row.get(c)) for c in num2): return 2
            return 2

        stages = to_score.apply(choose_stage_row, axis=1)
        yhat = np.zeros(len(to_score))

        msk1 = stages == 1
        if msk1.any():
            X = to_score.loc[msk1, num1 + catc]
            yhat[msk1] = m1.predict(pre1.transform(X))
        msk2 = ~msk1
        if msk2.any():
            X = to_score.loc[msk2, num2 + catc].copy()
            for c in num2:
                if c in X.columns and X[c].isna().any():
                    X[c] = X[c].fillna(X[c].median())
            yhat[msk2] = m2.predict(pre2.transform(X))

        yhat = np.clip(yhat * float(debias_factor), 0, None)
        to_score["forecast_qty"] = yhat
        out_rows.append(to_score[["store_id","sku_id","date","forecast_qty"]])

        cur = tmp.copy()
        cur.loc[cur["date"] == target_day, "qty"] = yhat

    out = pd.concat(out_rows, ignore_index=True).sort_values(["store_id","sku_id","date"])
    return out


# =========================
# 4) Public API – ORDER PLAN
# =========================

def order_plan_from_forecast(
    forecasts: pd.DataFrame,
    oof: pd.DataFrame,
    lead_time_days: int,
    service_z: Dict[str, float] = {"A":1.645,"B":1.282,"C":0.842}
) -> pd.DataFrame:
    """
    Build an order plan per (store_id, sku_id).

    Order quantity equals the sum of forecasts over the lead time plus safety stock,
    where safety stock is based on residual variability from OOF predictions.
    """
    f = forecasts.copy()
    if not {"store_id","sku_id","date","forecast_qty"}.issubset(f.columns):
        raise KeyError("forecasts must have columns: store_id, sku_id, date, forecast_qty")

    L_cut = f.groupby(["store_id","sku_id"]).apply(
        lambda d: d.nsmallest(max(int(lead_time_days),1), "date")["forecast_qty"].sum()
    ).rename("demand_LT").reset_index()

    resid = oof.copy()
    if {"store_id","sku_id","qty","yhat"}.issubset(resid.columns):
        resid["err"] = resid["qty"] - resid["yhat"]
        sigma = resid.groupby(["store_id","sku_id"])["err"].std().rename("sigma").reset_index()
    else:
        sigma = pd.DataFrame(columns=["store_id","sku_id","sigma"])

    plan = L_cut.merge(sigma, on=["store_id","sku_id"], how="left")
    plan["sigma"] = pd.to_numeric(plan["sigma"], errors="coerce")
    if plan["sigma"].notna().any():
        med = plan["sigma"].median()
        plan["sigma"] = plan["sigma"].fillna(med if not np.isnan(med) else 0.0)
    else:
        plan["sigma"] = 0.0

    z = float(service_z.get("A", 1.645))
    plan["safety_stock"] = z * plan["sigma"] * np.sqrt(max(int(lead_time_days),1))
    plan["order_qty"] = (plan["demand_LT"] + plan["safety_stock"]).round().astype(int)

    plan["cover_days"] = np.nan
    plan["date"] = f["date"].min()
    return plan[["store_id","sku_id","date","demand_LT","safety_stock","order_qty","cover_days"]]


# =========================
# 5) Save / Load helpers
# =========================

def save_artifact(artifact: Dict[str, Any], path_or_buf) -> None:
    joblib.dump(artifact, path_or_buf)

def load_artifact(path: str) -> Dict[str, Any]:
    return joblib.load(path)
