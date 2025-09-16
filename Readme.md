## Darkstore Demand Forecast

### Hyperlocal Product Demand Forecasting for Dark Stores
This project provides a simple, practical way to forecast short‑term SKU‑level demand for dark stores and turn those forecasts into replenishment order plans with safety stock. It includes a two‑stage forecasting pipeline and a Streamlit web app.

## Features
- **Two‑stage model**
  - **Stage 1**: Gradient boosting on rich features
  - **Stage 2**: Ridge regression fallback for sparse feature rows
- **Daily demand forecasts** per SKU × Store
- **Order plan generator** with safety stock based on forecast error volatility
- **Streamlit app** to:
  - Upload your dataset
  - Prepare features
  - Train the model
  - Choose forecast horizon (7–28 days)
  - Generate an order plan with a selected lead time

## Input data requirements
Your training CSV should include the following columns.

- **Required**
  - `date`: daily date in `YYYY-MM-DD` format
  - `store_nbr`: store identifier
  - `family`: product identifier (SKU or category)
  - `sales`: daily units sold (not revenue)

- **Optional (recommended)**
  - `onpromotion` (0/1)
  - `transactions` (store traffic per day)
  - `oil_price` (macro variable)
  - `holiday_flag` (0/1)

Notes:
- One row per day × store × SKU.
- Do not use weekly or monthly aggregates.

If you just want to try the app, use the included synthetic sample dataset: `sample_train.csv`.

## How to run locally
Clone the repo and install dependencies:

```bash
git clone https://github.com/<yourname>/darkstore-demand-forecast.git
cd darkstore-demand-forecast
pip install -r requirements.txt
```

Run the app:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

## Deploying
### Option 1: Streamlit Community Cloud (free)
1. Push this repo to GitHub
2. Go to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Connect your repo and choose `app.py` as the entrypoint
4. The app will build automatically and provide a public link

### Option 2: Other services
- Render
- Heroku
- Docker + Cloud Run (for production)

## Repo structure
```
darkstore-demand-forecast/
├── app.py                # Streamlit UI
├── two_stage.py          # Forecasting + order planning logic
├── smoke_test.py         # Simple import check
├── requirements.txt      # Dependencies
├── .gitignore
├── README.md
└── sample_train.csv      # Tiny sample dataset for quick demo
```

## Example output
- **Forecasts**: Predicted demand for each SKU × Store for the next H days
- **Order plan**: Lead‑time demand plus safety stock = recommended order quantity

## Tech stack
- Python 3.10+
- Streamlit
- scikit‑learn
- pandas, numpy

## Author
Built as a prototype for hyperlocal demand forecasting in dark stores. Feel free to fork, experiment, or extend with real retail datasets.


