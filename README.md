# Customer Inside Pro

Interactive Streamlit dashboard for customer exploration and churn-risk estimation using Pandas, Plotly and scikit-learn Logistic Regression.

## Features
- KPI overview and churn visualizations
- Interactive customer filters and CSV export
- Customer-level risk scoring form
- Held-out evaluation metrics and confusion matrix
- Documentation of demo-data and model limitations

## Run
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The CSV is synthetic portfolio demo data. Do not describe it as real customer data or use its scores for business decisions.
