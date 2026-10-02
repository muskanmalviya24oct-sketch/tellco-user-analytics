# TellCo User Analytics (Nexthikes IT Solutions)

Reusable code for the TellCo telecom due-diligence project: data cleaning,
feature engineering, clustering, and a Streamlit dashboard, built on top of
the analysis in `TellCo_User_Analytics.ipynb`.

## Project layout

```
telco_project/
├── telco_analytics/        # pip-installable package
│   ├── cleaning.py          # load + clean the raw Excel export
│   ├── overview.py          # Task 1 — handsets, per-user aggregates, PCA
│   ├── engagement.py        # Task 2 — engagement metrics + k-means
│   ├── experience.py        # Task 3 — TCP/RTT/throughput + k-means
│   ├── satisfaction.py      # Task 4 — scores, clustering
│   └── feature_store.py     # save/load named feature tables (parquet/csv)
├── dashboard/
│   └── app.py                # Streamlit dashboard (reads/writes the feature store)
├── feature_store/            # created on first run; holds saved feature tables
├── tests/                    # pytest unit tests (synthetic data, no Excel needed)
├── pyproject.toml
└── README.md
```

## Install

```bash
pip install -e .                 # core package
pip install -e ".[dashboard]"    # + streamlit, matplotlib, seaborn
pip install -e ".[test]"         # + pytest
pip install -e ".[tracking]"     # + mlflow, sqlalchemy, pymysql (Task 4.6/4.7)
```

(Installing all at once: `pip install -e ".[dashboard,test,tracking]"`)

## Run the tests

```bash
pytest tests/ -v
```

Tests run against small synthetic data generated in `tests/conftest.py` —
you don't need the real Excel file to run them.

## Run the dashboard

```bash
streamlit run dashboard/app.py
```

In the sidebar, set the path to your raw Excel file (e.g.
`telcom_data.xlsx`) and click **"Rebuild from Excel"** the first time — this
cleans the data, computes every feature table, and saves it to
`feature_store/`. On every later run, the dashboard loads instantly from the
feature store instead of recomputing from the 150k-row Excel file.

## Using the feature store directly

```python
from telco_analytics.feature_store import FeatureStore

fs = FeatureStore('feature_store')
fs.list()                               # see what's saved
scores = fs.load('satisfaction_scores')  # reload a table any time
```

## Using the package functions directly (e.g. in a notebook)

```python
from telco_analytics.cleaning import load_and_clean
from telco_analytics import overview, engagement, experience, satisfaction

df = load_and_clean('telcom_data.xlsx')
user = overview.user_aggregates(df)
eng_c, scaler, km, X = engagement.cluster_engagement(
    engagement.engagement_table(user)
)
```

This is the same logic used in `TellCo_User_Analytics.ipynb` — the notebook
and the dashboard now share one source of truth instead of duplicating code.

## Not included (per trainer's instructions)

CI/CD (Travis/GitHub Actions) and the Dockerfile were explicitly waived for
this submission and are not included here.
