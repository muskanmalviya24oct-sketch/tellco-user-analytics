"""A minimal, reusable feature store.

Stores named, versioned feature tables as Parquet files on disk (falls back
to CSV if the parquet engine isn't installed) plus a JSON index describing
each table (columns, row count, created time, free-text description). This
keeps per-user features computed once (overview/engagement/experience/
satisfaction) reusable across the dashboard, the notebook, and future
similar problems without recomputing them from the raw 150k-row export.

Usage
-----
    from telco_analytics.feature_store import FeatureStore
    fs = FeatureStore('feature_store')              # folder to store files in
    fs.save('engagement_scores', eng_df, description='Per-user engagement metrics + cluster')
    df = fs.load('engagement_scores')
    fs.list()                                        # -> DataFrame of what's stored
"""
import json
import os
from datetime import datetime, timezone

import pandas as pd


class FeatureStore:
    def __init__(self, root: str = 'feature_store'):
        self.root = root
        os.makedirs(self.root, exist_ok=True)
        self.index_path = os.path.join(self.root, '_index.json')
        if not os.path.exists(self.index_path):
            self._write_index({})

    # -- internal helpers ---------------------------------------------------
    def _read_index(self) -> dict:
        with open(self.index_path) as f:
            return json.load(f)

    def _write_index(self, index: dict) -> None:
        with open(self.index_path, 'w') as f:
            json.dump(index, f, indent=2, default=str)

    def _table_path(self, name: str, ext: str) -> str:
        return os.path.join(self.root, f'{name}.{ext}')

    # -- public API -----------------------------------------------------------
    def save(self, name: str, df: pd.DataFrame, description: str = '') -> str:
        """Save `df` under `name`. Uses Parquet if available, else CSV.
        Returns the file path written."""
        try:
            path = self._table_path(name, 'parquet')
            df.to_parquet(path)
            fmt = 'parquet'
        except Exception:
            path = self._table_path(name, 'csv')
            df.to_csv(path, index=(df.index.name is not None))
            fmt = 'csv'

        index = self._read_index()
        index[name] = {
            'path': os.path.basename(path),
            'format': fmt,
            'rows': int(len(df)),
            'columns': list(df.columns.astype(str)),
            'index_name': df.index.name,
            'description': description,
            'updated_at': datetime.now(timezone.utc).isoformat(),
        }
        self._write_index(index)
        return path

    def load(self, name: str) -> pd.DataFrame:
        index = self._read_index()
        if name not in index:
            raise KeyError(f"No feature table named '{name}' in {self.root}. "
                            f"Available: {list(index)}")
        meta = index[name]
        path = os.path.join(self.root, meta['path'])
        if meta['format'] == 'parquet':
            df = pd.read_parquet(path)
        else:
            idx_col = 0 if meta.get('index_name') else None
            df = pd.read_csv(path, index_col=idx_col)
        return df

    def list(self) -> pd.DataFrame:
        index = self._read_index()
        if not index:
            return pd.DataFrame(columns=['name', 'rows', 'columns', 'description', 'updated_at'])
        rows = [{'name': n, 'rows': m['rows'], 'columns': len(m['columns']),
                 'description': m['description'], 'updated_at': m['updated_at']}
                for n, m in index.items()]
        return pd.DataFrame(rows).sort_values('name').reset_index(drop=True)

    def exists(self, name: str) -> bool:
        return name in self._read_index()

    def delete(self, name: str) -> None:
        index = self._read_index()
        if name not in index:
            return
        path = os.path.join(self.root, index[name]['path'])
        if os.path.exists(path):
            os.remove(path)
        del index[name]
        self._write_index(index)
