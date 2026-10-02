"""Load and clean the raw TellCo xDR Excel export."""
import pandas as pd

APPS = ['Social Media', 'Google', 'Email', 'Youtube', 'Netflix', 'Gaming', 'Other']

APP_COLS = [f'{a} {d} (Bytes)' for a in APPS for d in ('DL', 'UL')]

NUM_COLS = (
    ['Dur. (ms)', 'Total DL (Bytes)', 'Total UL (Bytes)',
     'Avg RTT DL (ms)', 'Avg RTT UL (ms)',
     'Avg Bearer TP DL (kbps)', 'Avg Bearer TP UL (kbps)',
     'TCP DL Retrans. Vol (Bytes)', 'TCP UL Retrans. Vol (Bytes)']
    + APP_COLS
)

HANDSET_COLS = ['Handset Type', 'Handset Manufacturer', 'Last Location Name']


def load_raw(path: str) -> pd.DataFrame:
    """Read the raw Excel export as-is (no cleaning)."""
    return pd.read_excel(path)


def _treat_with_mean(frame: pd.DataFrame, cols) -> pd.DataFrame:
    """Replace missing values and IQR outliers in `cols` with the column mean.

    If a column's IQR is 0 (mostly identical/zero values), only NaNs are
    filled, since the IQR rule would otherwise flag every non-zero value.
    """
    frame = frame.copy()
    for c in cols:
        q1, q3 = frame[c].quantile([.25, .75])
        iqr = q3 - q1
        if iqr == 0:
            inlier = frame[c].notna()
        else:
            inlier = frame[c].between(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
        frame.loc[~inlier, c] = frame.loc[inlier, c].mean()
    return frame


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Full cleaning pipeline: drop the summary row, drop rows with no
    customer id, treat missing values/outliers with the column mean, and
    fill missing handset/location text with 'undefined'.
    """
    df = raw[raw['Start'].notna()].copy()           # drop trailing summary row
    df.columns = df.columns.str.strip()
    df = df.rename(columns={'MSISDN/Number': 'MSISDN'})

    df = df.dropna(subset=['MSISDN']).copy()
    df['MSISDN'] = df['MSISDN'].astype('int64')

    df = _treat_with_mean(df, [c for c in NUM_COLS if c in df.columns])
    for c in HANDSET_COLS:
        if c in df.columns:
            df[c] = df[c].fillna('undefined')

    for a in APPS:
        df[f'{a} Total'] = df[f'{a} DL (Bytes)'] + df[f'{a} UL (Bytes)']

    return df


def load_and_clean(path: str) -> pd.DataFrame:
    """Convenience wrapper: load_raw() + clean()."""
    return clean(load_raw(path))
