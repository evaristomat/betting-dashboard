# utils_datetime.py

import numpy as np
import pandas as pd


def ensure_datetime(df, date_column="date"):
    """Converte coluna de data para datetime tratando diferentes formatos."""
    if not np.issubdtype(df[date_column].dtype, np.datetime64):
        month_mapping = {
            "Jan": "Jan",
            "Fev": "Feb",
            "Mar": "Mar",
            "Abr": "Apr",
            "Mai": "May",
            "Jun": "Jun",
            "Jul": "Jul",
            "Ago": "Aug",
            "Set": "Sep",
            "Out": "Oct",
            "Nov": "Nov",
            "Dez": "Dec",
        }
        date_str = df[date_column].astype(str)
        for pt, en in month_mapping.items():
            date_str = date_str.str.replace(pt, en, regex=False)
        # tenta diferentes estratégias
        df[date_column] = pd.to_datetime(
            date_str,
            format="%d %b %Y %H:%M",
            dayfirst=True,
            errors="coerce",
        )
    return df
