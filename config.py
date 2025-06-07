# config.py
import matplotlib.pyplot as plt

# ----------------- CONFIGURAÇÕES GERAIS ----------------- #
BACKGROUND_COLOR = "#0E1117"
PARAMS = {
    "axes.labelcolor": "white",
    "axes.edgecolor": "white",
    "axes.facecolor": BACKGROUND_COLOR,
    "xtick.color": "white",
    "ytick.color": "white",
    "text.color": "white",
    "figure.facecolor": BACKGROUND_COLOR,
    "grid.color": "gray",
    "grid.linestyle": "--",
}


def apply_matplotlib_style():
    """Aplica o estilo definido em PARAMS para todos os plots matplotlib."""
    plt.rcParams.update(PARAMS)


# utils_datetime.py
import numpy as np
import pandas as pd


# ----------------- CONVERSÃO DE DATETIME ----------------- #
def ensure_datetime(df: pd.DataFrame, date_column: str = "date") -> pd.DataFrame:
    """
    Converte a coluna de data para datetime, tratando abreviações de mês em português.
    """
    if date_column in df.columns and not np.issubdtype(
        df[date_column].dtype, np.datetime64
    ):
        mapping = {
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
        s = df[date_column].astype(str)
        for pt, en in mapping.items():
            s = s.str.replace(pt, en, regex=False)
        df[date_column] = pd.to_datetime(
            s,
            format="%d %b %Y %H:%M",
            dayfirst=True,
            infer_datetime_format=True,
            errors="coerce",
        )
    return df


# data_loading.py
import pandas as pd
import streamlit as st
from utils_datetime import ensure_datetime


# ----------------- CARREGAMENTO DE DADOS ----------------- #
@st.cache_data
def load_pending_bets() -> pd.DataFrame:
    """Carrega apostas pendentes de múltiplos arquivos possíveis."""
    possible = [
        "bets/bets_atualizadas.csv",
        "bets/bets.csv",
        "bets_atualizadas.csv",
        "bets.csv",
    ]
    df = pd.DataFrame()
    for path in possible:
        try:
            df = pd.read_csv(path)
            break
        except FileNotFoundError:
            continue
    if df.empty:
        return df
    if "status" not in df.columns:
        df["status"] = "pending"
    df = df[df["status"] == "pending"].copy()
    if "date" in df.columns:
        df = ensure_datetime(df, "date")
        df = df.dropna(subset=["date"]).reset_index(drop=True)
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """Carrega dados de apostas finalizadas e faz conversões de tipo."""
    try:
        df = pd.read_csv("bets/bets_atualizadas_por_mapa.csv")
        # conversões numéricas
        for col in ["ROI", "odds", "profit", "bet_result", "game"]:
            if col in df.columns:
                df[col] = df[col].astype(str).str.replace("%", "").str.replace(",", ".")
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df = ensure_datetime(df, "date")
        return df[df["status"].isin(["win", "loss"])].reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


# data_processing.py
import pandas as pd
import streamlit as st


# ----------------- PROCESSAMENTO DE DADOS ----------------- #
def process_data(df: pd.DataFrame, min_roi: float, max_roi: float) -> pd.DataFrame:
    """
    Filtra por ROI, converte tipos e adiciona colunas auxiliares.
    """
    try:
        df = df.dropna(subset=["ROI", "profit", "odds", "status"])
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df = df[(df["ROI"] >= min_roi) & (df["ROI"] <= max_roi)]
        if df.empty:
            return df
        df = df.sort_values(["date", "game"]).reset_index(drop=True)
        df["cumulative_profit"] = df["profit"].cumsum()
        if "bet_line" in df.columns:
            df["bet_group"] = df["bet_line"].astype(str).str.split().str[0]
        else:
            df["bet_group"] = df.get("bet_type", "unknown")
        return df.dropna(subset=["bet_group"]).reset_index(drop=True)
    except Exception as e:
        st.error(f"Erro no processamento: {e}")
        return pd.DataFrame()
