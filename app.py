# Standard imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Web application framework
import streamlit as st

import plotly.graph_objects as go
from datetime import datetime, date, timedelta

# ----------------- CONFIGURATION ----------------- #
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
plt.rcParams.update(PARAMS)


# ----------------- DATA LOADING & PROCESSING FUNCTIONS ----------------- #
def ensure_datetime(df, date_column="date"):
    """Converte coluna de data para datetime tratando meses em português"""
    if not np.issubdtype(df[date_column].dtype, np.datetime64):
        # Mapeamento de meses português -> inglês
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

        # Converte meses portugueses para inglês
        df_temp = df.copy()
        date_str = df_temp[date_column].astype(str)

        for pt_month, en_month in month_mapping.items():
            date_str = date_str.str.replace(pt_month, en_month)

        # Tenta diferentes formatos após conversão
        try:
            # Formato: "22 May 2025 13:00"
            df[date_column] = pd.to_datetime(
                date_str, format="%d %b %Y %H:%M", errors="coerce"
            )
        except:
            try:
                # Formato mais flexível
                df[date_column] = pd.to_datetime(
                    date_str, dayfirst=True, errors="coerce"
                )
            except:
                # Fallback final
                df[date_column] = pd.to_datetime(
                    date_str, infer_datetime_format=True, errors="coerce"
                )

    return df


# @st.cache_data
def load_pending_bets():
    """Carrega apostas pendentes do arquivo bets.csv"""
    try:
        df_pending = pd.read_csv("bets/bets_atualizadas.csv")
        print(df_pending)

        # Filtra apenas apostas pendentes
        pending_bets = df_pending[df_pending["status"] == "pending"].copy()

        if len(pending_bets) > 0:
            # Converte data para datetime para ordenação
            pending_bets = ensure_datetime(pending_bets, "date")

            # Ordena por data (mais próxima primeiro)
            pending_bets = pending_bets.sort_values("date")

        return pending_bets

    except FileNotFoundError:
        return pd.DataFrame()  # Retorna DataFrame vazio se arquivo não existir
    except Exception as e:
        st.error(f"Erro ao carregar apostas pendentes: {e}")
        return pd.DataFrame()


# @st.cache_data
def load_data():
    """Carrega dados das apostas processadas"""
    try:
        df = pd.read_csv("bets/bets_atualizadas_por_mapa.csv")

        # Limpa e converte dados para tipos corretos
        if "ROI" in df.columns:
            df["ROI"] = df["ROI"].astype(str).str.replace("%", "").str.replace(",", ".")
            df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce").fillna(0)

        if "odds" in df.columns:
            df["odds"] = pd.to_numeric(df["odds"], errors="coerce").fillna(1.0)

        if "profit" in df.columns:
            df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)

        if "bet_result" in df.columns:
            df["bet_result"] = pd.to_numeric(df["bet_result"], errors="coerce").fillna(
                0
            )

        if "game" in df.columns:
            df["game"] = pd.to_numeric(df["game"], errors="coerce").fillna(1)

        if "status" in df.columns:
            df["status"] = df["status"].astype(str)

        df = ensure_datetime(df, "date")

        # Filtra apenas apostas finalizadas
        df = df[df["status"].isin(["win", "loss"])]

        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


def process_data(df, min_roi, max_roi):
    """Processa dados aplicando filtros de ROI"""
    try:
        # Remove linhas com valores NaN críticos
        df = df.dropna(subset=["ROI", "profit", "odds", "status"])

        # Garante tipos corretos
        df["ROI"] = pd.to_numeric(df["ROI"], errors="coerce")
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")

        # Remove linhas que se tornaram NaN após conversão
        df = df.dropna(subset=["ROI", "profit", "odds"])

        # Filtra por ROI
        df = df[(df["ROI"] >= min_roi) & (df["ROI"] <= max_roi)]

        if len(df) == 0:
            return df

        # Ordena por data e calcula lucro cumulativo
        df = df.sort_values(["date", "game"]).reset_index(drop=True)
        df["cumulative_profit"] = df["profit"].cumsum()

        # Cria grupo de apostas baseado no bet_line
        if "bet_line" in df.columns:
            df["bet_group"] = df["bet_line"].astype(str).str.split().str[0]
        else:
            df["bet_group"] = df["bet_type"] if "bet_type" in df.columns else "unknown"

        # Remove grupos NaN
        df = df[df["bet_group"].notna()]

        return df

    except Exception as e:
        st.error(f"Erro no processamento: {e}")
        return pd.DataFrame()


# ----------------- VISUALIZATION FUNCTIONS ----------------- #
def display_summary(df, roi_value):
    """Exibe resumo das estatísticas"""
    try:
        total_profit = (
            float(df["profit"].sum()) if not df["profit"].isna().all() else 0.0
        )
        total_bets = len(df)
        wins = len(df[df["status"] == "win"])
        losses = total_bets - wins
        win_rate = wins / total_bets if total_bets != 0 else 0.0
        avg_odd = float(df["odds"].mean()) if not df["odds"].isna().all() else 0.0

        # Garante que não há NaN
        if pd.isna(total_profit):
            total_profit = 0.0
        if pd.isna(avg_odd):
            avg_odd = 0.0
        if pd.isna(win_rate):
            win_rate = 0.0

        # Display em colunas
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1, 1, 1, 1, 1, 1, 1.5])
        col1.metric(label="ROI Chosen", value=f"{roi_value}%")
        col2.metric(label="Total Bets", value=f"{total_bets}")
        col3.metric(label="Wins", value=f"{wins}")
        col4.metric(label="Losses", value=f"{losses}")
        col5.metric(label="Win Rate", value=f"{win_rate * 100:.0f}%")
        col6.metric(label="Average Odd", value=f"{avg_odd:.2f}")
        col7.metric(label="Profit", value=f"{total_profit:.2f}U")

    except Exception as e:
        st.error(f"Erro no cálculo do resumo: {e}")


def bankroll_plot(df):
    """Gráfico de evolução do bankroll geral e por games"""
    if len(df) == 0:
        st.warning("No data available for bankroll plot.")
        return

    try:
        # Garantir que temos dados finalizados
        df_finished = df[df["status"].isin(["win", "loss"])].copy()

        if len(df_finished) == 0:
            st.warning("Nenhuma aposta finalizada encontrada.")
            return

        # Ordenar por data para evolução cronológica
        df_finished = df_finished.sort_values("date").reset_index(drop=True)

        # Verificar se temos a coluna game
        has_game_column = (
            "game" in df_finished.columns and not df_finished["game"].isna().all()
        )

        if not has_game_column:
            st.info("Coluna 'game' não encontrada. Mostrando apenas evolução geral.")
            plot_simple_bankroll_evolution(df_finished)
            return

        # Preparar dados para diferentes categorias
        categories = {}

        # Todos os games
        categories["All Games"] = df_finished.copy()

        # Games específicos
        game_values = df_finished["game"].dropna().unique()
        for game in sorted(game_values):
            game_data = df_finished[df_finished["game"] == game].copy()
            if len(game_data) > 0:
                categories[f"Game {int(game)}"] = game_data

        # Se só temos uma categoria, usar plot simples
        if len(categories) <= 1:
            plot_simple_bankroll_evolution(df_finished)
            return

        # Criar figura simples
        plt.figure(figsize=(14, 8))

        # Configurações
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        # ========== GRÁFICO: Evolução do Bankroll ==========
        plt.title(
            "Evolution of Bankroll by Game Category",
            fontsize=16,
            fontweight="bold",
            color="white",
        )

        # Calcular estatísticas para cada categoria (para usar depois na tabela)
        stats_data = []

        for i, (category, data) in enumerate(categories.items()):
            if len(data) == 0:
                continue

            # Resetar índice para sequência correta
            data = data.reset_index(drop=True)

            # Calcular lucro cumulativo
            data["cumulative_profit"] = data["profit"].cumsum()

            color = colors[i % len(colors)]

            # Linha principal (sem média móvel)
            plt.plot(
                range(len(data)),
                data["cumulative_profit"],
                marker="o",
                markersize=4,
                label=f"{category} (Final: {data['cumulative_profit'].iloc[-1]:.2f}U)",
                color=color,
                linewidth=2,
            )

            # Calcular estatísticas para tabela
            total_bets = len(data)
            wins = len(data[data["status"] == "win"])
            losses = len(data[data["status"] == "loss"])
            winrate = (wins / total_bets * 100) if total_bets > 0 else 0
            total_profit = data["profit"].sum()
            avg_roi = data["ROI"].mean() if "ROI" in data.columns else 0

            stats_data.append(
                {
                    "Category": category,
                    "Total Bets": total_bets,
                    "Wins": wins,
                    "Losses": losses,
                    "Win Rate (%)": winrate,
                    "Total Profit": total_profit,
                    "Avg ROI (%)": avg_roi,
                }
            )

        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        plt.ylabel("Cumulative Profit (units)", fontsize=12, color="white")
        plt.xlabel("Bet Sequence", fontsize=12, color="white")
        plt.legend(loc="best")  # Legenda dentro do gráfico
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        st.pyplot(plt.gcf())

    except Exception as e:
        st.error(f"Erro no gráfico de bankroll: {e}")
        # Fallback para plot simples
        plot_simple_bankroll_evolution(df)


def plot_simple_bankroll_evolution(df):
    """Função de fallback para plot simples sem separação por games"""
    try:
        df = df.reset_index(drop=True)
        df["cumulative_profit"] = df["profit"].cumsum()

        plt.figure(figsize=(12, 7))

        # Linha principal
        ax = sns.lineplot(
            data=df,
            x=df.index,
            y="cumulative_profit",
            marker="o",
            label="Cumulative Profit",
        )

        # Linha de referência zero
        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        plt.title("Evolution of Bankroll Over Bets", fontsize=16, fontweight="bold")
        plt.ylabel("Cumulative Profit (units)", fontsize=12)
        plt.xlabel("Bet Sequence", fontsize=12)
        plt.legend(loc="best")  # Legenda dentro do gráfico
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(ax.get_figure())

    except Exception as e:
        st.error(f"Erro no plot simples de bankroll: {e}")


def odds_plot(df: pd.DataFrame) -> None:
    """Gráfico de lucratividade por faixas de odds específicas."""
    if len(df) == 0 or df["odds"].isna().all():
        st.warning("No odds data available for analysis.")
        return

    # Definir as faixas de odds - Opção 5: Simples e Clara
    def categorize_odds(odd_value):
        if pd.isna(odd_value):
            return "Unknown"
        elif odd_value < 1.5:
            return "< 1.5"
        elif 1.5 <= odd_value < 2.0:
            return "1.5 - 1.99"
        elif 2.0 <= odd_value < 2.5:
            return "2.0 - 2.49"
        elif 2.5 <= odd_value < 3.0:
            return "2.5 - 2.99"
        else:
            return "≥ 3.0"

    # Aplicar categorização
    df = df.copy()
    df["odds_category"] = df["odds"].apply(categorize_odds)

    # Remover categoria "Unknown" se existir
    df = df[df["odds_category"] != "Unknown"]

    if len(df) == 0:
        st.warning("No valid odds data after categorization.")
        return

    # Calcular estatísticas por categoria
    odds_stats = (
        df.groupby("odds_category")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "status": lambda x: (x == "win").mean(),
                "odds": "mean",
            }
        )
        .round(3)
    )

    # Flatten column names
    odds_stats.columns = [
        "Total_Profit",
        "Total_Bets",
        "Avg_Profit_Per_Bet",
        "Win_Rate",
        "Avg_Odds",
    ]

    # Ordenar categorias de forma lógica
    category_order = ["< 1.5", "1.5 - 1.99", "2.0 - 2.49", "2.5 - 2.99", "≥ 3.0"]

    # Filtrar apenas categorias que existem nos dados
    existing_categories = [cat for cat in category_order if cat in odds_stats.index]
    odds_stats = odds_stats.reindex(existing_categories)

    # Remover linhas com NaN
    odds_stats = odds_stats.dropna()

    if len(odds_stats) == 0:
        st.warning("No valid data for odds profitability analysis.")
        return

    # Criar figura com subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Profitability Analysis by Odds Range", fontsize=16, fontweight="bold")

    # 1. Total Profit por categoria
    colors = ["green" if x >= 0 else "red" for x in odds_stats["Total_Profit"]]
    bars1 = ax1.bar(
        range(len(odds_stats)), odds_stats["Total_Profit"], color=colors, alpha=0.7
    )
    ax1.set_title("Total Profit by Odds Range")
    ax1.set_ylabel("Total Profit (U)")
    ax1.set_xticks(range(len(odds_stats)))
    ax1.set_xticklabels(odds_stats.index, rotation=45)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color="black", linestyle="--", alpha=0.5)

    # Anotar valores
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.2f}U",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 2. Win Rate por categoria
    bars2 = ax2.bar(
        range(len(odds_stats)), odds_stats["Win_Rate"], color="skyblue", alpha=0.7
    )
    ax2.set_title("Win Rate by Odds Range")
    ax2.set_ylabel("Win Rate")
    ax2.set_xticks(range(len(odds_stats)))
    ax2.set_xticklabels(odds_stats.index, rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Anotar percentuais
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.1%}",
            ha="center",
            va="bottom",
        )

    # 3. Lucro médio por aposta
    colors3 = ["green" if x >= 0 else "red" for x in odds_stats["Avg_Profit_Per_Bet"]]
    bars3 = ax3.bar(
        range(len(odds_stats)),
        odds_stats["Avg_Profit_Per_Bet"],
        color=colors3,
        alpha=0.7,
    )
    ax3.set_title("Average Profit per Bet")
    ax3.set_ylabel("Avg Profit per Bet (U)")
    ax3.set_xticks(range(len(odds_stats)))
    ax3.set_xticklabels(odds_stats.index, rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.axhline(0, color="black", linestyle="--", alpha=0.5)

    # Anotar valores
    for i, bar in enumerate(bars3):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{height:.3f}U",
            ha="center",
            va="bottom" if height >= 0 else "top",
        )

    # 4. Número de apostas por categoria
    bars4 = ax4.bar(
        range(len(odds_stats)), odds_stats["Total_Bets"], color="orange", alpha=0.7
    )
    ax4.set_title("Number of Bets by Odds Range")
    ax4.set_ylabel("Number of Bets")
    ax4.set_xticks(range(len(odds_stats)))
    ax4.set_xticklabels(odds_stats.index, rotation=45)
    ax4.grid(True, alpha=0.3)

    # Anotar valores
    for i, bar in enumerate(bars4):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()
    st.pyplot(fig)

    # Exibir tabela de estatísticas
    st.markdown("### 📊 Detailed Statistics by Odds Range")

    # Preparar tabela para exibição
    display_stats = odds_stats.copy()
    display_stats["Win_Rate"] = display_stats["Win_Rate"].apply(lambda x: f"{x:.1%}")
    display_stats["Total_Profit"] = display_stats["Total_Profit"].apply(
        lambda x: f"{x:.2f}U"
    )
    display_stats["Avg_Profit_Per_Bet"] = display_stats["Avg_Profit_Per_Bet"].apply(
        lambda x: f"{x:.3f}U"
    )
    display_stats["Avg_Odds"] = display_stats["Avg_Odds"].apply(lambda x: f"{x:.2f}")
    display_stats["Total_Bets"] = display_stats["Total_Bets"].astype(int)

    display_stats.columns = [
        "Total Profit",
        "Total Bets",
        "Avg Profit/Bet",
        "Win Rate",
        "Avg Odds",
    ]

    st.dataframe(display_stats, use_container_width=True)


def bet_groups_plot(df):
    """Gráfico de distribuição por grupos de apostas"""
    if len(df) == 0:
        st.warning("No data available for bet groups plot.")
        return

    grouped = df.groupby(["bet_group", "status"]).size().unstack().fillna(0)

    if "win" not in grouped.columns:
        grouped["win"] = 0
    if "loss" not in grouped.columns:
        grouped["loss"] = 0

    grouped["total"] = grouped["win"] + grouped["loss"]
    # Evita divisão por zero usando replace
    grouped["win_ratio"] = grouped["win"] / grouped["total"].replace(0, np.nan)
    grouped["loss_ratio"] = grouped["loss"] / grouped["total"].replace(0, np.nan)

    # Remove grupos sem dados válidos
    grouped = grouped.dropna(subset=["win_ratio", "loss_ratio"])

    if len(grouped) == 0:
        st.warning("No valid data for bet groups analysis.")
        return

    plt.figure(figsize=(12, 7))
    melted = grouped.reset_index().melt(
        id_vars=["bet_group"],
        value_vars=["win", "loss"],
        var_name="status",
        value_name="count",
    )

    ax = sns.barplot(
        data=melted,
        x="bet_group",
        y="count",
        hue="status",
        hue_order=["loss", "win"],
        palette={"win": "green", "loss": "red"},
    )

    # Adiciona percentuais nas barras
    for i, bar in enumerate(ax.patches):
        height = bar.get_height()
        if height > 0:
            group_idx = i % len(grouped)
            try:
                if i < len(grouped):  # Loss bars
                    percentage = grouped["loss_ratio"].iloc[group_idx]
                else:  # Win bars
                    percentage = grouped["win_ratio"].iloc[group_idx]

                if not np.isnan(percentage):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height / 2,
                        f"{percentage:.1%}",
                        ha="center",
                        va="center",
                        color="white",
                        fontsize=9,
                        weight="bold",
                    )
            except (IndexError, KeyError):
                continue

    plt.title("Distribution of Bet Groups with Wins and Losses")
    plt.ylabel("Number of Bets")
    plt.xlabel("Bet Group")
    plt.xticks(rotation=45)
    plt.legend(title="Status", loc="upper right")
    plt.tight_layout()
    st.pyplot(ax.get_figure())


def profit_plot(df):
    """Gráfico de lucro por grupo de apostas"""
    if len(df) == 0:
        st.warning("No data available for profit plot.")
        return

    plt.figure(figsize=(12, 7))

    # Calcula profit por bet_group e bet_type
    profit_data = df.groupby(["bet_group", "bet_type"])["profit"].sum().reset_index()

    if len(profit_data) > 0:
        ax = sns.barplot(
            data=profit_data,
            x="bet_group",
            y="profit",
            hue="bet_type",
            palette="viridis",
            errorbar=None,
        )

        # Anota barras com valores
        for p in ax.patches:
            height = p.get_height()
            if abs(height) > 0.01 and not np.isnan(
                height
            ):  # Só anota se valor significativo e não NaN
                ax.annotate(
                    f"{height:.2f}",
                    (p.get_x() + p.get_width() / 2.0, height),
                    ha="center",
                    va="bottom" if height > 0 else "top",
                    xytext=(0, 5 if height > 0 else -5),
                    textcoords="offset points",
                    fontsize=9,
                )

        plt.title("Profit by Bet Group and Type")
        plt.ylabel("Total Profit (U)")
        plt.xlabel("Bet Group")
        plt.xticks(rotation=45)
        plt.legend(title="Bet Type", loc="upper left")
    else:
        plt.text(
            0.5,
            0.5,
            "No data available",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=16,
        )
        plt.title("Profit by Bet Group and Type")

    plt.tight_layout()
    st.pyplot(ax.get_figure())


def create_bet_type_analysis_charts(df):
    """Cria gráficos de análise por tipo de aposta usando matplotlib"""
    try:
        if len(df) == 0:
            return []

        charts = []

        # 1. Total Profit by Bet Type
        bet_type_stats = (
            df.groupby("bet_type")["profit"].sum().sort_values(ascending=False)
        )

        if len(bet_type_stats) > 0:
            plt.figure(figsize=(10, 6))
            colors = ["green" if x >= 0 else "red" for x in bet_type_stats.values]
            bars = plt.bar(
                bet_type_stats.index, bet_type_stats.values, color=colors, alpha=0.7
            )

            # Anota valores nas barras
            for bar in bars:
                height = bar.get_height()
                plt.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{height:.2f}U",
                    ha="center",
                    va="bottom" if height > 0 else "top",
                )

            plt.title("Total Profit by Bet Type")
            plt.xlabel("Bet Type")
            plt.ylabel("Total Profit (U)")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            charts.append(("Total Profit by Bet Type", plt.gcf()))

        # 2. Total Profit by Bet Line Type
        if "bet_line" in df.columns:
            df_temp = df.copy()
            df_temp["bet_line_type"] = (
                df_temp["bet_line"].astype(str).str.split().str[0]
            )
            bet_line_stats = (
                df_temp.groupby("bet_line_type")["profit"]
                .sum()
                .sort_values(ascending=False)
            )

            if len(bet_line_stats) > 0:
                plt.figure(figsize=(12, 6))
                colors = ["green" if x >= 0 else "red" for x in bet_line_stats.values]
                bars = plt.bar(
                    bet_line_stats.index, bet_line_stats.values, color=colors, alpha=0.7
                )

                # Anota valores nas barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}U",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                    )

                plt.title("Total Profit by Bet Line Type")
                plt.xlabel("Bet Line Type")
                plt.ylabel("Total Profit (U)")
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                charts.append(("Total Profit by Bet Line Type", plt.gcf()))

        # 3. Combined Bet Type and Line (Top 10)
        if "bet_line" in df.columns:
            df_temp = df.copy()
            df_temp["combined_bet"] = (
                df_temp["bet_type"].astype(str)
                + " + "
                + df_temp["bet_line"].astype(str)
            )
            combined_stats = (
                df_temp.groupby("combined_bet")["profit"]
                .sum()
                .sort_values(ascending=False)
                .head(10)
            )

            if len(combined_stats) > 0:
                plt.figure(figsize=(14, 6))
                colors = ["green" if x >= 0 else "red" for x in combined_stats.values]
                bars = plt.bar(
                    range(len(combined_stats)),
                    combined_stats.values,
                    color=colors,
                    alpha=0.7,
                )

                # Anota valores nas barras
                for i, bar in enumerate(bars):
                    height = bar.get_height()
                    plt.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{height:.2f}U",
                        ha="center",
                        va="bottom" if height > 0 else "top",
                    )

                plt.title("Total Profit by Combined Bet Type and Line (Top 10)")
                plt.xlabel("Combined Bet (Ranked)")
                plt.ylabel("Total Profit (U)")
                plt.xticks(
                    range(len(combined_stats)),
                    [
                        name[:25] + "..." if len(name) > 25 else name
                        for name in combined_stats.index
                    ],
                    rotation=45,
                    ha="right",
                )
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                charts.append(("Combined Bet Analysis (Top 10)", plt.gcf()))

        return charts

    except Exception as e:
        st.error(f"Erro ao criar gráficos de análise: {e}")
        return []


def map_analysis_plot(df):
    """Gráfico de análise por mapa"""
    if "game" not in df.columns or len(df) == 0:
        st.warning("No map data available for analysis.")
        return

    plt.figure(figsize=(10, 6))

    # Análise por mapa
    map_stats = (
        df.groupby("game")
        .agg({"profit": "sum", "status": lambda x: (x == "win").mean()})
        .round(3)
    )

    # Remove mapas sem dados válidos
    map_stats = map_stats.dropna()

    if len(map_stats) == 0:
        st.warning("No valid map data for analysis.")
        return

    map_stats.columns = ["Total Profit", "Win Rate"]
    map_stats.index = [f"Map {i}" for i in map_stats.index]

    # Subplot com 2 gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de profit por mapa
    colors = ["green" if x >= 0 else "red" for x in map_stats["Total Profit"]]
    bars1 = ax1.bar(map_stats.index, map_stats["Total Profit"], color=colors, alpha=0.7)
    ax1.set_title("Total Profit by Map")
    ax1.set_ylabel("Total Profit (U)")
    ax1.tick_params(axis="x", rotation=45)
    ax1.grid(True, alpha=0.3)

    # Anota valores nas barras
    for bar in bars1:
        height = bar.get_height()
        if not np.isnan(height):
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

    # Gráfico de win rate por mapa
    bars2 = ax2.bar(map_stats.index, map_stats["Win Rate"], color="skyblue", alpha=0.7)
    ax2.set_title("Win Rate by Map")
    ax2.set_ylabel("Win Rate")
    ax2.tick_params(axis="x", rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Anota percentuais
    for bar in bars2:
        height = bar.get_height()
        if not np.isnan(height):
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1%}",
                ha="center",
                va="bottom",
            )

    plt.tight_layout()
    st.pyplot(fig)


def get_melhores_apostas():
    """
    Pega as melhores apostas pendentes do DIA ATUAL baseado em ROI
    """
    try:
        df_pending = load_pending_bets()

        if len(df_pending) == 0:
            return pd.DataFrame()

        # Converte data para datetime
        df_pending = ensure_datetime(df_pending, "date")

        # Filtra apenas apostas do dia atual
        hoje = date.today()
        df_hoje = df_pending[df_pending["date"].dt.date == hoje].copy()

        if len(df_hoje) == 0:
            return pd.DataFrame()

        df_hoje["ROI_num"] = (
            df_hoje["ROI"].astype(str).str.replace("%", "").astype(float)
        )
        df_hoje["odds_num"] = pd.to_numeric(df_hoje["odds"], errors="coerce")
        df_hoje = df_hoje.dropna(subset=["ROI_num", "odds_num"])
        df_hoje["jogo_id"] = (
            df_hoje["t1"].astype(str) + " vs " + df_hoje["t2"].astype(str)
        )
        df_hoje["score"] = df_hoje["ROI_num"]

        melhores = (
            df_hoje.groupby("jogo_id")
            .apply(lambda x: x.nlargest(min(2, len(x)), "score"))
            .reset_index(drop=True)
        )

        if len(melhores) > 10:
            melhores = melhores.nlargest(10, "score")

        return melhores

    except Exception as e:
        st.error(f"Erro ao carregar apostas do dia: {e}")
        return pd.DataFrame()


def get_apostas_amanha():
    """
    Pega as melhores apostas pendentes para AMANHÃ baseado em ROI
    """
    try:
        df_pending = load_pending_bets()

        if len(df_pending) == 0:
            return pd.DataFrame()

        # Converte data para datetime
        df_pending = ensure_datetime(df_pending, "date")

        # Filtra apenas apostas de amanhã
        amanha = date.today() + timedelta(days=1)
        df_amanha = df_pending[df_pending["date"].dt.date == amanha].copy()

        if len(df_amanha) == 0:
            return pd.DataFrame()

        df_amanha["ROI_num"] = (
            df_amanha["ROI"].astype(str).str.replace("%", "").astype(float)
        )
        df_amanha["odds_num"] = pd.to_numeric(df_amanha["odds"], errors="coerce")
        df_amanha = df_amanha.dropna(subset=["ROI_num", "odds_num"])
        df_amanha["jogo_id"] = (
            df_amanha["t1"].astype(str) + " vs " + df_amanha["t2"].astype(str)
        )
        df_amanha["score"] = df_amanha["ROI_num"]

        melhores = (
            df_amanha.groupby("jogo_id")
            .apply(lambda x: x.nlargest(min(2, len(x)), "score"))
            .reset_index(drop=True)
        )

        if len(melhores) > 10:
            melhores = melhores.nlargest(10, "score")

        return melhores

    except Exception as e:
        st.error(f"Erro ao carregar apostas de amanhã: {e}")
        return pd.DataFrame()


def display_apostas_do_dia():
    """
    Função para exibir apostas recomendadas do dia atual em formato de tabela
    """
    st.markdown("---")
    st.markdown("## 🌟 Apostas Recomendadas do Dia")

    melhores_apostas = get_melhores_apostas()

    if len(melhores_apostas) == 0:
        hoje = date.today()
        st.info(
            f"📭 Nenhuma aposta encontrada para hoje ({hoje.strftime('%d/%m/%Y')}). As apostas aparecem aqui apenas para o dia atual."
        )
        return

    display_cols = [
        "date",
        "league",
        "t1",
        "t2",
        "bet_type",
        "bet_line",
        "ROI",
        "odds",
        "House",
    ]
    available_cols = [col for col in display_cols if col in melhores_apostas.columns]

    tabela_apostas = melhores_apostas[available_cols].copy()
    tabela_apostas["Ranking"] = range(1, len(tabela_apostas) + 1)

    cols_ordem = ["Ranking"] + [
        col for col in available_cols if col in tabela_apostas.columns
    ]
    tabela_apostas = tabela_apostas[cols_ordem]

    rename_dict = {
        "date": "Data/Hora",
        "league": "Liga",
        "t1": "Time 1",
        "t2": "Time 2",
        "bet_type": "Tipo Aposta",
        "bet_line": "Linha",
        "odds": "Odd",
        "House": "Casa",
    }

    tabela_apostas = tabela_apostas.rename(columns=rename_dict)

    total_apostas = len(melhores_apostas)
    roi_medio = melhores_apostas["ROI_num"].mean()
    melhor_roi = melhores_apostas["ROI_num"].max()
    hoje = date.today()

    st.write(
        f"**📅 {hoje.strftime('%d/%m/%Y')} • 📊 {total_apostas} apostas selecionadas** • ROI médio: **{roi_medio:.1f}%** • Melhor ROI: **{melhor_roi:.1f}%**"
    )

    st.dataframe(tabela_apostas, use_container_width=True, hide_index=True, height=400)


def display_apostas_amanha():
    """
    Função para exibir apostas recomendadas de amanhã em formato de tabela
    """
    st.markdown("---")
    st.markdown("## 🌅 Apostas Recomendadas para Amanhã")

    apostas_amanha = get_apostas_amanha()

    if len(apostas_amanha) == 0:
        amanha = date.today() + timedelta(days=1)
        st.info(
            f"📭 Nenhuma aposta encontrada para amanhã ({amanha.strftime('%d/%m/%Y')})."
        )
        return

    display_cols = [
        "date",
        "league",
        "t1",
        "t2",
        "bet_type",
        "bet_line",
        "ROI",
        "odds",
        "House",
    ]
    available_cols = [col for col in display_cols if col in apostas_amanha.columns]

    tabela_apostas = apostas_amanha[available_cols].copy()
    tabela_apostas["Ranking"] = range(1, len(tabela_apostas) + 1)

    cols_ordem = ["Ranking"] + [
        col for col in available_cols if col in tabela_apostas.columns
    ]
    tabela_apostas = tabela_apostas[cols_ordem]

    rename_dict = {
        "date": "Data/Hora",
        "league": "Liga",
        "t1": "Time 1",
        "t2": "Time 2",
        "bet_type": "Tipo Aposta",
        "bet_line": "Linha",
        "odds": "Odd",
        "House": "Casa",
    }

    tabela_apostas = tabela_apostas.rename(columns=rename_dict)

    total_apostas = len(apostas_amanha)
    roi_medio = apostas_amanha["ROI_num"].mean()
    melhor_roi = apostas_amanha["ROI_num"].max()
    amanha = date.today() + timedelta(days=1)

    st.write(
        f"**📅 {amanha.strftime('%d/%m/%Y')} • 📊 {total_apostas} apostas selecionadas** • ROI médio: **{roi_medio:.1f}%** • Melhor ROI: **{melhor_roi:.1f}%**"
    )

    st.dataframe(tabela_apostas, use_container_width=True, hide_index=True, height=400)


def display_key_insights(df):
    """
    Exibe insights automáticos baseados nos dados de apostas
    """
    if len(df) == 0:
        return

    st.markdown("---")
    st.subheader("🔍 Key Insights")

    try:
        # ===== ANÁLISES PRINCIPAIS =====

        # 1. Liga mais lucrativa
        liga_stats = (
            df.groupby("league")
            .agg({"profit": ["sum", "count"], "status": lambda x: (x == "win").mean()})
            .round(2)
        )
        liga_stats.columns = ["Total_Profit", "Total_Bets", "Win_Rate"]
        liga_stats = liga_stats[liga_stats["Total_Bets"] >= 3]  # Mínimo 3 apostas

        if len(liga_stats) > 0:
            melhor_liga = liga_stats.loc[liga_stats["Total_Profit"].idxmax()]
            melhor_liga_nome = liga_stats["Total_Profit"].idxmax()

        # 2. Time mais lucrativo (combinando t1 e t2)
        times_t1 = df.groupby("t1")["profit"].sum()
        times_t2 = df.groupby("t2")["profit"].sum()

        # Combina profits de quando o time é t1 ou t2
        all_teams = set(df["t1"].unique()) | set(df["t2"].unique())
        team_profits = {}
        for team in all_teams:
            profit_t1 = times_t1.get(team, 0)
            profit_t2 = times_t2.get(team, 0)
            team_profits[team] = profit_t1 + profit_t2

        if team_profits:
            melhor_time = max(team_profits, key=team_profits.get)
            melhor_time_profit = team_profits[melhor_time]

        # 3. Tipo de aposta mais lucrativo
        bet_type_stats = (
            df.groupby("bet_type")
            .agg({"profit": ["sum", "count"], "status": lambda x: (x == "win").mean()})
            .round(2)
        )
        bet_type_stats.columns = ["Total_Profit", "Total_Bets", "Win_Rate"]
        bet_type_stats = bet_type_stats[bet_type_stats["Total_Bets"] >= 3]

        if len(bet_type_stats) > 0:
            melhor_bet_type = bet_type_stats.loc[
                bet_type_stats["Total_Profit"].idxmax()
            ]
            melhor_bet_type_nome = bet_type_stats["Total_Profit"].idxmax()

        # 4. Casa de apostas mais lucrativa
        house_stats = (
            df.groupby("House")
            .agg({"profit": ["sum", "count"], "status": lambda x: (x == "win").mean()})
            .round(2)
        )
        house_stats.columns = ["Total_Profit", "Total_Bets", "Win_Rate"]
        house_stats = house_stats[house_stats["Total_Bets"] >= 3]

        if len(house_stats) > 0:
            melhor_house = house_stats.loc[house_stats["Total_Profit"].idxmax()]
            melhor_house_nome = house_stats["Total_Profit"].idxmax()

        # 5. Faixa de ROI mais eficiente
        df["roi_range"] = pd.cut(
            df["ROI"],
            bins=[0, 5, 10, 15, 20, 100],
            labels=["0-5%", "5-10%", "10-15%", "15-20%", "20%+"],
        )
        roi_stats = (
            df.groupby("roi_range")
            .agg({"profit": ["sum", "count"], "status": lambda x: (x == "win").mean()})
            .round(2)
        )
        roi_stats.columns = ["Total_Profit", "Total_Bets", "Win_Rate"]
        roi_stats = roi_stats.dropna()

        if len(roi_stats) > 0:
            melhor_roi_range = roi_stats.loc[roi_stats["Total_Profit"].idxmax()]
            melhor_roi_range_nome = roi_stats["Total_Profit"].idxmax()

        # ===== DISPLAY DOS INSIGHTS =====

        # Primeira linha de métricas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if len(liga_stats) > 0:
                st.metric(
                    "🏆 Liga Mais Lucrativa",
                    melhor_liga_nome,
                    f"{melhor_liga['Total_Profit']:.2f}U",
                    help=f"Win Rate: {melhor_liga['Win_Rate']:.1%} | {melhor_liga['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("🏆 Liga Mais Lucrativa", "N/A", "0.00U")

        with col2:
            if team_profits:
                st.metric(
                    "⭐ Time Mais Lucrativo",
                    melhor_time,
                    f"{melhor_time_profit:.2f}U",
                    help="Soma dos lucros quando este time participa",
                )
            else:
                st.metric("⭐ Time Mais Lucrativo", "N/A", "0.00U")

        with col3:
            if len(bet_type_stats) > 0:
                st.metric(
                    "🎯 Tipo Aposta Top",
                    melhor_bet_type_nome,
                    f"{melhor_bet_type['Total_Profit']:.2f}U",
                    help=f"Win Rate: {melhor_bet_type['Win_Rate']:.1%} | {melhor_bet_type['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("🎯 Tipo Aposta Top", "N/A", "0.00U")

        with col4:
            if len(house_stats) > 0:
                st.metric(
                    "🏠 Casa Mais Lucrativa",
                    melhor_house_nome,
                    f"{melhor_house['Total_Profit']:.2f}U",
                    help=f"Win Rate: {melhor_house['Win_Rate']:.1%} | {melhor_house['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("🏠 Casa Mais Lucrativa", "N/A", "0.00U")

        # Segunda linha de métricas
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            if len(roi_stats) > 0:
                st.metric(
                    "📊 ROI Range Ideal",
                    melhor_roi_range_nome,
                    f"{melhor_roi_range['Total_Profit']:.2f}U",
                    help=f"Win Rate: {melhor_roi_range['Win_Rate']:.1%} | {melhor_roi_range['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("📊 ROI Range Ideal", "N/A", "0.00U")

        with col2:
            # Streak atual
            df_sorted = df.sort_values(["date", "game"]).reset_index(drop=True)
            current_streak = 0
            streak_type = ""

            if len(df_sorted) > 0:
                last_status = df_sorted.iloc[-1]["status"]
                for i in range(len(df_sorted) - 1, -1, -1):
                    if df_sorted.iloc[i]["status"] == last_status:
                        current_streak += 1
                    else:
                        break
                streak_type = "W" if last_status == "win" else "L"

            st.metric(
                "🔥 Streak Atual",
                f"{current_streak}{streak_type}",
                f"{'🟢' if streak_type == 'W' else '🔴'}",
                help="Sequência atual de vitórias (W) ou derrotas (L)",
            )

        with col3:
            # Melhor mês
            if "date" in df.columns and not df["date"].isna().all():
                df["month_year"] = df["date"].dt.strftime("%m/%Y")
                month_stats = (
                    df.groupby("month_year")["profit"]
                    .sum()
                    .sort_values(ascending=False)
                )
                if len(month_stats) > 0:
                    melhor_mes = month_stats.index[0]
                    melhor_mes_profit = month_stats.iloc[0]
                    st.metric(
                        "📅 Melhor Mês",
                        melhor_mes,
                        f"{melhor_mes_profit:.2f}U",
                        help="Mês com maior lucro total",
                    )
                else:
                    st.metric("📅 Melhor Mês", "N/A", "0.00U")
            else:
                st.metric("📅 Melhor Mês", "N/A", "0.00U")

        with col4:
            # Odds sweet spot - usando mesma categorização do gráfico principal
            def categorize_odds_insights(odd_value):
                if pd.isna(odd_value):
                    return None
                elif odd_value < 1.5:
                    return "< 1.5"
                elif 1.5 <= odd_value < 2.0:
                    return "1.5 - 1.99"
                elif 2.0 <= odd_value < 2.5:
                    return "2.0 - 2.49"
                elif 2.5 <= odd_value < 3.0:
                    return "2.5 - 2.99"
                else:
                    return "≥ 3.0"

            df["odds_range"] = df["odds"].apply(categorize_odds_insights)
            odds_stats = (
                df.groupby("odds_range")
                .agg({"profit": "sum", "status": lambda x: (x == "win").mean()})
                .round(2)
            )
            odds_stats = odds_stats.dropna()

            if len(odds_stats) > 0:
                melhor_odds_range = odds_stats.loc[odds_stats["profit"].idxmax()]
                melhor_odds_range_nome = odds_stats["profit"].idxmax()
                st.metric(
                    "🎲 Odds Sweet Spot",
                    melhor_odds_range_nome,
                    f"{melhor_odds_range['profit']:.2f}U",
                    help=f"Win Rate: {melhor_odds_range['status']:.1%}",
                )
            else:
                st.metric("🎲 Odds Sweet Spot", "N/A", "0.00U")

        # ===== ANÁLISES DETALHADAS =====

        # Tabelas de top performers
        st.markdown("### 📋 Top Performers Detalhado")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🏆 Top 5 Ligas por Lucro**")
            if len(liga_stats) > 0:
                top_ligas = liga_stats.nlargest(5, "Total_Profit")[
                    ["Total_Profit", "Win_Rate", "Total_Bets"]
                ]
                top_ligas.columns = ["Lucro (U)", "Win Rate", "Apostas"]
                top_ligas["Win Rate"] = top_ligas["Win Rate"].apply(
                    lambda x: f"{x:.1%}"
                )
                top_ligas["Apostas"] = top_ligas["Apostas"].astype(int)
                st.dataframe(top_ligas, use_container_width=True)
            else:
                st.info("Dados insuficientes")

        with col2:
            st.markdown("**⭐ Top 5 Times por Lucro**")
            if team_profits:
                top_teams_df = pd.DataFrame(
                    list(team_profits.items()), columns=["Time", "Lucro"]
                )
                top_teams_df = top_teams_df.nlargest(5, "Lucro").set_index("Time")
                top_teams_df.columns = ["Lucro (U)"]
                st.dataframe(top_teams_df, use_container_width=True)
            else:
                st.info("Dados insuficientes")

        # Insights adicionais em texto
        st.markdown("### 💡 Insights Automáticos")

        insights = []

        # Insight sobre consistency
        if len(liga_stats) > 0:
            consistent_leagues = liga_stats[liga_stats["Win_Rate"] >= 0.6]
            if len(consistent_leagues) > 0:
                best_consistent = consistent_leagues.loc[
                    consistent_leagues["Total_Profit"].idxmax()
                ]
                insights.append(
                    f"🎯 **Liga mais consistente**: {consistent_leagues['Total_Profit'].idxmax()} com {best_consistent['Win_Rate']:.1%} de win rate e {best_consistent['Total_Profit']:.2f}U de lucro"
                )

        # Insight sobre ROI vs Volume
        if len(roi_stats) > 1:
            high_volume_roi = roi_stats[
                roi_stats["Total_Bets"] >= roi_stats["Total_Bets"].median()
            ]
            if len(high_volume_roi) > 0:
                best_volume_roi = high_volume_roi.loc[
                    high_volume_roi["Total_Profit"].idxmax()
                ]
                insights.append(
                    f"📊 **Melhor ROI com alto volume**: {high_volume_roi['Total_Profit'].idxmax()} com {best_volume_roi['Total_Bets']:.0f} apostas e {best_volume_roi['Total_Profit']:.2f}U"
                )

        # Insight sobre houses
        if len(house_stats) > 1:
            reliable_houses = house_stats[house_stats["Win_Rate"] >= 0.5]
            if len(reliable_houses) > 0:
                insights.append(
                    f"🏠 **Casas mais confiáveis**: {len(reliable_houses)} casas com win rate ≥ 50%"
                )

        for insight in insights:
            st.markdown(f"- {insight}")

        if not insights:
            st.info("Execute mais apostas para gerar insights automáticos detalhados.")

    except Exception as e:
        st.error(f"Erro ao gerar insights: {e}")


# ----------------- MAIN APPLICATION LOGIC ----------------- #
def main():
    try:
        df = load_data()
        if df.empty:
            st.error(
                "Nenhum dado encontrado. Verifique se o arquivo existe em 'bets/bets_atualizadas_por_mapa.csv'"
            )
            return

        st.title("Betting Statistics Dashboard - Classic Style")

        # Filtro de Tier (TIER 1 e TIER 2 apenas)
        tier1_leagues = [
            "LCK",
            "LPL",
            "LTA",
            "LEC",
            "LCS",
            "MSI",
            "Worlds",
            "VCT Champions",
            "VCT Masters",
        ]
        tier2_leagues = ["LPLOL", "LVP", "NLC", "LCP", "LFL", "LVP SL", "Prime League"]

        all_leagues = df["league"].unique().tolist()

        # Identifica quais ligas estão disponíveis
        tier1_available = [league for league in all_leagues if league in tier1_leagues]
        tier2_available = [
            league
            for league in all_leagues
            if league in tier2_leagues or league not in tier1_leagues
        ]

        tier_filter = st.selectbox(
            "Tier de Liga:",
            options=["Todos", "🏆 TIER 1", "🥈 TIER 2"],
            index=0,
            help="TIER 1: Ligas principais (LCK, LPL, LEC, etc.)\nTIER 2: Ligas regionais (LPLOL, LVP, NLC, LCP, etc.)",
        )

        # Aplica filtro de tier
        if tier_filter == "🏆 TIER 1":
            selected_leagues = tier1_available if tier1_available else all_leagues
        elif tier_filter == "🥈 TIER 2":
            selected_leagues = tier2_available if tier2_available else all_leagues
        else:
            selected_leagues = all_leagues

        df = df[df["league"].isin(selected_leagues)]

        if df.empty:
            st.write("No data available for the selected tier.")
            return

        # Filtro de Mês - com tratamento robusto
        try:
            # Verifica se há datas válidas
            valid_dates = df["date"].notna().sum()

            if valid_dates == 0:
                st.info("Filtro de mês não disponível (datas inválidas).")
                available_month_names = ["All Months"]
                selected_month_name = st.selectbox(
                    "Select a month:", options=available_month_names
                )
            else:
                df["month"] = df["date"].dt.month
                df = df.dropna(subset=["month"])
                available_months = sorted(
                    [m for m in df["month"].unique() if not pd.isna(m)]
                )

                months = {
                    0: "All Months",
                    1: "January",
                    2: "February",
                    3: "March",
                    4: "April",
                    5: "May",
                    6: "June",
                    7: "July",
                    8: "August",
                    9: "September",
                    10: "October",
                    11: "November",
                    12: "December",
                }

                if available_months:
                    available_month_names = [months[0]] + [
                        months[int(month)] for month in available_months
                    ]
                else:
                    available_month_names = [months[0]]

                selected_month_name = st.selectbox(
                    "Select a month:", options=available_month_names
                )

                if selected_month_name != "All Months":
                    selected_month_num = list(months.values()).index(
                        selected_month_name
                    )
                    df = df[df["month"] == selected_month_num]

        except Exception as e:
            st.warning(f"Erro no filtro de mês: {e}")

        # Filtro de ROI
        try:
            df = df.dropna(subset=["ROI"])
            df = df[df["ROI"].notna()]

            if len(df) == 0:
                st.error("Nenhum dado válido após filtros.")
                return

            min_available_roi = float(df["ROI"].min())
            max_available_roi = float(df["ROI"].max())

            if pd.isna(min_available_roi) or pd.isna(max_available_roi):
                st.error("Valores de ROI inválidos.")
                return

            if min_available_roi == max_available_roi:
                st.write(f"Apenas um ROI disponível: {min_available_roi}%")
                chosen_roi = min_available_roi
            else:
                chosen_roi = st.slider(
                    "Choose Minimum ROI (%)",
                    int(10),
                    int(max_available_roi),
                    int(10),
                )
        except Exception as e:
            st.error(f"Erro no filtro de ROI: {e}")
            return

        # Processa dados
        try:
            processed_df = process_data(df, chosen_roi, max_available_roi)

            if processed_df.empty:
                st.write(f"No data available for ROI >= {chosen_roi}%")
                return

            # Remove NaN críticos
            processed_df = processed_df.dropna(subset=["profit", "odds", "status"])

            if len(processed_df) == 0:
                st.error("Nenhum dado válido após processamento.")
                return

        except Exception as e:
            st.error(f"Erro no processamento de dados: {e}")
            return

        # Exibe resumo
        display_summary(processed_df, chosen_roi)

        # Exibe gráficos
        st.markdown("---")
        st.subheader("📈 Bankroll Evolution")
        try:
            bankroll_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico de bankroll: {e}")

        display_key_insights(processed_df)

        display_apostas_do_dia()

        display_apostas_amanha()

        # Seção de Apostas em Vigor
        st.markdown("---")
        st.subheader("🔄 Apostas em Vigor")

        try:
            # Carrega todas as apostas
            all_bets = (
                load_pending_bets()
            )  # Assumindo que esta função carrega o CSV completo

            # Filtra apenas apostas com status 'pending'
            pending_bets = all_bets[all_bets["status"] == "pending"].copy()

            if len(pending_bets) > 0:
                # Converte a coluna 'date' para datetime para ordenação correta
                try:
                    pending_bets["date_parsed"] = pd.to_datetime(
                        pending_bets["date"], format="%d %b %Y %H:%M"
                    )
                except:
                    # Fallback para outros formatos de data possíveis
                    pending_bets["date_parsed"] = pd.to_datetime(
                        pending_bets["date"], errors="coerce"
                    )

                # Ordena por data (mais próxima primeiro)
                pending_bets = pending_bets.sort_values("date_parsed", ascending=True)

                # Remove a coluna auxiliar de data parseada
                pending_bets = pending_bets.drop("date_parsed", axis=1)

                st.write(
                    f"📊 **{len(pending_bets)} apostas pendentes** (ordenadas por data mais próxima)"
                )

                # Seleciona colunas mais relevantes para exibição
                display_cols = [
                    col
                    for col in [
                        "date",
                        "league",
                        "t1",
                        "t2",
                        "bet_type",
                        "bet_line",
                        "ROI",
                        "odds",
                        "House",
                    ]
                    if col in pending_bets.columns
                ]

                # Mostra tabela de apostas pendentes
                st.dataframe(
                    pending_bets[display_cols],
                    use_container_width=True,
                    height=400,
                    hide_index=True,
                )
            else:
                st.info(
                    "📭 Nenhuma aposta pendente encontrada no arquivo bets/bets.csv"
                )

        except Exception as e:
            st.error(f"Erro ao carregar apostas pendentes: {e}")
            st.error(
                "Verifique se o arquivo bets/bets.csv existe e contém a coluna 'status'"
            )

        # Dados Detalhados
        st.markdown("---")
        st.subheader("📊 Dados Detalhados")

        try:
            st.markdown("#### 📋 Todas as Apostas Processadas")
            st.write(
                f"**{len(processed_df)} apostas** (ordenadas por data mais recente primeiro)"
            )

            # Ordena por data (mais recente primeiro)
            display_df = processed_df.copy()
            if pd.api.types.is_datetime64_any_dtype(display_df["date"]):
                display_df = display_df.sort_values("date", ascending=False)

            display_cols = [
                col
                for col in [
                    "date",
                    "league",
                    "t1",
                    "t2",
                    "bet_type",
                    "bet_line",
                    "bet_result",
                    "odds",
                    "status",
                    "profit",
                    "game",
                ]
                if col in display_df.columns
            ]

            # Mostra todas as apostas com scroll
            st.dataframe(
                display_df[display_cols],
                use_container_width=True,
                height=500,
            )
        except Exception as e:
            st.error(f"Erro ao mostrar dados detalhados: {e}")

        st.markdown("---")
        st.subheader("🎯 Profitability Analysis by Odds Range")
        try:
            odds_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico de odds: {e}")

        st.markdown("---")
        st.subheader("📊 Bet Groups Distribution")
        try:
            bet_groups_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico de grupos: {e}")

        st.markdown("---")
        st.subheader("💰 Profit Analysis")
        try:
            profit_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico de profit: {e}")

        st.markdown("---")
        st.subheader("🗺️ Map Analysis")
        try:
            map_analysis_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico de mapas: {e}")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# ----------------- ENTRY POINT ----------------- #
if __name__ == "__main__":
    main()
