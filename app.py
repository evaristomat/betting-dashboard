# Standard imports
import numpy as np
import pandas as pd

# Visualization imports
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# Web application framework
import streamlit as st

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


#@st.cache_data
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


#@st.cache_data
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

        # ========== TABELA DE ESTATÍSTICAS DETALHADAS ==========
        st.subheader("📊 Detailed Statistics by Game Category")

        if stats_data:
            # Criar DataFrame a partir dos dados coletados
            stats_df = pd.DataFrame(stats_data)

            display_stats = stats_df.copy()
            display_stats["Win Rate (%)"] = display_stats["Win Rate (%)"].round(2)
            display_stats["Total Profit"] = display_stats["Total Profit"].round(2)
            display_stats["Avg ROI (%)"] = display_stats["Avg ROI (%)"].round(2)

            st.dataframe(display_stats, use_container_width=True)

            # Insights automáticos
            st.subheader("🔍 Key Insights")

            if len(display_stats) > 1:
                # Melhor categoria por profit
                best_profit_idx = display_stats["Total Profit"].idxmax()
                best_profit_cat = display_stats.loc[best_profit_idx, "Category"]
                best_profit_value = display_stats.loc[best_profit_idx, "Total Profit"]

                # Melhor winrate
                best_winrate_idx = display_stats["Win Rate (%)"].idxmax()
                best_winrate_cat = display_stats.loc[best_winrate_idx, "Category"]
                best_winrate_value = display_stats.loc[best_winrate_idx, "Win Rate (%)"]

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "🏆 Best Profit Category",
                        best_profit_cat,
                        f"{best_profit_value:.2f} units",
                    )

                with col2:
                    st.metric(
                        "🎯 Best Win Rate",
                        best_winrate_cat,
                        f"{best_winrate_value:.1f}%",
                    )

                with col3:
                    total_games = len(
                        [
                            cat
                            for cat in categories.keys()
                            if "Game" in cat and cat != "All Games"
                        ]
                    )
                    st.metric("🎮 Game Categories", total_games, "detected")

                # Análise comparativa adicional
                if total_games >= 2:
                    game_categories = display_stats[
                        display_stats["Category"].str.contains("Game ")
                    ].copy()
                    if len(game_categories) >= 2:
                        st.write("**🎮 Game-by-Game Analysis:**")

                        best_game_profit = game_categories.loc[
                            game_categories["Total Profit"].idxmax()
                        ]
                        worst_game_profit = game_categories.loc[
                            game_categories["Total Profit"].idxmin()
                        ]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.success(
                                f"📈 **Most Profitable**: {best_game_profit['Category']} with {best_game_profit['Total Profit']:.2f}U"
                            )
                        with col2:
                            if worst_game_profit["Total Profit"] < 0:
                                st.error(
                                    f"📉 **Least Profitable**: {worst_game_profit['Category']} with {worst_game_profit['Total Profit']:.2f}U"
                                )
                            else:
                                st.info(
                                    f"📊 **Least Profitable**: {worst_game_profit['Category']} with {worst_game_profit['Total Profit']:.2f}U"
                                )

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
    """Gráfico de análise por faixas de odds."""
    if len(df) == 0 or df["odds"].isna().all():
        st.warning("No odds data available for analysis.")
        return

    bins = [1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3]
    df["odds_bin"] = pd.cut(df["odds"], bins)
    grouped = (
        df.groupby(["odds_bin", "status"], observed=False).size().unstack().fillna(0)
    )

    if "win" not in grouped:
        grouped["win"] = 0
    if "loss" not in grouped:
        grouped["loss"] = 0

    mid_points = [(bins[i] + bins[i + 1]) / 2 for i in range(len(bins) - 1)]
    theoretical_probs = [1 / point for point in mid_points]
    grouped["total"] = grouped["win"] + grouped["loss"]
    grouped["win_ratio"] = grouped["win"] / grouped["total"].replace(
        0,
        np.nan,
    )  # Use NaN para divisão por zero
    grouped["edge"] = grouped["win_ratio"] - theoretical_probs

    # Remove linhas com NaN
    grouped = grouped.dropna(subset=["edge"])

    if len(grouped) == 0:
        st.warning("No valid data for odds analysis.")
        return

    plt.figure(figsize=(10, 7))

    # Fix for seaborn FutureWarning - use matplotlib instead of seaborn barplot
    colors = ["red" if x < 0 else "green" for x in grouped["edge"]]
    plt.bar(range(len(grouped)), grouped["edge"], color=colors)

    plt.axhline(0, color="black", linestyle="--")

    for i, value in enumerate(grouped["edge"]):
        if not np.isnan(value):
            plt.text(
                i,
                max(0, value),
                f"{value:.2%}",
                ha="center",
                va="bottom" if value > 0 else "top",
                fontsize=10,
            )

    plt.title("Edge Over Theoretical Probabilities by Odds Range")
    plt.ylabel("Edge (Win Rate - Theoretical Probability)")
    plt.xlabel("Odds Range")
    plt.xticks(range(len(grouped)), grouped.index, rotation=45)
    plt.tight_layout()
    fig = plt.gcf()
    st.pyplot(fig)
    plt.close()


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


def scatter_plot(df):
    """Gráfico de dispersão Fair Odds vs Actual Odds"""
    if "fair_odds" not in df.columns:
        st.info("Coluna 'fair_odds' não encontrada. Pulando gráfico de dispersão.")
        return

    colors = df["status"].map({"win": "green", "loss": "red"})

    plt.figure(figsize=(12, 6))
    plt.scatter(df["fair_odds"], df["odds"], alpha=0.6, c=colors, s=50)
    plt.title("Fair Odds vs Actual Odds with Win/Loss Overlay")
    plt.xlabel("Fair Odds")
    plt.ylabel("Actual Odds")

    # Linha de referência y=x
    max_val = max(df["fair_odds"].max(), df["odds"].max())
    min_val = min(df["fair_odds"].min(), df["odds"].min())
    plt.plot(
        [min_val, max_val], [min_val, max_val], color="blue", linestyle="--", alpha=0.7
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Win",
            markersize=10,
            markerfacecolor="green",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Loss",
            markersize=10,
            markerfacecolor="red",
        ),
    ]
    plt.legend(handles=legend_elements)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(plt.gcf())


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
                    int(min_available_roi),
                    int(max_available_roi),
                    int(min_available_roi),
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

        # Seção de Apostas em Vigor
        st.markdown("---")
        st.subheader("🔄 Apostas em Vigor")

        try:
            # Carrega todas as apostas
            all_bets = (
                load_pending_bets()
            )  # Assumindo que esta função carrega o CSV completo
            
            # Filtra apenas apostas com status 'pending'
            pending_bets = all_bets[all_bets['status'] == 'pending'].copy()

            if len(pending_bets) > 0:
                # Converte a coluna 'date' para datetime para ordenação correta
                try:
                    pending_bets['date_parsed'] = pd.to_datetime(pending_bets['date'], format='%d %b %Y %H:%M')
                except:
                    # Fallback para outros formatos de data possíveis
                    pending_bets['date_parsed'] = pd.to_datetime(pending_bets['date'], errors='coerce')
                
                # Ordena por data (mais próxima primeiro)
                pending_bets = pending_bets.sort_values('date_parsed', ascending=True)
                
                # Remove a coluna auxiliar de data parseada
                pending_bets = pending_bets.drop('date_parsed', axis=1)
                
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
                    hide_index=True
                )
            else:
                st.info(
                    "📭 Nenhuma aposta pendente encontrada no arquivo bets/bets.csv"
                )
            
        except Exception as e:
            st.error(f"Erro ao carregar apostas pendentes: {e}")
            st.error("Verifique se o arquivo bets/bets.csv existe e contém a coluna 'status'")

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
        st.subheader("🎯 Odds Analysis")
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

        st.markdown("---")
        st.subheader("⚖️ Fair vs Actual Odds")
        try:
            scatter_plot(processed_df)
        except Exception as e:
            st.error(f"Erro no gráfico scatter: {e}")


    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")


# ----------------- ENTRY POINT ----------------- #
if __name__ == "__main__":
    main()
