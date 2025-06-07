import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import date, timedelta

from config import load_pending_bets, ensure_datetime
from strategy import apply_strategy_to_pending_bets, get_strategy_summary


# ----------------- SUMÁRIO DE MÉTRICAS ----------------- #
def display_summary(df, roi_value, strategy_active=False):
    """Exibe resumo LIMPO das estatísticas - foco em ROI e Lucro"""
    try:
        total_profit = (
            float(df["profit"].sum()) if not df["profit"].isna().all() else 0.0
        )
        total_bets = len(df)
        wins = len(df[df["status"] == "win"])
        win_rate = wins / total_bets if total_bets != 0 else 0.0
        avg_odd = float(df["odds"].mean()) if not df["odds"].isna().all() else 0.0
        roi_real = (total_profit / total_bets * 100) if total_bets > 0 else 0.0

        # Garante que não há NaN
        if pd.isna(total_profit):
            total_profit = 0.0
        if pd.isna(avg_odd):
            avg_odd = 0.0
        if pd.isna(roi_real):
            roi_real = 0.0

        # Display em colunas - ajustado conforme estratégia ativa
        if strategy_active:
            # Quando estratégia ativa: foco total em performance
            col1, col2, col3, col4 = st.columns(4)
            col1.metric(label="💰 Lucro Total", value=f"{total_profit:.2f}U")
            col2.metric(label="📊 ROI Real", value=f"{roi_real:.1f}%")
            col3.metric(label="🎲 Total Apostas", value=f"{total_bets}")
            col4.metric(
                label="📈 Lucro/Aposta",
                value=f"{total_profit / total_bets:.3f}U"
                if total_bets > 0
                else "0.000U",
            )
        else:
            # Quando sem estratégia: inclui ROI escolhido e odd média
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric(label="💰 Lucro Total", value=f"{total_profit:.2f}U")
            col2.metric(label="📊 ROI Real", value=f"{roi_real:.1f}%")
            col3.metric(label="🎯 ROI Escolhido", value=f"{roi_value}%")
            col4.metric(label="🎲 Total Apostas", value=f"{total_bets}")
            col5.metric(label="📈 Odd Média", value=f"{avg_odd:.2f}")

    except Exception as e:
        st.error(f"Erro no cálculo do resumo: {e}")


# ----------------- GRÁFICOS BÁSICOS ----------------- #


def bankroll_plot(df):
    """Gráfico de evolução do bankroll geral e por games"""
    if len(df) == 0:
        st.warning("Nenhum dado disponível para gráfico de bankroll.")
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
        categories["Todos os Games"] = df_finished.copy()

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
            "Evolução do Bankroll por Categoria de Game",
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
                    "Categoria": category,
                    "Total Apostas": total_bets,
                    "Vitórias": wins,
                    "Derrotas": losses,
                    "Taxa Vitória (%)": winrate,
                    "Lucro Total": total_profit,
                    "ROI Médio (%)": avg_roi,
                }
            )

        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
        plt.ylabel("Lucro Cumulativo (unidades)", fontsize=12, color="white")
        plt.xlabel("Sequência de Apostas", fontsize=12, color="white")
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
            label="Lucro Cumulativo",
        )

        # Linha de referência zero
        plt.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        plt.title(
            "Evolução do Bankroll ao Longo das Apostas", fontsize=16, fontweight="bold"
        )
        plt.ylabel("Lucro Cumulativo (unidades)", fontsize=12)
        plt.xlabel("Sequência de Apostas", fontsize=12)
        plt.legend(loc="best")  # Legenda dentro do gráfico
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(ax.get_figure())

    except Exception as e:
        st.error(f"Erro no plot simples de bankroll: {e}")


# ----------------- GRÁFICOS AVANÇADOS ----------------- #


def league_profit_plot(df: pd.DataFrame) -> None:
    """Gráfico de lucro total por liga."""
    if df.empty or "league" not in df:
        st.warning("Dados insuficientes para análise por liga.")
        return
    stats = (
        df.groupby("league")
        .agg(
            Lucro_Total=("profit", "sum"),
            Total_Apostas=("profit", "count"),
            Win_Rate=("status", lambda x: (x == "win").mean()),
        )
        .loc[lambda x: x["Total_Apostas"] >= 3]
        .sort_values("Lucro_Total")
    )
    if stats.empty:
        st.warning("Nenhuma liga com pelo menos 3 apostas.")
        return
    colors = ["green" if v >= 0 else "red" for v in stats["Lucro_Total"]]
    plt.figure(figsize=(12, 6))
    plt.barh(stats.index, stats["Lucro_Total"], color=colors)
    plt.axvline(0, color="black", linestyle="--")
    plt.xlabel("Lucro Total (U)")
    plt.title("Lucro Total por Liga")
    plt.tight_layout()
    st.pyplot(plt.gcf())


def odds_plot(df: pd.DataFrame) -> None:
    """Análise de lucratividade por faixas de odds."""
    if df.empty or df["odds"].isna().all():
        st.warning("Nenhum dado de odds disponível.")
        return

    def categorize(x):
        if x < 1.5:
            return "<1.5"
        elif x < 2.1:
            return "1.5-2.1"
        elif x < 2.5:
            return "2.0-2.49"
        elif x < 3.0:
            return "2.5-2.99"
        else:
            return "≥3.0"

    df_copy = df.copy()
    df_copy["odds_cat"] = df_copy["odds"].apply(categorize)
    stats = (
        df_copy.groupby("odds_cat")
        .agg(
            Lucro_Total=("profit", "sum"),
            Total_Apostas=("profit", "count"),
        )
        .reindex(["<1.5", "1.5-2.10", "2.0-2.49", "2.5-2.99", "≥3.0"])
        .dropna()
    )
    if stats.empty:
        st.warning("Dados insuficientes para análise de odds.")
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["green" if v >= 0 else "red" for v in stats["Lucro_Total"]]
    axes[0].bar(stats.index, stats["Lucro_Total"], color=colors)
    axes[0].set_title("Lucro por Faixa de Odds")
    axes[1].bar(stats.index, stats["Total_Apostas"], color="blue")
    axes[1].set_title("Número de Apostas por Faixa")
    for ax in axes:
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    st.pyplot(fig)


def bet_groups_plot(df):
    """Gráfico de distribuição por grupos de apostas"""
    if len(df) == 0:
        st.warning("Nenhum dado disponível para gráfico de grupos de apostas.")
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
        st.warning("Nenhum dado válido para análise de grupos de apostas.")
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

    plt.title("Distribuição de Grupos de Apostas com Vitórias e Derrotas")
    plt.ylabel("Número de Apostas")
    plt.xlabel("Grupo de Apostas")
    plt.xticks(rotation=45)
    plt.legend(title="Status", loc="upper right")
    plt.tight_layout()
    st.pyplot(ax.get_figure())


def profit_plot(df):
    """Gráfico de lucro por grupo de apostas"""
    if len(df) == 0:
        st.warning("Nenhum dado disponível para gráfico de lucro.")
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

        plt.title("Lucro por Grupo e Tipo de Aposta")
        plt.ylabel("Lucro Total (U)")
        plt.xlabel("Grupo de Apostas")
        plt.xticks(rotation=45)
        plt.legend(title="Tipo de Aposta", loc="upper left")
    else:
        plt.text(
            0.5,
            0.5,
            "Nenhum dado disponível",
            ha="center",
            va="center",
            transform=plt.gca().transAxes,
            fontsize=16,
        )
        plt.title("Lucro por Grupo e Tipo de Aposta")

    plt.tight_layout()
    st.pyplot(ax.get_figure())


def map_analysis_plot(df):
    """Gráfico de análise por mapa"""
    if "game" not in df.columns or len(df) == 0:
        st.warning("Nenhum dado de mapa disponível para análise.")
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
        st.warning("Nenhum dado válido de mapa para análise.")
        return

    map_stats.columns = ["Lucro Total", "Taxa Vitória"]
    map_stats.index = [f"Mapa {i}" for i in map_stats.index]

    # Subplot com 2 gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Gráfico de profit por mapa
    colors = ["green" if x >= 0 else "red" for x in map_stats["Lucro Total"]]
    bars1 = ax1.bar(map_stats.index, map_stats["Lucro Total"], color=colors, alpha=0.7)
    ax1.set_title("Lucro Total por Mapa")
    ax1.set_ylabel("Lucro Total (U)")
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
    bars2 = ax2.bar(
        map_stats.index, map_stats["Taxa Vitória"], color="skyblue", alpha=0.7
    )
    ax2.set_title("Taxa de Vitória por Mapa")
    ax2.set_ylabel("Taxa de Vitória")
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


# ----------------- APOSTAS RECOMENDADAS COM ESTRATÉGIA ----------------- #


def get_melhores_apostas(apply_strategy_filter=False):
    """
    Pega as melhores apostas pendentes do DIA ATUAL

    Args:
        apply_strategy_filter (bool): Se True, aplica a estratégia otimizada
    """
    try:
        df_pending = load_pending_bets()
        print(f"DEBUG get_melhores_apostas - Apostas carregadas: {len(df_pending)}")

        if len(df_pending) == 0:
            return pd.DataFrame()

        # Converte data para datetime
        df_pending = ensure_datetime(df_pending, "date")

        # Remove apostas com datas inválidas
        df_hoje = df_pending[df_pending["date"].notna()].copy()
        print(f"DEBUG get_melhores_apostas - Apostas com datas válidas: {len(df_hoje)}")

        if len(df_hoje) == 0:
            return pd.DataFrame()

        # Filtra apenas apostas do dia atual
        hoje = pd.Timestamp.now().date()
        print(f"DEBUG get_melhores_apostas - Data de hoje: {hoje}")

        # Converte coluna de data para date para comparação
        df_hoje["date_only"] = df_hoje["date"].dt.date

        # Filtra por hoje
        df_hoje = df_hoje[df_hoje["date_only"] == hoje].copy()
        print(f"DEBUG get_melhores_apostas - Apostas de hoje: {len(df_hoje)}")

        if len(df_hoje) == 0:
            print("DEBUG get_melhores_apostas - Nenhuma aposta encontrada para hoje")
            return pd.DataFrame()

        # Aplica estratégia se solicitado
        if apply_strategy_filter:
            print("DEBUG get_melhores_apostas - Aplicando estratégia otimizada")
            df_hoje = apply_strategy_to_pending_bets(df_hoje)

            if len(df_hoje) == 0:
                print(
                    "DEBUG get_melhores_apostas - Nenhuma aposta aprovada pela estratégia"
                )
                return pd.DataFrame()

        # Processa dados
        df_hoje["ROI_num"] = (
            df_hoje["ROI"].astype(str).str.replace("%", "").astype(float)
        )
        df_hoje["odds_num"] = pd.to_numeric(df_hoje["odds"], errors="coerce")
        df_hoje = df_hoje.dropna(subset=["ROI_num", "odds_num"])
        df_hoje["jogo_id"] = (
            df_hoje["t1"].astype(str) + " vs " + df_hoje["t2"].astype(str)
        )
        df_hoje["score"] = df_hoje["ROI_num"]

        print(
            f"DEBUG get_melhores_apostas - Apostas após processamento: {len(df_hoje)}"
        )

        # Seleciona melhores apostas por jogo
        melhores = (
            df_hoje.groupby("jogo_id")
            .apply(lambda x: x.nlargest(min(2, len(x)), "score"))
            .reset_index(drop=True)
        )

        # Limita a 10 apostas
        if len(melhores) > 10:
            melhores = melhores.nlargest(10, "score")

        print(
            f"DEBUG get_melhores_apostas - Melhores apostas retornadas: {len(melhores)}"
        )
        return melhores

    except Exception as e:
        print(f"DEBUG get_melhores_apostas - Erro: {e}")
        st.error(f"Erro ao carregar apostas do dia: {e}")
        return pd.DataFrame()


def get_apostas_amanha(apply_strategy_filter=False):
    """
    Pega as melhores apostas pendentes para AMANHÃ

    Args:
        apply_strategy_filter (bool): Se True, aplica a estratégia otimizada
    """
    try:
        df_pending = load_pending_bets()
        print(f"DEBUG get_apostas_amanha - Apostas carregadas: {len(df_pending)}")

        if len(df_pending) == 0:
            return pd.DataFrame()

        # Converte data para datetime
        df_pending = ensure_datetime(df_pending, "date")

        # Remove apostas com datas inválidas
        df_amanha = df_pending[df_pending["date"].notna()].copy()
        print(f"DEBUG get_apostas_amanha - Apostas com datas válidas: {len(df_amanha)}")

        if len(df_amanha) == 0:
            return pd.DataFrame()

        # Filtra apenas apostas de amanhã
        amanha = pd.Timestamp.now().date() + timedelta(days=1)
        print(f"DEBUG get_apostas_amanha - Data de amanhã: {amanha}")

        # Converte coluna de data para date para comparação
        df_amanha["date_only"] = df_amanha["date"].dt.date

        # Filtra por amanhã
        df_amanha = df_amanha[df_amanha["date_only"] == amanha].copy()
        print(f"DEBUG get_apostas_amanha - Apostas de amanhã: {len(df_amanha)}")

        if len(df_amanha) == 0:
            print("DEBUG get_apostas_amanha - Nenhuma aposta encontrada para amanhã")
            return pd.DataFrame()

        # Aplica estratégia se solicitado
        if apply_strategy_filter:
            print("DEBUG get_apostas_amanha - Aplicando estratégia otimizada")
            df_amanha = apply_strategy_to_pending_bets(df_amanha)

            if len(df_amanha) == 0:
                print(
                    "DEBUG get_apostas_amanha - Nenhuma aposta aprovada pela estratégia"
                )
                return pd.DataFrame()

        # Processa dados
        df_amanha["ROI_num"] = (
            df_amanha["ROI"].astype(str).str.replace("%", "").astype(float)
        )
        df_amanha["odds_num"] = pd.to_numeric(df_amanha["odds"], errors="coerce")
        df_amanha = df_amanha.dropna(subset=["ROI_num", "odds_num"])
        df_amanha["jogo_id"] = (
            df_amanha["t1"].astype(str) + " vs " + df_amanha["t2"].astype(str)
        )
        df_amanha["score"] = df_amanha["ROI_num"]

        print(
            f"DEBUG get_apostas_amanha - Apostas após processamento: {len(df_amanha)}"
        )

        # Seleciona melhores apostas por jogo
        melhores = (
            df_amanha.groupby("jogo_id")
            .apply(lambda x: x.nlargest(min(2, len(x)), "score"))
            .reset_index(drop=True)
        )

        # Limita a 10 apostas
        if len(melhores) > 10:
            melhores = melhores.nlargest(10, "score")

        print(
            f"DEBUG get_apostas_amanha - Melhores apostas retornadas: {len(melhores)}"
        )
        return melhores

    except Exception as e:
        print(f"DEBUG get_apostas_amanha - Erro: {e}")
        st.error(f"Erro ao carregar apostas de amanhã: {e}")
        return pd.DataFrame()


def display_apostas_section(
    df: pd.DataFrame, title: str, date_label: str, strategy_applied: bool = False
) -> None:
    """
    Exibe seção de apostas com indicação se estratégia foi aplicada
    """
    if strategy_applied:
        strategy_badge = "🎯 **ESTRATÉGIA APLICADA**"
        title_with_strategy = f"{title} - {strategy_badge}"
    else:
        title_with_strategy = title

    st.markdown("---")
    st.markdown(f"## {title_with_strategy}")

    if df.empty:
        if strategy_applied:
            st.info(f"📭 Nenhuma aposta aprovada pela estratégia para {date_label}.")
        else:
            st.info(f"📭 Nenhuma aposta encontrada para {date_label}.")
        return

    # Mostra resumo da estratégia se aplicada
    if strategy_applied:
        st.success(f"✅ {len(df)} apostas aprovadas pela Estratégia Otimizada")

        # Breakdown rápido e limpo
        if len(df) > 0 and "grouped_market" in df.columns:
            market_breakdown = df["grouped_market"].value_counts()

            # Só mostra os top 3 mercados para manter limpo
            top_markets = market_breakdown.head(3)
            breakdown_text = " | ".join(
                [
                    f"{market.replace('UNDER - ', 'U-').replace('OVER - ', 'O-')}: {count}"
                    for market, count in top_markets.items()
                ]
            )
            st.info(f"🎯 **Top Mercados**: {breakdown_text}")

            # ROI estimado médio se disponível
            if "estimated_roi" in df.columns:
                avg_roi = df["estimated_roi"].mean()
                st.info(f"📊 **ROI Médio Estimado**: {avg_roi:.1f}%")

    # Exibe tabela de apostas
    cols = [
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

    # Adiciona colunas da estratégia se disponíveis
    if strategy_applied and "grouped_market" in df.columns:
        cols.insert(-2, "grouped_market")  # Antes de ROI

    avail = [c for c in cols if c in df.columns]
    tab = df[avail].copy()
    tab["Ranking"] = range(1, len(tab) + 1)
    display_cols = ["Ranking"] + avail

    rename = {
        "date": "Data/Hora",
        "league": "Liga",
        "t1": "Time 1",
        "t2": "Time 2",
        "bet_type": "Tipo Aposta",
        "bet_line": "Linha",
        "grouped_market": "Mercado",
        "odds": "Odd",
        "House": "Casa",
    }

    st.dataframe(
        tab[display_cols].rename(columns=rename),
        use_container_width=True,
        hide_index=True,
    )


def display_apostas_do_dia(strategy_active: bool = False) -> None:
    """
    Exibe apostas recomendadas do dia

    Args:
        strategy_active (bool): Se True, aplica filtros da estratégia otimizada
    """
    melhores = get_melhores_apostas(apply_strategy_filter=strategy_active)
    display_apostas_section(
        melhores,
        "🌟 Apostas Recomendadas do Dia",
        "hoje",
        strategy_applied=strategy_active,
    )


def display_apostas_amanha(strategy_active: bool = False) -> None:
    """
    Exibe apostas recomendadas para amanhã

    Args:
        strategy_active (bool): Se True, aplica filtros da estratégia otimizada
    """
    melhores = get_apostas_amanha(apply_strategy_filter=strategy_active)
    display_apostas_section(
        melhores,
        "🌅 Apostas Recomendadas para Amanhã",
        "amanhã",
        strategy_applied=strategy_active,
    )


def display_strategy_summary():
    """
    Exibe resumo SUPER LIMPO da estratégia - apenas o essencial
    """
    strategy_info = get_strategy_summary()

    st.markdown("### 🎯 Estratégia Completa Ativa")

    # Apenas ROI esperado - o que realmente importa
    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        st.metric(
            "💰 ROI Esperado",
            strategy_info["expected_roi"],
            help="ROI histórico da estratégia aplicada",
        )

    with col2:
        st.metric(
            "🎯 Status",
            "ATIVA",
            help="Estratégia aplicada a todos os dados e recomendações",
        )

    with col3:
        st.info(
            "📊 **Filtros aplicados:** Mercados lucrativos + ROI ≥10% + Odds otimizadas"
        )

    # Critérios em formato compacto (se necessário)
    with st.expander("📋 Ver Critérios Detalhados"):
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**🎯 Mercados Aprovados:**")
            for market in strategy_info["criteria"]["markets"]:
                st.markdown(f"• {market}")

        with col2:
            st.markdown("**📊 ROI Ranges:**")
            for roi_range in strategy_info["criteria"]["roi_ranges"]:
                st.markdown(f"• {roi_range}")

            st.markdown("**🎲 Odds:**")
            st.markdown("• Todas exceto muito_baixa")


def display_key_insights(df: pd.DataFrame) -> None:
    """
    Exibe insights automáticos baseados nos dados de apostas
    """
    if len(df) == 0:
        return

    st.markdown("---")
    st.subheader("🔍 Insights Principais")

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
            df.groupby("roi_range", observed=False)
            .agg({"profit": ["sum", "count"]})
            .round(2)
        )
        roi_stats.columns = ["Total_Profit", "Total_Bets"]
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
                    help=f"{melhor_liga['Total_Bets']:.0f} apostas",
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
                    help=f"{melhor_bet_type['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("🎯 Tipo Aposta Top", "N/A", "0.00U")

        with col4:
            if len(house_stats) > 0:
                st.metric(
                    "🏠 Casa Mais Lucrativa",
                    melhor_house_nome,
                    f"{melhor_house['Total_Profit']:.2f}U",
                    help=f"{melhor_house['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("🏠 Casa Mais Lucrativa", "N/A", "0.00U")

        # Segunda linha de métricas - mais focada
        col1, col2, col3 = st.columns(3)

        with col1:
            if len(roi_stats) > 0:
                st.metric(
                    "📊 ROI Range Ideal",
                    melhor_roi_range_nome,
                    f"{melhor_roi_range['Total_Profit']:.2f}U",
                    help=f"{melhor_roi_range['Total_Bets']:.0f} apostas",
                )
            else:
                st.metric("📊 ROI Range Ideal", "N/A", "0.00U")

        with col2:
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

        with col3:
            # Odds sweet spot
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
            odds_stats = df.groupby("odds_range").agg({"profit": "sum"}).round(2)
            odds_stats = odds_stats.dropna()

            if len(odds_stats) > 0:
                melhor_odds_range = odds_stats.loc[odds_stats["profit"].idxmax()]
                melhor_odds_range_nome = odds_stats["profit"].idxmax()
                st.metric(
                    "🎲 Odds Sweet Spot",
                    melhor_odds_range_nome,
                    f"{melhor_odds_range['profit']:.2f}U",
                    help="Faixa de odds mais lucrativa",
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

        # Insights adicionais em texto - foco em ROI e Lucro
        st.markdown("### 💡 Insights Principais")

        insights = []

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
                    f"📊 **Melhor ROI com volume**: {high_volume_roi['Total_Profit'].idxmax()} com {best_volume_roi['Total_Bets']:.0f} apostas e {best_volume_roi['Total_Profit']:.2f}U"
                )

        # Insight sobre concentração de lucro
        if len(liga_stats) > 0:
            top_3_ligas = liga_stats.nlargest(3, "Total_Profit")
            total_profit_top3 = top_3_ligas["Total_Profit"].sum()
            percentage_top3 = (
                (total_profit_top3 / liga_stats["Total_Profit"].sum() * 100)
                if liga_stats["Total_Profit"].sum() != 0
                else 0
            )
            insights.append(
                f"🏆 **Concentração**: Top 3 ligas representam {percentage_top3:.1f}% do lucro total"
            )

        # Insight sobre mercados mais lucrativos
        if len(bet_type_stats) > 0:
            melhor_tipo = bet_type_stats.loc[bet_type_stats["Total_Profit"].idxmax()]
            insights.append(
                f"🎯 **Tipo de aposta mais lucrativo**: {bet_type_stats['Total_Profit'].idxmax()} com {melhor_tipo['Total_Profit']:.2f}U"
            )

        for insight in insights:
            st.markdown(f"• {insight}")

        if not insights:
            st.info("Execute mais apostas para gerar insights automáticos.")

    except Exception as e:
        st.error(f"Erro ao gerar insights: {e}")
