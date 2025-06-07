import streamlit as st
from datetime import date

from config import (
    apply_matplotlib_style,
    ensure_datetime,
    load_data,
    process_data,
)
from visualizations import (
    display_summary,
    bankroll_plot,
    league_profit_plot,
    odds_plot,
    bet_groups_plot,
    profit_plot,
    map_analysis_plot,
    display_apostas_do_dia,
    display_apostas_amanha,
    display_key_insights,
    display_strategy_summary,
)
from strategy import apply_optimized_strategy


def main():
    # ----- Estilo e cabeçalho -----
    apply_matplotlib_style()
    st.markdown(
        """
    <div style="text-align:center; margin-bottom:30px;">
      <h1 style="font-size:3.15em; color:#FFD700; margin:0;">🏆 BET.ANALYTICS PRO</h1>
      <div style="font-size:1.5em; color:#888; margin:10px 0;">━━━━━━━━━━━━━━━━━━━━━━━</div>
      <h3 style="color:#ccc; font-weight:300; margin:0;">
        V2.0 • by <a href="https://x.com/evaristomat" target="_blank" style="color:#1DA1F2;">Evaristo</a>
      </h3>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ----- Carrega dados -----
    df = load_data()
    if df.empty:
        st.error("Nenhum dado em 'bets/bets_atualizadas_por_mapa.csv'.")
        return

    # ----- Filtro de Liga -----
    tier1 = [
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
    tier2 = ["LPLOL", "LVP", "NLC", "LCP", "LFL", "LVP SL", "Prime League"]
    all_leagues = sorted(df["league"].dropna().unique())
    options = ["Todas as Ligas", "🏆 Tier 1", "🥈 Tier 2"] + [
        f"📍 {l}" for l in all_leagues
    ]
    choice = st.selectbox("Filtrar por Liga", options, index=0)

    if choice == "🏆 Tier 1":
        selected = [l for l in all_leagues if l in tier1]
    elif choice == "🥈 Tier 2":
        selected = [l for l in all_leagues if l in tier2 or l not in tier1]
    elif choice.startswith("📍 "):
        selected = [choice.replace("📍 ", "")]
    else:
        selected = all_leagues

    df = df[df["league"].isin(selected)]
    if df.empty:
        st.warning("Nenhum dado para a liga selecionada.")
        return

    # ----- Filtro de Mês -----
    df = ensure_datetime(df, "date")
    months = sorted(df["date"].dt.month.dropna().unique())
    month_map = {
        1: "Janeiro",
        2: "Fevereiro",
        3: "Março",
        4: "Abril",
        5: "Maio",
        6: "Junho",
        7: "Julho",
        8: "Agosto",
        9: "Setembro",
        10: "Outubro",
        11: "Novembro",
        12: "Dezembro",
    }
    sel_month = st.selectbox(
        "Selecionar mês", ["Todas"] + [month_map[m] for m in months]
    )
    if sel_month != "Todas":
        # obtém o número do mês a partir do nome
        num = list(month_map.values()).index(sel_month) + 1
        df = df[df["date"].dt.month == num]
        if df.empty:
            st.warning(f"Sem dados para {sel_month}.")
            return

    # ----- Toggle de Estratégia (ANTES dos filtros) -----
    st.markdown("---")
    st.markdown("### 🎯 Estratégia Otimizada")

    show_optimized = st.toggle(
        "🚀 Ativar Estratégia Otimizada COMPLETA",
        value=False,
        help="Aplica filtros da estratégia para capturar máximo lucro dos mercados selecionados",
    )

    # ----- Filtros Condicionais -----
    if not show_optimized:
        # ----- Filtro de ROI (apenas se estratégia não estiver ativa) -----
        max_roi = int(df["ROI"].max())
        default_min = max(10, int(df["ROI"].min()))
        chosen_roi = st.slider(
            "ROI Mínimo (%)", min_value=10, max_value=max_roi, value=default_min
        )
        df = df[df["ROI"] >= chosen_roi]
    else:
        # Se estratégia ativa, usa ROI mínimo padrão sem mostrar slider
        chosen_roi = 10
        df = df[df["ROI"] >= chosen_roi]

    # ----- Aplicar Estratégia e Processar Dados -----
    processed = process_data(
        df, chosen_roi, int(df["ROI"].max()) if len(df) > 0 else 100
    )
    if processed.empty:
        st.warning(f"Sem dados para os filtros aplicados.")
        return

    # ----- Aplicar Estratégia aos Dados -----
    if not show_optimized:
        df_to_show = processed.copy()
        st.info("📊 Visualizando **dados originais** filtrados por liga/mês/ROI.")
    else:
        df_to_show = apply_optimized_strategy(processed)

        if len(df_to_show) > 0:
            approval_rate = (len(df_to_show) / len(processed)) * 100
            st.success(
                f"✅ **Estratégia Ativa**: {len(df_to_show)} apostas aprovadas ({approval_rate:.1f}% do total)"
            )

            # Exibe resumo da estratégia - LIMPO
            display_strategy_summary()

        else:
            st.warning(
                "⚠️ Nenhuma aposta aprovada pela estratégia com os filtros atuais."
            )
            st.info("💡 Tente relaxar os filtros de liga/mês ou aguarde mais dados.")

    # ----- Sumário e Gráficos -----
    st.markdown("---")
    display_summary(df_to_show, chosen_roi, strategy_active=show_optimized)

    st.markdown("---")
    st.subheader("📈 Evolução do Bankroll")
    bankroll_plot(df_to_show)

    # ----- Apostas Recomendadas (com estratégia aplicada se ativa) -----
    display_apostas_do_dia(strategy_active=show_optimized)
    display_apostas_amanha(strategy_active=show_optimized)

    # ----- Key Insights -----
    display_key_insights(df_to_show)

    st.markdown("---")
    st.subheader("🏆 Lucro por Liga")
    league_profit_plot(df_to_show)

    st.markdown("---")
    st.subheader("🎲 Análise de Odds")
    odds_plot(df_to_show)

    st.markdown("---")
    st.subheader("📊 Grupos de Apostas")
    bet_groups_plot(df_to_show)

    st.markdown("---")
    st.subheader("💰 Lucro por Grupo e Tipo")
    profit_plot(df_to_show)

    st.markdown("---")
    st.subheader("🗺️ Análise por Mapa")
    map_analysis_plot(df_to_show)

    # ----- Tabela Completa -----
    st.markdown("---")
    if show_optimized:
        st.subheader("📋 Apostas da Estratégia Otimizada")
    else:
        st.subheader("📋 Todas as Apostas (Filtradas)")

    cols = [
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

    # Adiciona colunas da estratégia se ativa (apenas as mais importantes)
    if show_optimized and len(df_to_show) > 0:
        strategy_cols = [
            "grouped_market",
            "est_roi_category",
        ]  # Removido odds_category e direction para ficar mais limpo
        for col in strategy_cols:
            if col in df_to_show.columns:
                cols.insert(-2, col)  # Antes de game

    avail = [c for c in cols if c in df_to_show.columns]

    # Renomeia colunas para melhor apresentação (apenas as que estamos usando)
    rename_dict = {"grouped_market": "Mercado", "est_roi_category": "ROI Range"}

    df_display = df_to_show[avail].copy()
    df_display = df_display.rename(columns=rename_dict)

    st.dataframe(
        df_display.sort_values("date", ascending=False),
        use_container_width=True,
        height=400,
    )

    # ----- Informações Adicionais da Estratégia (SIMPLIFICADO) -----
    if show_optimized and len(df_to_show) > 0:
        st.markdown("---")
        st.markdown("### 📊 Performance da Estratégia")

        col1, col2, col3 = st.columns(3)

        # Apenas os top 3 mercados para manter limpo
        if "grouped_market" in df_to_show.columns:
            market_breakdown = df_to_show["grouped_market"].value_counts()
            with col1:
                st.markdown("**🎯 Top Mercados:**")
                for market, count in market_breakdown.head(3).items():
                    clean_name = market.replace("UNDER - ", "U-").replace(
                        "OVER - ", "O-"
                    )
                    st.markdown(f"• {clean_name}: {count}")

        # Performance geral mais limpa
        with col2:
            if "profit" in df_to_show.columns:
                total_profit = df_to_show["profit"].sum()
                roi_real = (
                    (total_profit / len(df_to_show) * 100) if len(df_to_show) > 0 else 0
                )
                st.markdown("**💰 Performance:**")
                st.markdown(f"• **{total_profit:.2f}U** de lucro")
                st.markdown(f"• **{roi_real:.1f}%** ROI real")

        # Taxa de aprovação
        with col3:
            if len(processed) > 0:
                approval_rate = (len(df_to_show) / len(processed)) * 100
                rejected = len(processed) - len(df_to_show)
                st.markdown("**📊 Seletividade:**")
                st.markdown(f"• **{approval_rate:.1f}%** aprovadas")
                st.markdown(f"• **{rejected}** apostas rejeitadas")


if __name__ == "__main__":
    main()
