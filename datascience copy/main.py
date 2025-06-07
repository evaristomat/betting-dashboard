import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")


def complete_betting_analysis_with_backtest(file_path):
    """Análise completa focada em LUCRO REAL + backtest da estratégia otimizada com ranges de ROI.
    VERSÃO COM INHIBITORS INCLUÍDOS - Odds ajustadas com nova categoria media_alta."""

    # Carrega dados
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y %H:%M", errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    print("🎯 ANÁLISE COMPLETA FOCADA EM LUCRO REAL + BACKTEST OTIMIZADO")
    print("✅ INCLUINDO MERCADOS DE INHIBITOR + ODDS AJUSTADAS")
    print("=" * 80)

    # Categorização de mercado
    def categorize_market(bet_type, bet_line):
        lt = str(bet_type).lower()
        ll = str(bet_line).lower()
        direction = "UNDER" if "under" in lt else "OVER"
        if "kill" in ll:
            market = "KILLS"
        elif "dragon" in ll:
            market = "DRAGONS"
        elif "tower" in ll:
            market = "TOWERS"
        elif "duration" in ll or "tempo" in ll:
            market = "DURATION"
        elif "baron" in ll:
            market = "BARONS"
        elif "inhibitor" in ll:
            market = "INHIBITORS"
        else:
            market = "OUTROS"
        return direction, market, f"{direction} - {market}"

    df[["direction", "market_type", "grouped_market"]] = df.apply(
        lambda r: categorize_market(r["bet_type"], r["bet_line"]),
        axis=1,
        result_type="expand",
    )

    # Nova categorização de odds com faixa media_alta
    def categorize_odds(odds):
        if pd.isna(odds):
            return "N/A"
        if odds <= 1.3:
            return "muito_baixa (0.0~~1.3)"
        elif odds <= 1.6:
            return "baixa (1.3~~1.6)"
        elif odds <= 2.0:
            return "media (1.6~~2.0)"
        elif odds <= 2.5:
            return "media_alta (2.0~~2.5)"  # nova categoria
        elif odds < 3.0:
            return "alta (2.5~~3.0)"
        else:
            return "muito_alta (3.0~~∞)"

    df["odds_category"] = df["odds"].apply(categorize_odds)

    # Métricas gerais (COM INHIBITORS)
    total = len(df)
    profit_total = df["profit"].sum()
    roi_avg = profit_total / total * 100
    win_rate = df["status"].eq("win").mean() * 100

    print(f"\n📊 DADOS GERAIS (COM INHIBITORS):")
    print(f"   Total de apostas: {total}")
    print(f"   Lucro total: {profit_total:.2f} unidades")
    print(f"   ROI médio: {roi_avg:.2f}%")
    print(f"   Win rate geral: {win_rate:.1f}%")

    # Mostrar distribuição de mercados incluindo INHIBITORS
    print(f"\n📈 DISTRIBUIÇÃO POR TIPO DE MERCADO:")
    market_distribution = df["market_type"].value_counts()
    for market, count in market_distribution.items():
        percentage = count / total * 100
        market_profit = df[df["market_type"] == market]["profit"].sum()
        print(
            f"   {market}: {count} apostas ({percentage:.1f}%) - Lucro: {market_profit:.2f}"
        )

    # ========================================================================
    # 1. ANÁLISE POR ROI ESTIMADO (RANGES MAIORES)
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("📈 ANÁLISE POR RANGES DE ROI ESTIMADO (COM INHIBITORS)")
    print("=" * 80)

    # Converter ROI estimado para numérico
    df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

    # Definir ranges de ROI maiores (cumulativos)
    def categorize_roi_ranges(roi):
        if pd.isna(roi):
            return "N/A"
        elif roi < 15:
            return "<15%"
        elif roi >= 30:
            return "≥30%"
        elif roi >= 25:
            return "≥25%"
        elif roi >= 20:
            return "≥20%"
        else:  # 15-19.99%
            return "15-20%"

    df["est_roi_category"] = df["estimated_roi"].apply(categorize_roi_ranges)

    # Análise por faixa de ROI estimado
    roi_est_analysis = (
        df.groupby("est_roi_category")
        .agg(
            {
                "profit": ["sum", "count"],
                "estimated_roi": "mean",
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    roi_est_analysis.columns = ["Total_Profit", "Bets", "Avg_Est_ROI", "Win_Rate"]
    roi_est_analysis["Real_ROI"] = (
        roi_est_analysis["Total_Profit"] / roi_est_analysis["Bets"] * 100
    ).round(2)
    roi_est_analysis["ROI_Accuracy"] = (
        roi_est_analysis["Real_ROI"] / roi_est_analysis["Avg_Est_ROI"] * 100
    ).round(1)
    roi_est_analysis = roi_est_analysis.sort_values("Real_ROI", ascending=False)

    print(f"\n🎲 PERFORMANCE POR RANGE DE ROI ESTIMADO:")
    print(
        f"{'Status':<3} {'Range':<8} {'Lucro':<10} {'ROI Real':<10} {'ROI Est':<10} {'Precisão':<10} {'Win Rate':<10} {'Apostas':<8}"
    )
    print("-" * 85)

    for range_roi, row in roi_est_analysis.iterrows():
        icon = "✅" if row["Real_ROI"] > 0 else "❌"
        print(
            f"{icon:<3} {range_roi:<8} {row['Total_Profit']:>8.2f} {row['Real_ROI']:>9.1f}% "
            f"{row['Avg_Est_ROI']:>9.1f}% {row['ROI_Accuracy']:>9.1f}% {row['Win_Rate']:>9.1f}% {int(row['Bets']):>8}"
        )

    # Selecionar ranges baseado APENAS em lucro e ROI real
    best_est_roi_categories = roi_est_analysis[
        (roi_est_analysis["Total_Profit"] > 1)  # Lucro mínimo de 1 unidade
        & (roi_est_analysis["Real_ROI"] > 0)  # ROI positivo
        & (roi_est_analysis["Bets"] >= 5)  # Volume mínimo
    ].index.tolist()

    print(
        f"\n💎 RANGES SELECIONADOS (BASEADO EM LUCRO REAL): {best_est_roi_categories}"
    )

    # Mostrar potencial de lucro
    total_profit_selected = roi_est_analysis.loc[
        best_est_roi_categories, "Total_Profit"
    ].sum()
    total_bets_selected = roi_est_analysis.loc[best_est_roi_categories, "Bets"].sum()
    weighted_roi = (
        (total_profit_selected / total_bets_selected * 100)
        if total_bets_selected > 0
        else 0
    )

    print(f"📊 POTENCIAL DOS RANGES SELECIONADOS:")
    print(f"   💰 Lucro total: {total_profit_selected:.2f} unidades")
    print(f"   🎲 Apostas: {int(total_bets_selected)}")
    print(f"   📈 ROI médio ponderado: {weighted_roi:.2f}%")

    # ========================================================================
    # 2. FILTRAR DADOS POR ROI ESTIMADO LUCRATIVO
    # ========================================================================
    df_filtered = df[df["est_roi_category"].isin(best_est_roi_categories)]

    print(f"\n📊 DADOS FILTRADOS POR RANGES DE ROI:")
    print(f"   Apostas originais: {len(df)}")
    print(f"   Apostas após filtro ROI: {len(df_filtered)}")
    print(f"   Taxa de aprovação: {len(df_filtered) / len(df) * 100:.1f}%")
    print(f"   Lucro filtrado: {df_filtered['profit'].sum():.2f} unidades")

    # ========================================================================
    # 3. ANÁLISE POR MERCADO (DADOS FILTRADOS) - INCLUINDO INHIBITORS
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("🏆 PERFORMANCE POR MERCADO (RANGES ROI FILTRADOS - COM INHIBITORS)")
    print("=" * 80)

    market_analysis = (
        df_filtered.groupby("grouped_market")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    market_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    market_analysis["ROI"] = (
        market_analysis["Total_Profit"] / market_analysis["Bets"] * 100
    ).round(2)
    market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n💰 PERFORMANCE POR MERCADO (ORDENADO POR LUCRO):")
    for mercado, row in market_analysis.iterrows():
        if row["Total_Profit"] >= 3:
            icon = "💎"  # Mercados muito lucrativos
        elif row["Total_Profit"] > 0:
            icon = "✅"  # Mercados lucrativos
        else:
            icon = "❌"  # Mercados com prejuízo

        # Destaque especial para INHIBITORS
        if "INHIBITOR" in mercado:
            icon = "🎯"  # Mercado especial INHIBITORS

        print(
            f"{icon} {mercado:<25} 💰 {row['Total_Profit']:>7.2f} | 📈 {row['ROI']:>6.1f}% | "
            f"🎯 {row['Win_Rate']:>5.1f}% | 🎲 {int(row['Bets'])}"
        )

    # Análise específica de INHIBITORS
    inhibitors_data = df_filtered[df_filtered["market_type"] == "INHIBITORS"]
    if len(inhibitors_data) > 0:
        inhibitors_profit = inhibitors_data["profit"].sum()
        inhibitors_count = len(inhibitors_data)
        inhibitors_roi = inhibitors_profit / inhibitors_count * 100
        inhibitors_wr = (inhibitors_data["status"] == "win").mean() * 100

        print(f"\n🎯 ANÁLISE ESPECÍFICA DE INHIBITORS:")
        print(f"   💰 Lucro total: {inhibitors_profit:.2f} unidades")
        print(f"   🎲 Total de apostas: {inhibitors_count}")
        print(f"   📈 ROI: {inhibitors_roi:.2f}%")
        print(f"   🎯 Win Rate: {inhibitors_wr:.1f}%")
        print(
            f"   💡 Contribuição para lucro total: {inhibitors_profit / df_filtered['profit'].sum() * 100:.1f}%"
        )

    # Análise de concentração de lucro
    profitable_markets_analysis = market_analysis[market_analysis["Total_Profit"] > 0]
    total_profitable_volume = profitable_markets_analysis["Total_Profit"].sum()

    print(f"\n📊 CONCENTRAÇÃO DE LUCRO POR MERCADO:")
    for mercado, row in profitable_markets_analysis.head(3).iterrows():
        percentage = row["Total_Profit"] / total_profitable_volume * 100
        print(
            f"   💎 {mercado}: {percentage:.1f}% do lucro total ({row['Total_Profit']:.2f} units)"
        )

    # ========================================================================
    # 4. ANÁLISE POR FAIXA DE ODDS (DADOS FILTRADOS) - NOVA CATEGORIZAÇÃO
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("📊 PERFORMANCE POR FAIXA DE ODDS (RANGES ROI FILTRADOS - ODDS AJUSTADAS)")
    print("=" * 80)

    odds_analysis = (
        df_filtered.groupby("odds_category")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    odds_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    odds_analysis["ROI"] = (
        odds_analysis["Total_Profit"] / odds_analysis["Bets"] * 100
    ).round(2)
    odds_analysis = odds_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n💰 PERFORMANCE POR FAIXA DE ODDS (ORDENADO POR LUCRO):")
    for odds_cat, row in odds_analysis.iterrows():
        if row["Total_Profit"] >= 5:
            icon = "💎"  # Faixas muito lucrativas
        elif row["Total_Profit"] > 0:
            icon = "✅"  # Faixas lucrativas
        else:
            icon = "❌"  # Faixas com prejuízo

        print(
            f"{icon} {odds_cat:<25} 💰 {row['Total_Profit']:>7.2f} | 📈 {row['ROI']:>6.1f}% | "
            f"🎯 {row['Win_Rate']:>5.1f}% | 🎲 {int(row['Bets'])}"
        )

    # Destacar performance da nova categoria media_alta
    if "media_alta (2.0~~2.5)" in odds_analysis.index:
        media_alta_data = odds_analysis.loc["media_alta (2.0~~2.5)"]
        print(f"\n🆕 NOVA CATEGORIA MEDIA_ALTA (2.0~~2.5):")
        print(f"   💰 Lucro: {media_alta_data['Total_Profit']:.2f} unidades")
        print(f"   📈 ROI: {media_alta_data['ROI']:.2f}%")
        print(f"   🎯 Win Rate: {media_alta_data['Win_Rate']:.1f}%")
        print(f"   🎲 Apostas: {int(media_alta_data['Bets'])}")

    # Identificar faixas de odds mais lucrativas
    odds_profit_ranking = odds_analysis.sort_values("Total_Profit", ascending=False)
    print(f"\n🏆 TOP 3 FAIXAS DE ODDS POR LUCRO:")
    for i, (odds_cat, row) in enumerate(odds_profit_ranking.head(3).iterrows(), 1):
        print(
            f"   {i}. {odds_cat}: {row['Total_Profit']:.2f} units ({row['ROI']:.1f}% ROI)"
        )

    # ========================================================================
    # 5. ANÁLISE POR DIREÇÃO (DADOS FILTRADOS)
    # ========================================================================
    direction_analysis = (
        df_filtered.groupby("direction")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    direction_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    direction_analysis["ROI"] = (
        direction_analysis["Total_Profit"] / direction_analysis["Bets"] * 100
    ).round(2)

    print(f"\n🔽 PERFORMANCE UNDER vs OVER (RANGES ROI FILTRADOS - COM INHIBITORS):")
    for direction, row in direction_analysis.iterrows():
        icon = "✅" if row["Total_Profit"] > 0 else "❌"
        print(
            f"{icon} {direction:<5} Lucro: {row['Total_Profit']:>7.2f} | ROI: {row['ROI']:>6.1f}% | "
            f"Win Rate: {row['Win_Rate']:>5.1f}% | Apostas: {int(row['Bets'])}"
        )

    # ========================================================================
    # 6. DEFINIÇÃO DOS CRITÉRIOS DA ESTRATÉGIA OTIMIZADA
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("🎯 DEFININDO ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    # Critérios baseados em LUCRO REAL, não em precisão estatística
    profitable_markets = market_analysis[
        (market_analysis["Total_Profit"] > 1)  # Lucro mínimo de 1 unidade
        & (market_analysis["ROI"] > 5)  # ROI mínimo de 5%
        & (market_analysis["Bets"] >= 3)  # Volume mínimo
    ].index.tolist()

    profitable_odds = odds_analysis[
        (odds_analysis["Total_Profit"] > 0)  # Qualquer lucro positivo
        & (odds_analysis["ROI"] > 0)  # ROI positivo
    ].index.tolist()

    profitable_directions = direction_analysis[
        direction_analysis["Total_Profit"] > 1  # Lucro mínimo de 1 unidade
    ].index.tolist()

    print(f"\n💎 CRITÉRIOS DA ESTRATÉGIA OTIMIZADA (BASEADO EM LUCRO REAL):")
    print(f"   💰 Ranges ROI lucrativos: {best_est_roi_categories}")
    print(f"   🏆 Mercados mais lucrativos: {profitable_markets}")
    print(f"   📊 Faixas de odds lucrativas: {profitable_odds}")
    print(f"   🎯 Direções mais lucrativas: {profitable_directions}")

    # Verificar se INHIBITORS está nos mercados lucrativos
    inhibitors_in_strategy = any("INHIBITOR" in market for market in profitable_markets)
    print(
        f"\n🎯 INHIBITORS na estratégia: {'✅ SIM' if inhibitors_in_strategy else '❌ NÃO'}"
    )

    # Mostrar potencial de lucro dos critérios selecionados
    selected_market_profit = (
        market_analysis.loc[profitable_markets, "Total_Profit"].sum()
        if profitable_markets
        else 0
    )
    selected_odds_profit = (
        odds_analysis.loc[profitable_odds, "Total_Profit"].sum()
        if profitable_odds
        else 0
    )

    print(f"\n📈 POTENCIAL DOS CRITÉRIOS SELECIONADOS:")
    print(f"   💰 Lucro dos mercados selecionados: {selected_market_profit:.2f} units")
    print(f"   📊 Lucro das odds selecionadas: {selected_odds_profit:.2f} units")

    # ========================================================================
    # 7. APLICAR ESTRATÉGIA OTIMIZADA (BACKTEST)
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("🚀 BACKTEST DA ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    def apply_optimized_strategy(row):
        return (
            row["est_roi_category"] in best_est_roi_categories
            and row["grouped_market"] in profitable_markets
            and row["odds_category"] in profitable_odds
            and row["direction"] in profitable_directions
        )

    # Aplicar apenas nos dados já filtrados por ROI
    df_filtered["strategy_approved"] = df_filtered.apply(
        apply_optimized_strategy, axis=1
    )

    optimized_bets = df_filtered[df_filtered["strategy_approved"]]
    rejected_bets = df_filtered[~df_filtered["strategy_approved"]]

    # Métricas do backtest
    opt_profit = optimized_bets["profit"].sum()
    opt_count = len(optimized_bets)
    opt_roi = (opt_profit / opt_count * 100) if opt_count > 0 else 0
    opt_wr = (optimized_bets["status"] == "win").mean() * 100 if opt_count > 0 else 0

    rej_profit = rejected_bets["profit"].sum()
    rej_count = len(rejected_bets)
    rej_roi = (rej_profit / rej_count * 100) if rej_count > 0 else 0

    print(f"\n📊 RESULTADOS DO BACKTEST:")
    print(f"   📈 Apostas no filtro ROI: {len(df_filtered)}")
    print(f"   ✅ Apostas aprovadas pela estratégia: {opt_count}")
    print(f"   ❌ Apostas rejeitadas: {rej_count}")
    print(f"   📊 Taxa de aprovação final: {opt_count / len(df_filtered) * 100:.1f}%")

    # Verificar quantas apostas INHIBITORS foram aprovadas
    optimized_inhibitors = optimized_bets[optimized_bets["market_type"] == "INHIBITORS"]
    print(f"   🎯 Apostas INHIBITORS aprovadas: {len(optimized_inhibitors)}")
    if len(optimized_inhibitors) > 0:
        inhibitors_contribution = optimized_inhibitors["profit"].sum()
        print(
            f"   💰 Contribuição INHIBITORS: {inhibitors_contribution:.2f} units ({inhibitors_contribution / opt_profit * 100:.1f}%)"
        )

    # ========================================================================
    # 8. COMPARAÇÃO DETALHADA
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("📊 COMPARAÇÃO: TODAS vs RANGES ROI vs ESTRATÉGIA FINAL (COM INHIBITORS)")
    print("=" * 80)

    print(f"\n📊 TODAS AS APOSTAS:")
    print(f"   💰 Lucro: {profit_total:.2f} unidades")
    print(f"   🎲 Apostas: {total}")
    print(f"   💹 ROI: {roi_avg:.2f}%")
    print(f"   📊 Win Rate: {win_rate:.1f}%")

    filtered_profit = df_filtered["profit"].sum()
    filtered_roi = (
        (filtered_profit / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    )
    filtered_wr = (
        (df_filtered["status"] == "win").mean() * 100 if len(df_filtered) > 0 else 0
    )

    print(f"\n📈 APENAS RANGES ROI LUCRATIVOS:")
    print(f"   💰 Lucro: {filtered_profit:.2f} unidades")
    print(f"   🎲 Apostas: {len(df_filtered)}")
    print(f"   💹 ROI: {filtered_roi:.2f}%")
    print(f"   📊 Win Rate: {filtered_wr:.1f}%")

    print(f"\n🚀 ESTRATÉGIA FINAL OTIMIZADA:")
    print(f"   💰 Lucro: {opt_profit:.2f} unidades")
    print(f"   🎲 Apostas: {opt_count}")
    print(f"   💹 ROI: {opt_roi:.2f}%")
    print(f"   📊 Win Rate: {opt_wr:.1f}%")

    if rej_count > 0:
        print(f"\n❌ APOSTAS REJEITADAS (DENTRO DOS RANGES ROI):")
        print(f"   💰 Lucro evitado: {rej_profit:.2f} unidades")
        print(f"   🎲 Apostas: {rej_count}")
        print(f"   💹 ROI rejeitado: {rej_roi:.2f}%")

    # ========================================================================
    # 9. IMPACTO DA OTIMIZAÇÃO
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("💎 IMPACTO FINANCEIRO DA OTIMIZAÇÃO (COM INHIBITORS)")
    print("=" * 80)

    if opt_count > 0:
        # Comparação financeira com todas as apostas
        roi_improvement_all = opt_roi - roi_avg
        profit_multiplication = opt_profit / profit_total if profit_total > 0 else 0
        volume_reduction_all = (total - opt_count) / total * 100
        profit_per_bet_improvement = (
            (opt_profit / opt_count) / (profit_total / total) if total > 0 else 0
        )

        print(f"\n💰 GANHOS FINANCEIROS vs TODAS AS APOSTAS:")
        print(
            f"   📈 Multiplicação do ROI: {opt_roi / roi_avg:.1f}x ({roi_avg:.2f}% → {opt_roi:.2f}%)"
        )
        print(
            f"   💎 Multiplicação do lucro: {profit_multiplication:.1f}x ({profit_total:.2f} → {opt_profit:.2f} units)"
        )
        print(f"   🎯 Eficiência por aposta: {profit_per_bet_improvement:.1f}x melhor")
        print(f"   📉 Redução de volume: {volume_reduction_all:.1f}% (mais seletivo)")

        # Comparação apenas com ranges ROI
        roi_improvement_filtered = opt_roi - filtered_roi
        profit_concentration_filtered = (
            (opt_profit / filtered_profit * 100) if filtered_profit != 0 else 0
        )

        print(f"\n🔍 REFINAMENTO ADICIONAL vs RANGES ROI:")
        print(f"   📈 Ganho ROI adicional: {roi_improvement_filtered:+.2f}%")
        print(f"   💰 Concentração de lucro: {profit_concentration_filtered:.1f}%")

        # Projeção de lucro futuro
        if opt_count > 0:
            monthly_projection = (
                opt_profit / opt_count
            ) * 30  # Assumindo ~1 aposta/dia

            print(f"\n🚀 PROJEÇÃO DE LUCRO (30 apostas/mês):")
            print(f"   💰 Lucro esperado mensal: {monthly_projection:.2f} units")
            print(f"   📊 ROI esperado: {opt_roi:.2f}% por aposta")

    # ========================================================================
    # 10. RECOMENDAÇÕES FINAIS
    # ========================================================================
    print(f"\n" + "=" * 80)
    print("💡 RECOMENDAÇÕES ESTRATÉGICAS FINAIS (COM INHIBITORS)")
    print("=" * 80)

    print(f"\n💰 RESUMO EXECUTIVO (COM INHIBITORS - FOCADO EM LUCRO):")
    print(f"   🏆 Ranges mais lucrativos: {best_est_roi_categories}")
    print(
        f"   💎 Top 3 mercados: {profitable_markets[:3] if len(profitable_markets) >= 3 else profitable_markets}"
    )
    print(f"   📈 ROI final da estratégia: {opt_roi:.2f}%")
    print(
        f"   💰 Lucro por aposta otimizada: {(opt_profit / opt_count):.4f} units"
        if opt_count > 0
        else "   💰 Lucro por aposta: N/A"
    )
    print(
        f"   🎯 INHIBITORS incluídos: {'✅ SIM' if inhibitors_in_strategy else '❌ NÃO'}"
    )

    print(f"\n🚀 PLANO DE AÇÃO COM INHIBITORS:")
    print(f"   1. 💎 FOCO TOTAL nos ranges de ROI identificados")
    print(f"   2. 🎯 APROVEITAR mercados INHIBITORS quando disponíveis")
    print(f"   3. 📊 MONITORAR nova categoria media_alta (2.0~~2.5)")
    print(f"   4. 🔍 MANTER disciplina nos critérios de seleção")
    print(f"   5. 💰 REAVALIAR estratégia a cada 50 apostas")

    # Recomendações específicas baseadas nos dados
    if len(profitable_markets) > 0:
        top_market = profitable_markets[0]
        print(f"\n💡 ESTRATÉGIA PRIORITÁRIA:")
        print(f"   🥇 FOCO PRINCIPAL em: {top_market}")
        print(f"   📈 Aumentar volume nos mercados TOP 3")
        print(f"   🎯 Manter disciplina nos ranges de ROI")
        print(f"   🔄 Monitorar performance das novas odds")

    # Retorno dos resultados
    return {
        "original_data": df,  # Dataset COMPLETO com INHIBITORS
        "filtered_data": df_filtered,
        "optimized_bets": optimized_bets,
        "rejected_bets": rejected_bets,
        "roi_est_analysis": roi_est_analysis,
        "market_analysis": market_analysis,
        "odds_analysis": odds_analysis,
        "strategy_criteria": {
            "est_roi_ranges": best_est_roi_categories,
            "markets": profitable_markets,
            "odds": profitable_odds,
            "directions": profitable_directions,
        },
        "performance_metrics": {
            "all_bets_with_inhibitors": {
                "profit": profit_total,
                "count": total,
                "roi": roi_avg,
            },
            "filtered_by_roi_ranges": {
                "profit": filtered_profit,
                "count": len(df_filtered),
                "roi": filtered_roi,
            },
            "optimized_bets": {
                "profit": opt_profit,
                "count": opt_count,
                "roi": opt_roi,
            },
        },
        "inhibitors_included": True,
        "new_odds_categories": True,
    }


# Execução direta
if __name__ == "__main__":
    file_path = (
        "../bets/bets_atualizadas_por_mapa.csv"  # Ajuste o caminho conforme necessário
    )
    results = complete_betting_analysis_with_backtest(file_path)

    print(f"\n" + "=" * 80)
    print("✅ ANÁLISE COM INHIBITORS E ODDS AJUSTADAS FINALIZADA!")
    print("🎯 Estratégia completa com todos os mercados disponíveis")
    print("🆕 Nova categoria de odds media_alta (2.0~~2.5) incluída")
    print("=" * 80)
