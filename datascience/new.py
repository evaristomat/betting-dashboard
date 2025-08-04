import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")


def super_complete_betting_analysis(file_path):
    """
    Análise super completa de apostas com foco em ROI estimado e implementação de melhorias.
    """
    
    # ========================================================================
    # CARREGAMENTO E PREPARAÇÃO DOS DADOS
    # ========================================================================
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y %H:%M", errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    
    # Converter ROI estimado para numérico
    df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)
    
    print("=" * 100)
    print("🎯 ANÁLISE SUPER COMPLETA DE APOSTAS - FOCO EM ROI ESTIMADO E MELHORIAS")
    print("=" * 100)
    
    # Métricas gerais iniciais
    total_bets = len(df)
    total_profit = df["profit"].sum()
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    win_rate = df["status"].eq("win").mean() * 100
    
    print(f"\n📊 VISÃO GERAL DO DATASET:")
    print(f"   Total de apostas: {total_bets}")
    print(f"   Lucro total: {total_profit:.2f} unidades")
    print(f"   ROI geral: {overall_roi:.2f}%")
    print(f"   Win rate geral: {win_rate:.1f}%")
    print(f"   Período: {df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}")
    
    # ========================================================================
    # 1. ANÁLISE PROFUNDA POR FAIXAS DE ROI ESTIMADO
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📈 SEÇÃO 1: ANÁLISE PROFUNDA POR FAIXAS DE ROI ESTIMADO")
    print("=" * 100)
    
    # Criar faixas de ROI mais granulares
    def categorize_roi_detailed(roi):
        if pd.isna(roi):
            return "N/A"
        elif roi < 10:
            return "0-10%"
        elif roi < 15:
            return "10-15%"
        elif roi < 20:
            return "15-20%"
        elif roi < 25:
            return "20-25%"
        elif roi < 30:
            return "25-30%"
        elif roi < 35:
            return "30-35%"
        elif roi < 40:
            return "35-40%"
        elif roi < 50:
            return "40-50%"
        else:
            return "50%+"
    
    df["roi_range"] = df["estimated_roi"].apply(categorize_roi_detailed)
    
    # Análise detalhada por faixa
    roi_analysis = df.groupby("roi_range").agg({
        "profit": ["sum", "count", "mean", "std"],
        "odds": ["mean", "std"],
        "status": lambda x: (x == "win").mean() * 100,
        "estimated_roi": "mean"
    }).round(2)
    
    roi_analysis.columns = ["Total_Profit", "Bets", "Avg_Profit", "Std_Profit", 
                           "Avg_Odds", "Std_Odds", "Win_Rate", "Avg_Est_ROI"]
    roi_analysis["Real_ROI"] = (roi_analysis["Total_Profit"] / roi_analysis["Bets"] * 100).round(2)
    roi_analysis["ROI_Accuracy"] = (roi_analysis["Real_ROI"] / roi_analysis["Avg_Est_ROI"] * 100).round(1)
    roi_analysis["Sharpe_Ratio"] = (roi_analysis["Avg_Profit"] / roi_analysis["Std_Profit"]).round(2)
    
    # Ordenar por uma métrica customizada que considera lucro, volume e consistência
    roi_analysis["Score"] = (
        roi_analysis["Total_Profit"] * 0.4 + 
        roi_analysis["Real_ROI"] * 0.3 + 
        roi_analysis["Bets"] * 0.2 + 
        roi_analysis["Sharpe_Ratio"] * 0.1
    ).round(2)
    
    roi_analysis = roi_analysis.sort_values("Score", ascending=False)
    
    print(f"\n📊 ANÁLISE DETALHADA POR FAIXA DE ROI ESTIMADO:")
    print(f"{'Faixa':<10} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'ROI Est':<10} "
          f"{'Precisão':<10} {'WR':<8} {'Sharpe':<8} {'Score':<8}")
    print("-" * 96)
    
    for roi_range, row in roi_analysis.iterrows():
        icon = "🏆" if row["Score"] > 50 else "✅" if row["Real_ROI"] > 0 else "❌"
        print(f"{icon} {roi_range:<8} {row['Total_Profit']:>8.2f} {int(row['Bets']):>9} "
              f"{row['Real_ROI']:>9.1f}% {row['Avg_Est_ROI']:>9.1f}% "
              f"{row['ROI_Accuracy']:>9.1f}% {row['Win_Rate']:>7.1f}% "
              f"{row['Sharpe_Ratio']:>7.2f} {row['Score']:>7.1f}")
    
    # Análise de correlação entre ROI estimado e real
    correlation = df["estimated_roi"].corr(df["profit"])
    print(f"\n📈 Correlação entre ROI estimado e lucro real: {correlation:.3f}")
    
    # Definir melhores faixas baseado em múltiplos critérios
    best_roi_ranges = roi_analysis[
        (roi_analysis["Total_Profit"] > 5) &  # Lucro mínimo significativo
        (roi_analysis["Real_ROI"] > 10) &     # ROI real mínimo
        (roi_analysis["Bets"] >= 10) &        # Volume mínimo para significância
        (roi_analysis["Sharpe_Ratio"] > 0)    # Consistência positiva
    ].index.tolist()
    
    # Se não encontrar faixas com critérios rigorosos, relaxar um pouco
    if not best_roi_ranges:
        best_roi_ranges = roi_analysis[
            (roi_analysis["Total_Profit"] > 1) &
            (roi_analysis["Real_ROI"] > 5) &
            (roi_analysis["Bets"] >= 5)
        ].index.tolist()
    
    print(f"\n💎 FAIXAS DE ROI SELECIONADAS (ALTA PERFORMANCE): {best_roi_ranges}")
    
    # Estatísticas das faixas selecionadas
    if best_roi_ranges:
        selected_stats = roi_analysis.loc[best_roi_ranges]
        total_selected_profit = selected_stats["Total_Profit"].sum()
        total_selected_bets = selected_stats["Bets"].sum()
        weighted_roi = (total_selected_profit / total_selected_bets * 100) if total_selected_bets > 0 else 0
        
        print(f"\n📊 ESTATÍSTICAS DAS FAIXAS SELECIONADAS:")
        print(f"   💰 Lucro total: {total_selected_profit:.2f} unidades")
        print(f"   🎲 Total de apostas: {int(total_selected_bets)}")
        print(f"   📈 ROI médio ponderado: {weighted_roi:.2f}%")
        print(f"   📊 Percentual do total: {total_selected_bets / total_bets * 100:.1f}% das apostas")
    
    # ========================================================================
    # 2. FILTRAR DADOS PELAS MELHORES FAIXAS DE ROI
    # ========================================================================
    df_filtered = df[df["roi_range"].isin(best_roi_ranges)]
    
    print(f"\n🔍 DADOS FILTRADOS POR ROI ESTIMADO:")
    print(f"   Apostas originais: {len(df)}")
    print(f"   Apostas após filtro: {len(df_filtered)}")
    print(f"   Taxa de aprovação: {len(df_filtered) / len(df) * 100:.1f}%")
    
    # ========================================================================
    # 3. ANÁLISE COMPLETA POR BET_TYPE
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🎮 SEÇÃO 2: ANÁLISE COMPLETA POR BET_TYPE (DADOS FILTRADOS)")
    print("=" * 100)
    
    bet_type_analysis = df_filtered.groupby("bet_type").agg({
        "profit": ["sum", "count", "mean", "std"],
        "odds": ["mean", "min", "max"],
        "status": lambda x: (x == "win").mean() * 100,
        "estimated_roi": "mean"
    }).round(2)
    
    bet_type_analysis.columns = ["Total_Profit", "Bets", "Avg_Profit", "Std_Profit",
                                "Avg_Odds", "Min_Odds", "Max_Odds", "Win_Rate", "Avg_Est_ROI"]
    bet_type_analysis["ROI"] = (bet_type_analysis["Total_Profit"] / bet_type_analysis["Bets"] * 100).round(2)
    bet_type_analysis["Consistency"] = (bet_type_analysis["Avg_Profit"] / 
                                       (bet_type_analysis["Std_Profit"] + 0.01)).round(2)
    
    bet_type_analysis = bet_type_analysis.sort_values("Total_Profit", ascending=False)
    
    print(f"\n💰 PERFORMANCE POR BET_TYPE (TOP 10):")
    print(f"{'Tipo':<30} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'WR':<8} "
          f"{'Odds Méd':<10} {'Consist.':<10}")
    print("-" * 88)
    
    for bet_type, row in bet_type_analysis.head(10).iterrows():
        icon = "💎" if row["Total_Profit"] > 10 else "✅" if row["Total_Profit"] > 0 else "❌"
        print(f"{icon} {bet_type[:28]:<28} {row['Total_Profit']:>8.2f} "
              f"{int(row['Bets']):>9} {row['ROI']:>9.1f}% {row['Win_Rate']:>7.1f}% "
              f"{row['Avg_Odds']:>9.2f} {row['Consistency']:>9.2f}")
    
    # Identificar bet_types mais lucrativos
    profitable_bet_types = bet_type_analysis[
        (bet_type_analysis["Total_Profit"] > 2) &
        (bet_type_analysis["ROI"] > 10) &
        (bet_type_analysis["Bets"] >= 5)
    ].index.tolist()
    
    print(f"\n🏆 BET_TYPES MAIS LUCRATIVOS: {len(profitable_bet_types)} tipos identificados")
    
    # ========================================================================
    # 4. ANÁLISE COMPLETA POR BET_LINE
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📋 SEÇÃO 3: ANÁLISE COMPLETA POR BET_LINE (DADOS FILTRADOS)")
    print("=" * 100)
    
    # Categorizar mercados de forma mais detalhada
    def categorize_market_detailed(bet_type, bet_line):
        lt = str(bet_type).lower()
        ll = str(bet_line).lower()
        
        # Determinar direção
        direction = "UNDER" if "under" in lt else "OVER" if "over" in lt else "OTHER"
        
        # Determinar mercado específico
        if "kill" in ll:
            market = "KILLS"
            if "first" in ll:
                sub_market = "FIRST_KILL"
            elif "total" in ll:
                sub_market = "TOTAL_KILLS"
            else:
                sub_market = "KILLS_OTHER"
        elif "dragon" in ll:
            market = "DRAGONS"
            if "first" in ll:
                sub_market = "FIRST_DRAGON"
            elif "total" in ll:
                sub_market = "TOTAL_DRAGONS"
            else:
                sub_market = "DRAGONS_OTHER"
        elif "tower" in ll:
            market = "TOWERS"
            if "first" in ll:
                sub_market = "FIRST_TOWER"
            elif "total" in ll:
                sub_market = "TOTAL_TOWERS"
            else:
                sub_market = "TOWERS_OTHER"
        elif "baron" in ll:
            market = "BARONS"
            sub_market = "BARONS"
        elif "inhibitor" in ll:
            market = "INHIBITORS"
            sub_market = "INHIBITORS"
        elif "duration" in ll or "tempo" in ll:
            market = "DURATION"
            sub_market = "GAME_DURATION"
        else:
            market = "OUTROS"
            sub_market = "OUTROS"
        
        return direction, market, sub_market, f"{direction} - {sub_market}"
    
    df_filtered[["direction", "market", "sub_market", "detailed_market"]] = df_filtered.apply(
        lambda r: categorize_market_detailed(r["bet_type"], r["bet_line"]),
        axis=1,
        result_type="expand"
    )
    
    # Análise por mercado detalhado
    market_analysis = df_filtered.groupby("detailed_market").agg({
        "profit": ["sum", "count", "mean", "std"],
        "odds": ["mean", "min", "max"],
        "status": lambda x: (x == "win").mean() * 100,
        "estimated_roi": "mean"
    }).round(2)
    
    market_analysis.columns = ["Total_Profit", "Bets", "Avg_Profit", "Std_Profit",
                              "Avg_Odds", "Min_Odds", "Max_Odds", "Win_Rate", "Avg_Est_ROI"]
    market_analysis["ROI"] = (market_analysis["Total_Profit"] / market_analysis["Bets"] * 100).round(2)
    market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)
    
    print(f"\n💰 PERFORMANCE POR MERCADO DETALHADO (TOP 15):")
    print(f"{'Mercado':<35} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'WR':<8} {'Odds':<15}")
    print("-" * 98)
    
    for market, row in market_analysis.head(15).iterrows():
        icon = "🏆" if row["Total_Profit"] > 10 else "💎" if row["Total_Profit"] > 5 else "✅" if row["Total_Profit"] > 0 else "❌"
        odds_range = f"{row['Min_Odds']:.2f}-{row['Max_Odds']:.2f}"
        print(f"{icon} {market[:33]:<33} {row['Total_Profit']:>8.2f} "
              f"{int(row['Bets']):>9} {row['ROI']:>9.1f}% {row['Win_Rate']:>7.1f}% "
              f"{odds_range:>14}")
    
    # Análise por tipo de mercado principal
    main_market_analysis = df_filtered.groupby("market").agg({
        "profit": ["sum", "count"],
        "status": lambda x: (x == "win").mean() * 100
    }).round(2)
    
    main_market_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    main_market_analysis["ROI"] = (main_market_analysis["Total_Profit"] / 
                                   main_market_analysis["Bets"] * 100).round(2)
    main_market_analysis = main_market_analysis.sort_values("Total_Profit", ascending=False)
    
    print(f"\n📊 RESUMO POR TIPO DE MERCADO PRINCIPAL:")
    for market, row in main_market_analysis.iterrows():
        percentage = row["Bets"] / len(df_filtered) * 100
        print(f"   {market}: {row['Total_Profit']:.2f} lucro | {int(row['Bets'])} apostas ({percentage:.1f}%) | "
              f"ROI: {row['ROI']:.1f}% | WR: {row['Win_Rate']:.1f}%")
    
    # ========================================================================
    # 5. ANÁLISE POR FAIXA DE ODDS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📊 SEÇÃO 4: ANÁLISE POR FAIXA DE ODDS (DADOS FILTRADOS)")
    print("=" * 100)
    
    # Categorização detalhada de odds
    def categorize_odds_detailed(odds):
        if pd.isna(odds):
            return "N/A"
        elif odds <= 1.20:
            return "1.00-1.20"
        elif odds <= 1.40:
            return "1.20-1.40"
        elif odds <= 1.60:
            return "1.40-1.60"
        elif odds <= 1.80:
            return "1.60-1.80"
        elif odds <= 2.00:
            return "1.80-2.00"
        elif odds <= 2.25:
            return "2.00-2.25"
        elif odds <= 2.50:
            return "2.25-2.50"
        elif odds <= 3.00:
            return "2.50-3.00"
        elif odds <= 4.00:
            return "3.00-4.00"
        else:
            return "4.00+"
    
    df_filtered["odds_range"] = df_filtered["odds"].apply(categorize_odds_detailed)
    
    odds_analysis = df_filtered.groupby("odds_range").agg({
        "profit": ["sum", "count", "mean"],
        "status": lambda x: (x == "win").mean() * 100,
        "estimated_roi": "mean"
    }).round(2)
    
    odds_analysis.columns = ["Total_Profit", "Bets", "Avg_Profit", "Win_Rate", "Avg_Est_ROI"]
    odds_analysis["ROI"] = (odds_analysis["Total_Profit"] / odds_analysis["Bets"] * 100).round(2)
    odds_analysis["Expected_WR"] = (100 / df_filtered.groupby("odds_range")["odds"].mean()).round(1)
    odds_analysis["WR_Diff"] = (odds_analysis["Win_Rate"] - odds_analysis["Expected_WR"]).round(1)
    
    odds_analysis = odds_analysis.sort_values("Total_Profit", ascending=False)
    
    print(f"\n📊 PERFORMANCE POR FAIXA DE ODDS:")
    print(f"{'Faixa':<12} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'WR Real':<10} "
          f"{'WR Esp':<10} {'Dif WR':<10}")
    print("-" * 82)
    
    for odds_range, row in odds_analysis.iterrows():
        icon = "💎" if row["Total_Profit"] > 5 else "✅" if row["Total_Profit"] > 0 else "❌"
        print(f"{icon} {odds_range:<10} {row['Total_Profit']:>8.2f} {int(row['Bets']):>9} "
              f"{row['ROI']:>9.1f}% {row['Win_Rate']:>9.1f}% {row['Expected_WR']:>9.1f}% "
              f"{row['WR_Diff']:>+9.1f}%")
    
    # ========================================================================
    # 6. ANÁLISE TEMPORAL
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📅 SEÇÃO 5: ANÁLISE TEMPORAL (DADOS FILTRADOS)")
    print("=" * 100)
    
    # Análise por mês
    df_filtered["month"] = df_filtered["date"].dt.to_period("M")
    monthly_analysis = df_filtered.groupby("month").agg({
        "profit": ["sum", "count"],
        "status": lambda x: (x == "win").mean() * 100
    }).round(2)
    
    monthly_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    monthly_analysis["ROI"] = (monthly_analysis["Total_Profit"] / monthly_analysis["Bets"] * 100).round(2)
    
    print(f"\n📅 PERFORMANCE MENSAL:")
    for month, row in monthly_analysis.iterrows():
        icon = "✅" if row["Total_Profit"] > 0 else "❌"
        print(f"{icon} {month}: Lucro: {row['Total_Profit']:.2f} | Apostas: {int(row['Bets'])} | "
              f"ROI: {row['ROI']:.1f}% | WR: {row['Win_Rate']:.1f}%")
    
    # ========================================================================
    # 7. DEFINIÇÃO DA ESTRATÉGIA OTIMIZADA FINAL
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🎯 SEÇÃO 6: ESTRATÉGIA OTIMIZADA FINAL")
    print("=" * 100)
    
    # Definir critérios finais baseados nas análises
    profitable_markets = market_analysis[
        (market_analysis["Total_Profit"] > 3) &
        (market_analysis["ROI"] > 15) &
        (market_analysis["Bets"] >= 5)
    ].index.tolist()
    
    profitable_odds = odds_analysis[
        (odds_analysis["Total_Profit"] > 2) &
        (odds_analysis["ROI"] > 10)
    ].index.tolist()
    
    # Aplicar estratégia final
    def apply_final_strategy(row):
        return (
            row["roi_range"] in best_roi_ranges and
            row["detailed_market"] in profitable_markets and
            row["odds_range"] in profitable_odds
        )
    
    df_filtered["final_approved"] = df_filtered.apply(apply_final_strategy, axis=1)
    
    final_bets = df_filtered[df_filtered["final_approved"]]
    rejected_bets = df_filtered[~df_filtered["final_approved"]]
    
    # ========================================================================
    # 8. REPORT FINAL COMPLETO E COMPARATIVO
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📊 REPORT FINAL COMPLETO E COMPARATIVO")
    print("=" * 100)
    
    # Cálculo de métricas para cada nível
    levels = {
        "1. Todas as Apostas": {
            "data": df,
            "profit": df["profit"].sum(),
            "count": len(df),
            "roi": (df["profit"].sum() / len(df) * 100) if len(df) > 0 else 0,
            "wr": (df["status"] == "win").mean() * 100
        },
        "2. Filtro ROI Estimado": {
            "data": df_filtered,
            "profit": df_filtered["profit"].sum(),
            "count": len(df_filtered),
            "roi": (df_filtered["profit"].sum() / len(df_filtered) * 100) if len(df_filtered) > 0 else 0,
            "wr": (df_filtered["status"] == "win").mean() * 100
        },
        "3. Estratégia Final": {
            "data": final_bets,
            "profit": final_bets["profit"].sum() if len(final_bets) > 0 else 0,
            "count": len(final_bets),
            "roi": (final_bets["profit"].sum() / len(final_bets) * 100) if len(final_bets) > 0 else 0,
            "wr": (final_bets["status"] == "win").mean() * 100 if len(final_bets) > 0 else 0
        }
    }
    
    print(f"\n📊 COMPARAÇÃO ENTRE NÍVEIS DE FILTRO:")
    print(f"{'Nível':<25} {'Lucro':<12} {'Apostas':<10} {'ROI':<10} {'Win Rate':<10} {'% Original':<12}")
    print("-" * 79)
    
    for level_name, metrics in levels.items():
        percentage = (metrics["count"] / levels["1. Todas as Apostas"]["count"] * 100) if levels["1. Todas as Apostas"]["count"] > 0 else 0
        icon = "🏆" if "Final" in level_name else "📊"
        print(f"{icon} {level_name:<23} {metrics['profit']:>10.2f} {metrics['count']:>9} "
              f"{metrics['roi']:>9.1f}% {metrics['wr']:>9.1f}% {percentage:>11.1f}%")
    
    # Análise de melhoria incremental
    print(f"\n📈 MELHORIA INCREMENTAL:")
    
    if len(df_filtered) > 0:
        roi_improvement_1 = levels["2. Filtro ROI Estimado"]["roi"] - levels["1. Todas as Apostas"]["roi"]
        profit_retention_1 = (levels["2. Filtro ROI Estimado"]["profit"] / 
                             levels["1. Todas as Apostas"]["profit"] * 100) if levels["1. Todas as Apostas"]["profit"] != 0 else 0
        
        print(f"   Filtro ROI Estimado vs Todas:")
        print(f"      • Melhoria ROI: {roi_improvement_1:+.2f}%")
        print(f"      • Retenção de lucro: {profit_retention_1:.1f}%")
        print(f"      • Redução de volume: {100 - (len(df_filtered) / len(df) * 100):.1f}%")
    
    if len(final_bets) > 0:
        roi_improvement_2 = levels["3. Estratégia Final"]["roi"] - levels["2. Filtro ROI Estimado"]["roi"]
        profit_concentration = (levels["3. Estratégia Final"]["profit"] / 
                               levels["2. Filtro ROI Estimado"]["profit"] * 100) if levels["2. Filtro ROI Estimado"]["profit"] != 0 else 0
        
        print(f"\n   Estratégia Final vs Filtro ROI:")
        print(f"      • Melhoria ROI adicional: {roi_improvement_2:+.2f}%")
        print(f"      • Concentração de lucro: {profit_concentration:.1f}%")
        print(f"      • Seletividade adicional: {100 - (len(final_bets) / len(df_filtered) * 100):.1f}%")
    
    # Resumo executivo
    print(f"\n{'=' * 100}")
    print("💎 RESUMO EXECUTIVO E RECOMENDAÇÕES")
    print("=" * 100)
    
    print(f"\n🎯 CRITÉRIOS FINAIS DA ESTRATÉGIA OTIMIZADA:")
    print(f"   📈 Faixas de ROI Estimado: {best_roi_ranges}")
    print(f"   🏆 Total de mercados aprovados: {len(profitable_markets)}")
    print(f"   📊 Faixas de odds aprovadas: {profitable_odds}")
    