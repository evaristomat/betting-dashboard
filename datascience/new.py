import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def super_complete_betting_analysis(file_path):
    """
    Análise completa de apostas com filtro fixo de ROI >= 14% e foco em lucro absoluto.
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
    print("🎯 ANÁLISE COMPLETA DE APOSTAS - FILTRO ROI >= 14%")
    print("=" * 100)

    # Métricas gerais iniciais
    total_bets = len(df)
    total_profit = df["profit"].sum()
    win_rate = df["status"].eq("win").mean() * 100

    print(f"\n📊 VISÃO GERAL DO DATASET:")
    print(f"   Total de apostas: {total_bets}")
    print(f"   Lucro total: {total_profit:.2f} unidades")
    print(f"   Win rate geral: {win_rate:.1f}%")
    print(
        f"   Período: {df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}"
    )

    # ========================================================================
    # 1. ANÁLISE POR RANGES DE ROI (INFORMATIVA)
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📈 SEÇÃO 1: ANÁLISE POR RANGES DE ROI ESTIMADO")
    print("=" * 100)

    # Análise por ranges cumulativos
    roi_ranges = [0, 10, 14, 15, 20, 25, 30, 35, 40, 50]

    print(f"\n📊 PERFORMANCE POR RANGE DE ROI ESTIMADO:")
    print(
        f"{'Range':<10} {'Apostas':<10} {'% Total':<10} {'Lucro':<12} {'ROI Real':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 87)

    for min_roi in roi_ranges:
        df_range = df[df["estimated_roi"] >= min_roi]
        if len(df_range) > 0:
            range_profit = df_range["profit"].sum()
            range_bets = len(df_range)
            range_wr = (df_range["status"] == "win").mean() * 100
            profit_per_bet = range_profit / range_bets
            percentage = range_bets / total_bets * 100
            roi_real = (range_profit / range_bets * 100) if range_bets > 0 else 0

            icon = "🎯" if min_roi == 14 else "✅" if range_profit > 0 else "❌"
            print(
                f"{icon} {f'{min_roi}%+':<8} {range_bets:<10} {percentage:>8.1f}% "
                f"{range_profit:>11.2f} {roi_real:>9.1f}% {range_wr:>9.1f}% {profit_per_bet:>14.3f}"
            )

    # ========================================================================
    # 2. APLICAR FILTRO FIXO ROI >= 14%
    # ========================================================================
    ROI_THRESHOLD = 20
    df_filtered = df[df["estimated_roi"] >= ROI_THRESHOLD]

    print(f"\n🔍 APLICANDO FILTRO ROI >= {ROI_THRESHOLD}%:")
    print(f"   Apostas originais: {len(df)}")
    print(f"   Apostas após filtro: {len(df_filtered)}")
    print(f"   Taxa de aprovação: {len(df_filtered) / len(df) * 100:.1f}%")
    print(f"   Lucro total filtrado: {df_filtered['profit'].sum():.2f} unidades")
    print(f"   Win rate filtrado: {(df_filtered['status'] == 'win').mean() * 100:.1f}%")

    # ========================================================================
    # 3. ANÁLISE COMPLETA POR BET_TYPE
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🎮 SEÇÃO 2: ANÁLISE POR BET_TYPE (ROI >= 14%)")
    print("=" * 100)

    bet_type_analysis = (
        df_filtered.groupby("bet_type")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "odds": ["mean", "min", "max"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    bet_type_analysis.columns = [
        "Total_Profit",
        "Bets",
        "Avg_Profit",
        "Avg_Odds",
        "Min_Odds",
        "Max_Odds",
        "Win_Rate",
    ]
    bet_type_analysis = bet_type_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n💰 PERFORMANCE POR BET_TYPE:")
    print(
        f"{'Tipo':<30} {'Lucro':<10} {'Apostas':<10} {'Win Rate':<10} {'Odds Médias':<12} {'Lucro/Aposta':<15}"
    )
    print("-" * 97)

    for bet_type, row in bet_type_analysis.iterrows():
        icon = (
            "💎"
            if row["Total_Profit"] > 10
            else "✅"
            if row["Total_Profit"] > 0
            else "❌"
        )
        profit_per_bet = row["Total_Profit"] / row["Bets"] if row["Bets"] > 0 else 0
        print(
            f"{icon} {bet_type[:28]:<28} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>9} {row['Win_Rate']:>9.1f}% {row['Avg_Odds']:>11.2f} "
            f"{profit_per_bet:>14.3f}"
        )

    # ========================================================================
    # 4. ANÁLISE DETALHADA POR MERCADO (BET_LINE)
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📋 SEÇÃO 3: ANÁLISE POR MERCADO DETALHADO (ROI >= 14%)")
    print("=" * 100)

    # Categorizar mercados
    def categorize_market_detailed(bet_type, bet_line):
        lt = str(bet_type).lower()
        ll = str(bet_line).lower()

        direction = "UNDER" if "under" in lt else "OVER" if "over" in lt else "OTHER"

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

    df_filtered[["direction", "market", "sub_market", "detailed_market"]] = (
        df_filtered.apply(
            lambda r: categorize_market_detailed(r["bet_type"], r["bet_line"]),
            axis=1,
            result_type="expand",
        )
    )

    # Análise por mercado detalhado
    market_analysis = (
        df_filtered.groupby("detailed_market")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "odds": ["mean", "min", "max"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    market_analysis.columns = [
        "Total_Profit",
        "Bets",
        "Avg_Profit",
        "Avg_Odds",
        "Min_Odds",
        "Max_Odds",
        "Win_Rate",
    ]
    market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n💰 TOP 20 MERCADOS POR LUCRO:")
    print(
        f"{'Mercado':<35} {'Lucro':<10} {'Apostas':<10} {'Win Rate':<10} {'Odds Range':<15} {'Lucro/Aposta':<15}"
    )
    print("-" * 110)

    for market, row in market_analysis.head(20).iterrows():
        if row["Bets"] >= 3:  # Mínimo de 3 apostas para relevância
            icon = (
                "🏆"
                if row["Total_Profit"] > 10
                else "💎"
                if row["Total_Profit"] > 5
                else "✅"
                if row["Total_Profit"] > 0
                else "❌"
            )
            odds_range = f"{row['Min_Odds']:.2f}-{row['Max_Odds']:.2f}"
            profit_per_bet = row["Total_Profit"] / row["Bets"]
            print(
                f"{icon} {market[:33]:<33} {row['Total_Profit']:>8.2f} "
                f"{int(row['Bets']):>9} {row['Win_Rate']:>9.1f}% {odds_range:>14} "
                f"{profit_per_bet:>14.3f}"
            )

    # Resumo por tipo de mercado principal
    main_market_analysis = (
        df_filtered.groupby("market")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    main_market_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    main_market_analysis = main_market_analysis.sort_values(
        "Total_Profit", ascending=False
    )

    print(f"\n📊 RESUMO POR TIPO DE MERCADO PRINCIPAL:")
    for market, row in main_market_analysis.iterrows():
        percentage = row["Bets"] / len(df_filtered) * 100
        profit_per_bet = row["Total_Profit"] / row["Bets"] if row["Bets"] > 0 else 0
        icon = "✅" if row["Total_Profit"] > 0 else "❌"
        print(
            f"   {icon} {market}: {row['Total_Profit']:.2f} lucro | {int(row['Bets'])} apostas ({percentage:.1f}%) | "
            f"WR: {row['Win_Rate']:.1f}% | Lucro/Aposta: {profit_per_bet:.3f}"
        )

    # ========================================================================
    # 5. ANÁLISE POR FAIXA DE ODDS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📊 SEÇÃO 4: ANÁLISE POR FAIXA DE ODDS (ROI >= 14%)")
    print("=" * 100)

    # Categorização de odds
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

    odds_analysis = (
        df_filtered.groupby("odds_range")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    odds_analysis.columns = ["Total_Profit", "Bets", "Avg_Profit", "Win_Rate"]
    odds_analysis["Expected_WR"] = (
        100 / df_filtered.groupby("odds_range")["odds"].mean()
    ).round(1)
    odds_analysis["WR_Diff"] = (
        odds_analysis["Win_Rate"] - odds_analysis["Expected_WR"]
    ).round(1)
    odds_analysis = odds_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n📊 PERFORMANCE POR FAIXA DE ODDS:")
    print(
        f"{'Faixa':<12} {'Lucro':<10} {'Apostas':<10} {'Win Rate':<10} {'WR Esperado':<12} {'Diferença':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 89)

    for odds_range, row in odds_analysis.iterrows():
        icon = (
            "💎"
            if row["Total_Profit"] > 5
            else "✅"
            if row["Total_Profit"] > 0
            else "❌"
        )
        profit_per_bet = row["Total_Profit"] / row["Bets"] if row["Bets"] > 0 else 0
        print(
            f"{icon} {odds_range:<10} {row['Total_Profit']:>8.2f} {int(row['Bets']):>9} "
            f"{row['Win_Rate']:>9.1f}% {row['Expected_WR']:>11.1f}% {row['WR_Diff']:>+9.1f}% "
            f"{profit_per_bet:>14.3f}"
        )

    # ========================================================================
    # 6. ANÁLISE TEMPORAL
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📅 SEÇÃO 5: ANÁLISE TEMPORAL (ROI >= 14%)")
    print("=" * 100)

    # Análise por mês
    df_filtered["month"] = df_filtered["date"].dt.to_period("M")
    monthly_analysis = (
        df_filtered.groupby("month")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    monthly_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]

    print(f"\n📅 PERFORMANCE MENSAL:")
    for month, row in monthly_analysis.iterrows():
        icon = "✅" if row["Total_Profit"] > 0 else "❌"
        profit_per_bet = row["Total_Profit"] / row["Bets"] if row["Bets"] > 0 else 0
        print(
            f"{icon} {month}: Lucro: {row['Total_Profit']:.2f} | Apostas: {int(row['Bets'])} | "
            f"WR: {row['Win_Rate']:.1f}% | Lucro/Aposta: {profit_per_bet:.3f}"
        )

    # ========================================================================
    # 7. ANÁLISE DE COMBINAÇÕES VENCEDORAS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🏆 SEÇÃO 6: TOP COMBINAÇÕES VENCEDORAS")
    print("=" * 100)

    # Criar combinações de mercado + odds
    df_filtered["combination"] = (
        df_filtered["detailed_market"] + " @ " + df_filtered["odds_range"]
    )

    combination_analysis = (
        df_filtered.groupby("combination")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    combination_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    combination_analysis = combination_analysis[
        combination_analysis["Bets"] >= 3
    ]  # Mínimo 3 apostas
    combination_analysis = combination_analysis.sort_values(
        "Total_Profit", ascending=False
    )

    print(f"\n💰 TOP 15 COMBINAÇÕES MAIS LUCRATIVAS (mínimo 3 apostas):")
    print(
        f"{'Combinação':<50} {'Lucro':<10} {'Apostas':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 95)

    for combo, row in combination_analysis.head(15).iterrows():
        if row["Total_Profit"] > 0:
            profit_per_bet = row["Total_Profit"] / row["Bets"]
            print(
                f"✅ {combo[:48]:<48} {row['Total_Profit']:>8.2f} {int(row['Bets']):>9} "
                f"{row['Win_Rate']:>9.1f}% {profit_per_bet:>14.3f}"
            )

    # ========================================================================
    # 8. RESUMO EXECUTIVO E RECOMENDAÇÕES
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("💎 RESUMO EXECUTIVO E RECOMENDAÇÕES")
    print("=" * 100)

    # Identificar melhores critérios
    profitable_markets = market_analysis[
        (market_analysis["Total_Profit"] > 3) & (market_analysis["Bets"] >= 5)
    ].index.tolist()[:10]  # Top 10 mercados

    profitable_odds = odds_analysis[
        (odds_analysis["Total_Profit"] > 2) & (odds_analysis["Bets"] >= 10)
    ].index.tolist()

    # Estatísticas finais
    filtered_profit = df_filtered["profit"].sum()
    filtered_bets = len(df_filtered)
    filtered_wr = (df_filtered["status"] == "win").mean() * 100

    print(f"\n📊 ESTATÍSTICAS COM FILTRO ROI >= 14%:")
    print(f"   💰 Lucro total: {filtered_profit:.2f} unidades")
    print(f"   🎲 Total de apostas: {filtered_bets}")
    print(f"   🎯 Win Rate: {filtered_wr:.1f}%")
    print(f"   💵 Lucro por aposta: {filtered_profit / filtered_bets:.3f} unidades")
    print(
        f"   📈 Média de apostas por dia: {filtered_bets / ((df['date'].max() - df['date'].min()).days + 1):.1f}"
    )

    print(f"\n🎯 CRITÉRIOS RECOMENDADOS PARA O BOT:")
    print(f"   1. ROI Estimado mínimo: 14%")
    print(
        f"   2. Focar nos {len(profitable_markets)} mercados mais lucrativos identificados"
    )
    print(
        f"   3. Priorizar odds nas faixas: {profitable_odds[:3] if len(profitable_odds) >= 3 else profitable_odds}"
    )

    print(f"\n💰 TOP 5 MERCADOS RECOMENDADOS:")
    for i, market in enumerate(profitable_markets[:5], 1):
        market_data = market_analysis.loc[market]
        print(
            f"   {i}. {market}: {market_data['Total_Profit']:.2f} lucro ({int(market_data['Bets'])} apostas)"
        )

    # Análise de risco
    print(f"\n⚠️ ANÁLISE DE RISCO:")

    # Calcular sequências de perdas
    df_filtered_sorted = df_filtered.sort_values("date")
    losing_streaks = []
    current_streak = 0

    for status in df_filtered_sorted["status"]:
        if status != "win":
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0

    if current_streak > 0:
        losing_streaks.append(current_streak)

    max_losing_streak = max(losing_streaks) if losing_streaks else 0

    print(f"   • Maior sequência de perdas: {max_losing_streak} apostas")
    print(
        f"   • Drawdown máximo estimado: {max_losing_streak * 1.5:.1f} unidades (com stake de 1.5)"
    )

    # Projeções
    daily_avg = filtered_bets / ((df["date"].max() - df["date"].min()).days + 1)
    profit_per_bet = filtered_profit / filtered_bets

    print(f"\n📊 PROJEÇÕES DE LUCRO:")
    print(f"   • Lucro esperado por dia: {daily_avg * profit_per_bet:.2f} unidades")
    print(
        f"   • Lucro esperado mensal (30 dias): {daily_avg * profit_per_bet * 30:.2f} unidades"
    )
    print(
        f"   • Lucro esperado anual (365 dias): {daily_avg * profit_per_bet * 365:.2f} unidades"
    )

    print(f"\n✅ CHECKLIST DE IMPLEMENTAÇÃO:")
    print(f"   ☐ Configurar filtro de ROI >= 14% no bot")
    print(f"   ☐ Implementar lista de mercados aprovados")
    print(f"   ☐ Configurar faixas de odds prioritárias")
    print(f"   ☐ Adicionar sistema de logs para monitoramento")
    print(f"   ☐ Implementar stop-loss diário")
    print(f"   ☐ Criar alertas para performance abaixo do esperado")

    print(f"\n{'=' * 100}")
    print("🏁 ANÁLISE COMPLETA FINALIZADA COM SUCESSO!")
    print("=" * 100)

    # Retornar resultados estruturados
    return {
        "summary": {
            "total_bets": total_bets,
            "total_profit": total_profit,
            "filtered_bets": filtered_bets,
            "filtered_profit": filtered_profit,
            "filtered_wr": filtered_wr,
            "roi_threshold": ROI_THRESHOLD,
        },
        "filters": {
            "roi_threshold": ROI_THRESHOLD,
            "profitable_markets": profitable_markets,
            "profitable_odds": profitable_odds,
        },
        "dataframes": {"original": df, "filtered": df_filtered},
        "analysis": {
            "bet_type_analysis": bet_type_analysis,
            "market_analysis": market_analysis,
            "odds_analysis": odds_analysis,
            "monthly_analysis": monthly_analysis,
            "combination_analysis": combination_analysis,
        },
    }


# Execução direta
if __name__ == "__main__":
    file_path = (
        "../bets/bets_atualizadas_por_mapa.csv"  # Ajuste o caminho conforme necessário
    )
    results = super_complete_betting_analysis(file_path)

    print(f"\n💾 Resultados salvos na variável 'results' com as seguintes chaves:")
    print(f"   • summary: Resumo geral da análise")
    print(f"   • filters: Critérios de filtro para o bot")
    print(f"   • dataframes: DataFrames original e filtrado")
    print(f"   • analysis: Todas as análises detalhadas")
