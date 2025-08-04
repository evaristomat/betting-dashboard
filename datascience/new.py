import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def complete_betting_analysis_with_leagues(file_path):
    """
    Análise completa de apostas com foco em mercados lucrativos e análise de ligas.
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
    print("🎯 ANÁLISE COMPLETA DE APOSTAS - SEM FILTRO DE ROI")
    print("=" * 100)

    # ========================================================================
    # 1. ANÁLISE GERAL SEM FILTRO DE ROI
    # ========================================================================
    total_bets = len(df)
    total_profit = df["profit"].sum()
    win_rate = df["status"].eq("win").mean() * 100
    roi_real_total = (total_profit / total_bets * 100) if total_bets > 0 else 0

    print(f"\n📊 VISÃO GERAL COMPLETA (TODAS AS APOSTAS):")
    print(f"   Total de apostas: {total_bets}")
    print(f"   Lucro total: {total_profit:.2f} unidades")
    print(f"   Win rate geral: {win_rate:.1f}%")
    print(f"   ROI real médio: {roi_real_total:.1f}%")
    print(f"   Lucro por aposta: {total_profit / total_bets:.3f} unidades")
    print(
        f"   Período: {df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}"
    )

    # ========================================================================
    # 2. IDENTIFICAR MERCADOS LUCRATIVOS (BET_TYPE)
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("💰 IDENTIFICAÇÃO DE MERCADOS LUCRATIVOS (BET_TYPE)")
    print("=" * 100)

    bet_type_analysis = (
        df.groupby("bet_type")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "odds": ["mean"],
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
        "Win_Rate",
    ]
    bet_type_analysis["ROI_Real"] = (
        bet_type_analysis["Total_Profit"] / bet_type_analysis["Bets"] * 100
    ).round(1)
    bet_type_analysis = bet_type_analysis.sort_values("Total_Profit", ascending=False)

    # Identificar mercados lucrativos
    profitable_bet_types = bet_type_analysis[
        bet_type_analysis["Total_Profit"] > 0
    ].index.tolist()

    print(f"\n📊 TODOS OS MERCADOS (BET_TYPE):")
    print(
        f"{'Tipo':<35} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'Win Rate':<10} {'Odds Médias':<12}"
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
        print(
            f"{icon} {bet_type[:33]:<33} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>9} {row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}% "
            f"{row['Avg_Odds']:>11.2f}"
        )

    print(f"\n✅ MERCADOS LUCRATIVOS IDENTIFICADOS: {len(profitable_bet_types)}")
    print(
        f"❌ MERCADOS COM PREJUÍZO: {len(bet_type_analysis) - len(profitable_bet_types)}"
    )

    # ========================================================================
    # 3. ANÁLISE DE ROI APÓS FILTRAR MERCADOS LUCRATIVOS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📈 ANÁLISE DE ROI COM APENAS MERCADOS LUCRATIVOS")
    print("=" * 100)

    # Filtrar apenas mercados lucrativos
    df_profitable = df[df["bet_type"].isin(profitable_bet_types)]

    profitable_bets = len(df_profitable)
    profitable_profit = df_profitable["profit"].sum()
    profitable_wr = (df_profitable["status"] == "win").mean() * 100
    profitable_roi = (
        (profitable_profit / profitable_bets * 100) if profitable_bets > 0 else 0
    )

    print(f"\n📊 ESTATÍSTICAS COM APENAS MERCADOS LUCRATIVOS:")
    print(
        f"   Apostas em mercados lucrativos: {profitable_bets} ({profitable_bets / total_bets * 100:.1f}% do total)"
    )
    print(f"   Lucro em mercados lucrativos: {profitable_profit:.2f} unidades")
    print(f"   Win rate em mercados lucrativos: {profitable_wr:.1f}%")
    print(f"   ROI real em mercados lucrativos: {profitable_roi:.1f}%")
    print(f"   Lucro por aposta: {profitable_profit / profitable_bets:.3f} unidades")

    # Análise de ROI por faixas nos mercados lucrativos
    roi_ranges = [0, 10, 14, 15, 20, 25, 30, 35, 40, 50]

    print(f"\n📊 ANÁLISE POR FAIXA DE ROI (APENAS MERCADOS LUCRATIVOS):")
    print(
        f"{'Range':<10} {'Apostas':<10} {'% Total':<10} {'Lucro':<12} {'ROI Real':<10} {'Win Rate':<10}"
    )
    print("-" * 72)

    for min_roi in roi_ranges:
        df_range = df_profitable[df_profitable["estimated_roi"] >= min_roi]
        if len(df_range) > 0:
            range_profit = df_range["profit"].sum()
            range_bets = len(df_range)
            range_wr = (df_range["status"] == "win").mean() * 100
            percentage = range_bets / profitable_bets * 100
            roi_real = (range_profit / range_bets * 100) if range_bets > 0 else 0

            print(
                f"{f'{min_roi}%+':<10} {range_bets:<10} {percentage:>8.1f}% "
                f"{range_profit:>11.2f} {roi_real:>9.1f}% {range_wr:>9.1f}%"
            )

    # ========================================================================
    # 4. ANÁLISE DE LIGAS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🏆 ANÁLISE POR LIGAS (TODAS AS APOSTAS)")
    print("=" * 100)

    # Extrair liga do evento (assumindo formato "Team1 vs Team2 - League")
    def extract_league(event):
        if pd.isna(event):
            return "Unknown"
        parts = str(event).split(" - ")
        return parts[-1] if len(parts) > 1 else "Unknown"

    df["league"] = df["event"].apply(extract_league)

    # Análise por liga
    league_analysis = (
        df.groupby("league")
        .agg(
            {
                "profit": ["sum", "count", "mean"],
                "status": lambda x: (x == "win").mean() * 100,
                "odds": ["mean"],
            }
        )
        .round(2)
    )

    league_analysis.columns = [
        "Total_Profit",
        "Bets",
        "Avg_Profit",
        "Win_Rate",
        "Avg_Odds",
    ]
    league_analysis["ROI_Real"] = (
        league_analysis["Total_Profit"] / league_analysis["Bets"] * 100
    ).round(1)
    league_analysis = league_analysis.sort_values("Total_Profit", ascending=False)

    # Ligas lucrativas
    profitable_leagues = league_analysis[
        league_analysis["Total_Profit"] > 0
    ].index.tolist()

    print(f"\n📊 TOP 20 LIGAS POR LUCRO:")
    print(
        f"{'Liga':<40} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'Win Rate':<10}"
    )
    print("-" * 80)

    for league, row in league_analysis.head(20).iterrows():
        icon = (
            "🏆"
            if row["Total_Profit"] > 10
            else "✅"
            if row["Total_Profit"] > 0
            else "❌"
        )
        print(
            f"{icon} {league[:38]:<38} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>9} {row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}%"
        )

    print(f"\n✅ LIGAS LUCRATIVAS: {len(profitable_leagues)}")
    print(f"❌ LIGAS COM PREJUÍZO: {len(league_analysis) - len(profitable_leagues)}")

    # ========================================================================
    # 5. ANÁLISE FOCADA APENAS NOS LUCRATIVOS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("💎 ANÁLISE FOCADA: APENAS MERCADOS E LIGAS LUCRATIVOS")
    print("=" * 100)

    # Filtrar por mercados E ligas lucrativas
    df_super_profitable = df[
        (df["bet_type"].isin(profitable_bet_types))
        & (df["league"].isin(profitable_leagues))
    ]

    super_profitable_bets = len(df_super_profitable)
    super_profitable_profit = df_super_profitable["profit"].sum()
    super_profitable_wr = (df_super_profitable["status"] == "win").mean() * 100
    super_profitable_roi = (
        (super_profitable_profit / super_profitable_bets * 100)
        if super_profitable_bets > 0
        else 0
    )

    print(f"\n📊 ESTATÍSTICAS COM MERCADOS E LIGAS LUCRATIVOS:")
    print(
        f"   Total de apostas: {super_profitable_bets} ({super_profitable_bets / total_bets * 100:.1f}% do total)"
    )
    print(f"   Lucro total: {super_profitable_profit:.2f} unidades")
    print(f"   Win rate: {super_profitable_wr:.1f}%")
    print(f"   ROI real: {super_profitable_roi:.1f}%")
    print(
        f"   Lucro por aposta: {super_profitable_profit / super_profitable_bets:.3f} unidades"
    )

    # Melhor faixa de ROI nos lucrativos
    print(f"\n📊 MELHOR FAIXA DE ROI (MERCADOS+LIGAS LUCRATIVOS):")

    best_roi_analysis = []
    for min_roi in roi_ranges:
        df_roi = df_super_profitable[df_super_profitable["estimated_roi"] >= min_roi]
        if len(df_roi) >= 10:  # Mínimo 10 apostas para relevância
            roi_profit = df_roi["profit"].sum()
            roi_bets = len(df_roi)
            roi_wr = (df_roi["status"] == "win").mean() * 100
            roi_real = (roi_profit / roi_bets * 100) if roi_bets > 0 else 0

            best_roi_analysis.append(
                {
                    "ROI_Min": min_roi,
                    "Profit": roi_profit,
                    "Bets": roi_bets,
                    "ROI_Real": roi_real,
                    "Win_Rate": roi_wr,
                    "Profit_Per_Bet": roi_profit / roi_bets,
                }
            )

    if best_roi_analysis:
        best_roi_df = pd.DataFrame(best_roi_analysis)
        best_roi_df = best_roi_df.sort_values("Profit_Per_Bet", ascending=False)

        print(
            f"\n{'ROI Mín':<10} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
        )
        print("-" * 75)

        for _, row in best_roi_df.iterrows():
            print(
                f"{row['ROI_Min']:>7}%+ {row['Profit']:>9.2f} {int(row['Bets']):>9} "
                f"{row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}% {row['Profit_Per_Bet']:>14.3f}"
            )

        # Identificar melhor faixa
        best_roi = best_roi_df.iloc[0]["ROI_Min"]
        print(f"\n🎯 MELHOR FAIXA DE ROI IDENTIFICADA: {best_roi}%+")

    # Top combinações lucrativas
    print(f"\n🏆 TOP 10 COMBINAÇÕES LUCRATIVAS (MERCADO + LIGA):")

    df_super_profitable["combination"] = (
        df_super_profitable["bet_type"] + " @ " + df_super_profitable["league"]
    )

    combo_analysis = (
        df_super_profitable.groupby("combination")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    combo_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    combo_analysis = combo_analysis[combo_analysis["Bets"] >= 3]  # Mínimo 3 apostas
    combo_analysis["Profit_Per_Bet"] = (
        combo_analysis["Total_Profit"] / combo_analysis["Bets"]
    )
    combo_analysis = combo_analysis.sort_values("Total_Profit", ascending=False)

    print(
        f"\n{'Combinação':<60} {'Lucro':<10} {'Apostas':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 105)

    for combo, row in combo_analysis.head(10).iterrows():
        print(
            f"{combo[:58]:<58} {row['Total_Profit']:>8.2f} {int(row['Bets']):>9} "
            f"{row['Win_Rate']:>9.1f}% {row['Profit_Per_Bet']:>14.3f}"
        )

    # ========================================================================
    # 6. RESUMO EXECUTIVO E RECOMENDAÇÕES
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("💎 RESUMO EXECUTIVO E RECOMENDAÇÕES ESTRATÉGICAS")
    print("=" * 100)

    print(f"\n📊 COMPARAÇÃO DE CENÁRIOS:")
    print(f"{'Cenário':<40} {'Apostas':<10} {'Lucro':<10} {'ROI':<10} {'Win Rate':<10}")
    print("-" * 80)

    scenarios = [
        ("Todas as apostas", total_bets, total_profit, roi_real_total, win_rate),
        (
            "Apenas mercados lucrativos",
            profitable_bets,
            profitable_profit,
            profitable_roi,
            profitable_wr,
        ),
        (
            "Mercados + Ligas lucrativas",
            super_profitable_bets,
            super_profitable_profit,
            super_profitable_roi,
            super_profitable_wr,
        ),
    ]

    for scenario, bets, profit, roi, wr in scenarios:
        print(f"{scenario:<40} {bets:<10} {profit:>9.2f} {roi:>9.1f}% {wr:>9.1f}%")

    print(f"\n🎯 ESTRATÉGIA RECOMENDADA:")
    print(
        f"   1. FOCAR APENAS em mercados lucrativos: {len(profitable_bet_types)} tipos identificados"
    )
    print(
        f"   2. PRIORIZAR ligas lucrativas: {len(profitable_leagues[:10])} principais ligas"
    )
    if best_roi_analysis:
        print(f"   3. APLICAR filtro de ROI mínimo: {best_roi}%")
    print(
        f"   4. EVITAR {len(bet_type_analysis) - len(profitable_bet_types)} mercados com histórico negativo"
    )

    # Top 5 mercados mais lucrativos
    print(f"\n💰 TOP 5 MERCADOS MAIS LUCRATIVOS:")
    for i, (bet_type, row) in enumerate(bet_type_analysis.head(5).iterrows(), 1):
        if row["Total_Profit"] > 0:
            print(
                f"   {i}. {bet_type}: {row['Total_Profit']:.2f} lucro | "
                f"{int(row['Bets'])} apostas | ROI: {row['ROI_Real']:.1f}%"
            )

    # Top 5 ligas mais lucrativas
    print(f"\n🏆 TOP 5 LIGAS MAIS LUCRATIVAS:")
    for i, (league, row) in enumerate(league_analysis.head(5).iterrows(), 1):
        if row["Total_Profit"] > 0:
            print(
                f"   {i}. {league}: {row['Total_Profit']:.2f} lucro | "
                f"{int(row['Bets'])} apostas | ROI: {row['ROI_Real']:.1f}%"
            )

    print(f"\n⚠️ GESTÃO DE RISCO:")

    # Calcular maior sequência de perdas nos lucrativos
    df_super_profitable_sorted = df_super_profitable.sort_values("date")
    losing_streaks = []
    current_streak = 0

    for status in df_super_profitable_sorted["status"]:
        if status != "win":
            current_streak += 1
        else:
            if current_streak > 0:
                losing_streaks.append(current_streak)
            current_streak = 0

    max_losing_streak = max(losing_streaks) if losing_streaks else 0

    print(f"   • Maior sequência de perdas (lucrativos): {max_losing_streak} apostas")
    print(f"   • Win rate nos lucrativos: {super_profitable_wr:.1f}%")
    print(f"   • Sugestão de bankroll: {max_losing_streak * 2} unidades mínimas")

    print(f"\n📈 POTENCIAL DE CRESCIMENTO:")
    improvement = (
        ((super_profitable_roi - roi_real_total) / roi_real_total * 100)
        if roi_real_total > 0
        else 0
    )
    print(f"   • Melhoria de ROI ao filtrar: +{improvement:.1f}%")
    print(
        f"   • Redução de apostas: -{100 - (super_profitable_bets / total_bets * 100):.1f}%"
    )
    print(
        f"   • Aumento de eficiência: {super_profitable_profit / super_profitable_bets:.3f} vs {total_profit / total_bets:.3f} unidades/aposta"
    )

    print(f"\n{'=' * 100}")
    print("🏁 ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 100)

    return {
        "summary": {
            "total": {
                "bets": total_bets,
                "profit": total_profit,
                "roi": roi_real_total,
                "wr": win_rate,
            },
            "profitable_markets": {
                "bets": profitable_bets,
                "profit": profitable_profit,
                "roi": profitable_roi,
                "wr": profitable_wr,
            },
            "super_profitable": {
                "bets": super_profitable_bets,
                "profit": super_profitable_profit,
                "roi": super_profitable_roi,
                "wr": super_profitable_wr,
            },
        },
        "filters": {
            "profitable_bet_types": profitable_bet_types,
            "profitable_leagues": profitable_leagues[:20],  # Top 20 ligas
            "best_roi_threshold": best_roi if best_roi_analysis else 14,
        },
        "analysis": {
            "bet_type_analysis": bet_type_analysis,
            "league_analysis": league_analysis,
            "combo_analysis": combo_analysis,
        },
    }


# Execução
if __name__ == "__main__":
    file_path = "../bets/bets_atualizadas_por_mapa.csv"  # Ajuste conforme necessário
    results = complete_betting_analysis_with_leagues(file_path)

    print(f"\n💾 Resultados salvos em 'results' com:")
    print(f"   • summary: Resumos comparativos")
    print(f"   • filters: Mercados e ligas lucrativas")
    print(f"   • analysis: Análises detalhadas")
