import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def simplified_betting_analysis(file_path):
    """
    Análise simplificada focada em: bet_line (mercados), ROI e ligas
    """

    # ========================================================================
    # CARREGAMENTO E PREPARAÇÃO DOS DADOS
    # ========================================================================
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y %H:%M", errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

    print("=" * 100)
    print("🎯 ANÁLISE COMPLETA DE APOSTAS - IDENTIFICAÇÃO DE MERCADOS LUCRATIVOS")
    print("=" * 100)

    # ========================================================================
    # 1. ANÁLISE GERAL - TODAS AS APOSTAS
    # ========================================================================
    total_bets = len(df)
    total_profit = df["profit"].sum()
    win_rate = df["status"].eq("win").mean() * 100
    roi_real_total = (total_profit / total_bets * 100) if total_bets > 0 else 0

    print(f"\n📊 ESTATÍSTICAS GERAIS (TODAS AS APOSTAS):")
    print(f"   Total de apostas: {total_bets}")
    print(f"   Lucro total: {total_profit:.2f} unidades")
    print(f"   Win rate: {win_rate:.1f}%")
    print(f"   ROI real: {roi_real_total:.1f}%")
    print(
        f"   Período: {df['date'].min().strftime('%d/%m/%Y')} a {df['date'].max().strftime('%d/%m/%Y')}"
    )

    # ========================================================================
    # 2. ANÁLISE DE MERCADOS (BET_TYPE + BET_LINE)
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📋 ANÁLISE DETALHADA POR MERCADO (BET_TYPE + BET_LINE)")
    print("=" * 100)

    # Criar mercado combinando bet_type e bet_line (removendo valores numéricos)
    def clean_market(bet_type, bet_line):
        # Remover números e pontos decimais do bet_line
        import re

        cleaned_line = re.sub(r"\s*\d+\.?\d*\s*", "", str(bet_line)).strip()
        return f"{bet_type} - {cleaned_line}"

    df["market"] = df.apply(
        lambda row: clean_market(row["bet_type"], row["bet_line"]), axis=1
    )

    market_analysis = (
        df.groupby("market")
        .agg(
            {
                "profit": ["sum", "count"],
                "status": lambda x: (x == "win").mean() * 100,
                "odds": "mean",
            }
        )
        .round(2)
    )

    market_analysis.columns = ["Total_Profit", "Bets", "Win_Rate", "Avg_Odds"]
    market_analysis["ROI_Real"] = (
        market_analysis["Total_Profit"] / market_analysis["Bets"] * 100
    ).round(1)
    market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)

    # Filtrar apenas mercados com pelo menos 10 apostas para relevância estatística
    relevant_markets = market_analysis[market_analysis["Bets"] >= 10]

    print(f"\n📊 MERCADOS COM PELO MENOS 10 APOSTAS:")
    print(
        f"{'Mercado':<50} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'Win Rate':<10}"
    )
    print("-" * 90)

    for market, row in relevant_markets.iterrows():
        icon = (
            "💎"
            if row["Total_Profit"] > 10
            else "✅"
            if row["Total_Profit"] > 0
            else "❌"
        )
        print(
            f"{icon} {market[:48]:<48} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>9} {row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}%"
        )

    # Identificar mercados lucrativos
    profitable_markets = relevant_markets[
        relevant_markets["Total_Profit"] > 0
    ].index.tolist()

    print(f"\n✅ MERCADOS LUCRATIVOS: {len(profitable_markets)}")
    print(
        f"❌ MERCADOS COM PREJUÍZO: {len(relevant_markets) - len(profitable_markets)}"
    )

    # ========================================================================
    # 3. ANÁLISE DE FAIXAS DE ROI - APÓS FILTRAR MERCADOS LUCRATIVOS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📈 ANÁLISE POR FAIXAS DE ROI (APENAS MERCADOS LUCRATIVOS)")
    print("=" * 100)

    # Filtrar apenas mercados lucrativos
    df_profitable_markets = df[df["market"].isin(profitable_markets)]

    print(
        f"\n📊 Apostas em mercados lucrativos: {len(df_profitable_markets)} ({len(df_profitable_markets) / total_bets * 100:.1f}% do total)"
    )

    roi_ranges = [0, 10, 14, 15, 20, 25, 30, 35, 40, 50]
    roi_analysis = []

    print(f"\n📊 PERFORMANCE POR FAIXA DE ROI (MERCADOS LUCRATIVOS):")
    print(
        f"{'Faixa ROI':<12} {'Apostas':<10} {'Lucro':<12} {'ROI Real':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 79)

    for min_roi in roi_ranges:
        df_roi = df_profitable_markets[
            df_profitable_markets["estimated_roi"] >= min_roi
        ]
        if len(df_roi) > 0:
            roi_profit = df_roi["profit"].sum()
            roi_bets = len(df_roi)
            roi_wr = (df_roi["status"] == "win").mean() * 100
            roi_real = (roi_profit / roi_bets * 100) if roi_bets > 0 else 0
            profit_per_bet = roi_profit / roi_bets

            # Guardar se ROI real > 5%
            if roi_real > 5:
                roi_analysis.append(min_roi)

            icon = "✅" if roi_real > 5 else "❌"
            print(
                f"{icon} {f'{min_roi}%+':<10} {roi_bets:<10} {roi_profit:>11.2f} "
                f"{roi_real:>9.1f}% {roi_wr:>9.1f}% {profit_per_bet:>14.3f}"
            )

    # Determinar faixa mínima de ROI
    min_roi_threshold = roi_analysis[0] if roi_analysis else 0
    print(
        f"\n🎯 FAIXA DE ROI RECOMENDADA: {min_roi_threshold}%+ (todas com ROI real > 5%)"
    )

    # ========================================================================
    # 4. ANÁLISE POR LIGAS - APÓS FILTRAR MERCADOS LUCRATIVOS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🏆 ANÁLISE POR LIGAS (APENAS MERCADOS LUCRATIVOS)")
    print("=" * 100)

    league_analysis = (
        df_profitable_markets.groupby("league")
        .agg(
            {
                "profit": ["sum", "count"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    league_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    league_analysis["ROI_Real"] = (
        league_analysis["Total_Profit"] / league_analysis["Bets"] * 100
    ).round(1)
    league_analysis = league_analysis.sort_values("Total_Profit", ascending=False)

    # Ligas lucrativas
    profitable_leagues = league_analysis[
        league_analysis["Total_Profit"] > 0
    ].index.tolist()

    print(f"\n📊 TODAS AS LIGAS (EM MERCADOS LUCRATIVOS):")
    print(
        f"{'Liga':<20} {'Lucro':<10} {'Apostas':<10} {'ROI Real':<10} {'Win Rate':<10}"
    )
    print("-" * 60)

    for league, row in league_analysis.iterrows():
        icon = (
            "💎"
            if row["Total_Profit"] > 10
            else "✅"
            if row["Total_Profit"] > 0
            else "❌"
        )
        print(
            f"{icon} {league[:18]:<18} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>9} {row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}%"
        )

    print(f"\n✅ LIGAS LUCRATIVAS: {len(profitable_leagues)}")
    print(f"❌ LIGAS COM PREJUÍZO: {len(league_analysis) - len(profitable_leagues)}")

    # ========================================================================
    # 5. ANÁLISE AUTOMATIZADA DE MERCADOS EM LIGAS NEGATIVAS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🔍 ANÁLISE AUTOMATIZADA - MERCADOS LUCRATIVOS EM LIGAS NEGATIVAS")
    print("=" * 100)

    # Primeiro, analisar todas as ligas (não apenas mercados lucrativos)
    all_league_analysis = (
        df.groupby("league")
        .agg(
            {
                "profit": ["sum", "count"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    all_league_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    all_league_analysis["ROI_Real"] = (
        all_league_analysis["Total_Profit"] / all_league_analysis["Bets"] * 100
    ).round(1)

    # Identificar ligas negativas com pelo menos 20 apostas
    negative_leagues = all_league_analysis[
        (all_league_analysis["Total_Profit"] < 0) & (all_league_analysis["Bets"] >= 20)
    ].sort_values("Total_Profit", ascending=True)

    print(
        f"\n📊 LIGAS NEGATIVAS COM ≥20 APOSTAS IDENTIFICADAS: {len(negative_leagues)}"
    )

    if len(negative_leagues) > 0:
        print(
            f"{'Liga':<20} {'Prejuízo':<12} {'Apostas':<10} {'ROI':<10} {'Win Rate':<10}"
        )
        print("-" * 62)

        for league, row in negative_leagues.iterrows():
            print(
                f"❌ {league[:18]:<18} {row['Total_Profit']:>10.2f} "
                f"{int(row['Bets']):>9} {row['ROI_Real']:>9.1f}% {row['Win_Rate']:>9.1f}%"
            )

        # Análise detalhada de cada liga negativa
        print(f"\n{'=' * 100}")
        print("🔎 ANÁLISE DETALHADA - MERCADOS LUCRATIVOS EM CADA LIGA NEGATIVA")
        print("=" * 100)

        best_markets_summary = []

        for league in negative_leagues.index:
            df_league = df[df["league"] == league]

            print(f"\n{'=' * 80}")
            print(f"📊 LIGA: {league}")
            print(f"{'=' * 80}")

            # Análise de mercados para esta liga específica
            league_market_analysis = (
                df_league.groupby("market")
                .agg(
                    {
                        "profit": ["sum", "count"],
                        "status": lambda x: (x == "win").mean() * 100,
                        "odds": "mean",
                    }
                )
                .round(2)
            )

            league_market_analysis.columns = ["Profit", "Bets", "Win_Rate", "Avg_Odds"]
            league_market_analysis["ROI"] = (
                league_market_analysis["Profit"] / league_market_analysis["Bets"] * 100
            ).round(1)

            # Filtrar mercados com pelo menos 3 apostas e lucrativos
            profitable_markets_in_league = league_market_analysis[
                (league_market_analysis["Bets"] >= 3)
                & (league_market_analysis["Profit"] > 0)
            ].sort_values("Profit", ascending=False)

            total_league_bets = len(df_league)
            total_league_profit = df_league["profit"].sum()

            print(f"📈 Estatísticas da Liga:")
            print(f"   Total apostas: {total_league_bets}")
            print(f"   Prejuízo total: {total_league_profit:.2f}")
            print(f"   ROI: {(total_league_profit / total_league_bets * 100):.1f}%")

            if len(profitable_markets_in_league) > 0:
                print(
                    f"\n✅ MERCADOS LUCRATIVOS ENCONTRADOS: {len(profitable_markets_in_league)}"
                )
                print(
                    f"{'Mercado':<40} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'Win Rate':<10} {'Odds Média':<10}"
                )
                print("-" * 90)

                # Mostrar os 2 melhores mercados
                top_2_markets = profitable_markets_in_league.head(2)

                for i, (market, row) in enumerate(top_2_markets.iterrows(), 1):
                    pct_of_bets = row["Bets"] / total_league_bets * 100
                    print(
                        f"💎 {market[:38]:<38} {row['Profit']:>8.2f} "
                        f"{int(row['Bets']):>8} ({pct_of_bets:>4.1f}%) {row['ROI']:>8.1f}% "
                        f"{row['Win_Rate']:>8.1f}% {row['Avg_Odds']:>9.2f}"
                    )

                    # Adicionar ao resumo
                    best_markets_summary.append(
                        {
                            "Liga": league,
                            "Mercado": market,
                            "Lucro": row["Profit"],
                            "Apostas": int(row["Bets"]),
                            "ROI": row["ROI"],
                            "Win_Rate": row["Win_Rate"],
                            "Rank": i,
                        }
                    )

                # Mostrar impacto potencial
                total_profit_top2 = top_2_markets["Profit"].sum()
                total_bets_top2 = top_2_markets["Bets"].sum()

                print(f"\n🎯 IMPACTO DOS 2 MELHORES MERCADOS:")
                print(f"   Lucro combinado: {total_profit_top2:.2f}")
                print(
                    f"   Apostas combinadas: {int(total_bets_top2)} ({total_bets_top2 / total_league_bets * 100:.1f}% da liga)"
                )
                print(
                    f"   ROI médio: {(total_profit_top2 / total_bets_top2 * 100):.1f}%"
                )

            else:
                print(f"\n❌ NENHUM MERCADO LUCRATIVO ENCONTRADO")
                print("   Todos os mercados com ≥3 apostas são negativos")

        # ========================================================================
        # 6. TABELA RESUMO DOS MELHORES MERCADOS EM LIGAS NEGATIVAS
        # ========================================================================
        if best_markets_summary:
            print(f"\n{'=' * 100}")
            print("📋 TABELA RESUMO - 2 MELHORES MERCADOS POR LIGA NEGATIVA")
            print("=" * 100)

            df_summary = pd.DataFrame(best_markets_summary)

            print(
                f"{'Liga':<15} {'Rank':<5} {'Mercado':<35} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'Win Rate':<10}"
            )
            print("-" * 95)

            for _, row in df_summary.iterrows():
                icon = "🥇" if row["Rank"] == 1 else "🥈"
                print(
                    f"{row['Liga']:<15} {icon:<5} {row['Mercado'][:33]:<33} "
                    f"{row['Lucro']:>8.2f} {row['Apostas']:>9} "
                    f"{row['ROI']:>9.1f}% {row['Win_Rate']:>9.1f}%"
                )

            # Estatísticas agregadas
            total_potential_profit = df_summary["Lucro"].sum()
            total_potential_bets = df_summary["Apostas"].sum()
            avg_roi = df_summary["ROI"].mean()

            print(f"\n📊 ESTATÍSTICAS AGREGADAS DOS MELHORES MERCADOS:")
            print(f"   Total de mercados identificados: {len(df_summary)}")
            print(f"   Lucro potencial total: {total_potential_profit:.2f}")
            print(f"   Apostas totais: {total_potential_bets}")
            print(f"   ROI médio: {avg_roi:.1f}%")

            # Top 3 mercados mais lucrativos overall
            top_3_overall = df_summary.nlargest(3, "Lucro")
            print(f"\n🏆 TOP 3 MERCADOS MAIS LUCRATIVOS EM LIGAS NEGATIVAS:")
            for i, (_, row) in enumerate(top_3_overall.iterrows(), 1):
                print(
                    f"   {i}. {row['Liga']} - {row['Mercado']}: {row['Lucro']:.2f} lucro (ROI: {row['ROI']:.1f}%)"
                )

        else:
            print(f"\n❌ NENHUM MERCADO LUCRATIVO ENCONTRADO EM LIGAS NEGATIVAS")
            print(
                "   Todas as ligas negativas não possuem mercados lucrativos com volume significativo"
            )

    else:
        print(f"\n✅ NENHUMA LIGA NEGATIVA COM ≥20 APOSTAS ENCONTRADA")

    # ========================================================================
    # 7. ANÁLISE DE ODDS DOS MERCADOS LUCRATIVOS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📊 ANÁLISE DE FAIXAS DE ODDS - APENAS MERCADOS LUCRATIVOS")
    print("=" * 100)

    # Filtrar apenas mercados lucrativos já identificados
    df_profitable_markets_only = df[df["market"].isin(profitable_markets)]

    print(
        f"\n📈 Total de apostas em mercados lucrativos: {len(df_profitable_markets_only)}"
    )

    # Função para categorizar odds em faixas mais detalhadas
    def categorize_odds_detailed(odds):
        if pd.isna(odds):
            return "N/A"
        elif odds <= 1.30:
            return "1.00-1.30"
        elif odds <= 1.50:
            return "1.30-1.50"
        elif odds <= 1.70:
            return "1.50-1.70"
        elif odds <= 1.90:
            return "1.70-1.90"
        elif odds <= 2.10:
            return "1.90-2.10"
        elif odds <= 2.30:
            return "2.10-2.30"
        elif odds <= 2.50:
            return "2.30-2.50"
        elif odds <= 3.00:
            return "2.50-3.00"
        else:
            return "3.00+"

    df_profitable_markets_only["odds_range"] = df_profitable_markets_only["odds"].apply(
        categorize_odds_detailed
    )

    # Análise geral por faixa de odds
    odds_analysis = (
        df_profitable_markets_only.groupby("odds_range")
        .agg(
            {
                "profit": ["sum", "count"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        .round(2)
    )

    odds_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    odds_analysis["ROI"] = (
        odds_analysis["Total_Profit"] / odds_analysis["Bets"] * 100
    ).round(1)
    odds_analysis["Avg_Profit_Per_Bet"] = (
        odds_analysis["Total_Profit"] / odds_analysis["Bets"]
    ).round(3)

    # Ordenar por ordem de odds
    odds_order = [
        "1.00-1.30",
        "1.30-1.50",
        "1.50-1.70",
        "1.70-1.90",
        "1.90-2.10",
        "2.10-2.30",
        "2.30-2.50",
        "2.50-3.00",
        "3.00+",
    ]
    odds_analysis = odds_analysis.reindex(
        [o for o in odds_order if o in odds_analysis.index]
    )

    print(f"\n📊 PERFORMANCE POR FAIXA DE ODDS (MERCADOS LUCRATIVOS):")
    print(
        f"{'Faixa Odds':<12} {'Lucro':<10} {'Apostas':<10} {'ROI':<10} {'Win Rate':<10} {'Lucro/Aposta':<15}"
    )
    print("-" * 77)

    for odds_range, row in odds_analysis.iterrows():
        icon = "💎" if row["ROI"] > 20 else "✅" if row["ROI"] > 0 else "❌"
        percentage = row["Bets"] / len(df_profitable_markets_only) * 100
        print(
            f"{icon} {odds_range:<10} {row['Total_Profit']:>8.2f} "
            f"{int(row['Bets']):>8} ({percentage:>4.1f}%) {row['ROI']:>8.1f}% "
            f"{row['Win_Rate']:>8.1f}% {row['Avg_Profit_Per_Bet']:>13.3f}"
        )

    # Identificar faixas mais lucrativas
    profitable_odds_ranges = odds_analysis[odds_analysis["ROI"] > 10].index.tolist()

    print(
        f"\n✅ FAIXAS DE ODDS MAIS LUCRATIVAS (ROI > 10%): {len(profitable_odds_ranges)}"
    )
    for odds_range in profitable_odds_ranges:
        data = odds_analysis.loc[odds_range]
        print(
            f"   • {odds_range}: ROI {data['ROI']:.1f}% | {int(data['Bets'])} apostas"
        )

    # Resumo estratégico
    print(f"\n{'=' * 100}")
    print("🎯 RESUMO ESTRATÉGICO - ODDS EM MERCADOS LUCRATIVOS")
    print("=" * 100)

    # Calcular médias por categoria de odds
    low_odds_ranges = ["1.00-1.30", "1.30-1.50", "1.50-1.70"]
    medium_odds_ranges = ["1.70-1.90", "1.90-2.10"]
    high_odds_ranges = ["2.10-2.30", "2.30-2.50", "2.50-3.00", "3.00+"]

    # Filtrar apenas as faixas que existem no índice
    existing_low = [r for r in low_odds_ranges if r in odds_analysis.index]
    existing_medium = [r for r in medium_odds_ranges if r in odds_analysis.index]
    existing_high = [r for r in high_odds_ranges if r in odds_analysis.index]

    print(f"\n📊 AGRUPAMENTO POR CATEGORIA:")
    print(f"{'Categoria':<20} {'Lucro':<15} {'Apostas':<15} {'ROI':<15}")
    print("-" * 65)

    if existing_low:
        low_odds = odds_analysis.loc[existing_low, ["Total_Profit", "Bets"]].sum()
        if low_odds["Bets"] > 0:
            low_roi = low_odds["Total_Profit"] / low_odds["Bets"] * 100
            print(
                f"{'Odds Baixas (≤1.70)':<20} {low_odds['Total_Profit']:>13.2f} "
                f"{int(low_odds['Bets']):>14} {low_roi:>14.1f}%"
            )

    if existing_medium:
        medium_odds = odds_analysis.loc[existing_medium, ["Total_Profit", "Bets"]].sum()
        if medium_odds["Bets"] > 0:
            medium_roi = medium_odds["Total_Profit"] / medium_odds["Bets"] * 100
            print(
                f"{'Odds Médias':<20} {medium_odds['Total_Profit']:>13.2f} "
                f"{int(medium_odds['Bets']):>14} {medium_roi:>14.1f}%"
            )

    if existing_high:
        high_odds = odds_analysis.loc[existing_high, ["Total_Profit", "Bets"]].sum()
        if high_odds["Bets"] > 0:
            high_roi = high_odds["Total_Profit"] / high_odds["Bets"] * 100
            print(
                f"{'Odds Altas (>2.10)':<20} {high_odds['Total_Profit']:>13.2f} "
                f"{int(high_odds['Bets']):>14} {high_roi:>14.1f}%"
            )

    print(f"\n💡 RECOMENDAÇÕES DE ODDS:")
    if profitable_odds_ranges:
        print(f"   • Priorizar faixas: {', '.join(profitable_odds_ranges[:3])}")
        print(
            f"   • ROI médio nas faixas recomendadas: {odds_analysis.loc[profitable_odds_ranges, 'ROI'].mean():.1f}%"
        )

    # ========================================================================
    # 8. APLICAR TODOS OS FILTROS
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("🎯 APLICAÇÃO DOS FILTROS")
    print("=" * 100)

    # Aplicar filtros progressivamente
    df_filtered = df.copy()

    # Filtro 1: Mercados lucrativos
    df_filtered = df_filtered[df_filtered["market"].isin(profitable_markets)]
    print(f"\n✅ Após filtro de MERCADOS LUCRATIVOS: {len(df_filtered)} apostas")

    # Filtro 2: ROI mínimo
    df_filtered = df_filtered[df_filtered["estimated_roi"] >= min_roi_threshold]
    print(f"✅ Após filtro de ROI >= {min_roi_threshold}%: {len(df_filtered)} apostas")

    # Filtro 3: Ligas lucrativas
    df_filtered = df_filtered[df_filtered["league"].isin(profitable_leagues)]
    print(f"✅ Após filtro de LIGAS LUCRATIVAS: {len(df_filtered)} apostas")

    # ========================================================================
    # 9. TABELA COMPARATIVA FINAL
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("📊 TABELA COMPARATIVA - ANTES vs DEPOIS DOS FILTROS")
    print("=" * 100)

    # Calcular métricas após filtros
    filtered_bets = len(df_filtered)
    filtered_profit = df_filtered["profit"].sum()
    filtered_wr = (
        (df_filtered["status"] == "win").mean() * 100 if filtered_bets > 0 else 0
    )
    filtered_roi = (filtered_profit / filtered_bets * 100) if filtered_bets > 0 else 0

    # Criar tabela comparativa
    print(f"\n{'Métrica':<25} {'Antes':<15} {'Depois':<15} {'Variação':<15}")
    print("-" * 70)

    # Total de apostas
    variation_bets = (filtered_bets - total_bets) / total_bets * 100
    print(
        f"{'Total de Apostas':<25} {total_bets:<15} {filtered_bets:<15} {variation_bets:>+14.1f}%"
    )

    # Lucro total
    variation_profit = (
        ((filtered_profit - total_profit) / total_profit * 100)
        if total_profit != 0
        else 0
    )
    print(
        f"{'Lucro Total':<25} {f'{total_profit:.2f}':<15} {f'{filtered_profit:.2f}':<15} {variation_profit:>+14.1f}%"
    )

    # Win rate
    variation_wr = ((filtered_wr - win_rate) / win_rate * 100) if win_rate != 0 else 0
    print(
        f"{'Win Rate':<25} {f'{win_rate:.1f}%':<15} {f'{filtered_wr:.1f}%':<15} {variation_wr:>+14.1f}%"
    )

    # ROI real
    variation_roi = (
        ((filtered_roi - roi_real_total) / roi_real_total * 100)
        if roi_real_total != 0
        else 0
    )
    print(
        f"{'ROI Real':<25} {f'{roi_real_total:.1f}%':<15} {f'{filtered_roi:.1f}%':<15} {variation_roi:>+14.1f}%"
    )

    # Lucro por aposta
    profit_per_bet_before = total_profit / total_bets
    profit_per_bet_after = filtered_profit / filtered_bets if filtered_bets > 0 else 0
    variation_ppb = (
        ((profit_per_bet_after - profit_per_bet_before) / profit_per_bet_before * 100)
        if profit_per_bet_before != 0
        else 0
    )
    print(
        f"{'Lucro por Aposta':<25} {f'{profit_per_bet_before:.3f}':<15} {f'{profit_per_bet_after:.3f}':<15} {variation_ppb:>+14.1f}%"
    )

    # ========================================================================
    # 10. RESUMO EXECUTIVO
    # ========================================================================
    print(f"\n{'=' * 100}")
    print("💎 RESUMO EXECUTIVO")
    print("=" * 100)

    print(f"\n📊 FILTROS APLICADOS:")
    print(f"   1. {len(profitable_markets)} mercados lucrativos identificados")
    print(f"   2. ROI estimado mínimo: {min_roi_threshold}%")
    print(f"   3. {len(profitable_leagues)} ligas lucrativas")

    print(f"\n📈 IMPACTO DOS FILTROS:")
    print(
        f"   • Redução de {100 - (filtered_bets / total_bets * 100):.1f}% nas apostas"
    )
    print(f"   • Aumento de {variation_roi:.1f}% no ROI")
    print(f"   • Melhoria de {variation_ppb:.1f}% no lucro por aposta")

    print(f"\n🎯 TOP 5 MERCADOS RECOMENDADOS:")
    top_markets = relevant_markets[
        relevant_markets.index.isin(profitable_markets)
    ].head(5)
    for i, (market, row) in enumerate(top_markets.iterrows(), 1):
        print(
            f"   {i}. {market}: {row['Total_Profit']:.2f} lucro | ROI: {row['ROI_Real']:.1f}%"
        )

    # Adicionar resumo das oportunidades em ligas negativas
    if "best_markets_summary" in locals() and best_markets_summary:
        print(f"\n🔍 OPORTUNIDADES EM LIGAS NEGATIVAS:")
        print(
            f"   • {len(set([x['Liga'] for x in best_markets_summary]))} ligas negativas com mercados lucrativos"
        )
        print(f"   • {len(best_markets_summary)} mercados lucrativos identificados")
        print(
            f"   • Lucro potencial: {sum([x['Lucro'] for x in best_markets_summary]):.2f} unidades"
        )
        print(
            f"   • ROI médio: {sum([x['ROI'] for x in best_markets_summary]) / len(best_markets_summary):.1f}%"
        )

    print(f"\n{'=' * 100}")
    print("🏁 ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 100)

    return {
        "before": {
            "bets": total_bets,
            "profit": total_profit,
            "roi": roi_real_total,
            "win_rate": win_rate,
        },
        "after": {
            "bets": filtered_bets,
            "profit": filtered_profit,
            "roi": filtered_roi,
            "win_rate": filtered_wr,
        },
        "filters": {
            "markets": profitable_markets,
            "min_roi": min_roi_threshold,
            "leagues": profitable_leagues,
        },
        "negative_leagues_opportunities": best_markets_summary
        if "best_markets_summary" in locals()
        else [],
    }


# Execução
if __name__ == "__main__":
    file_path = "../bets/bets_atualizadas_por_mapa.csv"
    results = simplified_betting_analysis(file_path)
