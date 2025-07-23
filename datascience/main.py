import pandas as pd
import numpy as np
import warnings
from datetime import datetime

warnings.filterwarnings("ignore")


def betting_strategy_analysis(file_path):
    """
    Análise otimizada de apostas focada em estratégia e resultados práticos.
    Output limpo para tomada de decisão estratégica.
    """

    # ===== CARREGAMENTO E PREPARAÇÃO DOS DADOS =====
    df = pd.read_csv(file_path)

    # Tratamento de datas (conversão PT->EN para meses)
    month_translation = {
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

    for pt_month, en_month in month_translation.items():
        df["date"] = df["date"].str.replace(pt_month, en_month)

    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y %H:%M", errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

    # Categorização de mercados
    def categorize_market(bet_type, bet_line):
        lt, ll = str(bet_type).lower(), str(bet_line).lower()
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

    # Categorização de odds otimizada
    def categorize_odds(odds):
        if pd.isna(odds):
            return "N/A"
        if odds <= 1.3:
            return "muito_baixa"
        elif odds <= 1.6:
            return "baixa"
        elif odds <= 2.0:
            return "media"
        elif odds <= 2.5:
            return "media_alta"
        elif odds < 3.0:
            return "alta"
        else:
            return "muito_alta"

    df["odds_category"] = df["odds"].apply(categorize_odds)

    # Categorização de ROI estimado
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
        else:
            return "15-20%"

    df["est_roi_category"] = df["estimated_roi"].apply(categorize_roi_ranges)

    # ===== MÉTRICAS GERAIS =====
    total_profit = df["profit"].sum()
    total_bets = len(df)
    overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
    win_rate = (df["status"] == "win").mean() * 100

    print("🎯 ANÁLISE ESTRATÉGICA DE APOSTAS")
    print("=" * 60)
    print(f"📊 VISÃO GERAL:")
    print(f"   Total: {total_bets} apostas | Lucro: {total_profit:.2f} units")
    print(f"   ROI: {overall_roi:.1f}% | Win Rate: {win_rate:.1f}%")

    # ===== ANÁLISE POR RANGE DE ROI ESTIMADO =====
    roi_analysis = (
        df.groupby("est_roi_category")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    roi_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]
    roi_analysis["Real_ROI"] = (
        roi_analysis["Total_Profit"] / roi_analysis["Bets"] * 100
    ).round(2)
    roi_analysis = roi_analysis.sort_values("Total_Profit", ascending=False)

    # Selecionar ranges lucrativos
    profitable_roi_ranges = roi_analysis[
        (roi_analysis["Total_Profit"] > 1)
        & (roi_analysis["Real_ROI"] > 0)
        & (roi_analysis["Bets"] >= 5)
    ].index.tolist()

    print(f"\n📈 PERFORMANCE POR RANGE DE ROI:")
    for range_name, row in roi_analysis.iterrows():
        status = "✅" if row["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {range_name:<8} → {row['Total_Profit']:>6.1f}u | {row['Real_ROI']:>6.1f}% | {int(row['Bets']):>3} apostas"
        )

    print(f"\n💎 RANGES SELECIONADOS: {profitable_roi_ranges}")

    # ===== FILTRAR DADOS POR ROI LUCRATIVO =====
    df_filtered = df[df["est_roi_category"].isin(profitable_roi_ranges)]

    print(f"\n🔍 FILTRO APLICADO:")
    print(
        f"   Original: {len(df)} → Filtrado: {len(df_filtered)} apostas ({len(df_filtered) / len(df) * 100:.1f}%)"
    )
    print(f"   Lucro filtrado: {df_filtered['profit'].sum():.2f} units")

    # ===== ANÁLISE POR MERCADO (DADOS FILTRADOS) =====
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

    # Mercados lucrativos
    profitable_markets = market_analysis[
        (market_analysis["Total_Profit"] > 1)
        & (market_analysis["ROI"] > 5)
        & (market_analysis["Bets"] >= 3)
    ].index.tolist()

    print(f"\n🏆 PERFORMANCE POR MERCADO:")
    for market, row in market_analysis.head(10).iterrows():
        if row["Total_Profit"] >= 3:
            icon = "💎"
        elif row["Total_Profit"] > 0:
            icon = "✅"
        else:
            icon = "❌"

        print(
            f"   {icon} {market:<20} → {row['Total_Profit']:>6.1f}u | {row['ROI']:>6.1f}% | {int(row['Bets']):>3} apostas"
        )

    # ===== ANÁLISE POR ODDS =====
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

    profitable_odds = odds_analysis[
        (odds_analysis["Total_Profit"] > 0) & (odds_analysis["ROI"] > 0)
    ].index.tolist()

    print(f"\n📊 PERFORMANCE POR ODDS:")
    for odds_cat, row in odds_analysis.iterrows():
        status = "✅" if row["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {odds_cat:<12} → {row['Total_Profit']:>6.1f}u | {row['ROI']:>6.1f}% | {int(row['Bets']):>3} apostas"
        )

    # ===== ANÁLISE POR DIREÇÃO =====
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

    profitable_directions = direction_analysis[
        direction_analysis["Total_Profit"] > 1
    ].index.tolist()

    print(f"\n🔽 UNDER vs OVER:")
    for direction, row in direction_analysis.iterrows():
        status = "✅" if row["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {direction:<5} → {row['Total_Profit']:>6.1f}u | {row['ROI']:>6.1f}% | {int(row['Bets']):>3} apostas"
        )

    # ===== ESTRATÉGIA FINAL =====
    def apply_strategy(row):
        return (
            row["est_roi_category"] in profitable_roi_ranges
            and row["grouped_market"] in profitable_markets
            and row["odds_category"] in profitable_odds
            and row["direction"] in profitable_directions
        )

    df_filtered["approved"] = df_filtered.apply(apply_strategy, axis=1)
    optimized_bets = df_filtered[df_filtered["approved"]]

    opt_profit = optimized_bets["profit"].sum()
    opt_count = len(optimized_bets)
    opt_roi = (opt_profit / opt_count * 100) if opt_count > 0 else 0
    opt_wr = (optimized_bets["status"] == "win").mean() * 100 if opt_count > 0 else 0

    # ===== OUTPUT ESTRATÉGICO =====
    print(f"\n" + "=" * 60)
    print("🚀 ESTRATÉGIA OTIMIZADA")
    print("=" * 60)

    print(f"\n📋 CRITÉRIOS DA ESTRATÉGIA:")
    print(f"   📈 ROI Ranges: {profitable_roi_ranges}")
    print(
        f"   🎯 Mercados: {profitable_markets[:3] if len(profitable_markets) >= 3 else profitable_markets}"
    )
    print(f"   📊 Odds: {profitable_odds}")
    print(f"   🔽 Direções: {profitable_directions}")

    print(f"\n📊 COMPARAÇÃO DE RESULTADOS:")
    print(f"{'Cenário':<20} {'Apostas':<8} {'Lucro':<10} {'ROI':<8} {'Win Rate'}")
    print(f"{'-' * 55}")
    print(
        f"{'Todas as apostas':<20} {total_bets:<8} {total_profit:>8.1f}u {overall_roi:>6.1f}% {win_rate:>8.1f}%"
    )

    filtered_profit = df_filtered["profit"].sum()
    filtered_roi = (
        (filtered_profit / len(df_filtered) * 100) if len(df_filtered) > 0 else 0
    )
    filtered_wr = (
        (df_filtered["status"] == "win").mean() * 100 if len(df_filtered) > 0 else 0
    )

    print(
        f"{'Filtro ROI':<20} {len(df_filtered):<8} {filtered_profit:>8.1f}u {filtered_roi:>6.1f}% {filtered_wr:>8.1f}%"
    )
    print(
        f"{'Estratégia Final':<20} {opt_count:<8} {opt_profit:>8.1f}u {opt_roi:>6.1f}% {opt_wr:>8.1f}%"
    )

    # ===== RECOMENDAÇÕES =====
    print(f"\n💡 PRÓXIMOS PASSOS:")

    if opt_count > 0:
        roi_improvement = opt_roi - overall_roi
        efficiency = (
            (opt_profit / opt_count) / (total_profit / total_bets)
            if total_bets > 0
            else 0
        )

        print(f"   ✅ Estratégia validada: +{roi_improvement:.1f}% ROI")
        print(f"   📈 Eficiência: {efficiency:.1f}x melhor por aposta")
        print(f"   🎯 Foco nos {len(profitable_markets)} mercados principais")
        print(f"   📊 Manter disciplina nos ranges ROI selecionados")

        # Projeção mensal
        if opt_count > 0:
            monthly_projection = (opt_profit / opt_count) * 30
            print(f"   💰 Projeção mensal (30 apostas): {monthly_projection:.1f} units")
    else:
        print(f"   ⚠️  Critérios muito restritivos - revisar filtros")
        print(f"   🔄 Considerar relaxar alguns critérios")

    # Análise de mercados específicos
    if "INHIBITORS" in market_analysis.index:
        inhibitors_data = (
            market_analysis.loc["OVER - INHIBITORS"]
            if "OVER - INHIBITORS" in market_analysis.index
            else None
        )
        if inhibitors_data is not None and inhibitors_data["Total_Profit"] > 0:
            print(
                f"   🎯 INHIBITORS performando bem: {inhibitors_data['Total_Profit']:.1f}u ({inhibitors_data['ROI']:.1f}% ROI)"
            )

    print(f"\n" + "=" * 60)
    print("✅ ANÁLISE CONCLUÍDA - ESTRATÉGIA PRONTA PARA EXECUÇÃO")
    print("=" * 60)

    return {
        "strategy_criteria": {
            "roi_ranges": profitable_roi_ranges,
            "markets": profitable_markets,
            "odds": profitable_odds,
            "directions": profitable_directions,
        },
        "performance": {
            "all_bets": {
                "count": total_bets,
                "profit": total_profit,
                "roi": overall_roi,
            },
            "filtered": {
                "count": len(df_filtered),
                "profit": filtered_profit,
                "roi": filtered_roi,
            },
            "optimized": {"count": opt_count, "profit": opt_profit, "roi": opt_roi},
        },
        "data": {"original": df, "filtered": df_filtered, "optimized": optimized_bets},
    }


def get_current_month_data(file_path):
    """Filtra dados apenas do mês atual"""
    df = pd.read_csv(file_path)

    # Conversão de meses PT->EN
    month_translation = {
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

    for pt_month, en_month in month_translation.items():
        df["date"] = df["date"].str.replace(pt_month, en_month)

    df["date"] = pd.to_datetime(df["date"], format="%d %b %Y %H:%M")

    current_month = datetime.now().month
    current_year = datetime.now().year

    monthly_data = df[
        (df["date"].dt.month == current_month) & (df["date"].dt.year == current_year)
    ]

    print(f"📅 Análise do mês: {datetime.now().strftime('%B %Y')}")
    print(f"📊 Registros encontrados: {len(monthly_data)}")

    return monthly_data


# ===== EXECUÇÃO =====
if __name__ == "__main__":
    file_path = "../bets/bets_atualizadas_por_mapa.csv"

    # Opção 1: Análise do mês atual
    monthly_data = get_current_month_data(file_path)
    monthly_data.to_csv("temp_monthly.csv", index=False)
    results = betting_strategy_analysis("temp_monthly.csv")

    # Opção 2: Análise de todos os dados
    # results = betting_strategy_analysis(file_path)

    # Limpeza
    import os

    if os.path.exists("temp_monthly.csv"):
        os.remove("temp_monthly.csv")
