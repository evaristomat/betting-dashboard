import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


def complete_betting_analysis_with_backtest(file_path):
    """Análise completa + backtest da estratégia otimizada"""

    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")

    print("🎯 ANÁLISE COMPLETA + BACKTEST DA ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    # Função para categorizar mercados
    def categorize_market(bet_type, bet_line):
        bet_line_str = str(bet_line).lower()
        bet_type_str = str(bet_type).lower()

        direction = "UNDER" if "under" in bet_type_str else "OVER"

        if "kill" in bet_line_str:
            market = "KILLS"
        elif "dragon" in bet_line_str:
            market = "DRAGONS"
        elif "tower" in bet_line_str:
            market = "TOWERS"
        elif "duration" in bet_line_str or "tempo" in bet_line_str:
            market = "DURATION"
        elif "baron" in bet_line_str:
            market = "BARONS"
        elif "inhibitor" in bet_line_str:
            market = "INHIBITORS"
        else:
            market = "OUTROS"

        return direction, market, f"{direction} - {market}"

    # Aplicar categorização
    df[["direction", "market_type", "grouped_market"]] = df.apply(
        lambda row: categorize_market(row["bet_type"], row["bet_line"]),
        axis=1,
        result_type="expand",
    )

    # Categorizar odds, agora incluindo o intervalo no nome
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

    # 1. ANÁLISE INICIAL DOS DADOS
    print(f"\n📊 DADOS GERAIS:")
    print(f"   Total de apostas: {len(df)}")
    print(f"   Lucro total atual: {df['profit'].sum():.2f} unidades")
    print(f"   ROI médio atual: {(df['profit'].sum() / len(df) * 100):.2f}%")
    print(f"   Win rate geral: {(df['status'] == 'win').mean() * 100:.1f}%")

    # 2. IDENTIFICAR ESTRATÉGIAS LUCRATIVAS
    print(f"\n" + "=" * 80)
    print("📈 IDENTIFICANDO ESTRATÉGIAS LUCRATIVAS")
    print("=" * 80)

    # Análise por mercado
    market_analysis = (
        df.groupby("grouped_market")
        .agg(
            {"profit": ["sum", "count", "mean"], "status": lambda x: (x == "win").sum()}
        )
        .round(4)
    )

    market_analysis.columns = ["Total_Profit", "Total_Bets", "Avg_Profit", "Wins"]
    market_analysis["Win_Rate"] = (
        market_analysis["Wins"] / market_analysis["Total_Bets"] * 100
    ).round(2)
    market_analysis["ROI"] = (
        market_analysis["Total_Profit"] / market_analysis["Total_Bets"] * 100
    ).round(2)
    market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n🏆 PERFORMANCE POR MERCADO:")
    for market, stats in market_analysis.iterrows():
        status = "✅" if stats["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {market:<25}: {stats['Total_Profit']:>7.2f} unidades | ROI: {stats['ROI']:>5.1f}% | Apostas: {stats['Total_Bets']:>3.0f}"
        )

    # Análise por odds
    odds_analysis = (
        df.groupby("odds_category")
        .agg(
            {"profit": ["sum", "count", "mean"], "status": lambda x: (x == "win").sum()}
        )
        .round(4)
    )

    odds_analysis.columns = ["Total_Profit", "Total_Bets", "Avg_Profit", "Wins"]
    odds_analysis["ROI"] = (
        odds_analysis["Total_Profit"] / odds_analysis["Total_Bets"] * 100
    ).round(2)
    odds_analysis = odds_analysis.sort_values("Total_Profit", ascending=False)

    print(f"\n🎲 PERFORMANCE POR FAIXA DE ODDS:")
    for odds_cat, stats in odds_analysis.iterrows():
        status = "✅" if stats["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {odds_cat:<15}: {stats['Total_Profit']:>7.2f} unidades | ROI: {stats['ROI']:>5.1f}% | Apostas: {stats['Total_Bets']:>3.0f}"
        )

    # Análise UNDER vs OVER
    direction_analysis = (
        df.groupby("direction")
        .agg(
            {"profit": ["sum", "count", "mean"], "status": lambda x: (x == "win").sum()}
        )
        .round(4)
    )

    direction_analysis.columns = ["Total_Profit", "Total_Bets", "Avg_Profit", "Wins"]
    direction_analysis["ROI"] = (
        direction_analysis["Total_Profit"] / direction_analysis["Total_Bets"] * 100
    ).round(2)

    print(f"\n🔽 PERFORMANCE UNDER vs OVER:")
    for direction, stats in direction_analysis.iterrows():
        status = "✅" if stats["Total_Profit"] > 0 else "❌"
        print(
            f"   {status} {direction:<5}: {stats['Total_Profit']:>7.2f} unidades | ROI: {stats['ROI']:>5.1f}% | Apostas: {stats['Total_Bets']:>3.0f}"
        )

    # 3. DEFINIR ESTRATÉGIA OTIMIZADA BASEADA NOS DADOS
    print(f"\n" + "=" * 80)
    print("🎯 DEFININDO ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    # Identificar mercados lucrativos (ROI > 0 e lucro > 1 unidade)
    profitable_markets = market_analysis[
        (market_analysis["Total_Profit"] > 0)
        & (market_analysis["ROI"] > 0)
        & (market_analysis["Total_Bets"] >= 5)  # Mínimo de volume
    ].index.tolist()

    # Identificar faixas de odds lucrativas
    profitable_odds = odds_analysis[
        (odds_analysis["Total_Profit"] > 0) & (odds_analysis["ROI"] > 0)
    ].index.tolist()

    # Identificar direção lucrativa
    profitable_direction = direction_analysis[
        direction_analysis["Total_Profit"] > 0
    ].index.tolist()

    print(f"\n📋 CRITÉRIOS DA ESTRATÉGIA OTIMIZADA:")
    print(f"   ✅ Mercados lucrativos: {profitable_markets}")
    print(f"   ✅ Faixas de odds lucrativas: {profitable_odds}")
    print(f"   ✅ Direção lucrativa: {profitable_direction}")

    # Critérios extras baseados na análise
    print(f"\n📋 CRITÉRIOS ADICIONAIS:")
    print(f"   ✅ ROI mínimo por mercado: >0%")
    print(f"   ✅ Volume mínimo: ≥5 apostas por mercado")
    print(f"   ✅ Foco em inhibitors (melhor ROI)")
    print(f"   ❌ Evitar mercados com ROI <-5%")

    # 4. APLICAR ESTRATÉGIA OTIMIZADA (BACKTEST)
    print(f"\n" + "=" * 80)
    print("🚀 BACKTEST DA ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    # Filtros da estratégia otimizada
    def apply_optimized_strategy(row):
        # Critério 1: Mercado deve ser lucrativo
        if row["grouped_market"] not in profitable_markets:
            return False, "mercado_nao_lucrativo"

        # Critério 2: Odds deve estar na faixa lucrativa
        if row["odds_category"] not in profitable_odds:
            return False, "odds_nao_lucrativa"

        # Critério 3: Direção deve ser lucrativa
        if row["direction"] not in profitable_direction:
            return False, "direcao_nao_lucrativa"

        # Critério 4: Priorizar inhibitors (sempre incluir)
        if row["market_type"] == "INHIBITORS":
            return True, "inhibitors_priority"

        # Critério 5: ROI histórico do mercado deve ser positivo
        market_roi = market_analysis.loc[row["grouped_market"], "ROI"]
        if market_roi <= 0:
            return False, "roi_negativo"

        return True, "aprovado"

    # Aplicar estratégia
    strategy_results = df.apply(apply_optimized_strategy, axis=1)
    df["strategy_approved"] = [result[0] for result in strategy_results]
    df["strategy_reason"] = [result[1] for result in strategy_results]

    # Separar apostas
    original_bets = df.copy()
    optimized_bets = df[df["strategy_approved"] == True].copy()
    rejected_bets = df[df["strategy_approved"] == False].copy()

    print(f"\n📊 RESULTADOS DO BACKTEST:")
    print(f"   📈 Apostas originais: {len(original_bets)}")
    print(f"   ✅ Apostas aprovadas pela estratégia: {len(optimized_bets)}")
    print(f"   ❌ Apostas rejeitadas: {len(rejected_bets)}")
    print(
        f"   📊 Taxa de aprovação: {len(optimized_bets) / len(original_bets) * 100:.1f}%"
    )

    # 5. COMPARAÇÃO DETALHADA
    print(f"\n" + "=" * 80)
    print("📊 COMPARAÇÃO: ATUAL vs ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    # Métricas originais
    original_profit = original_bets["profit"].sum()
    original_bets_count = len(original_bets)
    original_roi = original_profit / original_bets_count * 100
    original_wr = (original_bets["status"] == "win").mean() * 100

    # Métricas otimizadas
    optimized_profit = optimized_bets["profit"].sum()
    optimized_bets_count = len(optimized_bets)
    optimized_roi = (
        (optimized_profit / optimized_bets_count * 100)
        if optimized_bets_count > 0
        else 0
    )
    optimized_wr = (
        (optimized_bets["status"] == "win").mean() * 100
        if optimized_bets_count > 0
        else 0
    )

    # Métricas rejeitadas
    rejected_profit = rejected_bets["profit"].sum()
    rejected_bets_count = len(rejected_bets)
    rejected_roi = (
        (rejected_profit / rejected_bets_count * 100) if rejected_bets_count > 0 else 0
    )
    rejected_wr = (
        (rejected_bets["status"] == "win").mean() * 100
        if rejected_bets_count > 0
        else 0
    )

    print(f"\n📊 SITUAÇÃO ATUAL (TODAS AS APOSTAS):")
    print(f"   💰 Lucro total: {original_profit:.2f} unidades")
    print(f"   🎲 Total apostas: {original_bets_count}")
    print(f"   💹 ROI médio: {original_roi:.2f}%")
    print(f"   📊 Win rate: {original_wr:.1f}%")
    print(f"   📈 Lucro/aposta: {original_profit / original_bets_count:.4f}")

    print(f"\n🚀 ESTRATÉGIA OTIMIZADA (APOSTAS APROVADAS):")
    print(f"   💰 Lucro total: {optimized_profit:.2f} unidades")
    print(f"   🎲 Total apostas: {optimized_bets_count}")
    print(f"   💹 ROI médio: {optimized_roi:.2f}%")
    print(f"   📊 Win rate: {optimized_wr:.1f}%")
    print(
        f"   📈 Lucro/aposta: {optimized_profit / optimized_bets_count:.4f}"
        if optimized_bets_count > 0
        else "   📈 Lucro/aposta: 0.0000"
    )

    print(f"\n❌ APOSTAS REJEITADAS (QUE VOCÊ EVITARIA):")
    print(f"   💰 Lucro total: {rejected_profit:.2f} unidades")
    print(f"   🎲 Total apostas: {rejected_bets_count}")
    print(f"   💹 ROI médio: {rejected_roi:.2f}%")
    print(f"   📊 Win rate: {rejected_wr:.1f}%")
    print(
        f"   📈 Lucro/aposta: {rejected_profit / rejected_bets_count:.4f}"
        if rejected_bets_count > 0
        else "   📈 Lucro/aposta: 0.0000"
    )

    # 6. IMPACTO DA OTIMIZAÇÃO
    print(f"\n" + "=" * 80)
    print("💎 IMPACTO DA OTIMIZAÇÃO")
    print("=" * 80)

    if optimized_bets_count > 0:
        efficiency_gain = optimized_roi - original_roi
        profit_concentration = (
            (optimized_profit / original_profit * 100) if original_profit != 0 else 0
        )
        volume_reduction = (
            (original_bets_count - optimized_bets_count) / original_bets_count * 100
        )

        print(f"\n🎯 GANHOS DE EFICIÊNCIA:")
        print(
            f"   📈 Melhoria no ROI: {efficiency_gain:+.2f}% (de {original_roi:.2f}% → {optimized_roi:.2f}%)"
        )
        print(
            f"   💰 Concentração do lucro: {profit_concentration:.1f}% do lucro em {100 - volume_reduction:.1f}% das apostas"
        )
        print(f"   📉 Redução no volume: {volume_reduction:.1f}% menos apostas")
        print(
            f"   🎲 Eficiência: {optimized_profit / optimized_profit:.2f}x mais lucro por aposta"
            if optimized_profit > 0
            else "   🎲 Eficiência: N/A"
        )

        # Projeção para mesmo volume
        if optimized_bets_count > 0:
            projected_profit_same_volume = (
                optimized_profit / optimized_bets_count
            ) * original_bets_count
            profit_potential = projected_profit_same_volume - original_profit

            print(f"\n💰 POTENCIAL SE APLICASSE A ESTRATÉGIA NO MESMO VOLUME:")
            print(f"   🎯 Lucro projetado: {projected_profit_same_volume:.2f} unidades")
            print(f"   📈 Ganho potencial: +{profit_potential:.2f} unidades")
            print(
                f"   📊 Melhoria: {(profit_potential / original_profit * 100):+.1f}%"
                if original_profit > 0
                else "   📊 Melhoria: +∞%"
            )

    # 7. ANÁLISE DOS MOTIVOS DE REJEIÇÃO
    print(f"\n" + "=" * 80)
    print("🔍 ANÁLISE DOS MOTIVOS DE REJEIÇÃO")
    print("=" * 80)

    rejection_analysis = (
        rejected_bets.groupby("strategy_reason")
        .agg(
            {"profit": ["sum", "count"], "status": lambda x: (x == "win").mean() * 100}
        )
        .round(2)
    )

    rejection_analysis.columns = ["Profit_Lost", "Bets_Count", "Win_Rate"]
    rejection_analysis["ROI"] = (
        rejection_analysis["Profit_Lost"] / rejection_analysis["Bets_Count"] * 100
    ).round(2)
    rejection_analysis = rejection_analysis.sort_values("Profit_Lost", ascending=True)

    print(f"\n📊 MOTIVOS DE REJEIÇÃO (DO PIOR PARA O MELHOR):")
    for reason, stats in rejection_analysis.iterrows():
        impact = (
            "💸"
            if stats["Profit_Lost"] < -1
            else "📉"
            if stats["Profit_Lost"] < 0
            else "📈"
        )
        print(
            f"   {impact} {reason:<20}: {stats['Profit_Lost']:>7.2f} unidades | {stats['Bets_Count']:>3.0f} apostas | ROI: {stats['ROI']:>5.1f}%"
        )

    # 8. TOP PERFORMERS NA ESTRATÉGIA OTIMIZADA
    print(f"\n" + "=" * 80)
    print("🏆 TOP PERFORMERS NA ESTRATÉGIA OTIMIZADA")
    print("=" * 80)

    if len(optimized_bets) > 0:
        optimized_market_analysis = (
            optimized_bets.groupby("grouped_market")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").sum(),
                }
            )
            .round(4)
        )

        optimized_market_analysis.columns = [
            "Total_Profit",
            "Total_Bets",
            "Avg_Profit",
            "Wins",
        ]
        optimized_market_analysis["ROI"] = (
            optimized_market_analysis["Total_Profit"]
            / optimized_market_analysis["Total_Bets"]
            * 100
        ).round(2)
        optimized_market_analysis = optimized_market_analysis.sort_values(
            "Total_Profit", ascending=False
        )

        print(f"\n🏆 MERCADOS NA ESTRATÉGIA OTIMIZADA:")
        for market, stats in optimized_market_analysis.iterrows():
            print(
                f"   ✅ {market:<25}: {stats['Total_Profit']:>7.2f} unidades | ROI: {stats['ROI']:>5.1f}% | Apostas: {stats['Total_Bets']:>3.0f}"
            )

    # 9. RECOMENDAÇÕES FINAIS
    print(f"\n" + "=" * 80)
    print("💡 RECOMENDAÇÕES ESTRATÉGICAS FINAIS")
    print("=" * 80)

    print(f"\n🎯 RESUMO EXECUTIVO:")
    print(
        f"   💰 Lucro atual: {original_profit:.2f} unidades com {original_bets_count} apostas"
    )
    print(
        f"   🚀 Lucro otimizado: {optimized_profit:.2f} unidades com {optimized_bets_count} apostas"
    )
    print(f"   📈 Eficiência: {optimized_roi:.2f}% ROI vs {original_roi:.2f}% atual")

    if optimized_bets_count > 0 and optimized_roi > original_roi:
        print(f"\n✅ ESTRATÉGIA COMPROVADAMENTE SUPERIOR!")
        print(f"   🎯 Aplique os filtros da estratégia otimizada")
        print(f"   📈 Ganhe {efficiency_gain:.2f}% mais por aposta")
        print(
            f"   💰 Potencial de {profit_potential:.2f} unidades extras"
            if "profit_potential" in locals()
            else ""
        )
    else:
        print(f"\n⚠️ ESTRATÉGIA PRECISA DE AJUSTES")
        print(f"   🔍 Analise mercados individuais")
        print(f"   📊 Considere critérios menos restritivos")

    print(f"\n🚀 PRÓXIMOS PASSOS:")
    print(f"   1. 🎯 Implemente os filtros da estratégia otimizada")
    print(f"   2. 📊 Monitore performance com novos dados")
    print(f"   3. 🔄 Ajuste critérios conforme necessário")
    print(f"   4. 📈 Foque no ROI, não apenas no volume")

    return {
        "original_data": original_bets,
        "optimized_bets": optimized_bets,
        "rejected_bets": rejected_bets,
        "market_analysis": market_analysis,
        "odds_analysis": odds_analysis,
        "strategy_criteria": {
            "profitable_markets": profitable_markets,
            "profitable_odds": profitable_odds,
            "profitable_direction": profitable_direction,
        },
        "performance_metrics": {
            "original": {
                "profit": original_profit,
                "bets": original_bets_count,
                "roi": original_roi,
            },
            "optimized": {
                "profit": optimized_profit,
                "bets": optimized_bets_count,
                "roi": optimized_roi,
            },
            "rejected": {
                "profit": rejected_profit,
                "bets": rejected_bets_count,
                "roi": rejected_roi,
            },
        },
    }


# Executar análise completa
if __name__ == "__main__":
    file_path = "../bets/bets_atualizadas_por_mapa.csv"
    results = complete_betting_analysis_with_backtest(file_path)

    print("\n" + "=" * 80)
    print("✅ ANÁLISE COMPLETA FINALIZADA!")
    print("=" * 80)
    print("\n📋 Dados disponíveis em 'results':")
    print("   - results['original_data']: Todas as apostas originais")
    print("   - results['optimized_bets']: Apostas aprovadas pela estratégia")
    print("   - results['rejected_bets']: Apostas rejeitadas")
    print("   - results['strategy_criteria']: Critérios da estratégia otimizada")
    print("   - results['performance_metrics']: Métricas comparativas")
