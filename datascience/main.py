import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")


class RefinedBettingAnalyzer:
    """Analisador refinado de apostas com foco em ROI otimizado"""

    def __init__(self):
        self.month_translation = {
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

    def categorize_roi_ranges_plus(self, roi: float) -> str:
        """NOVA: Ranges de ROI no formato 10+, 15+, 20+, etc."""
        if pd.isna(roi):
            return "N/A"
        elif roi >= 30:
            return "30+"
        elif roi >= 25:
            return "25+"
        elif roi >= 20:
            return "20+"
        elif roi >= 15:
            return "15+"
        elif roi >= 10:
            return "10+"
        else:
            return "<10"

    def categorize_odds_filtered(self, odds: float) -> str:
        """NOVA: Categorização de odds EXCLUINDO muito_baixa e muito_alta"""
        if pd.isna(odds):
            return "N/A"
        elif odds <= 1.6:
            return "baixa"
        elif odds <= 2.0:
            return "media"
        elif odds <= 2.5:
            return "media_alta"
        elif odds <= 3.0:
            return "alta"
        else:
            return "EXCLUIR"  # Marca para remoção

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categorização de mercados"""
        lt, ll = str(bet_type).lower(), str(bet_line).lower()

        direction = "UNDER" if "under" in lt else "OVER"

        if "kill" in ll:
            if "first" in ll or "primeiro" in ll:
                market = "FIRST_KILL"
            elif "team" in ll or "time" in ll or "equipe" in ll:
                market = "TEAM_KILLS"
            else:
                market = "KILLS"
        elif "dragon" in ll:
            if "first" in ll or "primeiro" in ll:
                market = "FIRST_DRAGON"
            elif "elder" in ll or "anciao" in ll or "ancião" in ll:
                market = "ELDER_DRAGON"
            else:
                market = "DRAGONS"
        elif "tower" in ll or "torre" in ll:
            if "first" in ll or "primeiro" in ll or "primeira" in ll:
                market = "FIRST_TOWER"
            else:
                market = "TOWERS"
        elif "duration" in ll or "tempo" in ll or "duração" in ll or "duracao" in ll:
            market = "DURATION"
        elif "baron" in ll:
            if "first" in ll or "primeiro" in ll:
                market = "FIRST_BARON"
            else:
                market = "BARONS"
        elif "inhibitor" in ll or "inibidor" in ll:
            market = "INHIBITORS"
        elif "herald" in ll or "arauto" in ll:
            market = "HERALD"
        elif "blood" in ll or "sangue" in ll:
            market = "FIRST_BLOOD"
        elif "gold" in ll or "ouro" in ll:
            market = "GOLD"
        elif "creep" in ll or "cs" in ll:
            market = "CREEPS"
        elif "assist" in ll or "assistencia" in ll:
            market = "ASSISTS"
        elif "death" in ll or "morte" in ll:
            market = "DEATHS"
        else:
            market = "OUTROS"

        return direction, market, f"{direction} - {market}"

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """NOVA: Carrega dados com filtros de odds aplicados"""
        df = pd.read_csv(file_path)

        # Conversão de datas
        for pt_month, en_month in self.month_translation.items():
            df["date"] = df["date"].str.replace(pt_month, en_month)

        df["date"] = pd.to_datetime(
            df["date"], format="%d %b %Y %H:%M", errors="coerce"
        )
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

        # Categorizações
        df[["direction", "market_type", "grouped_market"]] = df.apply(
            lambda r: self.categorize_market(r["bet_type"], r["bet_line"]),
            axis=1,
            result_type="expand",
        )

        df["odds_category"] = df["odds"].apply(self.categorize_odds_filtered)
        df["est_roi_category"] = df["estimated_roi"].apply(
            self.categorize_roi_ranges_plus
        )

        # NOVO: Filtrar odds muito_baixa e muito_alta
        df_filtered = df[df["odds_category"] != "EXCLUIR"].copy()

        print(f"📊 DADOS APÓS FILTRAGEM DE ODDS:")
        print(f"   Dados originais: {len(df)} apostas")
        print(f"   Dados filtrados: {len(df_filtered)} apostas")
        print(
            f"   Removidas: {len(df) - len(df_filtered)} apostas (odds muito baixas/altas)"
        )

        return df_filtered

    def analyze_roi_ranges_comprehensive(self, df: pd.DataFrame, title: str) -> Dict:
        """NOVA: Análise abrangente das faixas de ROI no formato 10+, 15+, etc."""
        print(f"\n" + "=" * 80)
        print(f"🔍 ANÁLISE DE ROI RANGES - {title.upper()}")
        print("=" * 80)

        roi_analysis = (
            df.groupby("est_roi_category")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": "mean",
                }
            )
            .round(2)
        )

        roi_analysis.columns = [
            "Total_Profit",
            "Bets",
            "Avg_Profit",
            "Win_Rate",
            "Avg_Est_ROI",
        ]
        roi_analysis["Real_ROI"] = (
            roi_analysis["Total_Profit"] / roi_analysis["Bets"] * 100
        ).round(2)
        roi_analysis["Efficiency"] = (
            roi_analysis["Real_ROI"] / roi_analysis["Avg_Est_ROI"]
        ).round(3)

        # Ordenar por ROI range
        roi_order = ["<10", "10+", "15+", "20+", "25+", "30+"]
        roi_analysis = roi_analysis.reindex(
            [cat for cat in roi_order if cat in roi_analysis.index]
        )

        print(f"\n📊 PERFORMANCE POR FAIXA DE ROI:")
        print(
            f"{'Faixa':<8} {'Lucro':<10} {'Apostas':<8} {'Win%':<6} {'ROI Real':<9} {'ROI Est.':<8} {'Eficiência'}"
        )
        print("-" * 75)

        best_roi_range = None
        best_efficiency = -999

        for roi_range, row in roi_analysis.iterrows():
            if row["Bets"] >= 20:  # Mínimo de apostas para consideração
                status = (
                    "🏆"
                    if row["Real_ROI"] >= 10
                    else "✅"
                    if row["Real_ROI"] > 0
                    else "❌"
                )
                efficiency_str = (
                    f"{row['Efficiency']:.2f}x"
                    if not pd.isna(row["Efficiency"])
                    else "N/A"
                )

                print(
                    f"{status} {roi_range:<6} {row['Total_Profit']:>8.1f}u "
                    f"{int(row['Bets']):>6} {row['Win_Rate']:>5.1f}% "
                    f"{row['Real_ROI']:>7.1f}% {row['Avg_Est_ROI']:>7.1f}% "
                    f"{efficiency_str:>9}"
                )

                # Identificar melhor faixa (ROI real > 5% e eficiência > 0.1)
                if (
                    row["Real_ROI"] > 5
                    and row["Efficiency"] > 0.1
                    and row["Efficiency"] > best_efficiency
                ):
                    best_efficiency = row["Efficiency"]
                    best_roi_range = roi_range

        print(f"\n🎯 MELHOR FAIXA DE ROI IDENTIFICADA: {best_roi_range}")
        if best_roi_range:
            best_data = roi_analysis.loc[best_roi_range]
            print(f"   📈 ROI Real: {best_data['Real_ROI']:.1f}%")
            print(f"   🎯 Eficiência: {best_data['Efficiency']:.2f}x")
            print(f"   📊 Apostas: {int(best_data['Bets'])}")
            print(f"   💰 Lucro: {best_data['Total_Profit']:.1f}u")

        return {
            "analysis": roi_analysis,
            "best_roi_range": best_roi_range,
            "best_efficiency": best_efficiency,
        }

    def analyze_markets_with_roi_filter(
        self, df: pd.DataFrame, roi_filter: str, title: str
    ) -> Dict:
        """NOVA: Análise de mercados usando filtro de ROI otimizado"""
        print(f"\n" + "=" * 80)
        print(f"🎯 ANÁLISE DE MERCADOS COM FILTRO ROI ≥ {roi_filter} - {title.upper()}")
        print("=" * 80)

        # Extrair número do filtro ROI
        roi_threshold = int(roi_filter.replace("+", ""))
        filtered_df = df[df["estimated_roi"] >= roi_threshold].copy()

        if len(filtered_df) == 0:
            print(f"❌ Nenhuma aposta encontrada com ROI ≥ {roi_threshold}%")
            return None

        print(f"📊 DADOS COM FILTRO ROI ≥ {roi_threshold}%:")
        print(f"   Total de apostas: {len(filtered_df)}")
        print(f"   Lucro total: {filtered_df['profit'].sum():.1f}u")
        print(
            f"   ROI real: {(filtered_df['profit'].sum() / len(filtered_df) * 100):.1f}%"
        )
        print(f"   Win Rate: {(filtered_df['status'] == 'win').mean() * 100:.1f}%")

        # Análise por mercado
        market_analysis = (
            filtered_df.groupby("grouped_market")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": "mean",
                    "odds": "mean",
                }
            )
            .round(2)
        )

        market_analysis.columns = [
            "Total_Profit",
            "Bets",
            "Avg_Profit",
            "Win_Rate",
            "Avg_Est_ROI",
            "Avg_Odds",
        ]
        market_analysis["Real_ROI"] = (
            market_analysis["Total_Profit"] / market_analysis["Bets"] * 100
        ).round(2)

        # Filtrar mercados com pelo menos 5 apostas
        market_analysis = market_analysis[market_analysis["Bets"] >= 5]
        market_analysis = market_analysis.sort_values("Total_Profit", ascending=False)

        print(f"\n🏆 TOP MERCADOS (ROI ≥ {roi_threshold}%, mín. 5 apostas):")
        print(
            f"{'Mercado':<25} {'Lucro':<8} {'Apostas':<8} {'Win%':<6} {'ROI Real':<8} {'ROI Est.':<8} {'Odds'}"
        )
        print("-" * 85)

        profitable_markets = []
        for market, row in market_analysis.head(15).iterrows():
            status = (
                "🏆"
                if row["Total_Profit"] >= 5
                else "✅"
                if row["Total_Profit"] > 0
                else "❌"
            )
            print(
                f"{status} {market:<23} {row['Total_Profit']:>6.1f}u "
                f"{int(row['Bets']):>6} {row['Win_Rate']:>5.1f}% "
                f"{row['Real_ROI']:>6.1f}% {row['Avg_Est_ROI']:>6.1f}% "
                f"{row['Avg_Odds']:>5.2f}"
            )

            if row["Total_Profit"] > 0:
                profitable_markets.append(market)

        # Análise por direção
        direction_analysis = filtered_df.groupby("direction").agg(
            {
                "profit": ["sum", "count"],
                "status": lambda x: (x == "win").mean() * 100,
            }
        )
        direction_analysis.columns = ["Total_Profit", "Bets", "Win_Rate"]

        print(f"\n🔽 ANÁLISE POR DIREÇÃO (ROI ≥ {roi_threshold}%):")
        for direction, row in direction_analysis.iterrows():
            status = "✅" if row["Total_Profit"] > 0 else "❌"
            roi = (row["Total_Profit"] / row["Bets"] * 100) if row["Bets"] > 0 else 0
            print(
                f"   {status} {direction}: {row['Total_Profit']:+.1f}u | "
                f"{int(row['Bets'])} apostas | {row['Win_Rate']:.1f}% win | {roi:.1f}% ROI"
            )

        # Análise por odds filtradas
        odds_analysis = (
            filtered_df.groupby("odds_category")
            .agg(
                {
                    "profit": ["sum", "count"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "odds": "mean",
                }
            )
            .round(2)
        )
        odds_analysis.columns = ["Total_Profit", "Bets", "Win_Rate", "Avg_Odds"]
        odds_analysis["Real_ROI"] = (
            odds_analysis["Total_Profit"] / odds_analysis["Bets"] * 100
        ).round(2)

        print(f"\n📊 ANÁLISE POR ODDS (ROI ≥ {roi_threshold}%):")
        for odds_cat, row in odds_analysis.iterrows():
            if row["Bets"] > 0:
                status = "✅" if row["Total_Profit"] > 0 else "❌"
                print(
                    f"   {status} {odds_cat:<12} → {row['Total_Profit']:>6.1f}u | "
                    f"ROI: {row['Real_ROI']:>6.1f}% | {int(row['Bets'])} apostas | "
                    f"Win: {row['Win_Rate']:>5.1f}%"
                )

        return {
            "filtered_data": filtered_df,
            "market_analysis": market_analysis,
            "direction_analysis": direction_analysis,
            "odds_analysis": odds_analysis,
            "profitable_markets": profitable_markets,
            "total_profit": filtered_df["profit"].sum(),
            "total_bets": len(filtered_df),
            "roi_threshold": roi_threshold,
        }

    def print_strategic_recommendations(
        self,
        all_data_analysis: Dict,
        recent_data_analysis: Dict,
        best_roi_all: str,
        best_roi_recent: str,
    ):
        """NOVA: Recomendações estratégicas baseadas na análise refinada"""
        print("\n" + "=" * 80)
        print("🚀 RECOMENDAÇÕES ESTRATÉGICAS REFINADAS")
        print("=" * 80)

        print(f"\n📊 RESUMO EXECUTIVO:")
        print(f"   🎯 Melhor ROI (todos os dados): {best_roi_all}")
        print(f"   🎯 Melhor ROI (últimos 200): {best_roi_recent}")

        if all_data_analysis and recent_data_analysis:
            all_profit = all_data_analysis["total_profit"]
            recent_profit = recent_data_analysis["total_profit"]

            print(f"   💰 Lucro total (filtrado): {all_profit:.1f}u")
            print(f"   💰 Lucro recente (últimos 200): {recent_profit:.1f}u")

            # Mercados consistentes
            all_markets = set(all_data_analysis["profitable_markets"][:5])
            recent_markets = set(recent_data_analysis["profitable_markets"][:5])
            consistent_markets = all_markets.intersection(recent_markets)

            print(f"\n🎯 MERCADOS CONSISTENTEMENTE LUCRATIVOS:")
            for market in consistent_markets:
                all_roi = all_data_analysis["market_analysis"].loc[market, "Real_ROI"]
                recent_roi = recent_data_analysis["market_analysis"].loc[
                    market, "Real_ROI"
                ]
                print(
                    f"   🏆 {market}: {all_roi:.1f}% (total) | {recent_roi:.1f}% (recente)"
                )

            # Direções recomendadas
            print(f"\n🔽 DIREÇÕES RECOMENDADAS:")

            all_direction = all_data_analysis["direction_analysis"]
            recent_direction = recent_data_analysis["direction_analysis"]

            for direction in ["UNDER", "OVER"]:
                if (
                    direction in all_direction.index
                    and direction in recent_direction.index
                ):
                    all_profit_dir = all_direction.loc[direction, "Total_Profit"]
                    recent_profit_dir = recent_direction.loc[direction, "Total_Profit"]

                    status = (
                        "✅"
                        if all_profit_dir > 0 and recent_profit_dir > 0
                        else "⚠️"
                        if all_profit_dir > 0 or recent_profit_dir > 0
                        else "❌"
                    )
                    print(
                        f"   {status} {direction}: {all_profit_dir:+.1f}u (total) | {recent_profit_dir:+.1f}u (recente)"
                    )

            # Odds recomendadas
            print(f"\n📊 ODDS RECOMENDADAS:")
            all_odds = all_data_analysis["odds_analysis"]
            recent_odds = recent_data_analysis["odds_analysis"]

            for odds_cat in ["baixa", "media", "media_alta", "alta"]:
                if odds_cat in all_odds.index and odds_cat in recent_odds.index:
                    all_profit_odds = all_odds.loc[odds_cat, "Total_Profit"]
                    recent_profit_odds = recent_odds.loc[odds_cat, "Total_Profit"]

                    if all_profit_odds > 0 and recent_profit_odds > 0:
                        print(
                            f"   ✅ {odds_cat}: {all_profit_odds:+.1f}u (total) | {recent_profit_odds:+.1f}u (recente)"
                        )

        print(f"\n🎯 ESTRATÉGIA FINAL RECOMENDADA:")
        print(f"   1. Usar ROI mínimo: {best_roi_recent or best_roi_all}")
        print(f"   2. Focar nos mercados consistentes listados acima")
        print(f"   3. Priorizar direção com melhor performance em ambos os períodos")
        print(f"   4. Usar apenas odds que foram lucrativas em ambos os períodos")


def main():
    """Função principal refinada"""
    analyzer = RefinedBettingAnalyzer()

    # Caminho do arquivo
    file_path = "../bets/bets_atualizadas_por_mapa.csv"

    print("🚀 INICIANDO ANÁLISE REFINADA DE APOSTAS")
    print("=" * 60)

    # Carregar dados com filtros aplicados
    df = analyzer.load_and_preprocess_data(file_path)

    # 1. ANÁLISE DE ROI RANGES - TODOS OS DADOS
    roi_analysis_all = analyzer.analyze_roi_ranges_comprehensive(df, "TODOS OS DADOS")
    best_roi_all = roi_analysis_all["best_roi_range"]

    # 2. ANÁLISE DE ROI RANGES - ÚLTIMOS 200 JOGOS
    df_200 = df.head(200)
    roi_analysis_200 = analyzer.analyze_roi_ranges_comprehensive(
        df_200, "ÚLTIMOS 200 JOGOS"
    )
    best_roi_recent = roi_analysis_200["best_roi_range"]

    # 3. ANÁLISE DE MERCADOS COM FILTRO ROI OTIMIZADO - TODOS OS DADOS
    if best_roi_all:
        all_data_analysis = analyzer.analyze_markets_with_roi_filter(
            df, best_roi_all, "TODOS OS DADOS"
        )
    else:
        print("⚠️ Nenhuma faixa de ROI ótima encontrada para todos os dados")
        all_data_analysis = None

    # 4. ANÁLISE DE MERCADOS COM FILTRO ROI OTIMIZADO - ÚLTIMOS 200 JOGOS
    if best_roi_recent:
        recent_data_analysis = analyzer.analyze_markets_with_roi_filter(
            df_200, best_roi_recent, "ÚLTIMOS 200 JOGOS"
        )
    else:
        print("⚠️ Nenhuma faixa de ROI ótima encontrada para últimos 200 jogos")
        recent_data_analysis = None

    # 5. RECOMENDAÇÕES ESTRATÉGICAS FINAIS
    analyzer.print_strategic_recommendations(
        all_data_analysis, recent_data_analysis, best_roi_all, best_roi_recent
    )

    print("\n" + "=" * 80)
    print("✅ ANÁLISE REFINADA CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
