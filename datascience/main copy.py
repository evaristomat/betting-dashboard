import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore")


class EnhancedBettingAnalyzer:
    """Analisador aprimorado com múltiplas camadas de análise"""

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

    def categorize_roi_ranges(self, roi: float) -> str:
        """Ranges de ROI mais granulares para melhor visibilidade"""
        if pd.isna(roi):
            return "N/A"
        elif roi < 10:
            return "<10%"
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
        else:
            return "≥40%"

    def categorize_odds(self, odds: float) -> str:
        """Categorização de odds otimizada"""
        if pd.isna(odds):
            return "N/A"
        elif odds <= 1.3:
            return "muito_baixa"
        elif odds <= 1.6:
            return "baixa"
        elif odds <= 2.0:
            return "media"
        elif odds <= 2.5:
            return "media_alta"
        elif odds <= 3.0:
            return "alta"
        else:
            return "muito_alta"

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categorização de mercados"""
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

    def load_and_preprocess_data(self, file_path: str) -> pd.DataFrame:
        """Carrega e preprocessa os dados"""
        df = pd.read_csv(file_path)

        # Conversão de datas
        for pt_month, en_month in self.month_translation.items():
            df["date"] = df["date"].str.replace(pt_month, en_month)

        df["date"] = pd.to_datetime(
            df["date"], format="%d %b %Y %H:%M", errors="coerce"
        )
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

        # Categorizações
        df[["direction", "market_type", "grouped_market"]] = df.apply(
            lambda r: self.categorize_market(r["bet_type"], r["bet_line"]),
            axis=1,
            result_type="expand",
        )

        df["odds_category"] = df["odds"].apply(self.categorize_odds)
        df["est_roi_category"] = df["estimated_roi"].apply(self.categorize_roi_ranges)

        return df

    def analyze_roi_performance(self, df: pd.DataFrame) -> Dict:
        """Análise detalhada de performance por ROI"""
        roi_analysis = (
            df.groupby("est_roi_category")
            .agg(
                {
                    "profit": ["sum", "count"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": "mean",
                }
            )
            .round(2)
        )

        roi_analysis.columns = ["Total_Profit", "Bets", "Win_Rate", "Avg_Est_ROI"]
        roi_analysis["Real_ROI"] = (
            roi_analysis["Total_Profit"] / roi_analysis["Bets"] * 100
        ).round(2)
        roi_analysis["ROI_Efficiency"] = (
            roi_analysis["Real_ROI"] / roi_analysis["Avg_Est_ROI"]
        ).round(3)

        # Ordenar por lucro total
        roi_analysis = roi_analysis.sort_values("Total_Profit", ascending=False)

        return roi_analysis

    def analyze_market_performance(self, df: pd.DataFrame) -> Dict:
        """Análise detalhada por mercado"""
        market_analysis = (
            df.groupby("grouped_market")
            .agg(
                {
                    "profit": ["sum", "count", "mean", "std"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": ["mean", "std"],
                    "odds": "mean",
                }
            )
            .round(2)
        )

        market_analysis.columns = [
            "Total_Profit",
            "Bets",
            "Avg_Profit",
            "Profit_Std",
            "Win_Rate",
            "Avg_Est_ROI",
            "ROI_Std",
            "Avg_Odds",
        ]

        market_analysis["Real_ROI"] = (
            market_analysis["Total_Profit"] / market_analysis["Bets"] * 100
        ).round(2)
        market_analysis["Consistency"] = (
            market_analysis["Avg_Profit"] / market_analysis["Profit_Std"]
        ).round(2)
        market_analysis["Sharpe_Ratio"] = (
            market_analysis["Real_ROI"] / market_analysis["ROI_Std"]
        ).round(2)

        return market_analysis.sort_values("Total_Profit", ascending=False)

    def analyze_profitable_markets_roi(
        self, df: pd.DataFrame, profitable_markets: List[str]
    ) -> Dict:
        """Análise de ROI dentro dos mercados lucrativos"""
        df_profitable = df[df["grouped_market"].isin(profitable_markets)]

        roi_in_profitable = (
            df_profitable.groupby("est_roi_category")
            .agg(
                {
                    "profit": ["sum", "count"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "grouped_market": lambda x: list(x.unique()),
                }
            )
            .round(2)
        )

        roi_in_profitable.columns = ["Total_Profit", "Bets", "Win_Rate", "Markets"]
        roi_in_profitable["Real_ROI"] = (
            roi_in_profitable["Total_Profit"] / roi_in_profitable["Bets"] * 100
        ).round(2)

        return roi_in_profitable.sort_values("Total_Profit", ascending=False)

    def analyze_direction_depth(self, df: pd.DataFrame) -> Dict:
        """Análise aprofundada por direção"""
        direction_analysis = {}

        for direction in ["UNDER", "OVER"]:
            df_dir = df[df["direction"] == direction]
            if len(df_dir) == 0:
                continue

            analysis = (
                df_dir.groupby("market_type")
                .agg(
                    {
                        "profit": ["sum", "count"],
                        "status": lambda x: (x == "win").mean() * 100,
                        "estimated_roi": "mean",
                    }
                )
                .round(2)
            )

            analysis.columns = ["Total_Profit", "Bets", "Win_Rate", "Avg_Est_ROI"]
            analysis["Real_ROI"] = (
                analysis["Total_Profit"] / analysis["Bets"] * 100
            ).round(2)

            direction_analysis[direction] = analysis.sort_values(
                "Total_Profit", ascending=False
            )

        return direction_analysis

    def analyze_odds_performance(self, df: pd.DataFrame) -> Dict:
        """Análise detalhada por odds"""
        odds_analysis = (
            df.groupby("odds_category")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "odds": ["mean", "min", "max"],
                    "estimated_roi": "mean",
                }
            )
            .round(2)
        )

        odds_analysis.columns = [
            "Total_Profit",
            "Bets",
            "Avg_Profit",
            "Win_Rate",
            "Avg_Odds",
            "Min_Odds",
            "Max_Odds",
            "Avg_Est_ROI",
        ]

        odds_analysis["Real_ROI"] = (
            odds_analysis["Total_Profit"] / odds_analysis["Bets"] * 100
        ).round(2)
        odds_analysis["Risk_Reward"] = (
            odds_analysis["Win_Rate"] / 100 * odds_analysis["Avg_Odds"]
        ).round(2)

        return odds_analysis.sort_values("Total_Profit", ascending=False)

    def generate_strategy_recommendations(self, analysis_results: Dict) -> Dict:
        """Gera recomendações estratégicas baseadas na análise"""

        # Critérios base
        profitable_roi_ranges = analysis_results["roi_analysis"][
            (analysis_results["roi_analysis"]["Total_Profit"] > 1)
            & (analysis_results["roi_analysis"]["Real_ROI"] > 0)
            & (analysis_results["roi_analysis"]["Bets"] >= 5)
        ].index.tolist()

        profitable_markets = analysis_results["market_analysis"][
            (analysis_results["market_analysis"]["Total_Profit"] > 2)
            & (analysis_results["market_analysis"]["Real_ROI"] > 5)
            & (analysis_results["market_analysis"]["Bets"] >= 5)
        ].index.tolist()

        profitable_odds = analysis_results["odds_analysis"][
            (analysis_results["odds_analysis"]["Total_Profit"] > 0)
            & (analysis_results["odds_analysis"]["Real_ROI"] > 0)
        ].index.tolist()

        # Análise de direção
        direction_profit = {}
        for direction, data in analysis_results["direction_analysis"].items():
            direction_profit[direction] = data["Total_Profit"].sum()

        preferred_direction = (
            max(direction_profit, key=direction_profit.get)
            if direction_profit
            else "UNDER"
        )

        return {
            "roi_ranges": profitable_roi_ranges,
            "markets": profitable_markets,
            "odds": profitable_odds,
            "preferred_direction": preferred_direction,
            "direction_performance": direction_profit,
        }

    def print_enhanced_analysis(self, df: pd.DataFrame, analysis_results: Dict):
        """Output aprimorado da análise"""

        total_profit = df["profit"].sum()
        total_bets = len(df)
        overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
        win_rate = (df["status"] == "win").mean() * 100

        print("🎯 ANÁLISE ESTRATÉGICA APRIMORADA")
        print("=" * 70)
        print(f"📊 VISÃO GERAL:")
        print(f"   Total: {total_bets} apostas | Lucro: {total_profit:.2f} units")
        print(f"   ROI: {overall_roi:.1f}% | Win Rate: {win_rate:.1f}%")

        # ROI Analysis
        print(f"\n📈 PERFORMANCE DETALHADA POR ROI:")
        print(
            f"{'Range':<12} {'Lucro':<8} {'Apostas':<8} {'Win%':<6} {'ROI Real':<8} {'Eficiência':<10}"
        )
        print(f"{'-' * 65}")

        roi_analysis = analysis_results["roi_analysis"]
        for range_name, row in roi_analysis.iterrows():
            status = "✅" if row["Total_Profit"] > 0 else "❌"
            efficiency_str = (
                f"{row['ROI_Efficiency']:.2f}x"
                if not pd.isna(row["ROI_Efficiency"])
                else "N/A"
            )
            print(
                f"{status} {range_name:<10} {row['Total_Profit']:>6.1f}u {int(row['Bets']):>6} {row['Win_Rate']:>6.1f}% {row['Real_ROI']:>7.1f}% {efficiency_str:>9}"
            )

        # Market Analysis
        print(f"\n🏆 PERFORMANCE POR MERCADO (TOP 10):")
        print(
            f"{'Mercado':<20} {'Lucro':<8} {'ROI%':<6} {'Win%':<6} {'Apostas':<8} {'Sharpe':<7}"
        )
        print(f"{'-' * 70}")

        market_analysis = analysis_results["market_analysis"]
        for market, row in market_analysis.head(10).iterrows():
            if row["Total_Profit"] >= 5:
                icon = "🏆"
            elif row["Total_Profit"] >= 2:
                icon = "💎"
            elif row["Total_Profit"] > 0:
                icon = "✅"
            else:
                icon = "❌"

            sharpe_str = (
                f"{row['Sharpe_Ratio']:.2f}"
                if not pd.isna(row["Sharpe_Ratio"])
                else "N/A"
            )
            print(
                f"{icon} {market:<18} {row['Total_Profit']:>6.1f}u {row['Real_ROI']:>5.1f}% {row['Win_Rate']:>5.1f}% {int(row['Bets']):>6} {sharpe_str:>6}"
            )

        # Direction Analysis
        print(f"\n🔽 ANÁLISE APROFUNDADA POR DIREÇÃO:")
        direction_analysis = analysis_results["direction_analysis"]

        for direction, data in direction_analysis.items():
            total_dir_profit = data["Total_Profit"].sum()
            total_dir_bets = data["Bets"].sum()
            status = (
                "🏆"
                if total_dir_profit > 10
                else "✅"
                if total_dir_profit > 0
                else "❌"
            )

            print(
                f"\n{status} {direction} - Total: {total_dir_profit:.1f}u ({total_dir_bets} apostas)"
            )
            print(f"{'  Mercado':<15} {'Lucro':<8} {'ROI%':<6} {'Win%':<6} {'Apostas'}")
            print(f"  {'-' * 45}")

            for market, row in data.head(5).iterrows():
                sub_status = (
                    "  💎"
                    if row["Total_Profit"] > 3
                    else "  ✅"
                    if row["Total_Profit"] > 0
                    else "  ❌"
                )
                print(
                    f"{sub_status} {market:<13} {row['Total_Profit']:>6.1f}u {row['Real_ROI']:>5.1f}% {row['Win_Rate']:>5.1f}% {int(row['Bets']):>6}"
                )

        # Odds Analysis
        print(f"\n📊 PERFORMANCE POR ODDS:")
        odds_analysis = analysis_results["odds_analysis"]
        for odds_cat, row in odds_analysis.iterrows():
            status = "✅" if row["Total_Profit"] > 0 else "❌"
            print(
                f"   {status} {odds_cat:<12} → {row['Total_Profit']:>6.1f}u | ROI: {row['Real_ROI']:>5.1f}% | Risk/Reward: {row['Risk_Reward']:>4.2f} | {int(row['Bets'])} apostas"
            )

        # Strategy Recommendations
        recommendations = self.generate_strategy_recommendations(analysis_results)

        print(f"\n" + "=" * 70)
        print("🚀 RECOMENDAÇÕES ESTRATÉGICAS")
        print("=" * 70)

        print(f"\n📋 CRITÉRIOS OTIMIZADOS:")
        print(f"   📈 ROI Ranges lucrativos: {recommendations['roi_ranges']}")
        print(f"   🎯 Mercados TOP: {recommendations['markets'][:5]}")  # Top 5
        print(f"   📊 Odds válidas: {recommendations['odds']}")
        print(f"   🔽 Direção preferencial: {recommendations['preferred_direction']}")

        print(f"\n💡 INSIGHTS ESTRATÉGICOS:")

        # Performance comparison between directions
        dir_perf = recommendations["direction_performance"]
        if "UNDER" in dir_perf and "OVER" in dir_perf:
            under_profit = dir_perf["UNDER"]
            over_profit = dir_perf["OVER"]
            ratio = (
                abs(under_profit / over_profit) if over_profit != 0 else float("inf")
            )

            if under_profit > over_profit:
                print(
                    f"   🔽 UNDER domina: +{under_profit:.1f}u vs {over_profit:+.1f}u OVER (ratio {ratio:.1f}:1)"
                )
            else:
                print(
                    f"   🔼 OVER domina: +{over_profit:.1f}u vs {under_profit:+.1f}u UNDER (ratio {ratio:.1f}:1)"
                )

        # Best markets analysis
        if len(recommendations["markets"]) > 0:
            best_market = market_analysis.index[0]
            best_profit = market_analysis.iloc[0]["Total_Profit"]
            best_roi = market_analysis.iloc[0]["Real_ROI"]
            print(
                f"   🏆 Mercado #1: {best_market} ({best_profit:.1f}u, {best_roi:.1f}% ROI)"
            )

        # ROI efficiency
        best_roi_range = roi_analysis.index[0]
        best_roi_efficiency = roi_analysis.iloc[0]["ROI_Efficiency"]
        if not pd.isna(best_roi_efficiency):
            print(
                f"   📈 Range mais eficiente: {best_roi_range} ({best_roi_efficiency:.2f}x eficiência)"
            )

        return recommendations


def betting_strategy_analysis_enhanced(file_path: str) -> Dict:
    """Função principal da análise aprimorada"""

    analyzer = EnhancedBettingAnalyzer()

    # Carregar e preprocessar dados
    df = analyzer.load_and_preprocess_data(file_path)

    # Executar análises
    analysis_results = {
        "roi_analysis": analyzer.analyze_roi_performance(df),
        "market_analysis": analyzer.analyze_market_performance(df),
        "direction_analysis": analyzer.analyze_direction_depth(df),
        "odds_analysis": analyzer.analyze_odds_performance(df),
    }

    # Análise de ROI em mercados lucrativos
    profitable_markets = analysis_results["market_analysis"][
        (analysis_results["market_analysis"]["Total_Profit"] > 2)
        & (analysis_results["market_analysis"]["Real_ROI"] > 5)
    ].index.tolist()

    if profitable_markets:
        analysis_results["roi_in_profitable"] = analyzer.analyze_profitable_markets_roi(
            df, profitable_markets
        )

        print(f"\n💎 ROI DENTRO DOS MERCADOS LUCRATIVOS:")
        print(f"   Mercados considerados: {profitable_markets}")
        print(f"{'Range ROI':<12} {'Lucro':<8} {'Apostas':<8} {'Win%':<6} {'ROI Real'}")
        print(f"{'-' * 50}")

        for range_name, row in analysis_results["roi_in_profitable"].iterrows():
            status = "✅" if row["Total_Profit"] > 0 else "❌"
            print(
                f"{status} {range_name:<10} {row['Total_Profit']:>6.1f}u {int(row['Bets']):>6} {row['Win_Rate']:>5.1f}% {row['Real_ROI']:>6.1f}%"
            )

    # Imprimir análise completa
    recommendations = analyzer.print_enhanced_analysis(df, analysis_results)

    return {
        "data": df,
        "analysis": analysis_results,
        "recommendations": recommendations,
    }


def get_current_month_data(file_path: str) -> pd.DataFrame:
    """Filtra dados do mês atual"""
    analyzer = EnhancedBettingAnalyzer()
    df = analyzer.load_and_preprocess_data(file_path)

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

    # Análise do mês atual
    monthly_data = get_current_month_data(file_path)
    monthly_data.to_csv("temp_monthly.csv", index=False)
    results = betting_strategy_analysis_enhanced("temp_monthly.csv")

    # Limpeza
    import os

    if os.path.exists("temp_monthly.csv"):
        os.remove("temp_monthly.csv")
