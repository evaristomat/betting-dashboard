import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
from collections import defaultdict

warnings.filterwarnings("ignore")


class RollingBettingAnalyzer:
    """Analisador de apostas com janelas móveis (últimas N apostas)"""

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
        self.analysis_windows = [50, 100, 200, 300]  # Janelas de análise

    def categorize_roi_ranges(self, roi: float) -> str:
        """Ranges de ROI mais granulares"""
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
        """Categorização de odds"""
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
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

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

        # Adicionar colunas de análise temporal
        df["bet_number"] = range(len(df), 0, -1)  # Numeração reversa (mais recente = 1)
        df["hour"] = df["date"].dt.hour
        df["weekday"] = df["date"].dt.day_name()

        return df

    def analyze_window(self, df: pd.DataFrame, window_size: int) -> Dict:
        """Analisa uma janela específica de apostas"""
        df_window = df.head(window_size).copy()

        if len(df_window) == 0:
            return None

        total_profit = df_window["profit"].sum()
        total_bets = len(df_window)
        overall_roi = (total_profit / total_bets * 100) if total_bets > 0 else 0
        win_rate = (df_window["status"] == "win").mean() * 100

        # Análise por ROI
        roi_analysis = (
            df_window.groupby("est_roi_category")
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

        # Análise por mercado
        market_analysis = (
            df_window.groupby("grouped_market")
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
        market_analysis["Sharpe_Ratio"] = np.where(
            market_analysis["ROI_Std"] > 0,
            (market_analysis["Real_ROI"] / market_analysis["ROI_Std"]).round(2),
            np.nan,
        )

        # Análise por direção
        direction_analysis = df_window.groupby("direction").agg(
            {
                "profit": "sum",
                "status": lambda x: (x == "win").mean() * 100,
            }
        )

        # Análise por odds
        odds_analysis = (
            df_window.groupby("odds_category")
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

        # Análise temporal
        temporal_analysis = {
            "by_hour": df_window.groupby("hour")["profit"].sum().to_dict(),
            "by_weekday": df_window.groupby("weekday")["profit"].sum().to_dict(),
            "profit_trend": df_window["profit"].cumsum().tolist()[-10:],  # Últimas 10
        }

        return {
            "window_size": window_size,
            "total_profit": total_profit,
            "total_bets": total_bets,
            "overall_roi": overall_roi,
            "win_rate": win_rate,
            "roi_analysis": roi_analysis,
            "market_analysis": market_analysis,
            "direction_analysis": direction_analysis,
            "odds_analysis": odds_analysis,
            "temporal_analysis": temporal_analysis,
            "date_range": {
                "start": df_window["date"].min(),
                "end": df_window["date"].max(),
                "days": (df_window["date"].max() - df_window["date"].min()).days,
            },
        }

    def find_consistent_patterns(self, all_analyses: Dict[int, Dict]) -> Dict:
        """Encontra padrões consistentes entre diferentes janelas"""
        patterns = {
            "consistent_markets": defaultdict(int),
            "consistent_roi_ranges": defaultdict(int),
            "consistent_odds": defaultdict(int),
            "direction_preference": defaultdict(int),
            "performance_trend": {},
        }

        # Analisar cada janela
        for window, analysis in all_analyses.items():
            if not analysis:
                continue

            # Top 3 mercados
            top_markets = analysis["market_analysis"].nlargest(3, "Total_Profit")
            for market in top_markets.index:
                if top_markets.loc[market, "Total_Profit"] > 0:
                    patterns["consistent_markets"][market] += 1

            # ROI ranges lucrativos
            profitable_roi = analysis["roi_analysis"][
                analysis["roi_analysis"]["Total_Profit"] > 0
            ]
            for roi_range in profitable_roi.index:
                patterns["consistent_roi_ranges"][roi_range] += 1

            # Odds lucrativas
            profitable_odds = analysis["odds_analysis"][
                analysis["odds_analysis"]["Total_Profit"] > 0
            ]
            for odds_cat in profitable_odds.index:
                patterns["consistent_odds"][odds_cat] += 1

            # Direção preferencial
            best_direction = analysis["direction_analysis"]["profit"].idxmax()
            patterns["direction_preference"][best_direction] += 1

            # Trend de performance
            patterns["performance_trend"][window] = {
                "roi": analysis["overall_roi"],
                "win_rate": analysis["win_rate"],
                "profit": analysis["total_profit"],
            }

        return patterns

    def print_comparative_analysis(self, all_analyses: Dict[int, Dict], patterns: Dict):
        """Imprime análise comparativa entre períodos"""
        print("🚀 Iniciando análise dos últimos 200 jogos...")
        print(f"📊 Períodos de comparação: {self.analysis_windows}")

        # Dados gerais
        if 200 in all_analyses and all_analyses[200]:
            df_info = all_analyses[200]
            print(f"📅 Dados carregados: {df_info['total_bets']} apostas totais")
            print(f"🎯 Analisando os últimos 200 jogos (solicitados: 200)")

        # ROI nos mercados lucrativos para os últimos 200 jogos
        if 200 in all_analyses and all_analyses[200]:
            analysis = all_analyses[200]
            profitable_markets = analysis["market_analysis"][
                (analysis["market_analysis"]["Total_Profit"] > 2)
                & (analysis["market_analysis"]["Real_ROI"] > 5)
            ].index.tolist()

            if profitable_markets:
                print(f"\n💎 ROI DENTRO DOS MERCADOS LUCRATIVOS:")
                print(f"   Mercados considerados: {profitable_markets[:3]}")  # Top 3
                print(
                    f"{'Range ROI':<12} {'Lucro':<8} {'Apostas':<8} {'Win%':<6} {'ROI Real'}"
                )
                print(f"{'-' * 50}")

                # Filtrar dados apenas dos mercados lucrativos
                roi_profitable = analysis["roi_analysis"]
                for range_name, row in roi_profitable.iterrows():
                    if row["Bets"] > 0:
                        status = "✅" if row["Total_Profit"] > 0 else "❌"
                        print(
                            f"{status} {range_name:<10} {row['Total_Profit']:>6.1f}u "
                            f"{int(row['Bets']):>6} {row['Win_Rate']:>5.1f}% "
                            f"{row['Real_ROI']:>6.1f}%"
                        )

        # Análise estratégica detalhada para 200 jogos
        if 200 in all_analyses and all_analyses[200]:
            self._print_detailed_window_analysis(all_analyses[200])

        # Análise comparativa
        print("\n" + "=" * 80)
        print("📊 ANÁLISE COMPARATIVA DE PERÍODOS")
        print("=" * 80)

        print(
            f"\n{'Período':<15} {'Apostas':<8} {'Lucro':<8} {'ROI%':<8} {'Win%':<8} "
            f"{'Melhor Mercado':<20} {'Dir. Pref.':<10}"
        )
        print("-" * 80)

        for window in self.analysis_windows:
            if window in all_analyses and all_analyses[window]:
                analysis = all_analyses[window]
                best_market = analysis["market_analysis"]["Total_Profit"].idxmax()
                best_direction = analysis["direction_analysis"]["profit"].idxmax()

                print(
                    f"Últimos {window:<3}: {analysis['total_bets']:<8} "
                    f"{analysis['total_profit']:>7.1f}u {analysis['overall_roi']:>6.1f}% "
                    f"{analysis['win_rate']:>6.1f}% {best_market:<20} {best_direction:<10}"
                )

        # Padrões consistentes
        print("\n🔍 ANÁLISE DE CONSISTÊNCIA:")

        # Mercados consistentes
        consistent_markets = [
            m for m, count in patterns["consistent_markets"].items() if count >= 3
        ]  # Aparece em pelo menos 3 períodos
        print(f"\n   🎯 MERCADOS CONSISTENTES (aparecem no TOP 3 em ≥3 períodos):")
        for market in consistent_markets[:5]:
            periods = [
                w
                for w in self.analysis_windows
                if w in all_analyses
                and all_analyses[w]
                and market
                in all_analyses[w]["market_analysis"].nlargest(3, "Total_Profit").index
            ]
            print(f"      • {market}: Períodos {periods}")

        # Direção consistente
        print(f"\n   🔽 CONSISTÊNCIA DE DIREÇÃO:")
        total_windows = len(
            [w for w in self.analysis_windows if w in all_analyses and all_analyses[w]]
        )
        for direction, count in patterns["direction_preference"].items():
            percentage = (count / total_windows * 100) if total_windows > 0 else 0
            print(
                f"      • {direction} preferencial em {count}/{total_windows} períodos ({percentage:.1f}%)"
            )

        # Tendência de performance
        print(f"\n   📈 TENDÊNCIA DE ROI: ", end="")
        roi_values = [
            patterns["performance_trend"][w]["roi"]
            for w in sorted(patterns["performance_trend"].keys())
        ]
        if len(roi_values) >= 2:
            trend = "⬆️ Melhorando" if roi_values[-1] > roi_values[0] else "⬇️ Piorando"
            print(trend)
            print(
                f"      • Últimos {min(self.analysis_windows)} jogos: {roi_values[0]:.1f}%"
            )
            print(
                f"      • Últimos {max(self.analysis_windows)} jogos: {roi_values[-1]:.1f}%"
            )

    def _print_detailed_window_analysis(self, analysis: Dict):
        """Imprime análise detalhada de uma janela específica"""
        print(f"\n🎯 ANÁLISE ESTRATÉGICA - ÚLTIMOS {analysis['window_size']} JOGOS")
        print("=" * 70)
        print(f"📊 VISÃO GERAL:")
        print(
            f"   Total: {analysis['total_bets']} apostas | Lucro: {analysis['total_profit']:.2f} units"
        )
        print(
            f"   ROI: {analysis['overall_roi']:.1f}% | Win Rate: {analysis['win_rate']:.1f}%"
        )

        # Performance por ROI
        print(f"\n📈 PERFORMANCE DETALHADA POR ROI:")
        print(
            f"{'Range':<12} {'Lucro':<8} {'Apostas':<8} {'Win%':<6} {'ROI Real':<8} {'Eficiência':<10}"
        )
        print("-" * 65)

        roi_analysis = analysis["roi_analysis"].sort_values(
            "Total_Profit", ascending=False
        )
        for range_name, row in roi_analysis.iterrows():
            if row["Bets"] > 0:
                status = "✅" if row["Total_Profit"] > 0 else "❌"
                efficiency_str = (
                    f"{row['ROI_Efficiency']:.2f}x"
                    if not pd.isna(row["ROI_Efficiency"])
                    else "N/A"
                )
                print(
                    f"{status} {range_name:<10} {row['Total_Profit']:>6.1f}u "
                    f"{int(row['Bets']):>6} {row['Win_Rate']:>6.1f}% "
                    f"{row['Real_ROI']:>7.1f}% {efficiency_str:>9}"
                )

        # Performance por mercado
        print(f"\n🏆 PERFORMANCE POR MERCADO (TOP 10):")
        print(
            f"{'Mercado':<20} {'Lucro':<8} {'ROI%':<6} {'Win%':<6} {'Apostas':<8} {'Sharpe':<7}"
        )
        print("-" * 70)

        market_analysis = analysis["market_analysis"].sort_values(
            "Total_Profit", ascending=False
        )
        for market, row in market_analysis.head(10).iterrows():
            if row["Bets"] > 0:
                icon = (
                    "🏆"
                    if row["Total_Profit"] >= 5
                    else "✅"
                    if row["Total_Profit"] > 0
                    else "❌"
                )
                sharpe_str = (
                    f"{row['Sharpe_Ratio']:.2f}"
                    if not pd.isna(row["Sharpe_Ratio"])
                    else "N/A"
                )
                print(
                    f"{icon} {market:<18} {row['Total_Profit']:>6.1f}u "
                    f"{row['Real_ROI']:>5.1f}% {row['Win_Rate']:>5.1f}% "
                    f"{int(row['Bets']):>6} {sharpe_str:>6}"
                )

        # Análise por direção
        print(f"\n🔽 ANÁLISE APROFUNDADA POR DIREÇÃO:")

        for direction in ["UNDER", "OVER"]:
            dir_data = analysis["market_analysis"][
                analysis["market_analysis"].index.str.startswith(direction)
            ]
            total_profit = dir_data["Total_Profit"].sum()
            total_bets = dir_data["Bets"].sum()

            status = "✅" if total_profit > 0 else "❌"
            print(
                f"\n{status} {direction} - Total: {total_profit:.1f}u ({int(total_bets)} apostas)"
            )

            if len(dir_data) > 0:
                print(
                    f"  {'Mercado':<15} {'Lucro':<8} {'ROI%':<6} {'Win%':<6} {'Apostas'}"
                )
                print(f"  {'-' * 45}")

                for market, row in dir_data.nlargest(5, "Total_Profit").iterrows():
                    market_name = market.split(" - ")[1]
                    sub_status = (
                        "💎"
                        if row["Total_Profit"] > 3
                        else "✅"
                        if row["Total_Profit"] > 0
                        else "❌"
                    )
                    print(
                        f"  {sub_status} {market_name:<13} {row['Total_Profit']:>6.1f}u "
                        f"{row['Real_ROI']:>5.1f}% {row['Win_Rate']:>5.1f}% "
                        f"{int(row['Bets']):>6}"
                    )

        # Performance por odds
        print(f"\n📊 PERFORMANCE POR ODDS:")
        for odds_cat, row in analysis["odds_analysis"].iterrows():
            if row["Bets"] > 0:
                status = "✅" if row["Total_Profit"] > 0 else "❌"
                risk_reward = row["Win_Rate"] / 100 * row["Avg_Odds"]
                print(
                    f"   {status} {odds_cat:<12} → {row['Total_Profit']:>6.1f}u | "
                    f"ROI: {row['Real_ROI']:>6.1f}% | Risk/Reward: {risk_reward:>4.2f} | "
                    f"{int(row['Bets'])} apostas"
                )

        # Recomendações estratégicas
        self._print_strategic_recommendations(analysis)

    def _print_strategic_recommendations(self, analysis: Dict):
        """Imprime recomendações estratégicas baseadas na análise"""
        print("\n" + "=" * 70)
        print("🚀 RECOMENDAÇÕES ESTRATÉGICAS")
        print("=" * 70)

        # Identificar critérios lucrativos
        profitable_roi = analysis["roi_analysis"][
            (analysis["roi_analysis"]["Total_Profit"] > 0)
            & (analysis["roi_analysis"]["Bets"] >= 2)
        ].index.tolist()

        profitable_markets = (
            analysis["market_analysis"][
                (analysis["market_analysis"]["Total_Profit"] > 1)
                & (analysis["market_analysis"]["Real_ROI"] > 5)
            ]
            .nlargest(5, "Total_Profit")
            .index.tolist()
        )

        profitable_odds = analysis["odds_analysis"][
            analysis["odds_analysis"]["Total_Profit"] > 0
        ].index.tolist()

        best_direction = analysis["direction_analysis"]["profit"].idxmax()

        print(f"\n📋 CRITÉRIOS OTIMIZADOS:")
        print(f"   📈 ROI Ranges lucrativos: {profitable_roi[:3]}")
        print(f"   🎯 Mercados TOP: {profitable_markets[:3]}")
        print(f"   📊 Odds válidas: {profitable_odds}")
        print(f"   🔽 Direção preferencial: {best_direction}")

        print(f"\n💡 INSIGHTS ESTRATÉGICOS:")

        # Comparação de direções
        under_profit = (
            analysis["direction_analysis"].loc["UNDER", "profit"]
            if "UNDER" in analysis["direction_analysis"].index
            else 0
        )
        over_profit = (
            analysis["direction_analysis"].loc["OVER", "profit"]
            if "OVER" in analysis["direction_analysis"].index
            else 0
        )

        if under_profit != 0 or over_profit != 0:
            if over_profit > under_profit:
                ratio = abs(under_profit / over_profit) if over_profit != 0 else 0
                print(
                    f"   🔼 OVER domina: {over_profit:+.1f}u vs {under_profit:+.1f}u UNDER (ratio {ratio:.1f}:1)"
                )
            else:
                ratio = abs(over_profit / under_profit) if under_profit != 0 else 0
                print(
                    f"   🔽 UNDER domina: {under_profit:+.1f}u vs {over_profit:+.1f}u OVER (ratio {ratio:.1f}:1)"
                )

        # Melhor mercado
        if len(profitable_markets) > 0:
            best_market = profitable_markets[0]
            best_data = analysis["market_analysis"].loc[best_market]
            print(
                f"   🏆 Mercado #1: {best_market} ({best_data['Total_Profit']:.1f}u, {best_data['Real_ROI']:.1f}% ROI)"
            )

        # ROI mais eficiente
        if len(analysis["roi_analysis"]) > 0:
            best_roi = analysis["roi_analysis"].nlargest(1, "ROI_Efficiency")
            if len(best_roi) > 0 and not pd.isna(best_roi.iloc[0]["ROI_Efficiency"]):
                print(
                    f"   📈 Range mais eficiente: {best_roi.index[0]} ({best_roi.iloc[0]['ROI_Efficiency']:.2f}x eficiência)"
                )


def main():
    """Função principal"""
    analyzer = RollingBettingAnalyzer()

    # Caminho do arquivo
    file_path = "../bets/bets_atualizadas_por_mapa.csv"

    # Carregar dados
    df = analyzer.load_and_preprocess_data(file_path)

    # Analisar diferentes janelas
    all_analyses = {}
    for window in analyzer.analysis_windows:
        if len(df) >= window:
            all_analyses[window] = analyzer.analyze_window(df, window)

    # Encontrar padrões consistentes
    patterns = analyzer.find_consistent_patterns(all_analyses)

    # Imprimir análise comparativa
    analyzer.print_comparative_analysis(all_analyses, patterns)

    print("\n" + "=" * 80)
    print("✅ ANÁLISE CONCLUÍDA!")
    print("=" * 80)


if __name__ == "__main__":
    main()
