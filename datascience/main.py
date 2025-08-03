import pandas as pd
import numpy as np
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

warnings.filterwarnings("ignore")


@dataclass
class AnalysisConfig:
    """Configurações da análise"""

    min_bets_for_roi: int = 20  # Mínimo de apostas para considerar uma faixa de ROI
    min_bets_for_market: int = 5  # Mínimo de apostas para considerar um mercado
    roi_ranges: List[int] = None  # Faixas de ROI para análise
    odds_ranges: List[Tuple[float, float, str]] = None  # Faixas de odds

    def __post_init__(self):
        if self.roi_ranges is None:
            self.roi_ranges = [10, 15, 20, 25, 30]
        if self.odds_ranges is None:
            self.odds_ranges = [
                (0, 1.6, "baixa"),
                (1.6, 2.0, "media"),
                (2.0, 2.5, "media_alta"),
                (2.5, 3.0, "alta"),
                (3.0, float("inf"), "muito_alta"),
            ]


class CleanBettingAnalyzer:
    """Analisador limpo e otimizado de apostas em eSports"""

    def __init__(self, config: AnalysisConfig = None):
        self.config = config or AnalysisConfig()
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

        # Mapeamento dos mercados baseado nos dados reais
        self.market_mapping = {
            "inhibitors": "INHIBITORS",
            "dragons": "DRAGONS",
            "barons": "BARONS",
            "towers": "TOWERS",
            "kills": "KILLS",
            "duration": "DURATION",
        }

    def load_data(self, file_path: str) -> pd.DataFrame:
        """Carrega e preprocessa os dados"""
        print("📂 Carregando dados...")
        df = pd.read_csv(file_path)

        # Conversão de datas
        for pt_month, en_month in self.month_translation.items():
            df["date"] = df["date"].str.replace(pt_month, en_month)

        df["date"] = pd.to_datetime(
            df["date"], format="%d %b %Y %H:%M", errors="coerce"
        )
        df = df.sort_values("date", ascending=False).reset_index(drop=True)

        # Conversões numéricas
        df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)

        # Extrair mercado e linha da bet_line
        df["market"] = df["bet_line"].apply(self._extract_market)
        df["line_value"] = df["bet_line"].str.extract(r"(\d+\.?\d*)")[0].astype(float)
        df["direction"] = df["bet_type"].str.upper()

        # Criar mercado agrupado
        df["grouped_market"] = df["direction"] + "_" + df["market"]

        # Categorizar odds e ROI
        df["odds_category"] = df["odds"].apply(self._categorize_odds)
        df["roi_category"] = df["estimated_roi"].apply(self._categorize_roi)

        print(f"✅ {len(df)} apostas carregadas")
        print(
            f"📅 Período: {df['date'].min().strftime('%Y-%m-%d')} até {df['date'].max().strftime('%Y-%m-%d')}"
        )
        print(f"💰 Lucro total: {df['profit'].sum():.2f}u")
        print(f"📊 ROI geral: {(df['profit'].sum() / len(df) * 100):.2f}%")

        return df

    def _extract_market(self, bet_line: str) -> str:
        """Extrai o tipo de mercado da bet_line"""
        bet_line_lower = bet_line.lower()

        for keyword, market_name in self.market_mapping.items():
            if keyword in bet_line_lower:
                return market_name

        # Se for game_duration
        if "game_duration" in bet_line_lower:
            return "DURATION"

        return "OTHER"

    def _categorize_odds(self, odds: float) -> str:
        """Categoriza odds em faixas"""
        if pd.isna(odds):
            return "N/A"

        for min_val, max_val, category in self.config.odds_ranges:
            if min_val <= odds < max_val:
                return category

        return "muito_alta"

    def _categorize_roi(self, roi: float) -> str:
        """Categoriza ROI em faixas"""
        if pd.isna(roi):
            return "N/A"

        for threshold in sorted(self.config.roi_ranges, reverse=True):
            if roi >= threshold:
                return f"{threshold}+"

        return f"<{self.config.roi_ranges[0]}"

    def analyze_overall_performance(self, df: pd.DataFrame) -> Dict:
        """Análise geral de performance"""
        print("\n" + "=" * 80)
        print("📊 ANÁLISE GERAL DE PERFORMANCE")
        print("=" * 80)

        total_bets = len(df)
        total_profit = df["profit"].sum()
        win_rate = (df["status"] == "win").mean() * 100
        roi = (total_profit / total_bets * 100) if total_bets > 0 else 0

        print(f"\n📈 Resumo Geral:")
        print(f"   Total de apostas: {total_bets}")
        print(f"   Lucro total: {total_profit:.2f}u")
        print(f"   Taxa de acerto: {win_rate:.1f}%")
        print(f"   ROI real: {roi:.2f}%")
        print(f"   ROI médio estimado: {df['estimated_roi'].mean():.1f}%")

        # Performance por direção
        direction_stats = (
            df.groupby("direction")
            .agg(
                {
                    "profit": ["sum", "count"],
                    "status": lambda x: (x == "win").mean() * 100,
                }
            )
            .round(2)
        )

        print(f"\n🎯 Performance por Direção:")
        for direction in direction_stats.index:
            profit = direction_stats.loc[direction, ("profit", "sum")]
            count = direction_stats.loc[direction, ("profit", "count")]
            win_pct = direction_stats.loc[direction, ("status", "<lambda>")]
            dir_roi = (profit / count * 100) if count > 0 else 0

            print(
                f"   {direction}: {profit:+.1f}u | {int(count)} apostas | "
                f"{win_pct:.1f}% win | {dir_roi:+.1f}% ROI"
            )

        return {
            "total_bets": total_bets,
            "total_profit": total_profit,
            "win_rate": win_rate,
            "roi": roi,
            "direction_stats": direction_stats,
        }

    def analyze_markets(
        self, df: pd.DataFrame, title: str = "TODOS OS MERCADOS"
    ) -> pd.DataFrame:
        """Análise detalhada por mercado"""
        print(f"\n" + "=" * 80)
        print(f"🎮 ANÁLISE DE MERCADOS - {title}")
        print("=" * 80)

        # Agrupar por mercado
        market_analysis = (
            df.groupby("grouped_market")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": "mean",
                    "odds": "mean",
                    "line_value": "mean",
                }
            )
            .round(2)
        )

        market_analysis.columns = [
            "profit_total",
            "bets",
            "profit_avg",
            "win_rate",
            "roi_est",
            "odds_avg",
            "line_avg",
        ]
        market_analysis["roi_real"] = (
            market_analysis["profit_total"] / market_analysis["bets"] * 100
        ).round(2)

        # Ordenar por lucro total
        market_analysis = market_analysis.sort_values("profit_total", ascending=False)

        print(
            f"\n{'Mercado':<20} {'Lucro':<10} {'Apostas':<8} {'Win%':<8} "
            f"{'ROI Real':<10} {'ROI Est':<10} {'Odds':<8} {'Linha'}"
        )
        print("-" * 90)

        for market, row in market_analysis.iterrows():
            if row["bets"] >= self.config.min_bets_for_market:
                status = (
                    "🏆"
                    if row["profit_total"] > 5
                    else "✅"
                    if row["profit_total"] > 0
                    else "❌"
                )
                print(
                    f"{status} {market:<18} {row['profit_total']:>8.1f}u "
                    f"{int(row['bets']):>6} {row['win_rate']:>6.1f}% "
                    f"{row['roi_real']:>8.1f}% {row['roi_est']:>8.1f}% "
                    f"{row['odds_avg']:>6.2f} {row['line_avg']:>6.1f}"
                )

        return market_analysis

    def analyze_roi_efficiency(self, df: pd.DataFrame) -> Dict:
        """Análise de eficiência por faixa de ROI"""
        print(f"\n" + "=" * 80)
        print("🎯 ANÁLISE DE EFICIÊNCIA POR FAIXA DE ROI")
        print("=" * 80)

        roi_analysis = (
            df.groupby("roi_category")
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
            "profit_total",
            "bets",
            "profit_avg",
            "win_rate",
            "roi_est_avg",
        ]
        roi_analysis["roi_real"] = (
            roi_analysis["profit_total"] / roi_analysis["bets"] * 100
        ).round(2)
        roi_analysis["efficiency"] = (
            roi_analysis["roi_real"] / roi_analysis["roi_est_avg"]
        ).round(3)

        # Ordenar por categoria de ROI
        roi_order = [f"<{self.config.roi_ranges[0]}"] + [
            f"{r}+" for r in self.config.roi_ranges
        ]
        roi_analysis = roi_analysis.reindex(
            [cat for cat in roi_order if cat in roi_analysis.index]
        )

        print(
            f"\n{'Faixa ROI':<12} {'Lucro':<10} {'Apostas':<8} {'Win%':<8} "
            f"{'ROI Real':<10} {'ROI Est':<10} {'Eficiência'}"
        )
        print("-" * 75)

        best_roi = None
        best_efficiency = 0

        for roi_cat, row in roi_analysis.iterrows():
            if row["bets"] >= self.config.min_bets_for_roi:
                status = (
                    "🏆"
                    if row["roi_real"] > 10
                    else "✅"
                    if row["roi_real"] > 0
                    else "❌"
                )
                eff_str = (
                    f"{row['efficiency']:.2f}x"
                    if not pd.isna(row["efficiency"])
                    else "N/A"
                )

                print(
                    f"{status} {roi_cat:<10} {row['profit_total']:>8.1f}u "
                    f"{int(row['bets']):>6} {row['win_rate']:>6.1f}% "
                    f"{row['roi_real']:>8.1f}% {row['roi_est_avg']:>8.1f}% "
                    f"{eff_str:>10}"
                )

                # Identificar melhor faixa
                if row["roi_real"] > 5 and row["efficiency"] > best_efficiency:
                    best_roi = roi_cat
                    best_efficiency = row["efficiency"]

        if best_roi:
            print(
                f"\n🎯 Melhor faixa de ROI: {best_roi} (Eficiência: {best_efficiency:.2f}x)"
            )

        return {
            "roi_analysis": roi_analysis,
            "best_roi_category": best_roi,
            "best_efficiency": best_efficiency,
        }

    def analyze_odds_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Análise de performance por faixa de odds"""
        print(f"\n" + "=" * 80)
        print("📊 ANÁLISE POR FAIXA DE ODDS")
        print("=" * 80)

        odds_analysis = (
            df.groupby("odds_category")
            .agg(
                {
                    "profit": ["sum", "count"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "odds": "mean",
                }
            )
            .round(2)
        )

        odds_analysis.columns = ["profit_total", "bets", "win_rate", "odds_avg"]
        odds_analysis["roi"] = (
            odds_analysis["profit_total"] / odds_analysis["bets"] * 100
        ).round(2)

        # Ordenar por categoria
        odds_order = [cat[2] for cat in self.config.odds_ranges]
        odds_analysis = odds_analysis.reindex(
            [cat for cat in odds_order if cat in odds_analysis.index]
        )

        print(
            f"\n{'Faixa Odds':<15} {'Lucro':<10} {'Apostas':<8} {'Win%':<8} "
            f"{'ROI':<8} {'Odds Média'}"
        )
        print("-" * 60)

        for odds_cat, row in odds_analysis.iterrows():
            if row["bets"] > 0:
                status = "✅" if row["profit_total"] > 0 else "❌"
                print(
                    f"{status} {odds_cat:<13} {row['profit_total']:>8.1f}u "
                    f"{int(row['bets']):>6} {row['win_rate']:>6.1f}% "
                    f"{row['roi']:>6.1f}% {row['odds_avg']:>8.2f}"
                )

        return odds_analysis

    def analyze_markets_with_roi_filter(
        self, df: pd.DataFrame, min_roi: float, title: str = "MERCADOS FILTRADOS"
    ) -> pd.DataFrame:
        """Análise de mercados aplicando filtro de ROI mínimo"""
        print(f"\n" + "=" * 80)
        print(f"🎯 {title} - ROI MÍNIMO: {min_roi}%")
        print("=" * 80)

        # Filtrar por ROI mínimo
        df_filtered = df[df["estimated_roi"] >= min_roi].copy()

        print(f"\n📊 Impacto do Filtro ROI ≥ {min_roi}%:")
        print(f"   Apostas originais: {len(df)}")
        print(
            f"   Apostas filtradas: {len(df_filtered)} ({len(df_filtered) / len(df) * 100:.1f}%)"
        )
        print(f"   Lucro sem filtro: {df['profit'].sum():.1f}u")
        print(f"   Lucro com filtro: {df_filtered['profit'].sum():.1f}u")
        print(f"   ROI sem filtro: {(df['profit'].sum() / len(df) * 100):.1f}%")
        print(
            f"   ROI com filtro: {(df_filtered['profit'].sum() / len(df_filtered) * 100):.1f}%"
        )

        if len(df_filtered) == 0:
            print("\n❌ Nenhuma aposta encontrada com este filtro de ROI")
            return pd.DataFrame()

        # Análise por mercado com filtro aplicado
        market_analysis = (
            df_filtered.groupby("grouped_market")
            .agg(
                {
                    "profit": ["sum", "count", "mean"],
                    "status": lambda x: (x == "win").mean() * 100,
                    "estimated_roi": "mean",
                    "odds": "mean",
                    "line_value": "mean",
                }
            )
            .round(2)
        )

        market_analysis.columns = [
            "profit_total",
            "bets",
            "profit_avg",
            "win_rate",
            "roi_est",
            "odds_avg",
            "line_avg",
        ]
        market_analysis["roi_real"] = (
            market_analysis["profit_total"] / market_analysis["bets"] * 100
        ).round(2)

        # Ordenar por lucro total
        market_analysis = market_analysis.sort_values("profit_total", ascending=False)

        print(
            f"\n{'Mercado':<20} {'Lucro':<10} {'Apostas':<8} {'Win%':<8} "
            f"{'ROI Real':<10} {'ROI Est':<10} {'Odds':<8} {'Linha'}"
        )
        print("-" * 90)

        for market, row in market_analysis.iterrows():
            if row["bets"] >= 3:  # Mínimo ainda menor para filtros restritivos
                status = (
                    "🏆"
                    if row["profit_total"] > 5
                    else "✅"
                    if row["profit_total"] > 0
                    else "❌"
                )
                print(
                    f"{status} {market:<18} {row['profit_total']:>8.1f}u "
                    f"{int(row['bets']):>6} {row['win_rate']:>6.1f}% "
                    f"{row['roi_real']:>8.1f}% {row['roi_est']:>8.1f}% "
                    f"{row['odds_avg']:>6.2f} {row['line_avg']:>6.1f}"
                )

        return market_analysis

    def generate_strategy_report(
        self, df: pd.DataFrame, lookback_periods: List[int] = None
    ) -> None:
        """Gera relatório estratégico completo"""
        if lookback_periods is None:
            lookback_periods = [100, 200, 500]

        print("\n" + "=" * 80)
        print("📋 RELATÓRIO ESTRATÉGICO COMPLETO")
        print("=" * 80)

        # Análise para diferentes períodos
        results = {}

        for period in [len(df)] + lookback_periods:
            if period <= len(df):
                df_period = df.head(period) if period < len(df) else df
                period_name = (
                    f"Últimas {period} apostas"
                    if period < len(df)
                    else "Todos os dados"
                )

                print(f"\n\n{'=' * 60}")
                print(f"📅 {period_name.upper()}")
                print(f"{'=' * 60}")

                # Análises
                overall = self.analyze_overall_performance(df_period)
                markets = self.analyze_markets(df_period, period_name)
                roi_eff = self.analyze_roi_efficiency(df_period)
                odds_perf = self.analyze_odds_performance(df_period)

                results[period] = {
                    "overall": overall,
                    "markets": markets,
                    "roi_efficiency": roi_eff,
                    "odds_performance": odds_perf,
                    "df_period": df_period,  # Guardar o dataframe para análise filtrada
                }

        # Aplicar análise com filtro de ROI ótimo
        self._analyze_with_optimal_roi_filter(results)

        # Recomendações finais
        self._print_final_recommendations(results, df)

    def _analyze_with_optimal_roi_filter(self, results: Dict) -> None:
        """Aplica análise com filtro de ROI ótimo identificado"""
        print("\n" + "=" * 80)
        print("🔍 ANÁLISE COM FILTRO DE ROI ÓTIMO")
        print("=" * 80)

        # Pegar o melhor ROI de todos os dados
        all_data_roi = results[list(results.keys())[0]]["roi_efficiency"]
        best_roi_category = all_data_roi["best_roi_category"]

        if best_roi_category and "+" in best_roi_category:
            optimal_roi = int(best_roi_category.replace("+", ""))

            print(f"\n🎯 ROI ótimo identificado: {optimal_roi}%+")
            print("Aplicando filtro para análise detalhada...")

            # Aplicar para todos os dados e período recente
            for period_key in [
                list(results.keys())[0],
                min(200, list(results.keys())[0]),
            ]:
                if period_key in results:
                    df_period = results[period_key]["df_period"]
                    period_name = (
                        "Todos os dados"
                        if period_key == len(df_period)
                        else f"Últimas {period_key} apostas"
                    )

                    self.analyze_markets_with_roi_filter(
                        df_period,
                        optimal_roi,
                        f"ANÁLISE FILTRADA - {period_name.upper()}",
                    )

    def _print_final_recommendations(self, results: Dict, df: pd.DataFrame) -> None:
        """Imprime recomendações estratégicas finais"""
        print("\n" + "=" * 80)
        print("🚀 RECOMENDAÇÕES ESTRATÉGICAS FINAIS")
        print("=" * 80)

        # Identificar padrões consistentes
        all_data_results = results[len(df)]
        recent_results = results.get(200, results.get(100))

        if recent_results:
            # Mercados consistentemente lucrativos
            all_markets = all_data_results["markets"]
            recent_markets = recent_results["markets"]

            profitable_all = set(all_markets[all_markets["profit_total"] > 0].index)
            profitable_recent = set(
                recent_markets[recent_markets["profit_total"] > 0].index
            )

            consistent_markets = profitable_all.intersection(profitable_recent)

            if consistent_markets:
                print(f"\n🎯 MERCADOS CONSISTENTEMENTE LUCRATIVOS:")
                for market in sorted(consistent_markets):
                    all_roi = all_markets.loc[market, "roi_real"]
                    recent_roi = recent_markets.loc[market, "roi_real"]
                    print(
                        f"   • {market}: {all_roi:.1f}% (histórico) | {recent_roi:.1f}% (recente)"
                    )

            # Melhor faixa de ROI
            best_roi_all = all_data_results["roi_efficiency"]["best_roi_category"]
            best_roi_recent = recent_results["roi_efficiency"]["best_roi_category"]

            print(f"\n📊 FAIXAS DE ROI RECOMENDADAS:")
            print(f"   • Histórico: {best_roi_all or 'Nenhuma identificada'}")
            print(f"   • Recente: {best_roi_recent or 'Nenhuma identificada'}")

            # Odds recomendadas
            print(f"\n🎲 FAIXAS DE ODDS LUCRATIVAS:")
            odds_all = all_data_results["odds_performance"]
            odds_recent = recent_results["odds_performance"]

            for odds_cat in odds_all.index:
                if odds_cat in odds_recent.index:
                    if (
                        odds_all.loc[odds_cat, "profit_total"] > 0
                        and odds_recent.loc[odds_cat, "profit_total"] > 0
                    ):
                        print(f"   • {odds_cat}: Consistentemente lucrativa")

        print(f"\n💡 ESTRATÉGIA FINAL SUGERIDA:")
        print(f"   1. Focar nos mercados consistentemente lucrativos listados")
        print(f"   2. Usar filtro de ROI baseado na análise de eficiência")
        print(f"   3. Manter-se nas faixas de odds que demonstram lucro consistente")
        print(f"   4. Monitorar tendências recentes vs. históricas para ajustes")


def main():
    """Função principal otimizada"""
    # Configurações
    config = AnalysisConfig(
        min_bets_for_roi=20,
        min_bets_for_market=5,
        roi_ranges=[10, 15, 20, 25, 30],
        odds_ranges=[
            (0, 1.6, "baixa"),
            (1.6, 2.0, "media"),
            (2.0, 2.5, "media_alta"),
            (2.5, 3.0, "alta"),
            (3.0, float("inf"), "muito_alta"),
        ],
    )

    # Inicializar analisador
    analyzer = CleanBettingAnalyzer(config)

    # Caminho do arquivo
    file_path = "../bets/bets_atualizadas_por_mapa.csv"

    print("🚀 INICIANDO ANÁLISE COMPLETA DE APOSTAS")
    print("=" * 60)

    try:
        # Carregar dados
        df = analyzer.load_data(file_path)

        # Gerar relatório estratégico completo
        analyzer.generate_strategy_report(df, lookback_periods=[100, 200, 500])

        print("\n" + "=" * 80)
        print("✅ ANÁLISE CONCLUÍDA COM SUCESSO!")
        print("=" * 80)

    except FileNotFoundError:
        print(f"❌ Erro: Arquivo não encontrado em {file_path}")
    except Exception as e:
        print(f"❌ Erro durante análise: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
