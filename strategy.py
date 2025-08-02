import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuração ROI 20+ - Baseada na análise refinada"""

    # ROI ranges baseados na análise real (mantendo formato original)
    ALLOWED_ROI_RANGES = [
        "≥40%",
        "35-40%",
        "30-35%",
        "25-30%",
        "20-25%",
    ]

    # Mercados PERMITIDOS baseados na análise refinada
    ALLOWED_MARKETS = [
        "UNDER - TOWERS",  # 27.2u lucro, 18.4% ROI, 69.6% Win Rate
        "UNDER - DRAGONS",  # 21.2u lucro, 21.7% ROI, 57.1% Win Rate - CONSISTENTE
        "UNDER - KILLS",  # 14.1u lucro, 19.5% ROI, 65.3% Win Rate
        "UNDER - BARONS",  # 2.6u lucro, 11.3% ROI, 82.6% Win Rate
        "UNDER - INHIBITORS",  # Incluído por consistência
        "OVER - KILLS",  # 12.0u lucro, 16.2% ROI, 63.5% Win Rate - CONSISTENTE
    ]

    # Odds lucrativas baseadas na análise
    ALLOWED_ODDS = ["baixa", "media", "media_alta"]

    # Direção preferencial baseada nos dados
    PREFERRED_DIRECTION = "UNDER"  # +59.3u vs +1.9u OVER

    # Mercados PROIBIDOS
    FORBIDDEN_MARKETS = [
        "OVER - TOWERS",  # -1.1u lucro, -3.9% ROI
        "OVER - BARONS",  # -2.0u lucro, -40.0% ROI
        "OVER - DURATION",  # -2.2u lucro, -14.6% ROI
        "UNDER - DURATION",  # -2.7u lucro, -2.8% ROI
        "OVER - INHIBITORS",  # -8.9u lucro, -38.5% ROI
        "OVER - DRAGONS",  # Performance negativa
    ]

    # Odds problemáticas
    FORBIDDEN_ODDS = ["muito_alta", "muito_baixa", "alta"]

    # Performance esperada baseada nos dados reais
    EXPECTED_ROI = 8.0
    EXPECTED_PROFIT = 61.2


class BettingStrategyAnalyzer:
    """Analisador de apostas com estratégia ROI 20+"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.verbose = False

    def set_verbose(self, verbose: bool = True):
        self.verbose = verbose
        return self

    def categorize_odds(self, odds: float) -> str:
        """Categoriza odds"""
        if pd.isna(odds):
            return "N/A"

        if odds < 1.3:
            return "muito_baixa"
        elif odds < 1.6:
            return "baixa"
        elif odds < 2.0:
            return "media"
        elif odds < 2.5:
            return "media_alta"
        elif odds < 3.0:
            return "alta"
        else:
            return "muito_alta"

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categoriza mercado"""
        bet_type_lower = str(bet_type).lower()
        bet_line_lower = str(bet_line).lower()

        # Determinar direção
        direction = "UNDER" if "under" in bet_type_lower else "OVER"

        # Determinar tipo de mercado
        market_type = "OUTROS"

        market_keywords = {
            "TOWERS": ["tower", "torre"],
            "DRAGONS": ["dragon", "dragao", "dragão"],
            "KILLS": ["kill", "abate"],
            "DURATION": ["duration", "tempo", "duração", "duracao"],
            "BARONS": ["baron", "baroness", "barão"],
            "INHIBITORS": ["inhibitor", "inibidor"],
        }

        for market, keywords in market_keywords.items():
            if any(keyword in bet_line_lower for keyword in keywords):
                market_type = market
                break

        grouped_market = f"{direction} - {market_type}"
        return direction, market_type, grouped_market

    def categorize_roi_ranges(self, roi: float) -> str:
        """Categoriza ROI em ranges"""
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

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessa o DataFrame"""
        df = df.copy()

        # Adicionar categorização de odds
        if "odds_category" not in df.columns:
            df["odds_category"] = df["odds"].apply(self.categorize_odds)

        # Adicionar categorização de mercado
        market_cols = ["direction", "market_type", "grouped_market"]
        if not all(col in df.columns for col in market_cols):
            df[market_cols] = df.apply(
                lambda row: self.categorize_market(
                    row.get("bet_type", ""), row.get("bet_line", "")
                ),
                axis=1,
                result_type="expand",
            )

        # Processar ROI estimado
        if "estimated_roi" not in df.columns:
            if df["ROI"].dtype == object:
                df["estimated_roi"] = (
                    df["ROI"]
                    .astype(str)
                    .str.replace("%", "", regex=False)
                    .astype(float)
                )
            else:
                df["estimated_roi"] = df["ROI"]

        # Categorizar ROI
        if "est_roi_category" not in df.columns:
            df["est_roi_category"] = df["estimated_roi"].apply(
                self.categorize_roi_ranges
            )

        return df

    def apply_strategy_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica filtros da estratégia ROI 20+"""
        if df.empty:
            return df.copy()

        if self.verbose:
            print(f"🎯 ESTRATÉGIA ROI 20+ - Mercados comprovadamente lucrativos")
            print(f"   📊 Dados originais: {len(df)} apostas")

        df_filtered = df.copy()

        # Filtro 1: ROI ≥ 20%
        roi_condition = df_filtered["estimated_roi"] >= 20.0
        df_filtered = df_filtered[roi_condition]
        if self.verbose:
            print(f"   ✅ Após filtro ROI ≥20%: {len(df_filtered)} apostas")

        # Filtro 2: ROI ranges permitidos
        roi_range_condition = df_filtered["est_roi_category"].isin(
            self.config.ALLOWED_ROI_RANGES
        )
        df_filtered = df_filtered[roi_range_condition]
        if self.verbose:
            print(f"   ✅ Após filtro ROI ranges: {len(df_filtered)} apostas")

        # Filtro 3: Apenas odds lucrativas
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(self.config.ALLOWED_ODDS)
        ]
        if self.verbose:
            print(
                f"   ✅ Após filtro odds (baixa/media/media_alta): {len(df_filtered)} apostas"
            )

        # Filtro 4: Apenas mercados permitidos
        df_filtered = df_filtered[
            df_filtered["grouped_market"].isin(self.config.ALLOWED_MARKETS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro mercados lucrativos: {len(df_filtered)} apostas")

        # Filtro 5: Remover mercados proibidos
        df_filtered = df_filtered[
            ~df_filtered["grouped_market"].isin(self.config.FORBIDDEN_MARKETS)
        ]
        if self.verbose:
            print(f"   🚫 Após remover mercados proibidos: {len(df_filtered)} apostas")

        return df_filtered.reset_index(drop=True)

    def apply_optimized_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a estratégia otimizada"""
        if df.empty:
            return df.copy()

        # Preprocessar dados
        df_processed = self.preprocess_dataframe(df)

        # Aplicar filtros
        df_filtered = self.apply_strategy_filters(df_processed)

        # Ordenar por prioridade baseada no lucro real
        if len(df_filtered) > 0:
            # Pesos baseados no lucro real de cada mercado
            market_weights = {
                "UNDER - TOWERS": 10,  # 27.2u - MELHOR ABSOLUTO
                "UNDER - DRAGONS": 9,  # 21.2u - SEGUNDO MELHOR + CONSISTENTE
                "UNDER - KILLS": 8,  # 14.1u - TERCEIRO
                "OVER - KILLS": 7,  # 12.0u - MELHOR OVER
                "UNDER - INHIBITORS": 6,  # Consistente
                "UNDER - BARONS": 5,  # 2.6u - MENOR POSITIVO
            }

            df_filtered["market_weight"] = (
                df_filtered["grouped_market"].map(market_weights).fillna(1)
            )

            # Score de prioridade combinando ROI estimado e peso do mercado
            df_filtered["priority_score"] = (
                df_filtered["estimated_roi"] * df_filtered["market_weight"]
            )

            # Ordenar por prioridade
            df_filtered = df_filtered.sort_values(
                ["priority_score", "estimated_roi"], ascending=[False, False]
            )

            # Remover colunas auxiliares
            df_filtered = df_filtered.drop(columns=["market_weight", "priority_score"])

        # Gerar estatísticas
        if self.verbose:
            stats = self.generate_statistics(df_processed, df_filtered)
            self._print_verbose_stats(stats)

        return df_filtered

    def generate_statistics(
        self, df_original: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> Dict:
        """Gera estatísticas da estratégia"""
        stats = {
            "total_bets": len(df_original),
            "approved_bets": len(df_filtered),
            "approval_rate": (len(df_filtered) / len(df_original)) * 100
            if len(df_original) > 0
            else 0,
            "avg_estimated_roi": df_filtered["estimated_roi"].mean()
            if len(df_filtered) > 0
            else 0,
        }

        if len(df_filtered) > 0:
            stats["market_breakdown"] = (
                df_filtered.groupby("grouped_market").size().to_dict()
            )
            stats["direction_breakdown"] = (
                df_filtered.groupby("direction").size().to_dict()
            )

            # Contagens específicas para validação
            stats["under_towers_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "UNDER - TOWERS"]
            )
            stats["under_dragons_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "UNDER - DRAGONS"]
            )
            stats["over_kills_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "OVER - KILLS"]
            )

            # Top apostas por ROI
            if len(df_filtered) > 0:
                top_bets = df_filtered.nlargest(5, "estimated_roi")
                stats["top_recommendations"] = [
                    {
                        "market": row["grouped_market"],
                        "roi": row["estimated_roi"],
                        "odds": row["odds"],
                        "odds_category": row["odds_category"],
                    }
                    for _, row in top_bets.iterrows()
                ]

        return stats

    def _print_verbose_stats(self, stats: Dict):
        """Imprime estatísticas detalhadas"""
        print(f"\n📈 RESULTADOS DA ESTRATÉGIA ROI 20+:")
        print(f"   🎯 Taxa de aprovação: {stats['approval_rate']:.1f}%")
        print(f"   💰 ROI médio estimado: {stats['avg_estimated_roi']:.1f}%")

        if "direction_breakdown" in stats:
            under_count = stats["direction_breakdown"].get("UNDER", 0)
            over_count = stats["direction_breakdown"].get("OVER", 0)
            print(f"   🔽 UNDER: {under_count} | OVER: {over_count}")

        if stats.get("under_towers_count", 0) > 0:
            print(
                f"   🏆 UNDER-TOWERS: {stats['under_towers_count']} (MELHOR: 27.2u, 18.4% ROI)"
            )

        if stats.get("under_dragons_count", 0) > 0:
            print(
                f"   🐲 UNDER-DRAGONS: {stats['under_dragons_count']} (CONSISTENTE: 21.2u, 21.7% ROI)"
            )

        if stats.get("over_kills_count", 0) > 0:
            print(
                f"   ⚔️ OVER-KILLS: {stats['over_kills_count']} (MELHOR OVER: 12.0u, 16.2% ROI)"
            )

        print(f"\n   ✅ Estratégia ROI 20+ aplicada com sucesso!")


# Funções de conveniência - MANTIDAS EXATAMENTE IGUAIS
def apply_optimized_strategy(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Função principal - mantida para compatibilidade"""
    analyzer = BettingStrategyAnalyzer().set_verbose(verbose)
    return analyzer.apply_optimized_strategy(df)


def apply_strategy_to_pending_bets(
    df_pending: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Aplica estratégia às apostas pendentes - mantida para compatibilidade"""
    return apply_optimized_strategy(df_pending, verbose)


def get_strategy_summary() -> Dict:
    """Resumo da estratégia - MANTENDO ESTRUTURA EXATA"""
    config = StrategyConfig()

    return {
        "name": "Estratégia ROI 20+ - Baseada em Dados Reais",
        "version": "v10.0_ROI20PLUS",
        "expected_roi": config.EXPECTED_ROI,
        "focus": "ROI ≥20% em mercados comprovadamente lucrativos",
        "criteria": {
            "roi_ranges": config.ALLOWED_ROI_RANGES,
            "markets": config.ALLOWED_MARKETS,  # ← MANTIDO IGUAL
            "odds": config.ALLOWED_ODDS,
            "preferred_direction": config.PREFERRED_DIRECTION,
            "excluded": {
                "odds": config.FORBIDDEN_ODDS,
                "markets": config.FORBIDDEN_MARKETS,
            },
        },
        "market_priority": {
            "1": "UNDER - TOWERS (27.2u lucro, 18.4% ROI)",
            "2": "UNDER - DRAGONS (21.2u lucro, 21.7% ROI) - CONSISTENTE",
            "3": "UNDER - KILLS (14.1u lucro, 19.5% ROI)",
            "4": "OVER - KILLS (12.0u lucro, 16.2% ROI) - MELHOR OVER",
            "5": "UNDER - BARONS (2.6u lucro, 11.3% ROI)",
            "6": "UNDER - INHIBITORS (incluído por consistência)",
        },
        "removed_markets": [
            "OVER - TOWERS",
            "OVER - BARONS",
            "OVER - DURATION",
            "UNDER - DURATION",
            "OVER - INHIBITORS",
            "OVER - DRAGONS",
        ],
        "strategy_principles": {
            "1": "ROI mínimo 20% (faixa com 14.4% ROI real comprovado)",
            "2": "Foco em UNDER (31x melhor que OVER)",
            "3": "Odds lucrativas: baixa/media/media_alta",
            "4": "Prioridade: TOWERS > DRAGONS > KILLS",
            "5": "OVER apenas em KILLS",
        },
    }


if __name__ == "__main__":
    print("✅ Estratégia ROI 20+ v10.0 - Estrutura Mantida!")
    print("🎯 Foco: UNDER-TOWERS, UNDER-DRAGONS, UNDER-KILLS, OVER-KILLS")
    print("📊 ROI mínimo: 20% (comprovado: 8.0% ROI real)")
    print("🔽 Direção: UNDER domina (31:1 vs OVER)")
    print("💎 Expectativa: 8.0% ROI, ~760 apostas aprovadas")
