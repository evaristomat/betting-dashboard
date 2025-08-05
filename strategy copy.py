import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class StrategyConfig:
    """Configuração da Estratégia ROI 20+ - Baseada em análise de dados reais"""

    # Nome e versão da estratégia
    name: str = "Estratégia ROI 20+ - Mercados Comprovados"
    version: str = "v11.0_ROI20_CLEAN"

    # Filtros de ROI
    min_roi: float = 20.0
    roi_ranges: List[str] = field(
        default_factory=lambda: ["≥40%", "35-40%", "30-35%", "25-30%", "20-25%"]
    )

    # Mercados aprovados com suas métricas reais
    allowed_markets: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "UNDER - TOWERS": {
                "lucro": 27.2,
                "roi": 18.4,
                "win_rate": 69.6,
                "prioridade": 10,
            },
            "UNDER - DRAGONS": {
                "lucro": 21.2,
                "roi": 21.7,
                "win_rate": 57.1,
                "prioridade": 9,
            },
            "UNDER - KILLS": {
                "lucro": 14.1,
                "roi": 19.5,
                "win_rate": 65.3,
                "prioridade": 8,
            },
            "OVER - KILLS": {
                "lucro": 12.0,
                "roi": 16.2,
                "win_rate": 63.5,
                "prioridade": 7,
            },
            "UNDER - INHIBITORS": {
                "lucro": 0.0,
                "roi": 0.0,
                "win_rate": 50.0,
                "prioridade": 6,
            },  # Mantido conforme solicitado
            "UNDER - BARONS": {
                "lucro": 2.6,
                "roi": 11.3,
                "win_rate": 82.6,
                "prioridade": 5,
            },
        }
    )

    # Odds permitidas
    allowed_odds_ranges: List[str] = field(
        default_factory=lambda: ["baixa", "media", "media_alta"]
    )

    # Mercados e odds proibidos
    forbidden_markets: List[str] = field(
        default_factory=lambda: [
            "OVER - TOWERS",
            "OVER - BARONS",
            "OVER - DURATION",
            "UNDER - DURATION",
            "OVER - INHIBITORS",
            "OVER - DRAGONS",
        ]
    )

    forbidden_odds: List[str] = field(
        default_factory=lambda: ["muito_alta", "muito_baixa", "alta"]
    )

    # Métricas esperadas
    expected_roi: float = 8.0
    expected_profit: float = 61.2
    expected_approval_rate: float = 48.0


class BettingStrategyAnalyzer:
    """Analisador de apostas com estratégia ROI 20+ refinada"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.verbose = False
        self._stats = {}

    def set_verbose(self, verbose: bool = True) -> "BettingStrategyAnalyzer":
        """Ativa/desativa modo verbose"""
        self.verbose = verbose
        return self

    def categorize_odds(self, odds: float) -> str:
        """Categoriza odds em faixas padronizadas"""
        if pd.isna(odds):
            return "N/A"

        ranges = [
            (0, 1.3, "muito_baixa"),
            (1.3, 1.6, "baixa"),
            (1.6, 2.0, "media"),
            (2.0, 2.5, "media_alta"),
            (2.5, 3.0, "alta"),
            (3.0, float("inf"), "muito_alta"),
        ]

        for min_val, max_val, category in ranges:
            if min_val <= odds < max_val:
                return category

        return "muito_alta"

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categoriza mercado de forma padronizada"""
        bet_type_lower = str(bet_type).lower().strip()
        bet_line_lower = str(bet_line).lower().strip()

        # Direção
        direction = "UNDER" if "under" in bet_type_lower else "OVER"

        # Mapeamento de mercados
        market_map = {
            "TOWERS": ["tower", "torre"],
            "DRAGONS": ["dragon", "dragao", "dragão"],
            "KILLS": ["kill", "abate", "eliminação"],
            "DURATION": ["duration", "tempo", "duração", "duracao", "game_duration"],
            "BARONS": ["baron", "barão"],
            "INHIBITORS": ["inhibitor", "inibidor"],
        }

        # Identificar mercado
        market_type = "OUTROS"
        for market, keywords in market_map.items():
            if any(kw in bet_line_lower for kw in keywords):
                market_type = market
                break

        grouped_market = f"{direction} - {market_type}"
        return direction, market_type, grouped_market

    def categorize_roi_ranges(self, roi: float) -> str:
        """Categoriza ROI em faixas estratégicas"""
        if pd.isna(roi):
            return "N/A"

        ranges = [
            (40, float("inf"), "≥40%"),
            (35, 40, "35-40%"),
            (30, 35, "30-35%"),
            (25, 30, "25-30%"),
            (20, 25, "20-25%"),
            (15, 20, "15-20%"),
            (10, 15, "10-15%"),
            (0, 10, "<10%"),
        ]

        for min_val, max_val, category in ranges:
            if min_val <= roi < max_val:
                return category

        return "<10%"

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessa DataFrame com todas as categorizações necessárias"""
        df = df.copy()

        # Categorizar odds
        if "odds_category" not in df.columns:
            df["odds_category"] = df["odds"].apply(self.categorize_odds)

        # Categorizar mercados
        if not all(
            col in df.columns for col in ["direction", "market_type", "grouped_market"]
        ):
            df[["direction", "market_type", "grouped_market"]] = df.apply(
                lambda row: self.categorize_market(
                    row.get("bet_type", ""), row.get("bet_line", "")
                ),
                axis=1,
                result_type="expand",
            )

        # Processar ROI
        if "estimated_roi" not in df.columns:
            if "ROI" in df.columns:
                if df["ROI"].dtype == object:
                    df["estimated_roi"] = df["ROI"].str.rstrip("%").astype(float)
                else:
                    df["estimated_roi"] = df["ROI"]
            else:
                df["estimated_roi"] = 0.0

        # Categorizar ROI
        if "est_roi_category" not in df.columns:
            df["est_roi_category"] = df["estimated_roi"].apply(
                self.categorize_roi_ranges
            )

        return df

    def apply_strategy_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica todos os filtros da estratégia de forma organizada"""
        if df.empty:
            return df.copy()

        initial_count = len(df)
        df_filtered = df.copy()

        if self.verbose:
            print(f"\n🎯 APLICANDO ESTRATÉGIA {self.config.name}")
            print(f"📊 Apostas iniciais: {initial_count}")

        # Filtro 1: ROI mínimo
        df_filtered = df_filtered[df_filtered["estimated_roi"] >= self.config.min_roi]
        if self.verbose:
            print(
                f"✅ Filtro ROI ≥{self.config.min_roi}%: {len(df_filtered)} apostas ({len(df_filtered) / initial_count * 100:.1f}%)"
            )

        # Filtro 2: Faixas de ROI permitidas
        df_filtered = df_filtered[
            df_filtered["est_roi_category"].isin(self.config.roi_ranges)
        ]
        if self.verbose:
            print(
                f"✅ Filtro faixas ROI: {len(df_filtered)} apostas ({len(df_filtered) / initial_count * 100:.1f}%)"
            )

        # Filtro 3: Odds permitidas
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(self.config.allowed_odds_ranges)
        ]
        if self.verbose:
            print(
                f"✅ Filtro odds permitidas: {len(df_filtered)} apostas ({len(df_filtered) / initial_count * 100:.1f}%)"
            )

        # Filtro 4: Mercados permitidos
        allowed_market_names = list(self.config.allowed_markets.keys())
        df_filtered = df_filtered[
            df_filtered["grouped_market"].isin(allowed_market_names)
        ]
        if self.verbose:
            print(
                f"✅ Filtro mercados aprovados: {len(df_filtered)} apostas ({len(df_filtered) / initial_count * 100:.1f}%)"
            )

        # Filtro 5: Remover mercados proibidos (redundância intencional para garantia)
        df_filtered = df_filtered[
            ~df_filtered["grouped_market"].isin(self.config.forbidden_markets)
        ]

        # Armazenar estatísticas
        self._stats = {
            "initial_count": initial_count,
            "final_count": len(df_filtered),
            "approval_rate": len(df_filtered) / initial_count * 100
            if initial_count > 0
            else 0,
        }

        return df_filtered.reset_index(drop=True)

    def apply_priority_sorting(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica ordenação por prioridade baseada no desempenho real"""
        if df.empty:
            return df

        df = df.copy()

        # Adicionar prioridade do mercado
        df["market_priority"] = df["grouped_market"].map(
            lambda x: self.config.allowed_markets.get(x, {}).get("prioridade", 0)
        )

        # Score composto: prioridade do mercado * ROI estimado
        df["priority_score"] = df["market_priority"] * df["estimated_roi"]

        # Ordenar por score e ROI
        df = df.sort_values(
            ["priority_score", "estimated_roi", "market_priority"],
            ascending=[False, False, False],
        )

        # Remover colunas auxiliares
        df = df.drop(columns=["market_priority", "priority_score"], errors="ignore")

        return df

    def apply_optimized_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a estratégia completa otimizada"""
        if df.empty:
            return df.copy()

        # Pipeline de processamento
        df_processed = self.preprocess_dataframe(df)
        df_filtered = self.apply_strategy_filters(df_processed)
        df_sorted = self.apply_priority_sorting(df_filtered)

        # Gerar estatísticas finais
        if self.verbose:
            self._print_final_statistics(df_processed, df_sorted)

        return df_sorted

    def _print_final_statistics(
        self, df_original: pd.DataFrame, df_filtered: pd.DataFrame
    ):
        """Imprime estatísticas finais da estratégia"""
        if df_filtered.empty:
            print("\n❌ Nenhuma aposta aprovada pela estratégia")
            return

        print(f"\n📈 RESULTADOS DA ESTRATÉGIA:")
        print(f"   Taxa de aprovação: {self._stats['approval_rate']:.1f}%")
        print(f"   Apostas aprovadas: {len(df_filtered)}")
        print(f"   ROI médio estimado: {df_filtered['estimated_roi'].mean():.1f}%")

        # Breakdown por mercado
        market_counts = df_filtered["grouped_market"].value_counts()
        print(f"\n📊 Distribuição por mercado:")
        for market, count in market_counts.items():
            info = self.config.allowed_markets.get(market, {})
            print(f"   {market}: {count} apostas (ROI real: {info.get('roi', 0):.1f}%)")

        # Top 3 apostas
        print(f"\n🏆 Top 3 apostas por ROI:")
        for i, (_, bet) in enumerate(df_filtered.head(3).iterrows(), 1):
            print(
                f"   {i}. {bet['grouped_market']} - ROI: {bet['estimated_roi']:.1f}% - Odds: {bet['odds']:.2f}"
            )

    def generate_statistics(
        self, df_original: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> Dict:
        """Gera estatísticas detalhadas (mantido para compatibilidade)"""
        stats = {
            "total_bets": len(df_original),
            "approved_bets": len(df_filtered),
            "approval_rate": self._stats.get("approval_rate", 0),
            "avg_estimated_roi": df_filtered["estimated_roi"].mean()
            if len(df_filtered) > 0
            else 0,
        }

        if len(df_filtered) > 0:
            # Breakdown por mercado e direção
            stats["market_breakdown"] = (
                df_filtered.groupby("grouped_market").size().to_dict()
            )
            stats["direction_breakdown"] = (
                df_filtered.groupby("direction").size().to_dict()
            )

            # Contagens específicas
            for market in ["UNDER - TOWERS", "UNDER - DRAGONS", "OVER - KILLS"]:
                key = market.lower().replace(" - ", "_") + "_count"
                stats[key] = len(df_filtered[df_filtered["grouped_market"] == market])

            # Top recomendações
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


# Funções de conveniência - MANTIDAS PARA COMPATIBILIDADE
def apply_optimized_strategy(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Função principal para aplicar estratégia otimizada"""
    analyzer = BettingStrategyAnalyzer().set_verbose(verbose)
    return analyzer.apply_optimized_strategy(df)


def apply_strategy_to_pending_bets(
    df_pending: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Aplica estratégia às apostas pendentes"""
    return apply_optimized_strategy(df_pending, verbose)


def get_strategy_summary() -> Dict:
    """Retorna resumo completo da estratégia"""
    config = StrategyConfig()

    return {
        "name": config.name,
        "version": config.version,
        "expected_roi": config.expected_roi,
        "focus": f"ROI ≥{config.min_roi}% em mercados comprovadamente lucrativos",
        "criteria": {
            "roi_ranges": config.roi_ranges,
            "markets": list(config.allowed_markets.keys()),
            "odds": config.allowed_odds_ranges,
            "preferred_direction": "UNDER",
            "excluded": {
                "odds": config.forbidden_odds,
                "markets": config.forbidden_markets,
            },
        },
        "market_priority": {
            str(i + 1): f"{market} ({info['lucro']:.1f}u lucro, {info['roi']:.1f}% ROI)"
            for i, (market, info) in enumerate(
                sorted(
                    config.allowed_markets.items(),
                    key=lambda x: x[1]["prioridade"],
                    reverse=True,
                )
            )
        },
        "removed_markets": config.forbidden_markets,
        "strategy_principles": {
            "1": f"ROI mínimo {config.min_roi}% (comprovado: {config.expected_roi}% ROI real)",
            "2": "Foco em UNDER (31x melhor que OVER)",
            "3": "Odds lucrativas: baixa/media/media_alta",
            "4": "Prioridade: TOWERS > DRAGONS > KILLS",
            "5": "OVER apenas em KILLS",
        },
    }


if __name__ == "__main__":
    # Teste da estratégia
    print(f"✅ {StrategyConfig().name} - {StrategyConfig().version}")
    print(
        "🎯 Mercados principais: UNDER-TOWERS, UNDER-DRAGONS, UNDER-KILLS, OVER-KILLS"
    )
    print(
        f"📊 ROI mínimo: {StrategyConfig().min_roi}% (ROI real esperado: {StrategyConfig().expected_roi}%)"
    )
    print("🔽 Direção dominante: UNDER (31:1 vs OVER)")
    print(
        f"💰 Expectativa: {StrategyConfig().expected_roi}% ROI, ~{StrategyConfig().expected_approval_rate}% aprovação"
    )
