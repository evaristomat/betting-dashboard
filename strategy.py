import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuração REFINADA - Apenas mercados comprovadamente lucrativos"""

    # ROI ranges baseados na análise real
    ALLOWED_ROI_RANGES = [
        "≥40%",  # 8.2u lucro, 24.7% ROI real
        "25-30%",  # 8.6u lucro, 48.0% ROI real
        "20-25%",  # 3.5u lucro, 9.1% ROI real
    ]

    # Mercados LUCRATIVOS baseados na análise real
    ALLOWED_MARKETS = [
        "OVER - KILLS",  # 11.3u | 38.8% ROI | 75.9% Win Rate - 🏆 MELHOR
        "UNDER - INHIBITORS",  # 4.8u  | 13.0% ROI | 62.2% Win Rate - 💎 SÓLIDO
        "UNDER - DRAGONS",  # 3.6u  | 15.8% ROI | 60.9% Win Rate - ✅ BOM
        "UNDER - KILLS",  # 0.8u  | 6.8%  ROI | 58.3% Win Rate - ✅ MARGINAL
        "UNDER - TOWERS",  # Mantido conforme solicitado
        "OVER - INHIBITORS",  # 0.5u  | 4.3%  ROI | 58.3% Win Rate - ✅ MARGINAL
    ]

    # Odds lucrativas baseadas na análise
    ALLOWED_ODDS = ["media"]  # Única categoria com lucro positivo (5.9u)

    # Direção preferencial baseada nos dados
    PREFERRED_DIRECTION = "OVER"  # 2.8u vs -0.1u UNDER

    # Mercados PROIBIDOS (removidos conforme solicitado)
    FORBIDDEN_MARKETS = [
        "OVER - DRAGONS",  # Removido conforme solicitado
        "UNDER - BARONS",  # Removido conforme solicitado
        "OVER - BARONS",  # Removido conforme solicitado
        "UNDER - DURATION",  # Removido conforme solicitado
        "OVER - DURATION",  # Removido conforme solicitado
        "OVER - TOWERS",  # Removido conforme solicitado
    ]

    # Odds problemáticas
    FORBIDDEN_ODDS = ["muito_alta", "muito_baixa", "alta"]

    # Performance esperada baseada nos dados reais
    EXPECTED_ROI = 15.0  # Estimativa conservadora
    EXPECTED_PROFIT = 20.0  # Estimativa conservadora


class BettingStrategyAnalyzer:
    """Analisador focado apenas em mercados lucrativos"""

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
        """Aplica filtros da estratégia focada em mercados lucrativos"""
        if df.empty:
            return df.copy()

        if self.verbose:
            print(
                f"🎯 ESTRATÉGIA REFINADA - Foco em mercados comprovadamente lucrativos"
            )
            print(f"   📊 Dados originais: {len(df)} apostas")

        df_filtered = df.copy()

        # Filtro 1: ROI ranges lucrativos
        roi_condition = df_filtered["est_roi_category"].isin(
            self.config.ALLOWED_ROI_RANGES
        )
        df_filtered = df_filtered[roi_condition]
        if self.verbose:
            print(
                f"   ✅ Após filtro ROI (≥40%, 25-30%, 20-25%): {len(df_filtered)} apostas"
            )

        # Filtro 2: Apenas odds "media" (única lucrativa)
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(self.config.ALLOWED_ODDS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro odds (apenas media): {len(df_filtered)} apostas")

        # Filtro 3: Apenas mercados permitidos
        df_filtered = df_filtered[
            df_filtered["grouped_market"].isin(self.config.ALLOWED_MARKETS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro mercados lucrativos: {len(df_filtered)} apostas")

        # Filtro 4: Remover mercados proibidos (garantia extra)
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
                "OVER - KILLS": 10,  # 11.3u - MELHOR ABSOLUTO
                "UNDER - INHIBITORS": 8,  # 4.8u - SEGUNDO MELHOR
                "UNDER - DRAGONS": 6,  # 3.6u - TERCEIRO
                "UNDER - KILLS": 4,  # 0.8u - MARGINAL
                "UNDER - TOWERS": 4,  # Mantido
                "OVER - INHIBITORS": 2,  # 0.5u - MARGINAL
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
            stats["over_kills_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "OVER - KILLS"]
            )
            stats["under_inhibitors_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "UNDER - INHIBITORS"]
            )
            stats["under_dragons_count"] = len(
                df_filtered[df_filtered["grouped_market"] == "UNDER - DRAGONS"]
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
        print(f"\n📈 RESULTADOS DA ESTRATÉGIA REFINADA:")
        print(f"   🎯 Taxa de aprovação: {stats['approval_rate']:.1f}%")
        print(f"   💰 ROI médio estimado: {stats['avg_estimated_roi']:.1f}%")

        if "direction_breakdown" in stats:
            under_count = stats["direction_breakdown"].get("UNDER", 0)
            over_count = stats["direction_breakdown"].get("OVER", 0)
            print(f"   🔽 UNDER: {under_count} | OVER: {over_count}")

        if stats.get("over_kills_count", 0) > 0:
            print(
                f"   🏆 OVER-KILLS encontradas: {stats['over_kills_count']} (MELHOR MERCADO - 11.3u, 38.8% ROI)"
            )

        if stats.get("under_inhibitors_count", 0) > 0:
            print(
                f"   💎 UNDER-INHIBITORS encontradas: {stats['under_inhibitors_count']} (4.8u, 13.0% ROI)"
            )

        if stats.get("under_dragons_count", 0) > 0:
            print(
                f"   🐲 UNDER-DRAGONS encontradas: {stats['under_dragons_count']} (3.6u, 15.8% ROI)"
            )

        print(f"\n   ✅ Estratégia aplicada com sucesso - Foco em mercados lucrativos!")


# Funções de conveniência - MANTIDAS PARA COMPATIBILIDADE
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
    """Resumo da estratégia atualizada"""
    config = StrategyConfig()

    return {
        "name": "Estratégia Refinada - Apenas Mercados Lucrativos",
        "version": "v9.0_FOCUSED",
        "expected_roi": config.EXPECTED_ROI,
        "focus": "Apenas mercados com histórico comprovado de lucro",
        "criteria": {
            "roi_ranges": config.ALLOWED_ROI_RANGES,
            "markets": config.ALLOWED_MARKETS,
            "odds": config.ALLOWED_ODDS,
            "preferred_direction": config.PREFERRED_DIRECTION,
            "excluded": {
                "odds": config.FORBIDDEN_ODDS,
                "markets": config.FORBIDDEN_MARKETS,
            },
        },
        "market_priority": {
            "1": "OVER - KILLS (11.3u, 38.8% ROI)",
            "2": "UNDER - INHIBITORS (4.8u, 13.0% ROI)",
            "3": "UNDER - DRAGONS (3.6u, 15.8% ROI)",
            "4": "UNDER - KILLS / UNDER - TOWERS (marginais)",
            "5": "OVER - INHIBITORS (0.5u, marginal)",
        },
        "removed_markets": [
            "OVER - DRAGONS",
            "UNDER/OVER - BARONS",
            "UNDER/OVER - DURATION",
            "OVER - TOWERS",
        ],
        "strategy_principles": {
            "1": "Foco absoluto em mercados lucrativos",
            "2": "ROI ranges comprovados (≥40%, 25-30%, 20-25%)",
            "3": "Apenas odds 'media' (1.6-2.0)",
            "4": "Prioridade por OVER-KILLS",
            "5": "Volume reduzido, qualidade aumentada",
        },
    }


if __name__ == "__main__":
    print("✅ Estratégia REFINADA v9.0 - Apenas Mercados Lucrativos!")
    print("🎯 Foco: OVER-KILLS, UNDER-INHIBITORS, UNDER-DRAGONS")
    print("🚫 Removidos: OVER-DRAGONS, BARONS, DURATION, OVER-TOWERS")
    print("📊 ROI esperado: 15-20% com volume reduzido")
    print("💎 Princípio: Qualidade > Quantidade")
