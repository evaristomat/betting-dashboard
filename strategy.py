import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuração SIMPLIFICADA - Foco no que REALMENTE importa"""

    # ROI ranges REFINADOS baseados na nova análise aprimorada
    ALLOWED_ROI_RANGES = [
        "20-25%",
        "15-20%",
        "≥40%",
        "10-15%",
    ]  # Apenas os lucrativos dentro dos mercados bons

    # Mercados REAIS ordenados por LUCRO REAL da sua análise
    ALLOWED_MARKETS = [
        "UNDER - KILLS",  # 15.1u | 55.9% ROI - 🏆 SEU MELHOR
        "UNDER - TOWERS",  # 10.1u | 21.5% ROI - 💎 SEGUNDO MELHOR
        "OVER - DRAGONS",  # 6.9u  | 31.4% ROI - ✅ ÚNICO OVER LUCRATIVO
        "UNDER - DURATION",  # 5.8u  | 14.6% ROI - ✅ SÓLIDO
        "UNDER - BARONS",  # 2.8u  | 11.2% ROI - ✅ BOM
        "UNDER - INHIBITORS",  # 2.6u  | 4.3% ROI - ✅ CONSISTENTE
        "UNDER - DRAGONS",  # 2.4u  | 9.8% ROI - ✅ ESTÁVEL
    ]

    # Odds: TODAS exceto muito_alta são válidas
    ALLOWED_ODDS = ["media", "media_alta", "baixa", "muito_baixa"]

    # FILTRO PRINCIPAL: Preferência ABSOLUTA por UNDER
    PREFERRED_DIRECTION = ["UNDER"]  # +38.8u vs -15.2u OVER

    # EXCEÇÃO: OVER só é permitido em DRAGONS (único OVER lucrativo)
    ALLOWED_OVER_MARKETS = ["OVER - DRAGONS"]  # 6.9u de lucro

    # Mercados para EVITAR (os únicos OVER com prejuízo)
    FORBIDDEN_MARKETS = [
        "OVER - DURATION",  # -1.0u | -100.0% ROI
        "OVER - BARONS",  # -3.5u | -14.0% ROI
        "OVER - INHIBITORS",  # -4.4u | -43.9% ROI
        "OVER - TOWERS",  # Não apareceu nos dados, mas evitar OVER em geral
        "OVER - KILLS",  # Não apareceu nos dados, mas evitar OVER em geral
    ]

    # Odds problemáticas
    FORBIDDEN_ODDS = ["muito_alta"]  # -3.5u | -14.0% ROI

    # Performance esperada REAL
    EXPECTED_ROI = 22.2  # ROI real da estratégia final (163 apostas)
    EXPECTED_PROFIT = 36.2  # Lucro real em units


class BettingStrategyAnalyzer:
    """Analisador SIMPLIFICADO - Foco em UNDER + OVER-DRAGONS"""

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

        if odds < 1.5:
            return "baixa"
        elif odds < 2.0:
            return "media"
        elif odds < 2.5:
            return "media_alta"
        else:
            return "muito_alta"

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categoriza mercado EXATAMENTE como na sua análise"""
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
        """Categoriza ROI com ranges refinados baseados na nova análise"""
        if pd.isna(roi):
            return "N/A"
        elif roi < 10:
            return "<10%"
        elif roi < 15:
            return "10-15%"
        elif roi < 20:
            return "15-20%"
        elif roi < 25:
            return "20-25%"  # SWEET SPOT identificado (+26.1u, 80% win rate)
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
        """Aplica filtros SIMPLIFICADOS - Foco em DIREÇÃO"""
        if df.empty:
            return df.copy()

        if self.verbose:
            print(f"🎯 ESTRATÉGIA SIMPLIFICADA - Foco em UNDER + OVER-DRAGONS")
            print(f"   📊 Dados originais: {len(df)} apostas")

        df_filtered = df.copy()

        # Filtro 1: ROI ranges REFINADOS (baseados na análise aprimorada)
        roi_condition = df_filtered["est_roi_category"].isin(
            self.config.ALLOWED_ROI_RANGES
        )
        df_filtered = df_filtered[roi_condition]
        if self.verbose:
            print(
                f"   ✅ Após filtro ROI otimizado (20-25%, 15-20%, ≥40%, 10-15%): {len(df_filtered)} apostas"
            )

        # Filtro 2: Odds REFINADAS (baseadas na nova análise)
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(["media", "baixa", "media_alta"])
        ]
        if self.verbose:
            print(
                f"   ✅ Após filtro odds (media, baixa, media_alta): {len(df_filtered)} apostas"
            )

        # Filtro 3: ESTRATÉGIA PRINCIPAL - Direção
        # Pegar TODOS os UNDER + apenas OVER-DRAGONS
        under_condition = df_filtered["direction"] == "UNDER"
        over_dragons_condition = df_filtered["grouped_market"] == "OVER - DRAGONS"

        direction_condition = under_condition | over_dragons_condition
        df_filtered = df_filtered[direction_condition]

        if self.verbose:
            under_count = len(df_filtered[df_filtered["direction"] == "UNDER"])
            over_dragons_count = len(
                df_filtered[df_filtered["grouped_market"] == "OVER - DRAGONS"]
            )
            print(f"   🔽 UNDER: {under_count} | OVER-DRAGONS: {over_dragons_count}")

        # Filtro 4: Remover apenas os OVER problemáticos específicos
        df_filtered = df_filtered[
            ~df_filtered["grouped_market"].isin(self.config.FORBIDDEN_MARKETS)
        ]
        if self.verbose:
            print(f"   🚫 Após remover OVER problemáticos: {len(df_filtered)} apostas")

        return df_filtered.reset_index(drop=True)

    def apply_optimized_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica a estratégia SIMPLIFICADA"""
        if df.empty:
            return df.copy()

        # Preprocessar dados
        df_processed = self.preprocess_dataframe(df)

        # Aplicar filtros
        df_filtered = self.apply_strategy_filters(df_processed)

        # Ordenar por prioridade REAL
        if len(df_filtered) > 0:
            # Pesos baseados na NOVA ANÁLISE REAL
            market_weights = {
                "UNDER - KILLS": 10,  # 27.8u - MELHOR ABSOLUTO (39.2% ROI, 76.1% WR)
                "UNDER - TOWERS": 8,  # 17.3u - SEGUNDO LUGAR (18.0% ROI, 69.8% WR)
                "UNDER - DRAGONS": 6,  # 7.8u - TERCEIRO (11.7% ROI, 55.2% WR)
                "OVER - DRAGONS": 4,  # 2.2u - ÚNICO OVER BOM (6.7% ROI, 51.5% WR)
                "UNDER - DURATION": 1,  # -1.2u - EVITAR (mas pode ter casos bons)
                "UNDER - BARONS": 1,  # -1.9u - EVITAR (mas pode ter casos bons)
            }

            df_filtered["market_weight"] = (
                df_filtered["grouped_market"].map(market_weights).fillna(0)
            )
            df_filtered["priority_score"] = (
                df_filtered["estimated_roi"] * df_filtered["market_weight"]
            )

            df_filtered = df_filtered.sort_values(
                ["priority_score", "estimated_roi"], ascending=[False, False]
            )
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
            stats["kills_count"] = len(
                df_filtered[df_filtered["market_type"] == "KILLS"]
            )
            stats["towers_count"] = len(
                df_filtered[df_filtered["market_type"] == "TOWERS"]
            )
            stats["dragons_count"] = len(
                df_filtered[df_filtered["market_type"] == "DRAGONS"]
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
        """Imprime estatísticas"""
        print(f"\n📈 RESULTADOS DA ESTRATÉGIA SIMPLIFICADA:")
        print(f"   🎯 Taxa de aprovação: {stats['approval_rate']:.1f}%")
        print(f"   💰 ROI médio estimado: {stats['avg_estimated_roi']:.1f}%")

        if "direction_breakdown" in stats:
            under_count = stats["direction_breakdown"].get("UNDER", 0)
            over_count = stats["direction_breakdown"].get("OVER", 0)
            print(f"   🔽 UNDER: {under_count} | OVER: {over_count}")

        if stats.get("kills_count", 0) > 0:
            print(
                f"   🏆 KILLS encontradas: {stats['kills_count']} (MELHOR MERCADO - 27.8u, 39.2% ROI)"
            )

        if stats.get("towers_count", 0) > 0:
            print(
                f"   💎 TOWERS encontradas: {stats['towers_count']} (SEGUNDO MELHOR - 17.3u, 18.0% ROI)"
            )

        if stats.get("dragons_count", 0) > 0:
            print(
                f"   🐲 DRAGONS encontradas: {stats['dragons_count']} (UNDER: 7.8u + OVER: 2.2u)"
            )

        print(f"   ✅ Estratégia REFINADA aplicada - Foco no range 20-25% ROI!")


# Funções de conveniência
def apply_optimized_strategy(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Função principal - SIMPLIFICADA"""
    analyzer = BettingStrategyAnalyzer().set_verbose(verbose)
    return analyzer.apply_optimized_strategy(df)


def apply_strategy_to_pending_bets(
    df_pending: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Aplica estratégia às apostas pendentes"""
    return apply_optimized_strategy(df_pending, verbose)


def get_strategy_summary() -> Dict:
    """Resumo da estratégia SIMPLIFICADA"""
    config = StrategyConfig()

    return {
        "name": "Estratégia Refinada - Baseada em Análise Aprimorada",
        "version": "v8.0_REFINED",
        "expected_roi": config.EXPECTED_ROI,
        "sweet_spot": "Range 20-25% ROI (+26.1u, 80% win rate)",
        "market_hierarchy": "KILLS > TOWERS > UNDER-DRAGONS > OVER-DRAGONS",
        "criteria": {
            "roi_ranges": config.ALLOWED_ROI_RANGES,  # Refinados: 20-25%, 15-20%, ≥40%, 10-15%
            "markets": config.ALLOWED_MARKETS,
            "odds": ["media", "baixa", "media_alta"],  # Refinadas baseadas na análise
            "preferred_direction": config.PREFERRED_DIRECTION,
            "excluded": {
                "odds": ["muito_alta", "muito_baixa"],  # Ambas negativas na análise
                "markets": config.FORBIDDEN_MARKETS,
            },
        },
        "performance_real": {
            "historical_roi": f"{config.EXPECTED_ROI}%",
            "historical_profit": f"{config.EXPECTED_PROFIT} units",
            "approval_rate": "26.5%",
            "efficiency": "Foco na direção, não micro-otimizações",
        },
        "strategy_focus": {
            "direction": "UNDER prioritário + OVER apenas em DRAGONS",
            "forbidden_over": ["DURATION", "BARONS", "INHIBITORS"],
            "top_markets": ["UNDER-KILLS", "UNDER-TOWERS", "OVER-DRAGONS"],
        },
    }


if __name__ == "__main__":
    print("✅ Estratégia SIMPLIFICADA v7.0!")
    print("🎯 Princípio: UNDER domina (+38.8u vs -15.2u OVER)")
    print("🐲 Exceção: OVER-DRAGONS único OVER lucrativo (+6.9u)")
    print("🔽 Foco: Direção da aposta, não micro-otimizações de ROI")
    print("📊 Resultado esperado: 22.2% ROI com 163 apostas selecionadas")
