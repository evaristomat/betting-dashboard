import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuração CORRIGIDA baseada na análise real de julho 2025"""

    # ROI ranges CORRETOS (baseados na sua análise real)
    ALLOWED_ROI_RANGES = ["≥20%", "≥25%"]  # CORRIGIDO: Apenas os lucrativos

    # Mercados REAIS da sua análise (ordenados por performance)
    ALLOWED_MARKETS = [
        "UNDER - TOWERS",  # 11.7u | 40.4% ROI - 🏆 SEU MELHOR
        "OVER - DRAGONS",  # 9.3u  | 66.4% ROI - 💎 EXCELENTE WIN RATE
        "UNDER - KILLS",  # 5.1u  | 34.2% ROI - ✅ SÓLIDO
        "UNDER - DURATION",  # 4.5u  | 37.2% ROI - ✅ BOM
        "UNDER - DRAGONS",  # 3.3u  | 36.7% ROI - ✅ CONSISTENTE
    ]

    # Faixas de odds CORRETAS (baseadas na sua análise)
    ALLOWED_ODDS = [
        "media",  # 12.0u | 15.6% ROI - 🏆 MELHOR VOLUME+LUCRO
        "media_alta",  # 5.8u  | 25.1% ROI - 💎 BOA MARGEM
        "baixa",  # 4.6u  | 20.8% ROI - ✅ LUCRATIVO
    ]

    # Direção DOMINANTE na sua análise
    PREFERRED_DIRECTION = ["UNDER"]  # +21.9u vs -2.1u OVER

    # Mercados para EVITAR (prejuízos reais da sua análise)
    FORBIDDEN_MARKETS = [
        "UNDER - BARONS",  # -0.0u | -0.2% ROI
        "OVER - TOWERS",  # -1.0u | -100.0% ROI
        "OVER - BARONS",  # -2.5u | -25.0% ROI
        "OVER - INHIBITORS",  # -2.6u | -32.2% ROI
        "UNDER - INHIBITORS",  # -2.6u | -12.6% ROI
    ]

    # Odds problemáticas para EVITAR
    FORBIDDEN_ODDS = [
        "muito_alta",  # -2.5u | -25.0% ROI
    ]

    # ROI ranges problemáticos para EVITAR
    FORBIDDEN_ROI_RANGES = ["15-20%", "<15%", "≥30%"]

    # Performance esperada REAL (baseada na estratégia final)
    EXPECTED_ROI = 37.9  # ROI real da estratégia otimizada
    TARGET_APPROVAL_RATE = 14.3  # 65/453 apostas (29.1% após filtro ROI)
    HISTORICAL_PROFIT = 24.6  # Lucro real em units


class BettingStrategyAnalyzer:
    """Analisador CORRIGIDO baseado nos dados reais de julho 2025"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.verbose = False

    def set_verbose(self, verbose: bool = True):
        """Ativa/desativa modo verbose para debugging"""
        self.verbose = verbose
        return self

    def categorize_odds(self, odds: float) -> str:
        """Categoriza odds - AJUSTADO para seus dados reais"""
        if pd.isna(odds):
            return "N/A"

        # Ranges ajustados baseados na sua performance real
        if odds < 1.5:
            return "baixa"  # 4.6u lucro na sua análise
        elif odds < 2.0:
            return "media"  # 12.0u lucro - SEU MELHOR
        elif odds < 2.5:
            return "media_alta"  # 5.8u lucro - BOA MARGEM
        else:
            return "muito_alta"  # -2.5u prejuízo - EVITAR

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categoriza mercado EXATAMENTE como na sua análise"""
        bet_type_lower = str(bet_type).lower()
        bet_line_lower = str(bet_line).lower()

        # Determinar direção
        direction = "UNDER" if "under" in bet_type_lower else "OVER"

        # Determinar tipo de mercado (EXATO da sua análise)
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
        """Categoriza ROI - EXATO da sua análise"""
        if pd.isna(roi):
            return "N/A"

        if roi >= 30:
            return "≥30%"  # EVITAR - prejuízo na sua análise
        elif roi >= 25:
            return "≥25%"  # PERMITIDO - lucrativo
        elif roi >= 20:
            return "≥20%"  # PERMITIDO - lucrativo
        elif roi >= 15:
            return "15-20%"  # EVITAR - prejuízo na sua análise
        else:
            return "<15%"  # EVITAR - prejuízo na sua análise

    def preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocessa o DataFrame adicionando colunas necessárias"""
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
        """Aplica os filtros CORRETOS baseados na sua análise real"""
        if df.empty:
            return df.copy()

        if self.verbose:
            print(f"🎯 APLICANDO ESTRATÉGIA CORRIGIDA - Baseada em Dados Reais")
            print(f"   📊 Dados originais: {len(df)} apostas")

        df_filtered = df.copy()

        # Filtro 1: APENAS ROI ranges lucrativos (≥20% e ≥25%)
        df_filtered = df_filtered[
            df_filtered["est_roi_category"].isin(self.config.ALLOWED_ROI_RANGES)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro ROI (≥20%, ≥25%): {len(df_filtered)} apostas")

        # Filtro 2: APENAS mercados que deram lucro REAL
        df_filtered = df_filtered[
            df_filtered["grouped_market"].isin(self.config.ALLOWED_MARKETS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro mercados lucrativos: {len(df_filtered)} apostas")

        # Filtro 3: APENAS odds lucrativas (media, media_alta, baixa)
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(self.config.ALLOWED_ODDS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro odds: {len(df_filtered)} apostas")

        # Filtro 4: Preferência FORTE por UNDER (+21.9u vs -2.1u)
        df_under = df_filtered[
            df_filtered["direction"].isin(self.config.PREFERRED_DIRECTION)
        ]
        if len(df_under) > 0:
            df_filtered = df_under
            if self.verbose:
                print(f"   🔽 Após priorizar UNDER: {len(df_filtered)} apostas")

        # Filtro 5: Remover mercados que causaram prejuízo REAL
        df_filtered = df_filtered[
            ~df_filtered["grouped_market"].isin(self.config.FORBIDDEN_MARKETS)
        ]
        if self.verbose:
            print(
                f"   🚫 Após remover mercados problemáticos: {len(df_filtered)} apostas"
            )

        # Filtro 6: Remover odds que causaram prejuízo
        df_filtered = df_filtered[
            ~df_filtered["odds_category"].isin(self.config.FORBIDDEN_ODDS)
        ]

        return df_filtered.reset_index(drop=True)

    def apply_optimized_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Função principal: aplica a estratégia CORRIGIDA baseada em dados reais
        """
        if df.empty:
            return df.copy()

        # Preprocessar dados
        df_processed = self.preprocess_dataframe(df)

        # Aplicar filtros
        df_filtered = self.apply_strategy_filters(df_processed)

        # Ordenar por prioridade REAL (baseada na sua análise)
        if len(df_filtered) > 0:
            # Pesos baseados no LUCRO REAL da sua análise
            market_weights = {
                "UNDER - TOWERS": 5,  # 11.7u - SEU MELHOR
                "OVER - DRAGONS": 4,  # 9.3u - SEGUNDO MELHOR
                "UNDER - KILLS": 3,  # 5.1u - TERCEIRO
                "UNDER - DURATION": 2,  # 4.5u - QUARTO
                "UNDER - DRAGONS": 1,  # 3.3u - QUINTO
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

        # Gerar estatísticas se verbose
        if self.verbose:
            stats = self.generate_statistics(df_processed, df_filtered)
            self._print_verbose_stats(stats)

        return df_filtered

    def generate_statistics(
        self, df_original: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> Dict:
        """Gera estatísticas da estratégia aplicada"""
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
            stats["odds_breakdown"] = (
                df_filtered.groupby("odds_category").size().to_dict()
            )
            stats["direction_breakdown"] = (
                df_filtered.groupby("direction").size().to_dict()
            )

            # Contagem dos top mercados
            stats["towers_count"] = len(
                df_filtered[df_filtered["market_type"] == "TOWERS"]
            )
            stats["dragons_count"] = len(
                df_filtered[df_filtered["market_type"] == "DRAGONS"]
            )

            # Top apostas por ROI
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
        """Imprime estatísticas baseadas nos dados reais"""
        print(f"\n📈 RESULTADOS DA ESTRATÉGIA CORRIGIDA:")
        print(f"   🎯 Taxa de aprovação: {stats['approval_rate']:.1f}%")
        print(f"   💰 ROI médio estimado: {stats['avg_estimated_roi']:.1f}%")

        if "direction_breakdown" in stats:
            under_count = stats["direction_breakdown"].get("UNDER", 0)
            over_count = stats["direction_breakdown"].get("OVER", 0)
            print(f"   🔽 UNDER: {under_count} | OVER: {over_count}")

        if stats.get("towers_count", 0) > 0:
            print(
                f"   🏆 TOWERS encontradas: {stats['towers_count']} (SEU MELHOR MERCADO)"
            )

        if "top_recommendations" in stats and len(stats["top_recommendations"]) > 0:
            print(f"\n   🏆 TOP RECOMENDAÇÕES:")
            for i, bet in enumerate(stats["top_recommendations"][:3], 1):
                if "TOWERS" in bet["market"]:
                    icon = "🏆"
                elif "DRAGONS" in bet["market"]:
                    icon = "💎"
                else:
                    icon = "✅"
                print(
                    f"      {i}. {icon} {bet['market']} - ROI: {bet['roi']:.1f}% - Odds: {bet['odds']:.2f}"
                )

        print(f"   ✅ Estratégia alinhada com dados reais aplicada!")


# Funções de conveniência mantidas para compatibilidade
def apply_optimized_strategy(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Função de conveniência - CORRIGIDA"""
    analyzer = BettingStrategyAnalyzer().set_verbose(verbose)
    return analyzer.apply_optimized_strategy(df)


def validate_strategy_criteria(df: pd.DataFrame) -> Dict:
    """Valida se os dados atendem aos critérios CORRETOS"""
    if df.empty:
        return {"valid": False, "message": "Dataset vazio"}

    required_columns = ["bet_type", "bet_line", "odds", "ROI"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "message": f"Colunas obrigatórias ausentes: {missing_columns}",
        }

    analyzer = BettingStrategyAnalyzer()
    df_strategy = analyzer.apply_optimized_strategy(df)
    stats = analyzer.generate_statistics(df, df_strategy)

    warnings = []

    if stats.get("towers_count", 0) == 0:
        warnings.append(
            "⚠️ Nenhuma aposta TOWERS encontrada (seu melhor mercado: 11.7u)"
        )

    if "direction_breakdown" in stats:
        over_count = stats["direction_breakdown"].get("OVER", 0)
        under_count = stats["direction_breakdown"].get("UNDER", 0)
        if over_count > under_count * 0.1:  # OVER deveria ser <10% do volume
            warnings.append("🚨 Muitas apostas OVER (UNDER gerou 21.9u vs -2.1u OVER)")

    return {
        "valid": True,
        "total_bets": stats["total_bets"],
        "approved_bets": stats["approved_bets"],
        "approval_rate": stats["approval_rate"],
        "avg_estimated_roi": stats["avg_estimated_roi"],
        "warnings": warnings,
        "message": f"Estratégia REAL aplicada. {stats['approved_bets']}/{stats['total_bets']} apostas aprovadas ({stats['approval_rate']:.1f}%)",
        "towers_count": stats.get("towers_count", 0),
    }


def get_strategy_summary() -> Dict:
    """Retorna resumo da estratégia CORRIGIDA baseada em dados reais"""
    config = StrategyConfig()

    return {
        "name": "Estratégia Baseada em Dados Reais - Julho 2025",
        "version": "v6.0_REAL",
        "expected_roi": f"{config.EXPECTED_ROI}%",
        "criteria": {
            "roi_ranges": config.ALLOWED_ROI_RANGES,
            "markets": config.ALLOWED_MARKETS,
            "odds": config.ALLOWED_ODDS,
            "preferred_direction": config.PREFERRED_DIRECTION,
            "excluded": {
                "odds": config.FORBIDDEN_ODDS,
                "markets": config.FORBIDDEN_MARKETS,
                "roi_ranges": config.FORBIDDEN_ROI_RANGES,
            },
        },
        "performance_real": {
            "historical_roi": f"{config.EXPECTED_ROI}%",
            "historical_profit": f"{config.HISTORICAL_PROFIT} units",
            "approval_rate": f"{config.TARGET_APPROVAL_RATE}%",
            "efficiency": "39.9x melhor ROI que estratégia original",
        },
        "corrections_v6": [
            "🔧 CORRIGIDO: Apenas ROI ≥20% e ≥25% (únicos lucrativos)",
            "🏆 CORRIGIDO: UNDER-TOWERS como #1 (11.7u real)",
            "💎 CORRIGIDO: OVER-DRAGONS como #2 (9.3u real)",
            "📊 CORRIGIDO: Odds 'media' como melhor (12.0u real)",
            "🚫 CORRIGIDO: Removidos TODOS os mercados com prejuízo real",
            "🔽 CORRIGIDO: UNDER preferência absoluta (+21.9u vs -2.1u)",
            "⚖️ ALINHADO: 100% baseado na sua análise de julho",
        ],
        "market_priority_real": get_real_market_ranking(),
    }


def get_real_market_ranking() -> List[Dict]:
    """Ranking EXATO baseado na sua análise real"""
    return [
        {
            "rank": 1,
            "market": "UNDER - TOWERS",
            "roi": 40.4,
            "profit": 11.7,
            "priority": "🏆 SEU MELHOR",
        },
        {
            "rank": 2,
            "market": "OVER - DRAGONS",
            "roi": 66.4,
            "profit": 9.3,
            "priority": "💎 EXCELENTE WIN RATE",
        },
        {
            "rank": 3,
            "market": "UNDER - KILLS",
            "roi": 34.2,
            "profit": 5.1,
            "priority": "✅ SÓLIDO",
        },
        {
            "rank": 4,
            "market": "UNDER - DURATION",
            "roi": 37.2,
            "profit": 4.5,
            "priority": "✅ BOM",
        },
        {
            "rank": 5,
            "market": "UNDER - DRAGONS",
            "roi": 36.7,
            "profit": 3.3,
            "priority": "✅ CONSISTENTE",
        },
        # Mercados para EVITAR (todos com prejuízo real)
        {
            "rank": -1,
            "market": "UNDER - INHIBITORS",
            "roi": -12.6,
            "profit": -2.6,
            "priority": "🚨 EVITAR",
        },
        {
            "rank": -2,
            "market": "OVER - INHIBITORS",
            "roi": -32.2,
            "profit": -2.6,
            "priority": "🚨 EVITAR",
        },
        {
            "rank": -3,
            "market": "OVER - BARONS",
            "roi": -25.0,
            "profit": -2.5,
            "priority": "🚨 EVITAR",
        },
        {
            "rank": -4,
            "market": "OVER - TOWERS",
            "roi": -100.0,
            "profit": -1.0,
            "priority": "🚨 NUNCA",
        },
    ]


if __name__ == "__main__":
    print("✅ Estratégia CORRIGIDA v6.0 - Baseada em Dados Reais!")
    print(f"🎯 ROI esperado: {StrategyConfig.EXPECTED_ROI}% (REAL da sua análise)")
    print(
        f"📊 {len(StrategyConfig.ALLOWED_MARKETS)} mercados aprovados (APENAS os lucrativos)"
    )
    print(
        f"🚫 {len(StrategyConfig.FORBIDDEN_MARKETS)} mercados proibidos (TODOS com prejuízo)"
    )
    print(f"🔽 Preferência ABSOLUTA: UNDER (+21.9u vs -2.1u OVER)")

    print(f"\n🔧 CORREÇÕES PRINCIPAIS:")
    print(f"   ❌ REMOVIDO: <15%, 15-20%, ≥30% ROI (todos com prejuízo)")
    print(f"   ✅ MANTIDO: Apenas ≥20% e ≥25% (únicos lucrativos)")
    print(f"   🏆 CORRIGIDO: UNDER-TOWERS como #1 (11.7u vs 9.9u KILLS)")
    print(f"   💎 ADICIONADO: OVER-DRAGONS como #2 (9.3u de lucro real)")
    print(f"   📊 CORRIGIDO: 'media' odds como melhor (12.0u vs 5.8u media_alta)")
    print(f"   🚫 REMOVIDO: TODOS os mercados com prejuízo real")

    print(f"\n🎯 ESTRATÉGIA AGORA 100% ALINHADA COM SEUS DADOS REAIS!")
