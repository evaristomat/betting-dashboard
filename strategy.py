import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class StrategyConfig:
    """Configuração centralizada da estratégia - CORRIGIDA baseada na análise real"""

    # ROI ranges lucrativos (CORRIGIDO - removendo ≥30% que era prejudicial)
    ALLOWED_ROI_RANGES = ["<15%", "≥20%", "≥25%", "15-20%"]

    # Mercados lucrativos REAIS (baseados na sua análise de julho)
    ALLOWED_MARKETS = [
        "UNDER - KILLS",  # 9.9u | 43.2% ROI - 🏆 TOP PERFORMER
        "UNDER - DRAGONS",  # 6.6u | 33.2% ROI - 💎 EXCELENTE
        "OVER - DRAGONS",  # 6.1u | 43.3% ROI - 🚀 ALTA PERFORMANCE
        "UNDER - DURATION",  # 3.6u | 33.1% ROI - ✅ SÓLIDO
        "UNDER - INHIBITORS",  # 2.2u | 6.1% ROI  - ✅ CONSISTENTE
        "UNDER - TOWERS",  # 1.8u | 5.7% ROI  - ✅ LUCRATIVO
        "OVER - BARONS",  # 1.5u | 13.6% ROI - ✅ POSITIVO
    ]

    # Faixas de odds lucrativas (baseadas na análise real)
    ALLOWED_ODDS = [
        "media_alta",  # 12.0u | 34.2% ROI - 🏆 MELHOR PERFORMER
        "media",  # 6.0u  | 6.1% ROI  - ✅ VOLUME + LUCRO
        "muito_alta",  # 1.5u  | 13.6% ROI - ✅ BOA MARGEM
    ]

    # Direção preferencial (UNDER domina completamente)
    PREFERRED_DIRECTION = ["UNDER"]

    # Mercados para evitar (perdas confirmadas)
    FORBIDDEN_MARKETS = [
        "OVER - INHIBITORS",  # -0.8u | -26.7% ROI
        "UNDER - BARONS",  # -2.1u | -9.8% ROI
        "OVER - TOWERS",  # -4.0u | -100.0% ROI
    ]

    # Odds problemáticas para evitar
    FORBIDDEN_ODDS = [
        "baixa",  # -0.6u | -1.4% ROI
        "muito_baixa",  # -2.1u | -18.8% ROI
    ]

    # Performance esperada (baseada na estratégia final da análise)
    EXPECTED_ROI = 25.5
    TARGET_APPROVAL_RATE = 41.3  # 93/225 apostas
    HISTORICAL_PROFIT = 23.7


class BettingStrategyAnalyzer:
    """Analisador da estratégia de apostas - VERSÃO CORRIGIDA"""

    def __init__(self, config: StrategyConfig = None):
        self.config = config or StrategyConfig()
        self.verbose = False

    def set_verbose(self, verbose: bool = True):
        """Ativa/desativa modo verbose para debugging"""
        self.verbose = verbose
        return self

    def categorize_odds(self, odds: float) -> str:
        """Categoriza odds em ranges otimizados (baseado na análise real)"""
        if pd.isna(odds):
            return "N/A"

        # Ranges baseados na performance real da análise
        if odds < 1.3:
            return "muito_baixa"
        elif odds < 1.6:
            return "baixa"
        elif odds < 2.0:
            return "media"
        elif odds < 2.5:
            return "media_alta"  # MELHOR CATEGORIA na análise
        elif odds < 3.0:
            return "alta"
        else:
            return "muito_alta"

    def categorize_market(self, bet_type: str, bet_line: str) -> Tuple[str, str, str]:
        """Categoriza mercado e direção da aposta"""
        bet_type_lower = str(bet_type).lower()
        bet_line_lower = str(bet_line).lower()

        # Determinar direção
        direction = "UNDER" if "under" in bet_type_lower else "OVER"

        # Determinar tipo de mercado com melhor detecção
        market_type = "OUTROS"

        # Mapeamento mais específico baseado na análise
        market_keywords = {
            "KILLS": ["kill", "abate"],
            "DRAGONS": ["dragon", "dragao", "dragão"],
            "TOWERS": ["tower", "torre"],
            "BARONS": ["baron", "baroness", "barão"],
            "INHIBITORS": ["inhibitor", "inibidor"],
            "DURATION": ["duration", "tempo", "duração", "duracao"],
        }

        for market, keywords in market_keywords.items():
            if any(keyword in bet_line_lower for keyword in keywords):
                market_type = market
                break

        grouped_market = f"{direction} - {market_type}"
        return direction, market_type, grouped_market

    def categorize_roi_ranges(self, roi: float) -> str:
        """Categoriza ROI em ranges (CORRIGIDOS baseados na análise)"""
        if pd.isna(roi):
            return "N/A"

        if roi >= 30:
            return "≥30%"  # EVITAR - era problemático na análise
        elif roi >= 25:
            return "≥25%"  # PERMITIDO
        elif roi >= 20:
            return "≥20%"  # PERMITIDO
        elif roi >= 15:
            return "15-20%"  # PERMITIDO
        else:
            return "<15%"  # PERMITIDO - tinha o melhor volume

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
        """Aplica os filtros da estratégia CORRIGIDA"""
        if df.empty:
            return df.copy()

        if self.verbose:
            print(f"🎯 APLICANDO ESTRATÉGIA CORRIGIDA v5.0")
            print(f"   📊 Dados originais: {len(df)} apostas")

        df_filtered = df.copy()

        # Filtro 1: ROI estimado (CORRIGIDO - sem ≥30%)
        df_filtered = df_filtered[
            df_filtered["est_roi_category"].isin(self.config.ALLOWED_ROI_RANGES)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro ROI: {len(df_filtered)} apostas")

        # Filtro 2: Mercados lucrativos REAIS
        df_filtered = df_filtered[
            df_filtered["grouped_market"].isin(self.config.ALLOWED_MARKETS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro mercados: {len(df_filtered)} apostas")

        # Filtro 3: Faixas de odds lucrativas
        df_filtered = df_filtered[
            df_filtered["odds_category"].isin(self.config.ALLOWED_ODDS)
        ]
        if self.verbose:
            print(f"   ✅ Após filtro odds: {len(df_filtered)} apostas")

        # Filtro 4: Preferência por UNDER (93% do lucro vem de UNDER)
        df_under = df_filtered[
            df_filtered["direction"].isin(self.config.PREFERRED_DIRECTION)
        ]
        if len(df_under) > 0:
            df_filtered = df_under
            if self.verbose:
                print(f"   🔽 Após filtro UNDER: {len(df_filtered)} apostas")

        # Filtro 5: Remover mercados proibidos (dupla checagem)
        df_filtered = df_filtered[
            ~df_filtered["grouped_market"].isin(self.config.FORBIDDEN_MARKETS)
        ]

        # Filtro 6: Remover odds problemáticas (dupla checagem)
        df_filtered = df_filtered[
            ~df_filtered["odds_category"].isin(self.config.FORBIDDEN_ODDS)
        ]

        return df_filtered.reset_index(drop=True)

    def generate_statistics(
        self, df_original: pd.DataFrame, df_filtered: pd.DataFrame
    ) -> Dict:
        """Gera estatísticas da estratégia aplicada"""
        if len(df_original) == 0:
            return {"error": "Dataset original vazio"}

        stats = {
            "total_bets": len(df_original),
            "approved_bets": len(df_filtered),
            "approval_rate": (len(df_filtered) / len(df_original)) * 100,
            "avg_estimated_roi": df_filtered["estimated_roi"].mean()
            if len(df_filtered) > 0
            else 0,
        }

        if len(df_filtered) > 0:
            # Breakdown por mercado
            stats["market_breakdown"] = (
                df_filtered.groupby("grouped_market").size().to_dict()
            )

            # Breakdown por odds
            stats["odds_breakdown"] = (
                df_filtered.groupby("odds_category").size().to_dict()
            )

            # Contagem de apostas KILLS (top performer)
            stats["kills_count"] = len(
                df_filtered[df_filtered["market_type"] == "KILLS"]
            )

            # Contagem de UNDER vs OVER
            stats["direction_breakdown"] = (
                df_filtered.groupby("direction").size().to_dict()
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

            # Verificar concentração nos top 3 mercados
            top_3_markets = ["UNDER - KILLS", "UNDER - DRAGONS", "OVER - DRAGONS"]
            top_3_count = sum(
                stats["market_breakdown"].get(market, 0) for market in top_3_markets
            )
            stats["top_3_concentration"] = (
                (top_3_count / len(df_filtered)) * 100 if len(df_filtered) > 0 else 0
            )

        return stats

    def apply_optimized_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Função principal: aplica a estratégia CORRIGIDA completa

        Returns:
            DataFrame com apenas as apostas aprovadas pela estratégia
        """
        if df.empty:
            return df.copy()

        # Preprocessar dados
        df_processed = self.preprocess_dataframe(df)

        # Aplicar filtros
        df_filtered = self.apply_strategy_filters(df_processed)

        # Ordenar por prioridade: ROI estimado + mercado
        if len(df_filtered) > 0:
            # Adicionar peso de prioridade por mercado
            market_weights = {
                "UNDER - KILLS": 3,  # Top performer
                "UNDER - DRAGONS": 2,  # Segundo melhor
                "OVER - DRAGONS": 2,  # Terceiro melhor
                "UNDER - DURATION": 1,  # Bom
                "UNDER - INHIBITORS": 1,  # Consistente
                "UNDER - TOWERS": 1,  # Lucrativo
                "OVER - BARONS": 1,  # Positivo
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

    def _print_verbose_stats(self, stats: Dict):
        """Imprime estatísticas detalhadas"""
        print(f"\n📈 RESULTADOS DA ESTRATÉGIA v5.0:")
        print(f"   🎯 Taxa de aprovação: {stats['approval_rate']:.1f}%")
        print(f"   💰 ROI médio estimado: {stats['avg_estimated_roi']:.1f}%")

        # Direção (UNDER vs OVER)
        if "direction_breakdown" in stats:
            under_count = stats["direction_breakdown"].get("UNDER", 0)
            over_count = stats["direction_breakdown"].get("OVER", 0)
            print(f"   🔽 UNDER: {under_count} | OVER: {over_count}")

        # KILLS (top performer)
        if stats["kills_count"] > 0:
            print(f"   🏆 KILLS encontradas: {stats['kills_count']} (TOP PERFORMER)")

        # Concentração nos top 3
        if "top_3_concentration" in stats:
            print(
                f"   🎯 Top 3 mercados: {stats['top_3_concentration']:.1f}% das apostas"
            )

        # Top recomendações
        if "top_recommendations" in stats and len(stats["top_recommendations"]) > 0:
            print(f"\n   🏆 TOP RECOMENDAÇÕES:")
            for i, bet in enumerate(stats["top_recommendations"][:3], 1):
                if "KILLS" in bet["market"]:
                    icon = "🏆"
                elif "DRAGONS" in bet["market"]:
                    icon = "💎"
                else:
                    icon = "✅"
                print(
                    f"      {i}. {icon} {bet['market']} - ROI: {bet['roi']:.1f}% - Odds: {bet['odds']:.2f}"
                )

        print(f"   ✅ Estratégia aplicada com sucesso!")


def apply_optimized_strategy(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """
    Função de conveniência para aplicar a estratégia CORRIGIDA
    Mantém compatibilidade com o código existente
    """
    analyzer = BettingStrategyAnalyzer().set_verbose(verbose)
    return analyzer.apply_optimized_strategy(df)


def apply_strategy_to_pending_bets(
    df_pending: pd.DataFrame, verbose: bool = False
) -> pd.DataFrame:
    """Aplica a estratégia às apostas pendentes"""
    return apply_optimized_strategy(df_pending, verbose)


def validate_strategy_criteria(df: pd.DataFrame) -> Dict:
    """Valida se os dados atendem aos critérios da estratégia CORRIGIDA"""
    if df.empty:
        return {"valid": False, "message": "Dataset vazio"}

    required_columns = ["bet_type", "bet_line", "odds", "ROI"]
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        return {
            "valid": False,
            "message": f"Colunas obrigatórias ausentes: {missing_columns}",
        }

    # Aplicar estratégia para validar
    analyzer = BettingStrategyAnalyzer()
    df_strategy = analyzer.apply_optimized_strategy(df)
    stats = analyzer.generate_statistics(df, df_strategy)

    # Verificações específicas
    warnings = []

    if stats["kills_count"] == 0:
        warnings.append("⚠️ Nenhuma aposta KILLS encontrada (mercado top performer)")

    if stats.get("top_3_concentration", 0) < 70:
        warnings.append("⚠️ Baixa concentração nos mercados top 3")

    # Verificar se há apostas OVER quando UNDER domina
    if "direction_breakdown" in stats:
        over_count = stats["direction_breakdown"].get("OVER", 0)
        under_count = stats["direction_breakdown"].get("UNDER", 0)
        if over_count > under_count:
            warnings.append("🚨 Muitas apostas OVER (UNDER performa 4x melhor)")

    return {
        "valid": True,
        "total_bets": stats["total_bets"],
        "approved_bets": stats["approved_bets"],
        "approval_rate": stats["approval_rate"],
        "avg_estimated_roi": stats["avg_estimated_roi"],
        "warnings": warnings,
        "message": f"Estratégia CORRIGIDA aplicada. {stats['approved_bets']}/{stats['total_bets']} apostas aprovadas ({stats['approval_rate']:.1f}%)",
        "kills_count": stats["kills_count"],
        "top_3_concentration": stats.get("top_3_concentration", 0),
    }


def get_strategy_summary() -> Dict:
    """Retorna resumo completo da estratégia CORRIGIDA v5.0"""
    config = StrategyConfig()

    return {
        "name": "Estratégia Corrigida v5.0 - Baseada em Dados Reais",
        "version": "v5.0",
        "expected_roi": f"{config.EXPECTED_ROI}%",
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
        "performance": {
            "historical_roi": f"{config.EXPECTED_ROI}%",
            "target_profit": f"{config.HISTORICAL_PROFIT} units",
            "target_approval_rate": f"{config.TARGET_APPROVAL_RATE}%",
            "efficiency": "136.5x melhor que estratégia base",
        },
        "corrections_v5": [
            "🔧 CORRIGIDO: ROI ≥30% removido (era prejudicial -52.8% ROI)",
            "📊 ATUALIZADO: Mercados baseados na análise real de julho",
            "🎯 ADICIONADO: Preferência por UNDER (+22.0u vs -5.3u OVER)",
            "📈 CORRIGIDO: Odds média_alta como melhor categoria (34.2% ROI)",
            "🏆 PRIORIZADO: KILLS como mercado top (9.9u lucro)",
            "🚫 REMOVIDO: Mercados que causavam prejuízo real",
            "⚖️ BALANCEADO: Critérios alinhados com performance real",
        ],
        "market_priority": get_market_priority_ranking(),
        "risk_warnings": [
            "🎯 FOCO: Top 3 mercados representam 67% do lucro total",
            "🔽 CRÍTICO: 93% do lucro vem de apostas UNDER",
            "⚠️ EVITAR: Odds baixa/muito_baixa causam prejuízo",
            "🚨 NUNCA: Apostar em OVER-TOWERS (-100% ROI)",
        ],
    }


def get_market_priority_ranking() -> List[Dict]:
    """Retorna ranking REAL baseado na análise de julho"""
    return [
        {
            "rank": 1,
            "market": "UNDER - KILLS",
            "roi": 43.2,
            "profit": 9.9,
            "priority": "🏆 TOP PERFORMER",
        },
        {
            "rank": 2,
            "market": "UNDER - DRAGONS",
            "roi": 33.2,
            "profit": 6.6,
            "priority": "💎 EXCELENTE",
        },
        {
            "rank": 3,
            "market": "OVER - DRAGONS",
            "roi": 43.3,
            "profit": 6.1,
            "priority": "🚀 ALTA PERFORMANCE",
        },
        {
            "rank": 4,
            "market": "UNDER - DURATION",
            "roi": 33.1,
            "profit": 3.6,
            "priority": "✅ SÓLIDO",
        },
        {
            "rank": 5,
            "market": "UNDER - INHIBITORS",
            "roi": 6.1,
            "profit": 2.2,
            "priority": "✅ CONSISTENTE",
        },
        {
            "rank": 6,
            "market": "UNDER - TOWERS",
            "roi": 5.7,
            "profit": 1.8,
            "priority": "✅ LUCRATIVO",
        },
        {
            "rank": 7,
            "market": "OVER - BARONS",
            "roi": 13.6,
            "profit": 1.5,
            "priority": "✅ POSITIVO",
        },
        # Mercados para EVITAR (baseado na análise real)
        {
            "rank": -1,
            "market": "OVER - TOWERS",
            "roi": -100.0,
            "profit": -4.0,
            "priority": "🚨 EVITAR SEMPRE",
        },
        {
            "rank": -2,
            "market": "UNDER - BARONS",
            "roi": -9.8,
            "profit": -2.1,
            "priority": "❌ EVITAR",
        },
        {
            "rank": -3,
            "market": "OVER - INHIBITORS",
            "roi": -26.7,
            "profit": -0.8,
            "priority": "❌ EVITAR",
        },
    ]


# Exemplo de uso da versão CORRIGIDA:
if __name__ == "__main__":
    # Usar a classe para máxima flexibilidade
    analyzer = BettingStrategyAnalyzer().set_verbose(True)

    # Ou usar as funções de conveniência para compatibilidade
    # df_approved = apply_optimized_strategy(df_bets, verbose=True)

    print("✅ Estratégia CORRIGIDA v5.0 carregada com sucesso!")
    print(f"🎯 ROI esperado: {StrategyConfig.EXPECTED_ROI}% (baseado em dados reais)")
    print(f"📊 {len(StrategyConfig.ALLOWED_MARKETS)} mercados aprovados")
    print(f"🚫 {len(StrategyConfig.FORBIDDEN_MARKETS)} mercados proibidos")
    print(f"🔽 Preferência: UNDER (93% do lucro)")

    # Mostrar diferenças da correção
    print(f"\n🔧 PRINCIPAIS CORREÇÕES:")
    print(f"   ❌ Removido ROI ≥30% (causava -52.8% ROI)")
    print(f"   ✅ UNDER-KILLS como #1 (9.9u lucro)")
    print(f"   ✅ Média-alta como melhor odds (34.2% ROI)")
    print(f"   🚫 Evitando mercados com prejuízo real")
