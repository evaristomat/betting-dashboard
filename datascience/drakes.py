import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings

warnings.filterwarnings("ignore")


class LoLDragonsAnalyzer:
    def __init__(self, csv_path):
        """
        Inicializa o analisador com o arquivo CSV
        """
        self.df = pd.read_csv(csv_path)
        self.prepare_data()

    def prepare_data(self):
        """
        Prepara e limpa os dados para análise
        """
        print("Preparando dados...")

        # Criar colunas derivadas para análise
        self.df["over_2_5_t1"] = (self.df["dragons_t1"] > 2.5).astype(int)
        self.df["over_2_5_t2"] = (self.df["dragons_t2"] > 2.5).astype(int)
        self.df["over_4_5_total"] = (self.df["total_dragons"] > 4.5).astype(int)

        # Time que tem mais dragões (1 = t1, 2 = t2, 0 = empate)
        self.df["most_dragons"] = np.where(
            self.df["dragons_t1"] > self.df["dragons_t2"],
            1,
            np.where(self.df["dragons_t1"] < self.df["dragons_t2"], 2, 0),
        )

        # Time que fez primeiro dragão
        self.df["first_dragon_team"] = np.where(
            self.df["firstdragon_t1"] == 1,
            1,
            np.where(self.df["firstdragon_t2"] == 1, 2, 0),
        )

        print(f"Dataset carregado: {len(self.df)} jogos")
        print(f"Colunas disponíveis: {list(self.df.columns)}")

    def basic_statistics(self):
        """
        Estatísticas básicas sobre dragões
        """
        print("\n" + "=" * 50)
        print("ESTATÍSTICAS BÁSICAS - DRAGÕES")
        print("=" * 50)

        print(
            f"Média de dragões por time: {self.df[['dragons_t1', 'dragons_t2']].mean().mean():.2f}"
        )
        print(
            f"Média total de dragões por partida: {self.df['total_dragons'].mean():.2f}"
        )
        print(f"Máximo de dragões em uma partida: {self.df['total_dragons'].max()}")

        print(f"\nDistribuição Over 2.5 dragões:")
        print(f"Team 1 Over 2.5: {self.df['over_2_5_t1'].mean() * 100:.1f}%")
        print(f"Team 2 Over 2.5: {self.df['over_2_5_t2'].mean() * 100:.1f}%")

        print(
            f"\nOver 4.5 dragões total: {self.df['over_4_5_total'].mean() * 100:.1f}%"
        )

        # Distribuição de dragões por partida
        dragon_dist = self.df["total_dragons"].value_counts().sort_index()
        print(f"\nDistribuição total de dragões:")
        for dragons, count in dragon_dist.items():
            percentage = (count / len(self.df)) * 100
            print(f"{int(dragons)} dragões: {count} jogos ({percentage:.1f}%)")

    def first_dragon_analysis(self):
        """
        Análise da correlação entre primeiro dragão e over/under
        """
        print("\n" + "=" * 50)
        print("ANÁLISE PRIMEIRO DRAGÃO vs OVER/UNDER")
        print("=" * 50)

        # Filtrar jogos onde houve primeiro dragão
        first_dragon_games = self.df[self.df["first_dragon_team"] != 0].copy()

        if len(first_dragon_games) == 0:
            print("Nenhum jogo com primeiro dragão encontrado!")
            return

        print(f"Jogos com primeiro dragão: {len(first_dragon_games)}")

        # Análise para o time que fez primeiro dragão
        first_dragon_over_2_5 = []
        first_dragon_most_dragons = []

        for _, row in first_dragon_games.iterrows():
            if row["first_dragon_team"] == 1:
                first_dragon_over_2_5.append(row["over_2_5_t1"])
                first_dragon_most_dragons.append(1 if row["most_dragons"] == 1 else 0)
            elif row["first_dragon_team"] == 2:
                first_dragon_over_2_5.append(row["over_2_5_t2"])
                first_dragon_most_dragons.append(1 if row["most_dragons"] == 2 else 0)

        over_2_5_rate = np.mean(first_dragon_over_2_5) * 100
        most_dragons_rate = np.mean(first_dragon_most_dragons) * 100

        print(f"\nTime que faz PRIMEIRO DRAGÃO:")
        print(f"Over 2.5 dragões: {over_2_5_rate:.1f}%")
        print(f"Fica com mais dragões: {most_dragons_rate:.1f}%")

        # Over 4.5 total quando há primeiro dragão
        over_4_5_with_first = first_dragon_games["over_4_5_total"].mean() * 100
        print(f"Over 4.5 total (jogos com 1º dragão): {over_4_5_with_first:.1f}%")

        return {
            "first_dragon_over_2_5": over_2_5_rate,
            "first_dragon_most_dragons": most_dragons_rate,
            "over_4_5_with_first_dragon": over_4_5_with_first,
        }

    def champion_influence_analysis(self, min_games=20):
        """
        Análise de influência dos campeões nos dragões
        """
        print("\n" + "=" * 50)
        print("ANÁLISE INFLUÊNCIA CAMPEÕES (GERAL)")
        print("=" * 50)

        positions = ["top", "jung", "mid", "adc", "sup"]
        champion_stats = {}

        for pos in positions:
            print(f"\n--- {pos.upper()} ---")

            # Coletar dados de todos os campeões nesta posição
            champ_data = []

            # Team 1
            for _, row in self.df.iterrows():
                champ = row[f"{pos}_t1"]
                if pd.notna(champ):
                    champ_data.append(
                        {
                            "champion": champ,
                            "dragons": row["dragons_t1"],
                            "over_2_5": row["over_2_5_t1"],
                            "most_dragons": 1 if row["most_dragons"] == 1 else 0,
                            "first_dragon": 1 if row["first_dragon_team"] == 1 else 0,
                        }
                    )

            # Team 2
            for _, row in self.df.iterrows():
                champ = row[f"{pos}_t2"]
                if pd.notna(champ):
                    champ_data.append(
                        {
                            "champion": champ,
                            "dragons": row["dragons_t2"],
                            "over_2_5": row["over_2_5_t2"],
                            "most_dragons": 1 if row["most_dragons"] == 2 else 0,
                            "first_dragon": 1 if row["first_dragon_team"] == 2 else 0,
                        }
                    )

            # Analisar por campeão
            champ_df = pd.DataFrame(champ_data)
            if len(champ_df) == 0:
                continue

            champ_grouped = (
                champ_df.groupby("champion")
                .agg(
                    {
                        "dragons": ["count", "mean"],
                        "over_2_5": "mean",
                        "most_dragons": "mean",
                        "first_dragon": "mean",
                    }
                )
                .round(3)
            )

            # Filtrar campeões com pelo menos min_games jogos
            champ_grouped = champ_grouped[
                champ_grouped[("dragons", "count")] >= min_games
            ]

            if len(champ_grouped) == 0:
                print(f"Nenhum campeão com {min_games}+ jogos")
                continue

            # Ordenar por média de dragões
            champ_grouped = champ_grouped.sort_values(
                ("dragons", "mean"), ascending=False
            )

            print(f"Top 10 campeões (com {min_games}+ jogos):")
            print(
                "Campeão | Jogos | Avg Dragons | Over 2.5% | Most Dragons% | First Dragon%"
            )
            print("-" * 80)

            for champ, stats in champ_grouped.head(10).iterrows():
                games = int(stats[("dragons", "count")])
                avg_dragons = stats[("dragons", "mean")]
                over_2_5_pct = stats[("over_2_5", "mean")] * 100
                most_dragons_pct = stats[("most_dragons", "mean")] * 100
                first_dragon_pct = stats[("first_dragon", "mean")] * 100

                print(
                    f"{champ:12} | {games:5} | {avg_dragons:10.2f} | {over_2_5_pct:8.1f}% | {most_dragons_pct:12.1f}% | {first_dragon_pct:11.1f}%"
                )

            champion_stats[pos] = champ_grouped

        return champion_stats

    def jungle_specific_analysis(self, min_games=20):
        """
        Análise específica e detalhada dos junglers
        """
        print("\n" + "=" * 50)
        print("ANÁLISE DETALHADA - JUNGLERS")
        print("=" * 50)

        # Coletar dados específicos de jungle
        jungle_data = []

        # Team 1
        for _, row in self.df.iterrows():
            jungler = row["jung_t1"]
            if pd.notna(jungler):
                jungle_data.append(
                    {
                        "jungler": jungler,
                        "dragons": row["dragons_t1"],
                        "over_2_5": row["over_2_5_t1"],
                        "most_dragons": 1 if row["most_dragons"] == 1 else 0,
                        "first_dragon": 1 if row["first_dragon_team"] == 1 else 0,
                        "total_dragons_game": row["total_dragons"],
                        "over_4_5_total": row["over_4_5_total"],
                        "game_length": row["gamelength"],
                    }
                )

        # Team 2
        for _, row in self.df.iterrows():
            jungler = row["jung_t2"]
            if pd.notna(jungler):
                jungle_data.append(
                    {
                        "jungler": jungler,
                        "dragons": row["dragons_t2"],
                        "over_2_5": row["over_2_5_t2"],
                        "most_dragons": 1 if row["most_dragons"] == 2 else 0,
                        "first_dragon": 1 if row["first_dragon_team"] == 2 else 0,
                        "total_dragons_game": row["total_dragons"],
                        "over_4_5_total": row["over_4_5_total"],
                        "game_length": row["gamelength"],
                    }
                )

        jungle_df = pd.DataFrame(jungle_data)

        if len(jungle_df) == 0:
            print("Nenhum dado de jungle encontrado!")
            return

        # Análise por jungler
        jungle_grouped = (
            jungle_df.groupby("jungler")
            .agg(
                {
                    "dragons": ["count", "mean", "std"],
                    "over_2_5": "mean",
                    "most_dragons": "mean",
                    "first_dragon": "mean",
                    "total_dragons_game": "mean",
                    "over_4_5_total": "mean",
                    "game_length": "mean",
                }
            )
            .round(3)
        )

        # Filtrar junglers com pelo menos min_games jogos
        jungle_filtered = jungle_grouped[
            jungle_grouped[("dragons", "count")] >= min_games
        ]

        if len(jungle_filtered) == 0:
            print(f"Nenhum jungler com {min_games}+ jogos")
            return jungle_grouped

        # Ordenar por diferentes métricas para análise
        print(f"\nJUNGLERS com {min_games}+ jogos - Ordenados por Média de Dragões:")
        print(
            "Jungler | Jogos | Avg Dragons | Over 2.5% | Most Dragons% | First Dragon% | Avg Total Game | Over 4.5 Game%"
        )
        print("-" * 120)

        jungle_by_dragons = jungle_filtered.sort_values(
            ("dragons", "mean"), ascending=False
        )

        for jungler, stats in jungle_by_dragons.head(15).iterrows():
            games = int(stats[("dragons", "count")])
            avg_dragons = stats[("dragons", "mean")]
            over_2_5_pct = stats[("over_2_5", "mean")] * 100
            most_dragons_pct = stats[("most_dragons", "mean")] * 100
            first_dragon_pct = stats[("first_dragon", "mean")] * 100
            avg_total_game = stats[("total_dragons_game", "mean")]
            over_4_5_game_pct = stats[("over_4_5_total", "mean")] * 100

            print(
                f"{jungler:10} | {games:5} | {avg_dragons:10.2f} | {over_2_5_pct:8.1f}% | {most_dragons_pct:12.1f}% | {first_dragon_pct:11.1f}% | {avg_total_game:13.2f} | {over_4_5_game_pct:11.1f}%"
            )

        # Análise de correlações
        print(f"\n--- CORRELAÇÕES IMPORTANTES ---")
        print(
            f"Correlação First Dragon vs Most Dragons: {jungle_df['first_dragon'].corr(jungle_df['most_dragons']):.3f}"
        )
        print(
            f"Correlação First Dragon vs Over 2.5: {jungle_df['first_dragon'].corr(jungle_df['over_2_5']):.3f}"
        )
        print(
            f"Correlação Duração do Jogo vs Total Dragons: {jungle_df['game_length'].corr(jungle_df['total_dragons_game']):.3f}"
        )

        return jungle_filtered

    def betting_insights(self):
        """
        Insights específicos para apostas
        """
        print("\n" + "=" * 50)
        print("INSIGHTS PARA APOSTAS")
        print("=" * 50)

        # Análise por duração do jogo
        print("\n--- ANÁLISE POR DURAÇÃO DO JOGO ---")

        # Criar bins para duração
        self.df["game_length_bin"] = pd.cut(
            self.df["gamelength"],
            bins=[0, 25, 30, 35, 100],
            labels=["<25min", "25-30min", "30-35min", ">35min"],
        )

        duration_analysis = (
            self.df.groupby("game_length_bin")
            .agg(
                {
                    "total_dragons": "mean",
                    "over_4_5_total": "mean",
                    "over_2_5_t1": "mean",
                    "over_2_5_t2": "mean",
                }
            )
            .round(3)
        )

        print("Duração | Avg Total Dragons | Over 4.5% | Over 2.5 T1% | Over 2.5 T2%")
        print("-" * 70)
        for duration, stats in duration_analysis.iterrows():
            print(
                f"{duration:8} | {stats['total_dragons']:16.2f} | {stats['over_4_5_total'] * 100:7.1f}% | {stats['over_2_5_t1'] * 100:10.1f}% | {stats['over_2_5_t2'] * 100:10.1f}%"
            )

        # Análise por resultado do primeiro dragão
        print("\n--- VALOR DAS APOSTAS APÓS PRIMEIRO DRAGÃO ---")

        first_dragon_games = self.df[self.df["first_dragon_team"] != 0]
        if len(first_dragon_games) > 0:
            fd_stats = {
                "games": len(first_dragon_games),
                "over_4_5_rate": first_dragon_games["over_4_5_total"].mean(),
                "team_with_first_over_2_5_rate": 0,
                "team_with_first_most_dragons_rate": 0,
            }

            # Calcular para o time que fez primeiro dragão
            first_dragon_team_over_2_5 = []
            first_dragon_team_most = []

            for _, row in first_dragon_games.iterrows():
                if row["first_dragon_team"] == 1:
                    first_dragon_team_over_2_5.append(row["over_2_5_t1"])
                    first_dragon_team_most.append(1 if row["most_dragons"] == 1 else 0)
                elif row["first_dragon_team"] == 2:
                    first_dragon_team_over_2_5.append(row["over_2_5_t2"])
                    first_dragon_team_most.append(1 if row["most_dragons"] == 2 else 0)

            fd_stats["team_with_first_over_2_5_rate"] = np.mean(
                first_dragon_team_over_2_5
            )
            fd_stats["team_with_first_most_dragons_rate"] = np.mean(
                first_dragon_team_most
            )

            print(f"Jogos analisados: {fd_stats['games']}")
            print(
                f"Over 4.5 total (quando há 1º dragão): {fd_stats['over_4_5_rate'] * 100:.1f}%"
            )
            print(
                f"Time com 1º dragão faz Over 2.5: {fd_stats['team_with_first_over_2_5_rate'] * 100:.1f}%"
            )
            print(
                f"Time com 1º dragão fica com mais dragões: {fd_stats['team_with_first_most_dragons_rate'] * 100:.1f}%"
            )

        # Recomendações
        print("\n--- RECOMENDAÇÕES PARA APOSTAS ---")
        overall_over_4_5 = self.df["over_4_5_total"].mean() * 100
        overall_over_2_5_any = (
            (self.df["over_2_5_t1"] == 1) | (self.df["over_2_5_t2"] == 1)
        ).mean() * 100

        print(f"1. Over 4.5 total: {overall_over_4_5:.1f}% dos jogos")
        print(f"2. Pelo menos um time Over 2.5: {overall_over_2_5_any:.1f}% dos jogos")

        if len(first_dragon_games) > 0:
            print(f"3. Após primeiro dragão, apostar no mesmo time para:")
            print(
                f"   - Over 2.5 dragões: {fd_stats['team_with_first_over_2_5_rate'] * 100:.1f}% de acerto"
            )
            print(
                f"   - Most dragons: {fd_stats['team_with_first_most_dragons_rate'] * 100:.1f}% de acerto"
            )

    def generate_visualizations(self):
        """
        Gera visualizações importantes
        """
        print("\n" + "=" * 50)
        print("GERANDO VISUALIZAÇÕES")
        print("=" * 50)

        plt.style.use("default")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(
            "Análise de Dragões - League of Legends", fontsize=16, fontweight="bold"
        )

        # 1. Distribuição total de dragões
        ax1 = axes[0, 0]
        dragon_counts = self.df["total_dragons"].value_counts().sort_index()
        ax1.bar(dragon_counts.index, dragon_counts.values, alpha=0.7, color="skyblue")
        ax1.set_title("Distribuição Total de Dragões por Partida")
        ax1.set_xlabel("Total de Dragões")
        ax1.set_ylabel("Número de Partidas")
        ax1.axvline(x=4.5, color="red", linestyle="--", label="Over 4.5")
        ax1.legend()

        # 2. Over 2.5 por time
        ax2 = axes[0, 1]
        over_2_5_data = [
            self.df["over_2_5_t1"].mean() * 100,
            self.df["over_2_5_t2"].mean() * 100,
        ]
        ax2.bar(
            ["Team 1", "Team 2"],
            over_2_5_data,
            alpha=0.7,
            color=["lightcoral", "lightgreen"],
        )
        ax2.set_title("Over 2.5 Dragões por Time")
        ax2.set_ylabel("Percentual (%)")
        ax2.axhline(y=50, color="red", linestyle="--", alpha=0.5)

        # 3. Relação duração vs dragões
        ax3 = axes[1, 0]
        ax3.scatter(
            self.df["gamelength"], self.df["total_dragons"], alpha=0.6, color="purple"
        )
        ax3.set_title("Duração do Jogo vs Total de Dragões")
        ax3.set_xlabel("Duração (minutos)")
        ax3.set_ylabel("Total de Dragões")

        # Linha de tendência
        z = np.polyfit(self.df["gamelength"], self.df["total_dragons"], 1)
        p = np.poly1d(z)
        ax3.plot(self.df["gamelength"], p(self.df["gamelength"]), "r--", alpha=0.8)

        # 4. Primeiro dragão vs resultado
        ax4 = axes[1, 1]
        first_dragon_games = self.df[self.df["first_dragon_team"] != 0]
        if len(first_dragon_games) > 0:
            # Calcular taxas de sucesso
            success_rates = []
            labels = []

            # Over 2.5 para time que fez primeiro dragão
            first_team_over_2_5 = []
            for _, row in first_dragon_games.iterrows():
                if row["first_dragon_team"] == 1:
                    first_team_over_2_5.append(row["over_2_5_t1"])
                elif row["first_dragon_team"] == 2:
                    first_team_over_2_5.append(row["over_2_5_t2"])

            success_rates.append(np.mean(first_team_over_2_5) * 100)
            labels.append("Over 2.5\n(1º Dragão)")

            # Most dragons para time que fez primeiro dragão
            first_team_most = []
            for _, row in first_dragon_games.iterrows():
                if row["first_dragon_team"] == 1:
                    first_team_most.append(1 if row["most_dragons"] == 1 else 0)
                elif row["first_dragon_team"] == 2:
                    first_team_most.append(1 if row["most_dragons"] == 2 else 0)

            success_rates.append(np.mean(first_team_most) * 100)
            labels.append("Most Dragons\n(1º Dragão)")

            # Over 4.5 total
            success_rates.append(first_dragon_games["over_4_5_total"].mean() * 100)
            labels.append("Over 4.5 Total\n(1º Dragão)")

            bars = ax4.bar(
                labels, success_rates, alpha=0.7, color=["gold", "orange", "red"]
            )
            ax4.set_title("Taxa de Sucesso Após Primeiro Dragão")
            ax4.set_ylabel("Taxa de Sucesso (%)")
            ax4.axhline(y=50, color="black", linestyle="--", alpha=0.5)

            # Adicionar valores nas barras
            for bar, rate in zip(bars, success_rates):
                ax4.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                )

        plt.tight_layout()
        plt.savefig("lol_dragons_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("Visualizações salvas em 'lol_dragons_analysis.png'")

    def run_complete_analysis(self):
        """
        Executa análise completa
        """
        print("=" * 60)
        print("ANÁLISE COMPLETA - DRAGÕES LEAGUE OF LEGENDS")
        print("=" * 60)

        # Executar todas as análises
        self.basic_statistics()
        first_dragon_stats = self.first_dragon_analysis()
        champion_stats = self.champion_influence_analysis(min_games=20)
        jungle_stats = self.jungle_specific_analysis(min_games=20)
        self.betting_insights()
        self.generate_visualizations()

        print("\n" + "=" * 60)
        print("ANÁLISE CONCLUÍDA!")
        print("=" * 60)

        return {
            "first_dragon": first_dragon_stats,
            "champions": champion_stats,
            "jungle": jungle_stats,
        }


# Função principal para executar
def main():
    """
    Função principal - altere o caminho do arquivo CSV aqui
    """
    # ALTERE AQUI O CAMINHO DO SEU ARQUIVO CSV
    csv_path = "lol_games_2025.csv"  # Substitua pelo caminho do seu arquivo

    try:
        # Criar analisador
        analyzer = LoLDragonsAnalyzer(csv_path)

        # Executar análise completa
        results = analyzer.run_complete_analysis()

        print("\nUse os métodos individuais para análises específicas:")
        print("- analyzer.basic_statistics()")
        print("- analyzer.first_dragon_analysis()")
        print("- analyzer.champion_influence_analysis(min_games=20)")
        print("- analyzer.jungle_specific_analysis(min_games=20)")
        print("- analyzer.betting_insights()")
        print("- analyzer.generate_visualizations()")

        return analyzer, results

    except FileNotFoundError:
        print(f"ERRO: Arquivo '{csv_path}' não encontrado!")
        print("Certifique-se de que o arquivo CSV está no diretório correto.")
        print("Altere a variável 'csv_path' no código para o caminho correto.")
        return None, None

    except Exception as e:
        print(f"ERRO: {str(e)}")
        return None, None


if __name__ == "__main__":
    analyzer, results = main()
