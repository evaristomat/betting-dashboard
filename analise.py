import pandas as pd
from collections import Counter


def explore_betting_data(file_path: str):
    """
    Script para explorar bet_type e bet_line do banco de dados
    """

    print("=" * 80)
    print("EXPLORAÇÃO DE BET_TYPE E BET_LINE")
    print("=" * 80)

    # Carregar dados
    df = pd.read_csv(file_path)

    print(f"\n📊 INFORMAÇÕES GERAIS:")
    print(f"Total de registros: {len(df)}")

    # BET_TYPE
    print(f"\n🏷️ VALORES ÚNICOS - BET_TYPE:")
    print("-" * 40)
    bet_types = df["bet_type"].value_counts()
    print(f"Total de tipos diferentes: {len(bet_types)}")
    print("\nTodos os valores (ordenados por frequência):")
    for bet_type, count in bet_types.items():
        print(f"  '{bet_type}': {count} ocorrências")

    # BET_LINE
    print(f"\n🎯 VALORES ÚNICOS - BET_LINE:")
    print("-" * 40)
    bet_lines = df["bet_line"].value_counts()
    print(f"Total de linhas diferentes: {len(bet_lines)}")
    print("\nTodos os valores (ordenados por frequência):")
    for bet_line, count in bet_lines.items():
        print(f"  '{bet_line}': {count} ocorrências")

    # Combinações bet_type + bet_line
    print(f"\n🔗 TODAS AS COMBINAÇÕES (BET_TYPE + BET_LINE):")
    print("-" * 60)
    combinations = (
        df.groupby(["bet_type", "bet_line"]).size().sort_values(ascending=False)
    )
    print(f"Total de combinações únicas: {len(combinations)}")
    print("\nTodas as combinações:")
    for i, ((bet_type, bet_line), count) in enumerate(combinations.items()):
        print(f"  {i + 1}. '{bet_type}' + '{bet_line}': {count} apostas")


# Para executar:
if __name__ == "__main__":
    # Substitua pelo caminho correto do seu arquivo
    file_path = "../bets/bets_atualizadas_por_mapa.csv"
    explore_betting_data(file_path)
