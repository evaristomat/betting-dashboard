# 🎲 Betting Statistics Dashboard

Dashboard interativo profissional para análise de apostas esportivas, criado com Streamlit e Plotly.

![Dashboard Preview](https://i.imgur.com/placeholder.png)

## ✨ Características

### 🎨 Visual
- **Tema escuro** moderno e profissional
- **Layout responsivo** que se adapta a qualquer tela
- **Cards informativos** com métricas principais
- **Gráficos interativos** com hover e zoom
- **Design idêntico** ao dashboard de referência

### 📊 Funcionalidades
- **Filtro por ROI**: Slider para filtrar apostas por ROI mínimo
- **Métricas em tempo real**: Win rate, lucro total, odds média
- **Gráfico de evolução**: Lucro cumulativo + média móvel
- **Dados detalhados**: Tabelas expandíveis com últimas apostas
- **Performance por jogo**: Estatísticas individuais por mapa

## 🚀 Instalação e Execução

### Método 1: Execução Automática
```bash
# Clone ou baixe os arquivos do projeto
# Execute o script automatizado:
python run_dashboard.py
```

### Método 2: Execução Manual
```bash
# 1. Instale as dependências
pip install streamlit pandas plotly numpy

# 2. Execute o dashboard
streamlit run app.py
```

### Método 3: Com Dados de Exemplo
```bash
# Se não tiver dados reais, gere dados de exemplo:
python sample_data_generator.py

# Depois execute o dashboard:
python run_dashboard.py
```

## 📁 Estrutura do Projeto

```
betting-dashboard/
├── app.py                              # 🎯 Dashboard principal
├── run_dashboard.py                    # 🚀 Script de execução
├── sample_data_generator.py            # 📊 Gerador de dados exemplo
├── requirements.txt                    # 📦 Dependências
├── future_features.py                  # 🔮 Funcionalidades futuras
├── README.md                          # 📖 Este arquivo
└── bets/
    └── bets_atualizadas_por_mapa.csv  # 📈 Seus dados (required)
```

## 📊 Formato dos Dados

O arquivo `bets/bets_atualizadas_por_mapa.csv` deve ter estas colunas:

| Coluna | Tipo | Descrição | Exemplo |
|--------|------|-----------|---------|
| `date` | string | Data e hora da aposta | "22 Mai 2025 13:00" |
| `league` | string | Liga/campeonato | "LCK", "LPL", "VCT" |
| `t1` | string | Time 1 | "T1", "Gen.G" |
| `t2` | string | Time 2 | "DRX", "Damwon" |
| `bet_type` | string | Tipo da aposta | "over", "under" |
| `bet_line` | string | Linha da aposta | "total_kills 25.5" |
| `bet_result` | number | Resultado real | 27.0, 4.0, 31.5 |
| `ROI` | number | ROI da aposta (%) | 15.5, 22.3 |
| `fair_odds` | number | Odds justas | 1.85, 2.10 |
| `odds` | number | Odds da casa | 1.90, 2.25 |
| `House` | string | Casa de aposta | "BET365", "Pinnacle" |
| `url` | string | URL da aposta | "url não disponível" |
| `status` | string | Resultado | "win", "loss" |
| `telegram` | number | Campo telegram | 0.0, 1.0 |
| `profit` | number | Lucro/prejuízo | 0.85, -1.0 |
| `game` | number | Número do mapa | 1, 2, 3, 4, 5 |

## 🎯 Como Usar

### 1. Carregamento Inicial
- Dashboard carrega automaticamente os dados do CSV
- Mostra métricas gerais no topo da página

### 2. Filtro por ROI
- Use o slider para filtrar apostas por ROI mínimo
- Métricas e gráfico atualizam automaticamente
- Valores de 0% a 65% disponíveis

### 3. Análise das Métricas
- **ROI Chosen**: Valor selecionado no filtro
- **Total Bets**: Apostas que passaram no filtro
- **Wins/Losses**: Vitórias e derrotas
- **Win Rate**: Taxa de acerto (%)
- **Average Odd**: Odd média das apostas
- **Total Profit**: Lucro total em unidades

### 4. Gráfico Interativo
- **Linha azul**: Evolução do lucro cumulativo
- **Linha vermelha**: Média móvel de 10 apostas
- **Hover**: Informações detalhadas ao passar mouse
- **Zoom**: Clique e arraste área para zoom
- **Pan**: Shift + clique para mover visualização

### 5. Dados Detalhados
- Expandir seção "📊 Dados Detalhados"
- Ver últimas 10 apostas em tabela
- Estatísticas resumidas por jogo/mapa

## 🔧 Personalização

### Alterar Tema/Cores
Edite as variáveis CSS em `app.py`:
```python
# Cores principais
background_color = "#1e2329"    # Fundo geral
card_color = "#2d3748"         # Cor dos cards
text_color = "#ffffff"         # Cor do texto
profit_color = "#48bb78"       # Verde para lucro
loss_color = "#f56565"         # Vermelho para prejuízo
```

### Adicionar Métricas
```python
# Exemplo: Nova métrica
with st.columns(1)[0]:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Nova Métrica</div>
        <div class="metric-value">{valor_calculado}</div>
    </div>
    """, unsafe_allow_html=True)
```

### Modificar Filtros
```python
# Exemplo: Adicionar filtro de liga
selected_leagues = st.multiselect(
    "Selecione as ligas:",
    options=df['league'].unique(),
    default=df['league'].unique()
)
filtered_df = df[df['league'].isin(selected_leagues)]
```

## 🔮 Funcionalidades Futuras

O arquivo `future_features.py` contém código para:

- 🔧 **Sidebar com filtros avançados** (data, liga, casa, tipo)
- 📊 **Gráficos adicionais** (heatmap, distribuição ROI, performance mensal)
- 📈 **Métricas avançadas** (Sharpe ratio, drawdown, Kelly criterion)
- 📥 **Export para Excel** com múltiplas abas
- 🚨 **Alertas automáticos** (sequências de derrotas, ROI baixo)
- 📱 **Layout mobile-friendly**

## 🌐 Deploy/Hospedagem

### Streamlit Cloud (Gratuito)
1. Faça upload para GitHub
2. Conecte em https://share.streamlit.io
3. Deploy automático

### Heroku
```bash
# Criar Procfile
echo "web: streamlit run app.py --server.port=\$PORT --server.address=0.0.0.0" > Procfile

# Deploy
git add .
git commit -m "Deploy dashboard"
git push heroku main
```

### Railway/Render
- Conecte repositório GitHub
- Configure build command: `pip install -r requirements.txt`
- Configure start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

## 🔍 Troubleshooting

### ❌ "File not found"
- Verifique se `bets/bets_atualizadas_por_mapa.csv` existe
- Use `python sample_data_generator.py` para gerar dados teste

### 📊 Gráfico vazio
- Confirme se há dados após aplicar filtro ROI
- Tente diminuir o valor mínimo do ROI

### 🐌 Performance lenta
- Limite número de apostas (últimas 1000)
- Use `@st.cache_data` em funções pesadas

### 🎨 Layout quebrado
- Atualize Streamlit: `pip install -U streamlit`
- Limpe cache: `streamlit cache clear`

## 📞 Suporte

- **Streamlit Docs**: https://docs.streamlit.io
- **Plotly Docs**: https://plotly.com/python/
- **Pandas Docs**: https://pandas.pydata.org/docs/

## 📄 Licença

MIT License - Veja arquivo LICENSE para detalhes.

---

**Desenvolvido com ❤️ usando Streamlit + Plotly**

*Dashboard profissional para análise de apostas esportivas*