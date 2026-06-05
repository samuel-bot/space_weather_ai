<div align="center">

# 🛰️ SpaceWeather AI

### Previsão Climática Inteligente com Machine Learning e Dados Satelitais

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![API](https://img.shields.io/badge/Open--Meteo-API-00B4D8?style=for-the-badge&logo=cloudflare&logoColor=white)](https://open-meteo.com)

<br/>

> **Global Solution 2026.1 — FIAP**  
> Disciplina: Cognitive Computing, Computer Vision and IoT Systems  
> Professor: Arnaldo Viana | Turma: 4º SI — Fevereiro

---

**[📓 Notebook](#-estrutura-do-notebook) • [🚀 Como Rodar](#-como-rodar) • [📊 Dados](#-sobre-os-dados) • [🤖 Modelos](#-modelos-treinados) • [🌐 API](#-integração-com-api-em-tempo-real) • [📈 Resultados](#-resultados)**

</div>

---

## 🌍 O Problema

Neste exato momento, mais de **3.000 satélites meteorológicos** estão em órbita ao redor da Terra, coletando continuamente dados atmosféricos — temperatura, pressão, umidade, cobertura de nuvens, velocidade do vento. Satélites como o **GOES-16 (NASA/NOAA)** e o **Meteosat (EUMETSAT)** transmitem milhares de leituras por hora para centros de processamento em Terra.

O desafio é: **como transformar esse volume massivo de dados em previsões climáticas precisas e automatizadas?**

Sistemas manuais de análise são lentos e não escalam. Com o crescimento acelerado da economia espacial e o aumento exponencial dos dados gerados, a inteligência artificial deixou de ser opcional — ela se tornou a única abordagem viável.

---

## 💡 A Solução

O **SpaceWeather AI** é um sistema de classificação e previsão climática baseado em Machine Learning, que:

1. **Aprende padrões atmosféricos** a partir de 13.200 registros de dados reais
2. **Classifica o tipo de clima** (☀️ Sunny, ⛅ Cloudy, 🌧️ Rainy, ❄️ Snowy) com mais de **91% de acurácia**
3. **Faz previsões em tempo real** consumindo dados atuais de qualquer cidade do mundo via API gratuita

O pipeline reproduz, em escala educacional, o mesmo processo que sistemas satelitais usam para gerar previsões climáticas globais: **captura de dados atmosféricos → pré-processamento → inferência via IA**.

---

## 🛰️ Conexão com o Tema Space Connect

O projeto está diretamente alinhado ao tema **Space Connect — Soluções Tecnológicas para Terra e Espaço**, explorando como dados de origem espacial podem gerar impacto real na Terra:

| Área | Aplicação do SpaceWeather AI |
|------|------------------------------|
| 🌾 **Agronegócio** | Alertas automáticos de chuva para decisões de irrigação e colheita |
| 🚨 **Defesa Civil** | Previsão de condições extremas para acionamento preventivo de evacuações |
| ✈️ **Logística** | Planejamento de rotas aéreas e marítimas com base no clima previsto |
| ⚡ **Energia** | Previsão de geração solar e eólica baseada em cobertura de nuvens e vento |
| 🏙️ **Cidades Inteligentes** | Monitoramento climático urbano em tempo real |

---

## 📊 Sobre os Dados

O dataset contém **13.200 registros** com 10 variáveis atmosféricas — as mesmas captadas por satélites meteorológicos em operação.

| Variável | Tipo | Descrição | Por que é usada |
|----------|------|-----------|-----------------|
| `Temperature` | Numérico | Temperatura em °C | Principal diferenciador: neve < 0°C, sol > 25°C |
| `Humidity` | Numérico | Umidade relativa (%) | Alta umidade → maior chance de chuva ou neve |
| `Wind Speed` | Numérico | Velocidade do vento (km/h) | Ventos fortes associados a frentes de mau tempo |
| `Precipitation (%)` | Numérico | Probabilidade de precipitação | Diretamente ligada a climas úmidos (Rainy, Snowy) |
| `Cloud Cover` | Categórico | Clear / Partly Cloudy / Cloudy / Overcast | Diferencia dias abertos de nublados ou chuvosos |
| `Atmospheric Pressure` | Numérico | Pressão em hPa | Baixa pressão = frente de mau tempo; alta = tempo estável |
| `UV Index` | Numérico | Índice UV (0–11) | Alto UV = poucas nuvens = tempo aberto |
| `Season` | Categórico | Primavera / Verão / Outono / Inverno | Contextualiza a leitura atmosférica |
| `Visibility (km)` | Numérico | Distância de visão horizontal | Baixa visibilidade indica neblina, chuva ou neve |
| `Location` | Categórico | Coastal / Inland / Mountain | O terreno influencia o microclima local |
| `Weather Type` 🎯 | Categórico | **Sunny / Cloudy / Rainy / Snowy** | **Variável alvo — o que o modelo prevê** |

### Distribuição das Classes

O dataset é balanceado, com aproximadamente **25% de cada tipo de clima** — ideal para treinar classificadores sem viés.

```
☀️  Sunny   →  ~3.300 registros (25%)
⛅  Cloudy  →  ~3.300 registros (25%)
🌧️  Rainy   →  ~3.300 registros (25%)
❄️  Snowy   →  ~3.300 registros (25%)
```

---

## 📁 Estrutura do Projeto

```
SpaceWeatherAI/
│
├── 📓 GS_SpaceWeatherAI.ipynb          # Notebook principal — código completo
├── 📊 weather_classification_data.csv  # Dataset com 13.200 registros atmosféricos
├── 📄 README.md                        # Este arquivo
│
└── 📈 plots/                           # Gráficos gerados pelo notebook
    ├── plot_01_distribuicao_alvo.png
    ├── plot_02_temperatura.png
    ├── plot_03_umidade_precipitacao.png
    ├── plot_04_correlacao.png
    ├── plot_05_categoricas.png
    ├── plot_06_comparacao_modelos.png
    ├── plot_07_confusion_matrix.png
    ├── plot_08_feature_importance.png
    └── plot_09_dashboard_previsoes.png
```

---

## 📓 Estrutura do Notebook

O notebook está organizado em **8 seções**, cada uma com propósito bem definido:

### Seção 1 — Instalação de Dependências
> Célula de `pip install` para garantir que todas as bibliotecas estejam disponíveis no ambiente.

### Seção 2 — Importação de Bibliotecas
> Carregamento de todas as dependências: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn e Requests.

### Seção 3 — Carregamento e Primeiro Olhar nos Dados
> Leitura do CSV, visualização das primeiras linhas, estatísticas descritivas e verificação de valores nulos.

### Seção 4 — Análise Exploratória (EDA)
> 5 visualizações interativas:
> - Distribuição e proporção da variável alvo
> - Temperatura por tipo de clima (boxplot + histograma)
> - Umidade e precipitação por tipo de clima
> - Mapa de correlação entre variáveis numéricas
> - Distribuição das variáveis categóricas

### Seção 5 — Pré-processamento
> - Label Encoding para variáveis categóricas
> - StandardScaler para normalização
> - Split 80% treino / 20% teste com estratificação

### Seção 6 — Treinamento e Comparação de Modelos
> Treinamento de 3 algoritmos com validação cruzada 5-fold:
> - Random Forest (200 estimadores)
> - Gradient Boosting (150 estimadores)
> - Logistic Regression (baseline)

### Seção 7 — Avaliação Detalhada do Melhor Modelo
> - Relatório de classificação por classe (precision, recall, F1-score)
> - Matriz de confusão absoluta e normalizada
> - Feature Importance — quais variáveis mais influenciaram o modelo

### Seção 8 — Previsão em Tempo Real com API
> - Integração com **Open-Meteo API** (gratuita, sem chave)
> - Geocoding automático — qualquer cidade do mundo pelo nome
> - Pipeline completo: cidade → API → pré-processamento → modelo → previsão
> - Dashboard visual com probabilidades por classe para múltiplas cidades

### Seção 9 — Relatório Final
> Documentação técnica completa: problema, metodologia, resultados, conclusão e conexão com o tema Space Connect.

---

## 🤖 Modelos Treinados

| Modelo | CV Accuracy (5-fold) | Test Accuracy | Observação |
|--------|---------------------|---------------|------------|
| **Random Forest** ⭐ | ~91% | **>91%** | Melhor modelo — selecionado para produção |
| Gradient Boosting | ~90% | ~90% | Alta performance, mais lento para treinar |
| Logistic Regression | ~85% | ~85% | Baseline — referência de comparação |

### Features Mais Importantes (Random Forest)

As 3 variáveis mais decisivas para as previsões foram:

1. 🌡️ **Temperature** — principal separador entre tipos climáticos
2. 🌧️ **Precipitation (%)** — diretamente ligada a climas úmidos
3. ☁️ **Cloud Cover** — diferencia dias abertos de fechados

Exatamente as variáveis primárias captadas por satélites meteorológicos — o que valida a escolha do dataset.

---

## 🌐 Integração com API em Tempo Real

O projeto utiliza a **[Open-Meteo API](https://open-meteo.com/)** para buscar dados atmosféricos reais:

- ✅ **Gratuita** — sem custos e sem necessidade de cadastro
- ✅ **Sem chave de acesso** — funciona imediatamente
- ✅ **Cobertura global** — qualquer cidade do mundo
- ✅ **Dados atuais** — temperatura, umidade, pressão, vento, UV, visibilidade

### Como funciona o pipeline

```
Nome da Cidade
      │
      ▼
Geocoding API ──► Latitude / Longitude
      │
      ▼
Open-Meteo API ──► Dados Atmosféricos Atuais
      │
      ▼
Pré-processamento ──► Label Encoding + StandardScaler
      │
      ▼
Random Forest ──► Tipo de Clima + Probabilidades
```

### Exemplo de Saída

```
═══════════════════════════════════════════════════════
  🛰️  SpaceWeather AI — Previsão para: SAO PAULO
═══════════════════════════════════════════════════════
📍 Localização: São Paulo, Brasil
   Coordenadas: -23.5505°N, -46.6333°E

📡 Dados Atmosféricos Atuais:
   🌡️  Temperatura      : 22.4 °C
   💧 Umidade          : 78 %
   💨 Vento            : 14.0 km/h
   🌧️  Precipitação     : 45 %
   ☁️  Nuvens           : 65% → 'cloudy'
   🔵 Pressão          : 1013.2 hPa
   ☀️  UV Index         : 3
   👁️  Visibilidade     : 9.2 km
   🍂 Estação          : Autumn

──────────────────────────────────────────────────────
  🤖 PREVISÃO DO MODELO: 🌧️  RAINY
──────────────────────────────────────────────────────
  ☀️  Sunny    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  8.2%
  ⛅  Cloudy   ████████░░░░░░░░░░░░░░░░░░░░░░  24.1%
  🌧️  Rainy    ████████████████████░░░░░░░░░░  62.5%
  ❄️  Snowy    ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░  5.2%
═══════════════════════════════════════════════════════
```

---

## 🚀 Como Rodar

### Pré-requisitos

- Python 3.10 ou superior
- Jupyter Notebook ou JupyterLab
- Conexão com a internet (para a integração com API)

### Passo a Passo

**1. Clone o repositório**
```bash
git clone https://github.com/seu-usuario/SpaceWeatherAI.git
cd SpaceWeatherAI
```

**2. Abra o notebook**
```bash
jupyter notebook GS_SpaceWeatherAI.ipynb
```

**3. Execute a primeira célula (instalação)**

A primeira célula do notebook instala todas as dependências automaticamente:
```python
import sys
!{sys.executable} -m pip install scikit-learn matplotlib seaborn requests pandas numpy --quiet
```

**4. Execute todas as células em ordem**

Use `Kernel > Restart & Run All` ou execute célula por célula de cima para baixo.

**5. Teste com sua cidade**

Na seção de API, altere a lista de cidades para testar onde quiser:
```python
cities = [
    ('Sao Paulo',  'inland'),
    ('Rio de Janeiro', 'coastal'),
    ('Gramado',    'mountain'),
    ('Manaus',     'inland'),
]
```

### Dependências

```
pandas>=2.0
numpy>=1.24
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
requests>=2.31
jupyter>=1.0
```

---

## 📈 Resultados

| Métrica | Valor |
|---------|-------|
| **Acurácia no Teste** | > 91% |
| **Validação Cruzada (5-fold)** | ~91% ± 0.5% |
| **F1-Score médio (weighted)** | > 0.91 |
| **Registros no Dataset** | 13.200 |
| **Features utilizadas** | 10 |
| **Classes previstas** | 4 (Sunny, Cloudy, Rainy, Snowy) |
| **Cidades testadas via API** | Global (qualquer cidade) |

---

## 🧰 Tecnologias Utilizadas

| Tecnologia | Versão | Uso no Projeto |
|------------|--------|----------------|
| **Python** | 3.10+ | Linguagem principal |
| **Pandas** | 2.0+ | Manipulação e análise de dados |
| **NumPy** | 1.24+ | Operações numéricas |
| **Matplotlib** | 3.7+ | Visualizações e gráficos |
| **Seaborn** | 0.12+ | Gráficos estatísticos |
| **Scikit-learn** | 1.3+ | Treinamento e avaliação dos modelos de ML |
| **Requests** | 2.31+ | Consumo da API meteorológica |
| **Jupyter Notebook** | — | Ambiente de desenvolvimento interativo |
| **Open-Meteo API** | — | Dados atmosféricos em tempo real (gratuita) |

---

## 🎓 Contexto Acadêmico

Este projeto foi desenvolvido para a **Global Solution 2026.1 da FIAP**, com tema **Space Connect**.

**Critérios atendidos (disciplina Cognitive Computing, Computer Vision and IoT):**

- ✅ Ciência de Dados / Machine Learning aplicado ao tema da GS
- ✅ Exploração e análise de dados com EDA completa
- ✅ Identificação de padrões e geração de insights relevantes
- ✅ Treinamento e avaliação de modelos preditivos com métricas
- ✅ Solução conectada ao problema proposto
- ✅ Resultados técnicos mensuráveis (acurácia, F1, matriz de confusão)
- ✅ Código reprodutível em notebook com gráficos e evidências

**ODS da ONU contemplados:**
- 🌱 ODS 13 — Ação contra a Mudança Global do Clima
- 🌾 ODS 2 — Fome Zero e Agricultura Sustentável
- 🏙️ ODS 11 — Cidades e Comunidades Sustentáveis
- ⚙️ ODS 9 — Indústria, Inovação e Infraestrutura

---

## 👥 Equipe

| Nome | RM |
|------|----|
| [ Samuel Mota Gama de Abreu] | [97803] |
| [Thiago Ratão Passerini] | [551351] |
| [Gabriel Valério Gouveia] | [552041] |

---

## 📄 Licença

Este projeto foi desenvolvido para fins educacionais como parte da Global Solution FIAP 2026.1.

---

<div align="center">

**🛰️ SpaceWeather AI**  
*Transformando dados espaciais em inteligência para a Terra*

[![FIAP](https://img.shields.io/badge/FIAP-Global%20Solution%202026.1-C2185B?style=for-the-badge)](https://fiap.com.br)

</div>