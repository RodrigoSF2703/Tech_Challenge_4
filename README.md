# Previsão de Preços Futuros com LSTM

Este projeto utiliza redes neurais LSTM (Long Short-Term Memory) para prever os preços de um ativo financeiro baseado em dados históricos. O modelo é treinado com dados de preços anteriores e utilizado para prever o comportamento futuro do ativo em um número determinado de dias.

## Objetivo

O objetivo deste projeto é construir um modelo de previsão para o preço de um ativo financeiro utilizando redes neurais LSTM. A solução proposta tem como foco a previsibilidade de séries temporais e foi configurada para:
1. **Carregar um modelo LSTM** treinado com dados históricos de preços.
2. **Prever o valor futuro** do ativo para o número de dias especificado.
3. **Fornecer uma API** para que outros sistemas possam realizar previsões com base no modelo treinado.

## Estrutura do Projeto

O projeto consiste nos seguintes arquivos principais:
- **main.py**: Arquivo principal que executa a API FastAPI.
- **predict.py**: Contém a lógica de previsão dos preços futuros com base no modelo LSTM treinado.
- **model/saved_model/lstm_model.h5**: Modelo LSTM treinado e salvo para uso em previsões.
- **data/scaled_data.pkl**: Dados escalonados para alimentar o modelo.

## Arquitetura do Modelo

O modelo LSTM é composto por:
1. **Camada LSTM** com unidades suficientes para aprender padrões temporais dos dados de entrada.
2. **Previsão de múltiplos dias**: A previsão é feita para os próximos dias, conforme especificado pelo usuário, baseado no modelo treinado com dados históricos.
3. **Saída**: Uma previsão para cada dia futuro solicitado.

O modelo é carregado diretamente de um arquivo salvo, e as previsões são feitas para o número de dias determinado.

## Como Funciona

### Endpoints da API

- **POST /predict**
  - **Descrição**: Realiza previsões de preços futuros com base no número de dias especificado.
  - **Parâmetros**:
    - **dias_para_prever**: Número de dias futuros para previsão.
  - **Exemplo de Requisição**:
    ```json
    {
      "dias_para_prever": 5
    }
    ```
  - **Exemplo de Resposta**:
    ```json
    {
      "predictions": [150, 151, 152, 153, 154]
    }
    ```

### Funcionamento Interno

1. **Carregamento do Modelo LSTM**: Ao fazer uma solicitação para o endpoint de previsão, o modelo LSTM é carregado do arquivo **lstm_model.h5**.
2. **Processamento dos Dados**: O modelo usa os dados históricos de preços escalonados (salvos em **scaled_data.pkl**) para fazer as previsões. Ele considera os últimos 60 dias (definido por `look_back`) para prever os próximos dias.
3. **Previsão**: A cada previsão, o modelo atualiza a sequência de entrada para os próximos dias, iterando até o número de previsões solicitado.

## Como Executar

### 0. Passo Inicial

Guia de Instalação e Execução do Aplicativo Localmente

### 1. Instalar as Dependências

Certifique-se de que você tem o **Python 3.x** instalado. Em seguida, instale as dependências necessárias utilizando o arquivo `requirements.txt`. Execute o seguinte comando:

```bash
pip install -r requirements.txt
