import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

def download_stock_data(symbol='INTC', output_file='data/dataset.csv', test_size=0.2):
    """
    Faz o download de dados históricos de uma empresa e realiza o pré-processamento.
    """
    print(f"Baixando dados de {symbol} ...")
    df = yf.download(symbol, period='max')

    # Salvar os dados em um arquivo CSV
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file)
    print(f"Dados salvos em: {output_file}")

    df = df[['Close']].dropna()  # Seleciona apenas a coluna 'Close' e remove valores nulos

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df)

    # Criar as variáveis X e y (X = preços anteriores, y = preço futuro)
    X, y = [], []
    for i in range(60, len(df_scaled)):  # Usando os 60 dias anteriores para prever o próximo
        X.append(df_scaled[i-60:i, 0])  # Adiciona os 60 dias anteriores
        y.append(df_scaled[i, 0])  # O preço do próximo dia

    X, y = np.array(X), np.array(y)

    # Dividir em conjunto de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    return X_train, X_test, y_train, y_test, scaler


if __name__ == "__main__":
    # Configurações padrão
    symbol = 'INTC'

    # Chamar a função para download
    download_stock_data(symbol)
