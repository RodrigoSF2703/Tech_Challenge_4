import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Adicionar o caminho do diretório 'data' para garantir que o Python consiga encontrá-lo
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'data'))

from download_data import download_stock_data

# Baixar e carregar dados diretamente
symbol = 'INTC'

df = download_stock_data(symbol=symbol)

# Função para pré-processar e dividir os dados
def preprocess_data(df, test_size=0.2):
    # Selecionar a coluna de fechamento para previsão
    data = df['Close'].values
    data = data.reshape(-1, 1)  # Transformar para formato 2D

    # Normalizar os dados
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)

    # Criar sequência de dados para LSTM
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(seq_length, len(data)):
            sequences.append(data[i-seq_length:i, 0])
        return np.array(sequences)

    seq_length = 60  # Use os últimos 60 dias para prever o próximo

    sequences = create_sequences(data_scaled, seq_length)
    X = sequences[:-1]  # Exclui o último valor para previsões
    y = data_scaled[seq_length:, 0]  # Valor de fechamento correspondente

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test, scaler

def evaluate_model():
    model_path = os.path.join("model", "saved_model", "lstm_model.h5")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"O modelo salvo não foi encontrado em {model_path}. Treine o modelo antes de avaliá-lo.")

    # Carregar o modelo salvo
    model = load_model(model_path)
    print("Modelo carregado com sucesso!")

    # Baixar e pré-processar os dados
    df = download_stock_data(symbol='INTC')
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)  # Função de pré-processamento

    # Fazer previsões
    y_pred = model.predict(X_test)

    # Desnormalizar as previsões
    y_pred = scaler.inverse_transform(y_pred)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Calcular métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAPE: {mape:.2f}%")

    # Visualizar os resultados
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label="Valores Reais", color="blue")
    plt.plot(y_pred, label="Previsões", color="red")
    plt.title("Comparação entre Valores Reais e Previstos")
    plt.xlabel("Amostras")
    plt.ylabel("Preço")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model()
