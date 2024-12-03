import os
import pickle
from model.lstm_model import create_model
from data.download_data import download_stock_data
from tensorflow.keras.callbacks import EarlyStopping

# Função para treinar o modelo
def train_model():
    symbol = 'INTC'
    output_file = 'data/dataset.csv'

    # Baixar e processar os dados
    X_train, X_test, y_train, y_test, scaler = download_stock_data(symbol, output_file)

    # Ajustar a forma dos dados para o modelo LSTM
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Salvar os dados escalonados em um arquivo pickle
    with open('data/scaled_data.pkl', 'wb') as f:
        pickle.dump(scaler, f)
        pickle.dump(X_train, f)
        pickle.dump(X_test, f)
        pickle.dump(y_train, f)
        pickle.dump(y_test, f)
    print("Dados escalonados salvos em 'scaled_data.pkl'")

    # Criar o modelo
    model = create_model((X_train.shape[1], 1))

    # EarlyStopping
    early_stopping = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)

    # Treinar o modelo
    model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

    # Salvar o modelo
    model_path = os.path.join("model", "saved_model", "lstm_model.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    model.save(model_path)
    print(f"Modelo salvo em: {model_path}")


if __name__ == "__main__":
    train_model()
