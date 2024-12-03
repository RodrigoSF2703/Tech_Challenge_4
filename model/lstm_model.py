# Importar as dependências
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def create_model(input_shape):
    model = Sequential()

    # Primeira camada LSTM com 50 unidades e sem regularização L2
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))  # Dropout para evitar overfitting

    # Segunda camada LSTM
    model.add(LSTM(50))  # Menos unidades
    model.add(Dropout(0.2))  # Dropout

    # Camada densa
    model.add(Dense(64, activation='relu'))  # Camada densa com ReLU
    model.add(Dense(1))  # Saída

    # Compilar o modelo
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')
    return model
