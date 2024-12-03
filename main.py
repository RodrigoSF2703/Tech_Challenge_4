from tensorflow.keras.models import load_model
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
import pickle


# Definindo a classe para a entrada
class PredictionInput(BaseModel):
    dias_para_prever: int  # Apenas o número de dias a ser previsto é enviado


# Função para prever os preços futuros com base no modelo treinado
def predict_future_price(dias_para_prever):
    # Carregar o modelo treinado
    model = load_model('model/saved_model/lstm_model.h5')  # Caminho para o modelo salvo

    # Carregar os dados processados (estes são os dados que você usou para treinar o modelo)
    with open('data/scaled_data.pkl', 'rb') as file:
        scaled_data = pickle.load(file)  # Dados escalonados para o modelo

    # O look_back fixo (últimos 60 dias, como no treinamento)
    look_back = 60

    # Preparar a entrada para o modelo com os dados escalonados
    current_input = scaled_data[-look_back:].reshape(1, look_back, 1)

    # Lista para armazenar as previsões dos próximos dias
    predictions = []

    # Prever para os dias solicitados
    for _ in range(dias_para_prever):
        next_day_prediction = model.predict(current_input)
        next_day_prediction = next_day_prediction[0][0]  # A previsão do próximo dia

        predictions.append(next_day_prediction)

        # Atualizar os dados de entrada para o próximo dia
        current_input = np.append(current_input[:, 1:, :], next_day_prediction.reshape(1, 1, 1), axis=1)

    # Converte as previsões para inteiros (caso necessário)
    predictions = [int(value) for value in predictions]

    return {"predictions": predictions}


# Criação da aplicação FastAPI
app = FastAPI()


# Rota para receber as previsões
@app.post("/predict")
def predict(data: PredictionInput):
    if data.dias_para_prever <= 0:
        return {"error": "O número de dias para previsão deve ser maior que zero."}
    return predict_future_price(data.dias_para_prever)
