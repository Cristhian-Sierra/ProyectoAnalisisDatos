from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI()

# Cargar el modelo pickle y el escalador
with open('modelo_knn_predict.pkl', 'rb') as file:
    model, scaler= pickle.load(file)


class PredictionInput(BaseModel):
    total_docs: int
    new_cites: int
    best_quartile: int
    total_refs: int
    sjr: float

@app.post("/predict")
async def predict(data: PredictionInput):
    # Convertir datos de entrada a un array numpy
    input_data = np.array([[data.total_docs, data.sjr, data.new_cites, data.best_quartile, data.total_refs]])

    # Escalar datos con MinMaxScaler
    input_data_scaled = scaler.transform(input_data)

    # Realizar la predicción con el modelo
    prediction = model.predict(input_data_scaled)
    
    # Devolver la predicción como respuesta
    return {"prediction": float(prediction[0])}

@app.get("/")
def read_root():
    return {"message": "Hello, this is the root endpoint!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn main.py:app --reload