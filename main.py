from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

 
# Cargar el modelo pickle y el escalador
with open('modelo_knn_predict.pkl', 'rb') as file1:
    model1, scaler1= pickle.load(file1)

# Cargar el modelo pickle
with open('modelo_knn_classifier.pkl', 'rb') as file2:
    model2 = pickle.load(file2)


# Definir el esquema Pydantic para las entradas de la API
class InputP(BaseModel):
    total_docs: int
    new_cites: int
    best_quartile: int
    total_refs: int
    sjr: float

class InputC(BaseModel):
    total_docs: int
    new_cites: int
    best_quartile: int
    total_refs: int

@app.post("/predict")
async def predict(data: InputP):
    # Convertir datos de entrada a un array numpy
    input_data= np.array([[data.total_docs, data.sjr, data.new_cites, data.best_quartile, data.total_refs]])

    # Escalar datos con MinMaxScaler
    input_data_scaled = scaler1.transform(input_data)

    # Realizar la predicción con el modelo
    prediction = model1.predict(input_data_scaled)
    
    # Devolver la predicción como respuesta
    return {"prediction": float(prediction[0])}
# Endpoint para realizar predicciones

@app.post("/classifier")
async def classifier(data: InputC):
    # Convertir datos de entrada a un array numpy
    input_data = np.array([[data.total_docs, data.new_cites, data.best_quartile, data.total_refs]])
    
    # Realizar la predicción con el modelo
    classifier = model2.predict(input_data)
    
    # Devolver la predicción como respuesta
    return {"classifier": classifier[0]}


@app.get("/")
def read_root():
    return {"message": "Hello, this is the root endpoint!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn main:app --reload
#ngrok http 8000