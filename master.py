from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
import uvicorn

app = FastAPI()


# Cargar el modelo pickle
with open('modelo_knn_classifier.pkl', 'rb') as file:
    model = pickle.load(file)


# Definir el esquema Pydantic para las entradas de la API
class ClassifierInput(BaseModel):
    total_docs: int
    new_cites: int
    best_quartile: int
    total_refs: int

# Endpoint para realizar predicciones
@app.post("/classifier")
async def classifier(data: ClassifierInput):
    # Convertir datos de entrada a un array numpy
    input_data = np.array([[data.total_docs, data.new_cites, data.best_quartile, data.total_refs]])
    
    # Realizar la predicción con el modelo
    classifier = model.predict(input_data)
    
    # Devolver la predicción como respuesta
    return {"classifier": classifier[0]}

@app.get("/")
def read_root():
    return {"message": "Hello, this is the root endpoint!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
# uvicorn main.py:app --reload