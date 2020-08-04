from machine import Machine
from fastapi import FastAPI
from pydantic import BaseModel


class Data(BaseModel):
    N_LINHA: list
    N_CATEGORIA: list
    N_PRODUTO: list
    QTD_ITEM: list
    V_VENDA: list
    V_CUSTO_VENDA: list
    V_PERC_MARGEM_T: list


app = FastAPI()
machine = Machine()

@app.get('/')
def read_root():
    return {"message": "api-machine"}

@app.post('/predict')
def predict(data: Data):
    pred = machine.predict(data.json())
    print(pred)
    return {"predict": str(pred)}
