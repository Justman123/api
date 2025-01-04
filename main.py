from fastapi import FastAPI
from controller import model

app = FastAPI()

app.include_router(model.router)

@app.get("/")
def read_root():
    return {"Hello World"}
