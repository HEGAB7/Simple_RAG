from utils import query
from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello, Working"}

@app.post('/query/{question}')
def qa(question):
    response = query(question)
    return {'Answer': response}


if __name__ == "__main__":
    uvicorn.run(app)