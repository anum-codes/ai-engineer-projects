from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel

app = FastAPI()

@app.get("/")
def hello_server():
    return {"message": "Hello from FastAPI server!"}