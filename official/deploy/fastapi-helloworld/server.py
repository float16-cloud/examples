import os
import uvicorn
from fastapi import FastAPI
app = FastAPI()

@app.get("/hello")
async def read_root():
    return {"message": "hi"}

async def main():
    config = uvicorn.Config(
        app, host="0.0.0.0", port=int(os.environ["PORT"])
    )
    server = uvicorn.Server(config)
    await server.serve()