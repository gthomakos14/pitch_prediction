from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from api.model_loader import load_artifacts
from api.routes import health, predict


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.bundle = load_artifacts()
    yield


app = FastAPI(title="Pitch Prediction API", version="0.1.0", lifespan=lifespan)
app.include_router(health.router)
app.include_router(predict.router)
