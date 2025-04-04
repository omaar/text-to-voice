from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from text_to_voice.api.endpoints import text_to_voice
from text_to_voice.core.config import settings

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description=settings.DESCRIPTION,
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rutas
app.include_router(text_to_voice.router, prefix="/api/v1", tags=["text-to-voice"])

@app.get("/")
async def root():
    return {"message": "Bienvenido a la API de Text to Voice"} 