# Text to Voice API

API REST para convertir texto a voz utilizando FastAPI.

## Características

- Conversión de texto a voz
- Soporte para múltiples idiomas
- Conversión de PDF a voz
- API RESTful con documentación automática

## Requisitos

- Python 3.8 o superior
- Poetry para gestión de dependencias

## Instalación

1. Clonar el repositorio:

```bash
git clone <url-del-repositorio>
cd text-to-voice
```

2. Instalar dependencias con Poetry:

```bash
poetry install
```

3. Activar el entorno virtual:

```bash
poetry shell
```

## Uso

1. Iniciar el servidor:

```bash
uvicorn text_to_voice.main:app --reload
```

2. Acceder a la documentación de la API:

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Endpoints

- POST `/api/v1/text-to-voice`: Convierte texto a voz
- POST `/api/v1/pdf-to-voice`: Convierte un PDF a voz

## Estructura del Proyecto

```
text-to-voice/
├── src/
│   └── text_to_voice/
│       ├── api/          # Endpoints de la API
│       ├── core/         # Configuración y constantes
│       ├── models/       # Modelos y esquemas
│       ├── services/     # Lógica de negocio
│       └── utils/        # Utilidades
├── tests/               # Tests unitarios
└── pyproject.toml       # Dependencias y configuración
```
