import logging
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from scalar_fastapi import get_scalar_api_reference
import uvicorn
import routes

logger = logging.getLogger("uvicorn.error")
logger.info("Server started on http://localhost:8080/docs")
tags_metadata = [
    {
        "name": routes.tag,
        "description": "Root and health check endpoints.",
    },
]

app = FastAPI(
    title="BookDB API",
    description="",
    version="0.6.7",
    openapi_url="/openapi.json",
    docs_url=None,
    redoc_url=None,
    debug=True,
    openapi_tags=tags_metadata,
)

app.include_router(
    routes.router,
)

# CORS Middleware
origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/docs", include_in_schema=False)
async def read_docs():
    return get_scalar_api_reference(
        openapi_url=app.openapi_url or "/openapi.json",
        title=app.title,
        hidden_clients={
            "dart": True,
            "ruby": True,
            "node": True,
            "php": True,
            "python": True,
            "c": True,
            "csharp": True,
            "clojure": True,
            "go": True,
            "http": True,
            "java": True,
            "kotlin": True,
            "objc": True,
            "ocaml": True,
            "powershell": True,
            "r": True,
            "swift": True,
        },
    )


# Run our uvicorn server on app startup
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)
