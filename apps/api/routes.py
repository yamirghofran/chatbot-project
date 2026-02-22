import logging

from fastapi import APIRouter
from pydantic import BaseModel

logger = logging.getLogger("uvicorn.error")

router = APIRouter()
tag = "Home"


@router.get("/", tags=[tag])
async def read_root():
    logger.debug("Test Hello World message.")
    return {"Hello": "World"}


# Test Post Endpoint
class Test(BaseModel):
    num: int
    txt: str


@router.post("/testfunc", tags=[tag], response_model=Test)
async def read_func(var: Test):
    var.num += 1
    var.txt = var.txt.upper()
    return var
