"""Liveness check for the client and CI."""

from importlib.metadata import version
from typing import Literal

from fastapi import APIRouter

from ryuk.api.schema import ApiModel

router = APIRouter(tags=["health"])


class Health(ApiModel):
    status: Literal["ok"]
    version: str


@router.get("/health", operation_id="getHealth")
def get_health() -> Health:
    return Health(status="ok", version=version("ryuk"))
