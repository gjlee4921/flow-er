"""Pydantic request/response models shared by the API and the frontend's generated client."""
from pydantic import BaseModel, Field


class NewExperiment(BaseModel):
    dataset: str = Field(description="A dataset name from GET /datasets, or a folder path under DATA_DIR")
    title: str | None = None


class ProjectPatch(BaseModel):
    title: str | None = None
    archived: bool | None = None


class ChatIn(BaseModel):
    text: str = ""
    attachments: list[str] = []


class LabChatIn(BaseModel):
    text: str
    screen: dict = Field(default_factory=dict, description="What the data lab shows now: file, axes, population, view")


class GateIn(BaseModel):
    name: str
    parent: str | None = None
    dims: list[str] = Field(min_length=1, max_length=2, description="Two parameters, or one for a range")
    vertices: list[list[float]] = Field(min_length=2, description="Polygon corners, or [[low], [high]] for a range")
    kind: str = Field("polygon", description="polygon, rect or range")


class QuadrantIn(BaseModel):
    parent: str | None = None
    dims: list[str] = Field(min_length=2, max_length=2)
    centre: list[float] = Field(min_length=2, max_length=2, description="Where the four quarters meet, in display units 0..1")
    names: list[str] | None = Field(None, description="Upper left, upper right, lower right, lower left")
