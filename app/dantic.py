from pydantic import BaseModel

class SegmentResult(BaseModel):
    cut1: float
    cut2: float


class EstimateResponse(BaseModel):

    segment: SegmentResult

    ate: list

    drcdf: list

    hei: dict