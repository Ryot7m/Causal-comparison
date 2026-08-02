from pydantic import BaseModel

class SegmentResult(BaseModel):
    cut1: float
    cut2: float
    
class AteResult(BaseModel):
    cls : int
    clsnum : int
    ate : float
    se : float
    ci_low : float
    ci_high : float
    
class DrcdfResult(BaseModel):
    seg: int
    c: int
    F1_dr: float
    F0_dr: float
    tau_c: float
    se_c: float
    ci_low: float
    ci_high: float
    
class HeiResult(BaseModel):
    score : float

class EstimateResponse(BaseModel):

    segment: SegmentResult

    ate: list[AteResult]

    drcdf: list[DrcdfResult]

    hei: HeiResult