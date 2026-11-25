from pydantic import BaseModel,Field

class FrameEvidence(BaseModel):
    frame_index:int=Field(ge=0)
    timestamp_seconds:float|None=None
    probability_manipulated:float=Field(ge=0,le=1)
    suspicious:bool

class VideoDetectionResponse(BaseModel):
    video_id:str
    prediction:str
    probability_manipulated:float=Field(ge=0,le=1)
    certainty_claim:bool=False
    frame_evidence:list[FrameEvidence]
    interpretation:str
