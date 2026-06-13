from pydantic import BaseModel, Field


class PitchContext(BaseModel):
    p_throws: str
    stand: str
    inning_topbot: str
    on_1b: bool
    on_2b: bool
    on_3b: bool
    balls: int = Field(ge=0, le=3)
    strikes: int = Field(ge=0, le=2)
    outs_when_up: int = Field(ge=0, le=2)
    inning: int = Field(ge=1)
    home_score: int = Field(ge=0)
    away_score: int = Field(ge=0)
    pitch_number: int = Field(ge=1)
    prev_pitches: list[str] = []


class PredictResponse(BaseModel):
    pitch_type: str
    probabilities: dict[str, float]
