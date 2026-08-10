# // schemas.py

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Literal

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

class ModelConfig(BaseModel):
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=1024, ge=1, le=8192)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    stop: Optional[List[str]] = None

    # Validación adicional si se requiere
    @validator('temperature')
    def validate_temperature(cls, v):
        if v < 0 or v > 2:
            raise ValueError('Temperature must be between 0 and 2')
        return v

class ModelResponse(BaseModel):
    text: str
    provider: str
    model: str
    usage: Optional[dict] = None
    error: Optional[str] = None
    success: bool = True