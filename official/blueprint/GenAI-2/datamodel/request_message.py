from typing import List, Optional, Union
from pydantic import BaseModel


class MessageInput(BaseModel):
    role: str
    content: Union[List[dict], str]


class ModelParamsOpenAI(BaseModel):
    messages: Optional[List[MessageInput]] = None
    model: Optional[str] = None
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 32768
    repetition_penalty: Optional[float] = 1.0
    end_id: Optional[int] = 2
    top_p: Optional[float] = 0.75
    top_k: Optional[int] = 40
    temperature: Optional[float] = 0.7
    stop: Optional[List[str]] = ["<bos>"]
    random_seed: Optional[int] = 2
    return_logs_prob: Optional[bool] = False
    response_format: Optional[dict] = None
    tools: Optional[List[dict]] = None
    tool_choice: Optional[str] = None
    reasoning_effort: Optional[str] = None


class EmbeddingParamsOpenAI(BaseModel):
    model: str
    input: str
    encoding_format: Optional[str] = "base64"
