from pydantic import BaseModel


class JsonResponse(BaseModel):
    text: str
