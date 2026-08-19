from fastapi import APIRouter
from pydantic import BaseModel

from port6.rag.chain import answer_query
from port6.schemas import AnswerResponse

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    message: str


@router.post("/query", response_model=AnswerResponse)
async def query(request: QueryRequest) -> AnswerResponse:
    return answer_query(request.message)
