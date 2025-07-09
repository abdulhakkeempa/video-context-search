from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    top_k: int = 5

class SearchResult(BaseModel):
    video_name: str
    timestamp: str

class SearchResponse(BaseModel):
    results: list[SearchResult]