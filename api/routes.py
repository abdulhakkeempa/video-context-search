from fastapi import APIRouter
from models.schema import SearchQuery, SearchResponse, SearchResult
from services.search import search

router = APIRouter()

@router.post("/search")
async def search_video(query: SearchQuery) -> SearchResponse:
    try:
        results = search(query.query, top_k=query.top_k)
        search_results = [SearchResult(**r) if isinstance(r, dict) else r for r in results]
        return SearchResponse(results=search_results)
    except Exception as e:
        return SearchResponse(results=[])
