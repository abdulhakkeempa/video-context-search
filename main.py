from fastapi import FastAPI
from api.routes import router as api_router

app = FastAPI(title="Video Context Search API")
app.include_router(api_router, prefix="/api", tags=["search"])

@app.get("/")
async def root():
    return {"message": "Welcome to the Video Context Search API. Use /api/search to perform searches."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")