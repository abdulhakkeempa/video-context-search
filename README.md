# Video Context Search

## What is this project?
This project is a video context search API that allows users to search for specific moments or scenes within a collection of videos using natural language queries. It leverages machine learning embeddings and FAISS (Facebook AI Similarity Search) for efficient similarity search over video frame representations. The API is built with FastAPI and uses Pydantic for data validation.

### How is it implemented?
- **Video Embeddings:** Each video is processed to extract frame-level embeddings using a text embedding model.
- **Indexing:** The embeddings are indexed using FAISS for searching using cosine similarity.
- **Metadata:** Video metadata (such as video name and timestamp) is stored in a pickle file for quick lookup.
- **API:** The FastAPI backend exposes a `/search` endpoint that takes a natural language query and returns the most relevant video segments.
- **Search:** When a query is received, it is embedded, searched against the FAISS index, and the top results are returned with video names and timestamps.

---

## How to set up and run locally

### Prerequisites
- Python 3.10 or higher
- [pip](https://pip.pypa.io/en/stable/)
- (Optional) [virtualenv](https://virtualenv.pypa.io/en/latest/) for isolated environments

### 1. Clone the repository
```powershell
git clone http://github.com/abdulhakkeempa/video-context-search.git
cd video-context-search
```

### 2. Create and activate a virtual environment (recommended)
```powershell
python -m venv venv
.\venv\Scripts\activate (Widnows)
source venv/bin/activate (Mac/Linux)
```

### 3. Install dependencies
```powershell
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

### 4. Generate embeddings from the video files
- Place your videos in `data/videos/` 
```powershell
python scripts/main.py
```

### 5. Ensure required data files are present
- `video_frames.index` (FAISS index file)
- `video_metadata.pkl` (Pickle file with video metadata)

### 6. Run the FastAPI server
```powershell
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`.

### 7. Test the API
You can use [Swagger UI](http://127.0.0.1:8000/docs) to interact with the `/api/search` endpoint.

---

## Example API Request
```json
POST /api/search
{
  "query": "A woman in water",
  "top_k": 5
}
```

---

## Project Structure
- `main.py` - FastAPI app entry point
- `api/routes.py` - API route definitions
- `services/` - Embedding, search, and indexing logic
- `models/schema.py` - Pydantic models
- `data/videos/` - Video files
- `video_frames.index` - FAISS index
- `video_metadata.pkl` - Video metadata

---

