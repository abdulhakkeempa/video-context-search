from services.embedding import generate_text_embedding
import numpy as np
import faiss
import pickle
from utils.timestamp import convert_second_to_timestamp

faiss_index = faiss.read_index("video_frames.index")

with open("video_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

def search(query: str, top_k: int = 5) -> list:
    try:
        query_vector = generate_text_embedding(query)

        distances, indices = faiss_index.search(np.array([query_vector], dtype=np.float32), k=top_k)

        results = []

        for i, index in enumerate(indices[0]):
            distance = distances[0][i]
            video_name, timestamp = metadata[index]
            print(f"Nearest neighbor {i+1}: {video_name} - {timestamp}, Distance {distance}")

            results.append({
                "video_name": video_name,
                "timestamp": convert_second_to_timestamp(timestamp),
                "distance": distance
            })

        return results

    except Exception as e:
        print(f"Error during search: {e}")
        raise e

# search("A woman in water", top_k=5)

# import cProfile
# import pstats

# cProfile.run("search('A woman in water', top_k=5)", "search_profile.prof")

# # Optionally, view sorted stats:
# stats = pstats.Stats("search_profile.prof")
# stats.strip_dirs().sort_stats("cumtime").print_stats(10)  # Top 10 slowest calls
