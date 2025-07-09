import torch
from torchvision import transforms
import numpy as np
import faiss
import os
import cv2
from PIL import Image
from tqdm import tqdm


os.environ['KMP_DUPLICATE_LIB_OK']='True'


# === Config ===
VIDEO_FOLDER = "data/videos"
device = "cuda" if torch.cuda.is_available() else "cpu"
DIM = 512  # Output dimension from CLIP
SKIP_SECONDS = 2

# === Model and Preprocessing ===
# Example using CLIP
import clip
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()

# === FAISS Setup ===
index = faiss.IndexFlatIP(DIM)  # or faiss.IndexIDMap if you want to associate IDs
metadata = []  # To store (video_name, timestamp) for each vector

# === Loop through videos ===
for filename in os.listdir(VIDEO_FOLDER):
    if not filename.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        continue

    video_path = os.path.join(VIDEO_FOLDER, filename)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Cannot open {filename}")
        continue

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps else 0

    print(f"\nProcessing: {filename}")

    t = 0
    while t < duration:
        frame_number = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()

        if not ret:
            print(f"Warning: Frame read failed at t={t}s")
            t += SKIP_SECONDS
            continue

        # Convert frame to tensor and get embedding
        with torch.no_grad():
            image_input = preprocess(Image.fromarray(frame)).unsqueeze(0).to(device)
            embedding = model.encode_image(image_input)
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)  # normalize
            embedding_np = embedding.cpu().numpy().astype("float32")  # FAISS needs float32

        index.add(embedding_np)  # Add to FAISS
        metadata.append((filename, round(t, 2)))  # Store metadata

        t += SKIP_SECONDS

    cap.release()

# Save index and metadata
faiss.write_index(index, "video_frames.index")

import pickle
with open("video_metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)

print(f"\nSaved FAISS index with {index.ntotal} embeddings.")
