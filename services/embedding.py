from typing import List
import clip
import torch
from PIL import Image
import numpy as np
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

device = "cuda" if torch.cuda.is_available() else "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)

def generate_text_embedding(query: str) -> List[float]:
    try:
        text = clip.tokenize([query]).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.cpu().numpy().tolist()[0]
    except Exception as e:
        print(f"Error generating text embedding: {e}")
        raise e


def generate_image_embedding(image: np.ndarray) -> List[float]:
    try:
        image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features.cpu().numpy().tolist()[0]
    except Exception as e:
        print(f"Error generating image embedding: {e}")
        raise e