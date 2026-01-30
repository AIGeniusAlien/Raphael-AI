from sentence_transformers import SentenceTransformer
import numpy as np

_MODEL = None

def get_model(model_name: str):
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def embed_texts(texts, model_name: str):
    model = get_model(model_name)
    vecs = model.encode(texts, normalize_embeddings=True)
    return np.array(vecs, dtype=np.float32)
