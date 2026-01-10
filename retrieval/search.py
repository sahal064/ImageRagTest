# import faiss
# import pickle
# import torch
# import open_clip
# import numpy as np

# # Load index + metadata
# index = faiss.read_index("db_index.faiss")
# with open("db_metadata.pkl", "rb") as f:
#     metadata = pickle.load(f)

# # Load CLIP once
# model, _, preprocess = open_clip.create_model_and_transforms(
#     "ViT-B-32", pretrained="laion2b_s34b_b79k"
# )
# tokenizer = open_clip.get_tokenizer("ViT-B-32")
# model.eval()

# def search_images(query, k=5):
#     with torch.no_grad():
#         tokens = tokenizer([query])
#         text_emb = model.encode_text(tokens)
#         text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

#     D, I = index.search(text_emb.cpu().numpy(), k)

#     return [metadata[i] for i in I[0]]
# search.py

import faiss
import pickle
import torch
import open_clip
import numpy as np
from typing import List, Dict, Set

# -------------------- Load DBs --------------------

index = faiss.read_index("db/db_index.faiss")

with open("db/db_metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

# -------------------- Load CLIP --------------------

model, preprocess, _ = open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

tokenizer = open_clip.get_tokenizer("ViT-B-32")
model.eval()

PROMPT_TEMPLATES = [
    "a photo of {}",
    "an image of {}",
    "a picture showing {}",
    "a photograph of {}"
]

# -------------------- Utility Functions --------------------

def normalize_query(query: str) -> str:
    return query.lower().strip()

def extract_query_tags(query: str) -> Set[str]:
    return set(query.split())

def tag_boosting(record: Dict, query: str) -> float:
    return 0.05 if query in record.get("tags", []) else 0.0

def metadata_search(metadata: List[Dict], query_tags: Set[str]) -> List[int]:
    matched_ids = []
    for record in metadata:
        if set(record.get("tags", [])) & query_tags:
            matched_ids.append(record["id"])
    return matched_ids

# -------------------- Core Search --------------------

def search_images(
    query: str,
    k: int = 5,
    learn_tags: bool = True
) -> List[Dict]:
    """
    Hybrid image search using:
    - Metadata filtering
    - CLIP embedding search
    - Tag boosting
    """

    query = normalize_query(query)
    query_tags = extract_query_tags(query)

    # ---------- Metadata pre-filter ----------
    meta_ids = metadata_search(metadata, query_tags)
    meta_results = [m for m in metadata if m["id"] in meta_ids]

    # ---------- CLIP embedding search ----------
    prompts = [t.format(query) for t in PROMPT_TEMPLATES]
    text_tokens = tokenizer(prompts)

    with torch.no_grad():
        text_emb = model.encode_text(text_tokens)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

        query_emb = text_emb.mean(dim=0, keepdim=True)
        query_emb = query_emb / query_emb.norm(dim=-1, keepdim=True)

    scores, indices = index.search(query_emb.cpu().numpy(), k)

    # ---------- Build results ----------
    results = []
    for idx, score in zip(indices[0], scores[0]):
        record = metadata[idx]
        boosted_score = float(score + tag_boosting(record, query))

        results.append({
            "id": record["id"],
            "path": record["path"],
            "caption": record.get("caption"),
            "tags": record.get("tags", []),
            "score": boosted_score,
            "source": record.get("source"),
            "data":record.get("data") if record["data"] else None,
            "inference":record.get("inference") if record["inference"] else None,

        })

    # ---------- Online tag learning ----------
    if learn_tags:
        for r in results[:3]:
            meta_record = metadata[r["id"]]
            if query not in meta_record["tags"]:
                meta_record["tags"].append(query)

        # persist updates
        with open("db/db_metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

    return results

