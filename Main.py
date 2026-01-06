import os 
import open_clip
import torch
from PIL import Image
import numpy as np
import faiss
import pickle
import cv2
import time


###############################Loading DBs###############################
index=faiss.read_index("image_index.faiss")
with open("image_metadata.pkl","rb") as f:
    metadata=pickle.load(f)


###############################All Functions###############################
def tag_boosting(record,query):
    return 0.05 if query in record["tags"] else 0.00 


def meta_serach(metadata,query_tags):
    matched_ids=[]
    for record in metadata:
        record_tags=set(record["tags"])
        if record_tags & query_tags:
            matched_ids.append(record["id"])
    
    return matched_ids


def normalize_q(query:str) -> str:
    return query.lower().strip()

def extract_query_tags(query):
    return set(query.split(" "))


model,preprocess,_=open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)
tokenzier=open_clip.get_tokenizer("ViT-B-32")
model.eval()
templates = [
    "a photo of {}",
    "an image of {}",
    "a picture showing {}",
    "a photograph of {}"
]

while True:
    Query=input("Enter The Cat Color to Search")
    t0 = time.time()

    Query=normalize_q(Query)
    query_tags=extract_query_tags(Query)

    ids=meta_serach(metadata,query_tags)

    f_list=[elem for elem in metadata if elem["id"] in ids]

    [print("This is the Final Result using Meta Search: ", p["path"]) for p in f_list]
    print("Time With Search using Metatags:", time.time() - t0)


    

    prompts=[t.format(Query) for t in templates]

    text_tokens=tokenzier(prompts)
    with torch.no_grad():
        text_emb=model.encode_text(text_tokens)

    text_emb/=text_emb.norm(dim=-1,keepdim=True)

    en_emb=text_emb.mean(dim=0,keepdim=True)
    en_emb/=en_emb.norm(dim=0,keepdim=True)



    scores,indices=index.search(en_emb.cpu().numpy(),k=5)
    top_indices=indices[0][:3]

    results=[]
    for idx,score in zip(indices[0],scores[0]):
        record=metadata[idx]
        score+=tag_boosting(record,Query)
        results.append({
            "score":float(score),
            "path":record["path"],
            "caption":record["caption"],
            "tags":record["tags"],
            "source":record["source"],

        })


    filtered=[
        r for r in results
        if Query in r["tags"]
    ]

    if len(filtered)==0:
        print("Unfiltered List")
        for r in results:
            print(r["path"],r["score"],r["caption"])
      

    elif len(filtered)>0:
        print("Filtered List")
        for f in filtered:
            print(f["path"],f["score"])
    print("Time Without Meta Search:", time.time() - t0)


     
    for idx in top_indices:
        
        if Query not in metadata[idx]["tags"]:
            metadata[idx]["tags"].append(Query)

    cont=input("Break?: ")
    if cont == "y":
        break
    else:
        continue


with open("cat_metadata.pkl","wb") as f:
    pickle.dump(metadata,f)