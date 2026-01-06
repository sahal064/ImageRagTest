import os 
import open_clip
import torch
from PIL import Image
import numpy as np
import faiss
import pickle
from transformers import BlipForConditionalGeneration,BlipProcessor

B_processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
B_model=BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

B_model.eval()

IMAGE_DIR="../images"

image_paths=[
    os.path.join(IMAGE_DIR,f)
    for f in os.listdir(IMAGE_DIR)
]

metadata=[]
for idx,path in enumerate(image_paths):
    img=Image.open(path).convert("RGB")
    inputs=B_processor(img,return_tensors="pt")
    with torch.no_grad():
        output=B_model.generate(**inputs)
    caption=B_processor.decode(output[0],skip_special_tokens=True)
    metadata.append({
        "id":idx,
        "path":path,
        "caption":caption,
        "tags":[],
        "source":None
    })


C_model,C_preprocess,_=open_clip.create_model_and_transforms(
    model_name="ViT-B-32",
    pretrained="laion2b_s34b_b79k"
)

# C_tokenzier=open_clip.get_tokenizer("ViT-B-32")
C_model.eval()

embeddings=[]

with torch.no_grad():
    for path in image_paths:
        img=Image.open(path).convert("RGB")
        img_tensor=C_preprocess(img).unsqueeze(0)

        emb=C_model.encode_image(img_tensor)
        emb=emb/emb.norm(dim=-1,keepdim=True)

        embeddings.append(emb.cpu().numpy())

image_embeddings=np.vstack(embeddings)

dim=image_embeddings.shape[1]
index=faiss.IndexFlatIP(dim)
print(metadata)
index.add(image_embeddings)
faiss.write_index(index,"try_index.faiss")
with open("cat_metadata.pkl","wb") as f:
    pickle.dump(metadata,f)
print("Total Images Indexed: ", index.ntotal)