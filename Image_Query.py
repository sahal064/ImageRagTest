import torch
from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image
import time
import json
from llama_cpp import Llama
# import os
# from transformers import logging as hf_logging
# hf_logging.set_verbosity_error()

# os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
# os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
# os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"


llm=Llama(
    model_path=r"C:\Users\sksha\.cache\huggingface\hub\models--microsoft--Phi-3-mini-4k-instruct-gguf\snapshots\a64113399c2f6b8ad3e11c394733a2ddadaa7f33\Phi-3-mini-4k-instruct-q4.gguf",
    n_ctx=1024,
    n_threads=4,
)
def clean_records(schema):
    schema["records"] = [
        r for r in schema["records"]
        if r["entity"] and str(r["entity"]).strip()
    ]
    return schema

print("Model Loaded Successfully!")
def inference_prompt(fact: str) -> str:
    print("Got The Schema ",fact)
    prompt= f"""
You are a data summarization assistant.
Do NOT add new facts.
Do NOT explain causes.P
List the Data Entities with Values
Only restate the given fact in one short, neutral sentence. Which is the Maximum and The Minimum Value Entity

Fact:
{fact}

Sentence:
""".strip()
    output = llm(
        prompt,
        max_tokens=40,
        temperature=0.0,
        stop=[]
    )

    return output["choices"][0]["text"].strip()
    




MODEL_NAME = "google/deplot"
def convert_to_schema(raw_text: str) -> dict:
    lines = raw_text.replace("<0x0A>", "\n").split("\n")
    lines = [l.strip() for l in lines if l.strip()]

    title = None
    if lines[0].lower().startswith("title"):
        title = lines.pop(0).split("|", 1)[1].strip()

    
    headers = [h.strip() for h in lines.pop(0).split("|")]

    entity_col = 0
    value_headers = headers[1:]

    records = []

    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        entity = parts[entity_col]

        values = {}
        for h, v in zip(value_headers, parts[1:]):
            try:
                values[h] = int(v)
            except ValueError:
                try:
                    values[h] = float(v)
                except ValueError:
                    values[h] = v

        records.append({
            "entity": entity,
            "values": values
        })

    d_schema={
        "chart": {
            "title": title
        },
        "headers": headers,
        "entity_column": entity_col,
        "records": records
    }
    return json.dumps(d_schema)

processor = Pix2StructProcessor.from_pretrained(MODEL_NAME)

model = Pix2StructForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    dtype=torch.float16,
)
model=model.to("cuda")
model.eval()
t0 = time.time()

image = Image.open("images/season.jpg")

# def query_image()
inputs = processor(
    images=image,
    text="Generate the underlying data table of this chart with organized labels like TITLE, ENTITY, VALUE, LEGEND, DESCRIPTION.",
    return_tensors="pt"
)
inputs = {k: v.to("cuda").half() for k, v in inputs.items()}


with torch.no_grad():   
    outputs = model.generate(
    **inputs,
    max_new_tokens=512
)

table_text = processor.decode(outputs[0], skip_special_tokens=True)
print(table_text)
print("Time: ",time.time()-t0)

headers=convert_to_schema(table_text)
# json_string = json.dumps(headers, indent=2)
# json_string=clean_records(json_string)
print(headers)

print("this is the Inference From The Data: ",inference_prompt(headers))
llm.close()
# print(headers)