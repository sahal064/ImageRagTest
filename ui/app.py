import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from PIL import Image

from agent.runner import run_agent
from retrieval.search import search_images

IMAGES_DIR = "images"

st.set_page_config(layout="wide")
st.title(" Image Search + Reasoing")

# ---------------- Sidebar ----------------
st.sidebar.header("Ask the Agent")
user_query = st.sidebar.text_input(
    "Search!!",
    placeholder="e.g. What was the increase in the population of India?"
)

# ---------------- Agent Q&A ----------------
if user_query:
    with st.spinner("Thinking..."):
        result = run_agent(user_query)

    st.subheader("Agent Answer")
    st.success(result["answer"])

    if result["images"]:
        st.subheader("Images Used")
        cols = st.columns(3)
        for idx, img in enumerate(result["images"]):
            col = cols[idx % 3]
            with col:
                path = img["path"]
                if os.path.exists(path):
                    st.image(Image.open(path), use_column_width=True)
                st.caption(img.get("caption"))

# ---------------- Image Browser ----------------
st.divider()
st.subheader(" Browse All Images")

image_files = [
    f for f in os.listdir(IMAGES_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

cols = st.columns(4)
for idx, img_name in enumerate(image_files):
    col = cols[idx % 4]
    with col:
        img_path = os.path.join(IMAGES_DIR, img_name)
        st.image(Image.open(img_path), caption=img_name, use_column_width=True)
