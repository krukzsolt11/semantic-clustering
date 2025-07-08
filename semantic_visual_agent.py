import streamlit as st
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import euclidean_distances
from PIL import Image
import os

# --- Load Models and Data ---
st.set_page_config(page_title="Semantic Visual Agent", layout="wide")
st.title("Semantic Visual Agent")
st.markdown("Welcome to the Semantic Visual Agent. Enter a text query to get started.")

# Load data and models
@st.cache_resource
def load_all():
    # Load embedding model
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Load cluster labels and image IDs (from HDBSCAN)
    cluster_data = np.load("./coco_dataset/llava_clusters_hdbscan_ollama.npz")
    image_ids = cluster_data["image_ids"]
    labels = cluster_data["labels"]

    embeddings = np.load("./coco_dataset/llava_embeddings.npz")["embeddings"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(embeddings)

    # Load Ollama-generated ontology (cluster keywords)
    with open("./coco_dataset/cluster_ontology_ollama.json", "r") as f:
        ontology = json.load(f)

    return embedder, image_ids, labels, embeddings, scaler, ontology, X_scaled

embedder, image_ids, labels, embeddings, scaler, ontology, X_scaled = load_all()

# --- Semantic Query Input ---
st.subheader("Enter a semantic query")
query = st.text_input("What are you looking for?", "people skiing")

if query:
    st.write(f"Received query: {query}")
    query_emb = embedder.encode([query])
    query_scaled = scaler.transform(query_emb)

    # Find closest point in the full embedding space
    distances = euclidean_distances(query_scaled, X_scaled)[0]
    top_idx = np.argsort(distances)[:3]
    top_image_ids = [image_ids[i] for i in top_idx if i < len(image_ids)]
    top_labels = [labels[i] for i in top_idx if i < len(labels)]

    # Show ontology (cluster topic keywords)
    if top_labels:
        topic_words = ontology.get(str(top_labels[0]), "[Unknown topic]")
        st.markdown(f"### 📍 Assigned Cluster: `{top_labels[0]}`")
        st.markdown("**Topic Keywords:** " + topic_words)

    # Load captions outside the loop
    with open("./coco_dataset/llava_captions.json", "r") as f_llava:
        llava_captions = json.load(f_llava)
    with open("./coco_dataset/annotations/captions_train2017.json", "r") as f_human:
        human_data = json.load(f_human)
        human_captions = {}
        for ann in human_data["annotations"]:
            img_key = str(ann["image_id"])
            if img_key not in human_captions:
                human_captions[img_key] = ann["caption"]

    sim_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Display results
    st.subheader("Closest Images in Embedding Space")
    cols = st.columns(len(top_image_ids))
    for i, img_id in enumerate(top_image_ids):
        img_str = str(int(img_id)).zfill(12)
        image_path = f"./coco_dataset/train/{img_str}.jpg"
        if not os.path.exists(image_path):
            image_path = f"./coco_dataset/test/{img_str}.jpg"
        if os.path.exists(image_path):
            img = Image.open(image_path)
            caption_llava = llava_captions.get(str(int(img_id)), "No LLaVA caption available.")
            caption_human = human_captions.get(str(int(img_id)), "No human caption available.")

            if caption_llava != "No LLaVA caption available." and caption_human != "No human caption available.":
                emb_llava = sim_model.encode(caption_llava, convert_to_tensor=True)
                emb_human = sim_model.encode(caption_human, convert_to_tensor=True)
                similarity = float(util.cos_sim(emb_llava, emb_human)[0][0])
                sim_text = f"**Similarity Score:** {similarity:.4f}"
            else:
                sim_text = "**Similarity Score:** N/A"

            cols[i].image(img, caption=f"ID: {img_str}", use_container_width=True)
            cols[i].markdown(f"**LLaVA Caption:** {caption_llava}")
            cols[i].markdown(f"**Human Caption:** {caption_human}")
            cols[i].markdown(sim_text)
        else:
            cols[i].markdown(f"Missing image: {img_str}")
