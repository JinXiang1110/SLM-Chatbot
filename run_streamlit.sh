#!/bin/bash
# === Activate conda ===
source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm

# === Go to your project folder ===
cd /home/zorin17/Desktop/LLM/

# === Run Streamlit app ===
streamlit run deployment.py --server.address=0.0.0.0 --server.port=8501
