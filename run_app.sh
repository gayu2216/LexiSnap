#!/bin/bash
# Run LexiSnap with the project venv (has torch, transformers, domain_updated)
cd "$(dirname "$0")"
hack_ai_env/bin/streamlit run app.py
