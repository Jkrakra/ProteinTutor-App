import os
import streamlit as st
import torch
import joblib
import ollama
import numpy as np
import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from txtai.embeddings import Embeddings
from txtai.pipeline import Labels
from transformers import pipeline, AutoTokenizer
st.set_page_config(layout="wide")

# --- 1. PROTEIN ANALYSIS CLASS ---


class ProteinTutor:
    def __init__(self, model_path, explainer_path, meta_path):
        self.model = joblib.load(model_path)
        self.explainer = joblib.load(explainer_path)
        self.feature_names = joblib.load(meta_path)

    def __call__(self, sequence):
        sequence = str(sequence).upper().strip()
        pa = ProteinAnalysis(sequence)
        feat_dict = pa.amino_acids_percent

        data_map = {
            "Length":len(sequence),
            "gravy": pa.gravy(),
            "iso_point": pa.isoelectric_point(),
            "molecular_weight": pa.molecular_weight(),
            **{a: feat_dict.get(a, 0) for a in "ACDEFGHIKLMNPQRSTVWY"}
        }

        features = [data_map[name] for name in self.feature_names]
        X = pd.DataFrame([features], columns=self.feature_names)

        pred = self.model.predict(X)[0]
        prob = self.model.predict_proba(X)[0]

        raw_shap = self.explainer.shap_values(X)
        explanation = raw_shap[int(pred)] if isinstance(
            raw_shap, list) else raw_shap

        return {
            "label": pred,
            "probability": max(prob),
            "explanation": np.array(explanation).flatten()
        }


@st.cache_resource
def initialize_backend():
    use_gpu = torch.cuda.is_available()

    tutor_model = ProteinTutor(
        model_path="protein_classifier.joblib",
        explainer_path="shap_explainer.joblib",
        meta_path="feature_meta.joblib"
    )

    classifier = Labels(
        "MoritzLaurer/ModernBERT-large-zeroshot-v2.0", gpu=use_gpu)

    embeddings = Embeddings(
        {"path": "sentence-transformers/all-MiniLM-L6-v2", "content": True})

    embeddings.index([
        (0, "EC 1: Oxidoreductases catalyze oxidation/reduction reactions.", None),
        (1, "EC 2: Transferases move functional groups.", None),
        (2, "EC 3: Hydrolases use water to break chemical bonds (e.g. Lysozyme).", None),
        (3, "EC 4: Lyases break bonds by means other than hydrolysis/oxidation.", None),
        (4, "EC 5: Isomerases catalyze structural shifts.", None),
        (5, "EC 6: Ligases join two large molecules together using ATP.", None),
        (6, "EC 7: Translocases move molecules across membranes.", None)
    ])

    return tutor_model, classifier, embeddings


tutor, classifier, embeddings = initialize_backend()


def ask_llama(prompt_text):
    response = ollama.chat(
        model='llama2',
        messages=[
            {'role': 'system', 'content': 'You are a professional Bio-Chemist assistant.'},
            {'role': 'user', 'content': prompt_text},
        ],
        options={'temperature': 0.7, 'num_predict': 200}
    )
    return response['message']['content']

with st.sidebar:
    st.header("🧬 Lab Input")
    sequence = st.text_area("Enter Protein Sequence:",
    value=st.session_state.get("sequence", ""),
    height=150)
    run_analysis = st.button("Run Analysis")

##Main User Interface
st.title("🧬 Protein AI Tutor")

if sequence and run_analysis:
    try:
        analysis = tutor(sequence)

        if analysis["label"] == 0:
            st.warning(f"⚠️ Non-Enzyme Detected. (Confidence: {analysis['probability']:.2%})")
        else:
            st.success(f"✅ Enzyme Confirmed! (Confidence: {analysis['probability']:.2%})")

            classes = ["Oxidoreductase", "Transferase", "Hydrolase", "Lyase", "Isomerase", "Ligase"]
            raw_pred= classifier(sequence, classes)
            st.write("Debug raw_pred", raw_pred)
            
            pred_index = raw_pred[0][0]
            prediction=classes[pred_index]
            st.write("DEBUG prediction:", prediction)

            fact_search = embeddings.search(f"Reaction of {prediction}", 1)
            fact = fact_search[0]['text'] if fact_search else "No specific EC fact found."

            st.session_state.prediction = prediction
            st.session_state.fact = fact
            st.session_state.sequence = sequence
            st.session_state.probability = analysis['probability']  # ← save this

    except Exception as e:
        st.error(f"Error analyzing sequence: {e}")

            # ✅ Completely outside try/except and if block
if st.session_state.get("prediction"):
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Predicted Class", st.session_state.prediction)
    with col2:
        st.metric("Model Confidence", f"{st.session_state.probability:.1%}")  # ← from session state
        
    st.divider()
    
    if st.button("Explain Mechanism"):
        with st.spinner("Analyzing biochemical properties..."):
            prompt = f"Explain why sequence {st.session_state.sequence[:20]}... is classified as a {st.session_state.prediction}. Reference this: {st.session_state.fact}"
            response_text = ask_llama(prompt)
            st.session_state.explanation = response_text

                                # ✅ Show explanation if it exists
if st.session_state.get("explanation"):
    st.info(st.session_state.explanation)
    
#Chat UI
st.divider()
st.subheader("Ask the Protein Tutor")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
try:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_input := st.chat_input("Ask about stability or active sites..."):
        st.session_state.chat_history.append({"role":"user", "content":user_input})
        
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                context = f"Sequence: {st.session_state.get('sequence', '')}" if st.session_state.get('sequence') else "No sequence provided"
                prompt = f"{context}\nUser: {user_input}"
                response = ask_llama(prompt)
            st.markdown(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
except Exception as e:
        st.error(f"Chat Error: {e}")