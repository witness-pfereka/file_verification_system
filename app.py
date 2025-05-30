import streamlit as st
import fitz  # PyMuPDF
import pytesseract
import cv2
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # Change for your local or deployment path

# === Utility Functions ===
def pdf_to_images(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    images = []
    for page_num in range(len(doc)):
        pix = doc[page_num].get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            img = img[:, :, :3]
        images.append(img)
    return images

def extract_text_from_images(images):
    text = ""
    for img in images:
        text += pytesseract.image_to_string(img)
    return text.strip()

def compare_images(img1, img2):
    img1 = cv2.resize(img1, (600, 800))
    img2 = cv2.resize(img2, (600, 800))
    diff = cv2.absdiff(img1, img2)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
    return diff, thresh

def save_vectorizer(text, path="File_Verification.pkl"):
    vectorizer = TfidfVectorizer()
    vectorizer.fit([text])
    with open(path, "wb") as f:
        pickle.dump(vectorizer, f)

def load_vectorizer(path="File_Verification.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)

def compute_similarity(vectorizer, text1, text2):
    vec1 = vectorizer.transform([text1])
    vec2 = vectorizer.transform([text2])
    return cosine_similarity(vec1, vec2)[0][0]

# === Streamlit App ===
st.set_page_config(page_title="Document Verification", layout="wide")
st.title("📄 Document Verification System")

col1, col2 = st.columns(2)
with col1:
    orig_pdf = st.file_uploader("Upload Original Document (PDF)", type=["pdf"], key="orig")
with col2:
    fake_pdf = st.file_uploader("Upload Document to Verify (PDF)", type=["pdf"], key="fake")

if orig_pdf and fake_pdf:
    st.info("Extracting text and images...")

    # Image + Text Extraction
    orig_images = pdf_to_images(orig_pdf)
    fake_images = pdf_to_images(fake_pdf)
    text_orig = extract_text_from_images(orig_images)
    text_fake = extract_text_from_images(fake_images)

    st.subheader("📝 OCR Text Samples")
    with st.expander("Original Document Text"):
        st.text(text_orig[:1000])
    with st.expander("Document to Verify Text"):
        st.text(text_fake[:1000])

    # Save and Load Vectorizer
    save_vectorizer(text_orig)
    vectorizer = load_vectorizer()

    # Similarity Check
    similarity = compute_similarity(vectorizer, text_orig, text_fake)
    st.subheader("📊 Text Similarity")
    st.metric("Similarity Score", f"{similarity:.4f}")

    if similarity > 0.9:
        st.success("✅ Document is likely legitimate.")
    elif similarity > 0.6:
        st.warning("⚠️ Document may be a modified copy.")
    else:
        st.error("❌ Document is likely fake.")

    # Visual Comparison
    st.subheader("🖼️ Visual Comparison (First Page Only)")
    diff, thresh = compare_images(orig_images[0], fake_images[0])
    col1, col2, col3 = st.columns(3)
    col1.image(cv2.cvtColor(orig_images[0], cv2.COLOR_BGR2RGB), caption="Original")
    col2.image(cv2.cvtColor(fake_images[0], cv2.COLOR_BGR2RGB), caption="To Verify")
    col3.image(thresh, caption="Difference Map", use_column_width=True)
else:
    st.info("Please upload both documents to begin verification.")
