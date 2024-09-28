import streamlit as st
from PIL import Image
import pytesseract
import cv2
import numpy as np
from transformers import pipeline
from fpdf import FPDF
import PyPDF2
import re
import os

# Explicitly set the Tesseract executable path
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Disable symlink warnings for Hugging Face cache if necessary
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Function to preprocess images (convert to grayscale, enhance contrast)
def preprocess_image(image):
    image_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    enhanced_image = cv2.equalizeHist(gray)
    return enhanced_image

# Function to highlight keyword in text
def highlight_text(text, keyword):
    highlighted_text = re.sub(f"({keyword})", r'**\1**', text, flags=re.IGNORECASE)
    return highlighted_text

# Attempt to load the summarization model and handle potential connection errors
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {str(e)}. Please check your internet connection or try again later.")
    model_loaded = False

# Streamlit app title
st.title('OCR and Document Search Application')

# Allow multiple files (images or PDFs) to be uploaded
uploaded_files = st.file_uploader("Upload images or PDFs", type=["png", "jpg", "jpeg", "pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        # Check if the file is a PDF or an image
        if uploaded_file.type == "application/pdf":
            # Extract text from the PDF
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            pdf_text = ""
            for page in pdf_reader.pages:
                pdf_text += page.extract_text()
            
            st.text_area(f"Extracted Text from {uploaded_file.name}", pdf_text)

            # Search functionality in PDF text
            search_term = st.text_input(f"Enter keyword to search in {uploaded_file.name}:")
            if search_term:
                if search_term.lower() in pdf_text.lower():
                    highlighted_text = highlight_text(pdf_text, search_term)
                    st.markdown(highlighted_text)
                else:
                    st.write(f"'{search_term}' not found in the text.")

        else:
            # Process the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded Image: {uploaded_file.name}")
            
            # Image preprocessing (grayscale and contrast enhancement)
            preprocessed_image = preprocess_image(image)
            st.image(preprocessed_image, caption="Preprocessed Image for Better OCR", use_column_width=True)

            # OCR: Extract text from the preprocessed image
            language = st.selectbox("Select language for OCR", ['eng', 'hin', 'spa', 'fra'], key=uploaded_file.name)
            text = pytesseract.image_to_string(preprocessed_image, lang=language)
            st.text_area(f"Extracted Text from {uploaded_file.name}", text)

            # Search functionality in image text
            search_term = st.text_input(f"Enter keyword to search in {uploaded_file.name}:")
            if search_term:
                if search_term.lower() in text.lower():
                    highlighted_text = highlight_text(text, search_term)
                    st.markdown(highlighted_text)
                else:
                    st.write(f"'{search_term}' not found in the text.")

            # Text summarization
            if model_loaded and st.button(f"Summarize Text in {uploaded_file.name}"):
                try:
                    # Adjust max_length based on input length and ensure a minimum
                    input_length = len(text.split())
                    max_length = max(50, min(150, input_length))  # Ensure at least 50, and no more than 150
                    max_new_tokens = max_length  # Set max new tokens to be the same as max_length

                    # Perform summarization with the adjusted max_length and max_new_tokens
                    summary = summarizer(text, max_length=max_length, min_length=1, max_new_tokens=max_new_tokens, clean_up_tokenization_spaces=True)
                    st.write("Summary of Extracted Text:")
                    st.write(summary[0]['summary_text'])
                except Exception as e:
                    st.error(f"Error during summarization: {str(e)}")

            # Download extracted text as .txt
            st.download_button(
                label="Download Text as .txt",
                data=text,
                file_name=f'{uploaded_file.name}_extracted_text.txt',
                mime='text/plain'
            )

            # Download extracted text as .pdf
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, text)
            pdf_output = pdf.output(dest='S').encode('latin1')
            st.download_button(
                label="Download as PDF",
                data=pdf_output,
                file_name=f'{uploaded_file.name}_extracted_text.pdf',
                mime="application/pdf"
            )
