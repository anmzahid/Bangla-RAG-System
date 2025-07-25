
import fitz  # PyMuPDF
import re
import os

def extract_with_pymupdf(pdf_path):
    if not os.path.exists(pdf_path):
        print(f"Error: PDF file not found at {pdf_path}")
        return ""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for i, page in enumerate(doc):
            # Extract text blocks to preserve some structure
            blocks = page.get_text("dict")["blocks"]
            page_text = ""
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text_span = span["text"].strip()
                            if text_span:
                                page_text += text_span + " "
            if page_text:
                text += f"\n--- Page {i+1} ---\n{page_text.strip()}"
            else:
                print(f"Warning: Page {i+1} is empty or unreadable")
        doc.close()
        return text
    except Exception as e:
        print(f"Error with PyMuPDF: {e}")
        return ""

def clean_text(text):
    # Retain meaningful Bengali content, remove garbled and irrelevant data
    # Keep Bengali Unicode (\u0980-\u09FF), spaces, and punctuation
    text = re.sub(r'[^\u0980-\u09FF\s।?!,]', '', text)  # Remove non-Bengali characters
    text = re.sub(r'\n\s*\n+', '\n', text)  # Normalize newlines
    text = re.sub(r'\s+', ' ', text)  # Normalize spaces
    text = re.sub(r'(\b\w+\b)\s*\1\s*', r'\1 ', text)  # Remove repeated words
    # Remove common garbage patterns and metadata
    text = re.sub(r'क्ष+|प्रु+|वर्ह+', '', text)  # Remove garbled repetitions
    text = re.sub(r'\b\d+\b|\bHSC26\b|\bBangla1st\b|\bPaper\b|\bPage\b', '', text, flags=re.IGNORECASE)  # Remove page numbers, titles
    text = text.strip()
    return text

def is_text_meaningful(text):
    words = text.split()
    unique_words = set(words)
    if len(unique_words) < 20 or len(text) < 200:
        print("Warning: Text has too few unique words or is too short")
        return False
    if len(re.findall(r'क्ष+|प्रु+|वर्ह+', text)) > 0:
        print("Warning: Text contains garbled characters")
        return False
    return True

if __name__ == "__main__":
    pdf_path = "../Data/HSC26-Bangla1st-Paper.pdf"
    output_path = "../Data/extracted_text.txt"
    
    print("Extracting text with PyMuPDF...")
    extracted_text = extract_with_pymupdf(pdf_path)
    
    if extracted_text and is_text_meaningful(extracted_text):
        cleaned_text = clean_text(extracted_text)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(cleaned_text)
        print(f"Text extracted and cleaned, saved to {output_path}")
    else:
        print("Failed to extract meaningful text. Consider OCR or manual inspection.")
        # Optional OCR fallback (uncomment and install dependencies if needed)
        # from pdf2image import convert_from_path
        # import pytesseract
        # images = convert_from_path(pdf_path)
        # text = ""
        # for i, image in enumerate(images):
        #     text += pytesseract.image_to_string(image, lang='ben', config='--psm 6')
        # cleaned_text = clean_text(text)
        # with open(output_path, "w", encoding="utf-8") as f:
        #     f.write(cleaned_text)
        # print(f"Text extracted with OCR and saved to {output_path}")
