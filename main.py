from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
import fitz
import os
from dotenv import load_dotenv
import pdfplumber

# OCR fallback
from pdf2image import convert_from_path
import pytesseract

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def generate_study_content(text: str):
    # IMPORTANT: don't send huge text in one go; see chunking note below
    prompt = f"""
You are a helpful study assistant. Based on the following study material, provide:

1. SUMMARY: A clear concise summary (max 150 words)
2. KEY CONCEPTS: List the 5 most important concepts
3. PRACTICE QUESTIONS: Generate 5 practice questions (mix of MCQ and short answer)
4. KEY TERMS: List important terms and their definitions

Study Material:
{text}

Format your response clearly with the headers: SUMMARY, KEY CONCEPTS, PRACTICE QUESTIONS, KEY TERMS
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

def extract_text_pdfplumber(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text += (page.extract_text() or "") + "\n"
    return text.strip()

def extract_text_pymupdf(path: str) -> str:
    doc = fitz.open(path)
    parts = []
    for page in doc:
        parts.append(page.get_text("text") or "")
    return "\n".join(parts).strip()

def extract_text_ocr(path: str) -> str:
    # Render PDF pages to images, then OCR each page.
    images = convert_from_path(path, dpi=300)  # higher DPI => better OCR, slower
    ocr_parts = []
    for img in images:
        ocr_parts.append(pytesseract.image_to_string(img))
    return "\n".join(ocr_parts).strip()

def smart_extract_text(path: str) -> tuple[str, str]:
    """
    Returns (text, method_used)
    """
    text = extract_text_pdfplumber(path)
    if len(text) > 50:
        return text, "pdfplumber"

    text = extract_text_pymupdf(path)
    if len(text) > 50:
        return text, "pymupdf"

    text = extract_text_ocr(path)
    if len(text) > 50:
        return text, "ocr"

    return "", "none"

@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    contents = await file.read()

    temp_path = "temp.pdf"
    with open(temp_path, "wb") as f:
        f.write(contents)

    text, method = smart_extract_text(temp_path)

    if not text.strip():
        return {"error": "Could not extract text from this PDF (even after OCR).", "method": method}

    # OPTIONAL: include method in response for debugging
    return {"method": method, "result": generate_study_content(text)}