from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from groq import Groq
import fitz
import os
from dotenv import load_dotenv
import pdfplumber
import re

# OCR fallback
from pdf2image import convert_from_path
import pytesseract

# VIDEO + TEXT
import subprocess
import json

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

############################
# STUDY CONTENT GENERATION
############################
def generate_study_content(text: str):
    prompt = f"""
You are a helpful study assistant. Based on the following study material, provide:

1. SUMMARY: A clear concise summary (max 150 words)
2. KEY CONCEPTS: List the 5 most important concepts
3. PRACTICE QUESTIONS: Generate 5 practice questions
4. KEY TERMS: Important terms and definitions

Study Material:
{text}

Format with headers: SUMMARY, KEY CONCEPTS, PRACTICE QUESTIONS, KEY TERMS
"""
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

############################
# PDF TEXT EXTRACTION
############################
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
    images = convert_from_path(path, dpi=250)
    ocr_parts = []
    for img in images:
        ocr_parts.append(pytesseract.image_to_string(img))
    return "\n".join(ocr_parts).strip()

def smart_extract_text(path: str):
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

############################
# PDF ROUTE
############################
@app.post("/process-pdf")
async def process_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "temp.pdf"

    with open(temp_path, "wb") as f:
        f.write(contents)

    text, method = smart_extract_text(temp_path)

    if not text.strip():
        return {"error": "Could not extract text", "method": method}

    return {"method": method, "result": generate_study_content(text)}

############################
# VIDEO GENERATION ROUTE
############################
@app.post("/generate-video")
async def generate_video(text: str = Form(...)):

    ########################################
    # STEP 1: Generate viral script
    ########################################
    script_prompt = f"""
Turn these study notes into a short viral TikTok-style explanation.

STRICT RULES:
- Make it about 35–45 seconds when spoken
- Around 80–120 words maximum
- Fast-paced and engaging
- Explain clearly but quickly
- Focus only on most interesting concepts
- Make it sound exciting and smart
- No filler or slow intro

Notes:
{text}
"""
    script_response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": script_prompt}]
    )

    script = script_response.choices[0].message.content.strip().replace("\n"," ")

    ########################################
    # STEP 2: ElevenLabs voice
    ########################################
    from elevenlabs.client import ElevenLabs
    el_client = ElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))

    audio = el_client.text_to_speech.convert(
        text=script,
        voice_id="JBFqnCBsd6RMkjVDRZzb",
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128"
    )

    with open("voiceover.mp3", "wb") as f:
        for chunk in audio:
            f.write(chunk)

    ########################################
    # STEP 3: Get audio duration
    ########################################
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", "voiceover.mp3"],
        capture_output=True,
        text=True
    )

    audio_info = json.loads(probe.stdout)
    duration = float(audio_info["format"]["duration"])


    ########################################
    # STEP 4: Create CLEAN TikTok captions
    ########################################
    ########################################
    # STEP 4: PERFECT CLEAN CAPTIONS
    ########################################
    import textwrap

    # Clean script properly
    clean_script = script.replace("\n", " ").replace("\r", " ")
    clean_script = re.sub(r'\s+', ' ', clean_script).strip()

    # Split into chunks of ~6 words each (TikTok style)
    words = clean_script.split()
    chunks = [" ".join(words[i:i+6]) for i in range(0, len(words), 6)]

    total_words = len(words)
    filters = []
    current_time = 0

    for chunk in chunks:
        word_count = len(chunk.split())

        # time proportional to words
        display_time = (word_count / total_words) * duration

        start = round(current_time, 2)
        end = round(current_time + display_time, 2)
        current_time += display_time

        # escape for ffmpeg safely
        safe_text = chunk.upper()
        safe_text = safe_text.replace("'", "").replace(":", "")
        safe_text = safe_text.replace(",", "")
        safe_text = safe_text.replace(".", "")
        
        filters.append(
            f"drawtext=fontfile=/System/Library/Fonts/Supplemental/Arial.ttf:"
            f"text='{safe_text}':"
            f"fontcolor=white:fontsize=70:"
            f"borderw=5:bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y=h-300:"
            f"enable='between(t,{start},{end})'"
        )

    filter_complex = ",".join(filters)

    ########################################
    # STEP 5: Render video
    ########################################
    subprocess.run([
        "ffmpeg","-y",
        "-stream_loop","-1",
        "-i","background.mp4",
        "-i","voiceover.mp3",
        "-vf", filter_complex,
        "-map","0:v",
        "-map","1:a",
        "-c:v","libx264",
        "-c:a","aac",
        "-shortest",
        "output_video.mp4"
    ], check=True)

    ########################################
    return FileResponse(
        "output_video.mp4",
        media_type="video/mp4",
        filename="studybrain_video.mp4"
    )