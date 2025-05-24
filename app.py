from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
from transformers import pipeline
import logging
from pathlib import Path
import huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get cache directories from environment variables
HF_CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", "/home/appuser/.cache/huggingface")
WHISPER_CACHE_DIR = os.getenv("XDG_CACHE_HOME", "/home/appuser/.cache") + "/whisper"

# Ensure cache directories exist
Path(HF_CACHE_DIR).mkdir(parents=True, exist_ok=True)
Path(WHISPER_CACHE_DIR).mkdir(parents=True, exist_ok=True)

# Set Hugging Face cache directory
huggingface_hub.constants.HF_HUB_CACHE = HF_CACHE_DIR

app = FastAPI(title="TranscriptoCast AI (Demo)")

# Load models once at startup
try:
    logger.info("Loading Whisper model...")
    whisper_model = whisper.load_model("base", download_root=WHISPER_CACHE_DIR)
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {str(e)}")

try:
    logger.info("Loading summarization model...")
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        cache_dir=HF_CACHE_DIR,
        local_files_only=False
    )
    logger.info("Summarization model loaded successfully")
except Exception as e:
    logger.error(f"Error loading summarization model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load summarization model: {str(e)}")

try:
    logger.info("Loading translation model...")
    translator = pipeline(
        "translation",
        model="facebook/mbart-large-50-many-to-many-mmt",
        cache_dir=HF_CACHE_DIR,
        local_files_only=False
    )
    logger.info("Translation model loaded successfully")
except Exception as e:
    logger.error(f"Error loading translation model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load translation model: {str(e)}")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    result = whisper_model.transcribe(tmp_path)
    os.remove(tmp_path)
    return JSONResponse(content={"text": result["text"]})

@app.post("/summarize")
async def summarize(text: str = Form(...)):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return JSONResponse(content={"summary": summary[0]["summary_text"]})

@app.post("/translate")
async def translate(text: str = Form(...), src_lang: str = Form("en_XX"), tgt_lang: str = Form("fr_XX")):
    translation = translator(text, src_lang=src_lang, tgt_lang=tgt_lang)
    return JSONResponse(content={"translation": translation[0]["translation_text"]})

@app.get("/")
def root():
    return {"message": "Welcome to TranscriptoCast AI Hugging Face Space!"} 