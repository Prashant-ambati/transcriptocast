import os
import sys
import shutil
from pathlib import Path

# Force cache directories before any imports
CACHE_BASE = "/opt/cache"
HF_CACHE_DIR = f"{CACHE_BASE}/huggingface"
WHISPER_CACHE_DIR = f"{CACHE_BASE}/whisper"

# Ensure cache directories exist and are writable
for cache_dir in [HF_CACHE_DIR, WHISPER_CACHE_DIR]:
    try:
        path = Path(cache_dir)
        path.mkdir(parents=True, exist_ok=True)
        # Try to create a test file to verify write permissions
        test_file = path / ".test_write"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        print(f"Error setting up cache directory {cache_dir}: {e}", file=sys.stderr)
        sys.exit(1)

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HOME"] = HF_CACHE_DIR
os.environ["HF_DATASETS_CACHE"] = HF_CACHE_DIR
os.environ["HF_HUB_CACHE"] = HF_CACHE_DIR
os.environ["HF_CACHE_HOME"] = HF_CACHE_DIR
os.environ["XDG_CACHE_HOME"] = CACHE_BASE

# Now import other modules
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
import whisper
import tempfile
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import logging
import huggingface_hub

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Force Hugging Face to use our cache directory
huggingface_hub.constants.HF_HUB_CACHE = HF_CACHE_DIR
huggingface_hub.constants.HF_HOME = HF_CACHE_DIR

app = FastAPI(title="TranscriptoCast AI (Demo)")

def load_model_with_retry(model_name, model_type, max_retries=3):
    """Load a model with retry logic and proper cache handling."""
    for attempt in range(max_retries):
        try:
            if model_type == "whisper":
                return whisper.load_model(model_name, download_root=WHISPER_CACHE_DIR)
            else:
                # For Hugging Face models, load tokenizer and model separately
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    cache_dir=HF_CACHE_DIR,
                    local_files_only=False,
                    use_auth_token=False
                )
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    cache_dir=HF_CACHE_DIR,
                    local_files_only=False,
                    use_auth_token=False
                )
                return pipeline(
                    model_type,
                    model=model,
                    tokenizer=tokenizer,
                    device=-1  # Use CPU
                )
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            # Clean up any partial downloads
            try:
                if model_type == "whisper":
                    shutil.rmtree(WHISPER_CACHE_DIR, ignore_errors=True)
                else:
                    shutil.rmtree(HF_CACHE_DIR, ignore_errors=True)
            except Exception as cleanup_error:
                logger.error(f"Error cleaning up cache: {cleanup_error}")

# Load models once at startup
try:
    logger.info("Loading Whisper model...")
    whisper_model = load_model_with_retry("base", "whisper")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Error loading Whisper model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load Whisper model: {str(e)}")

try:
    logger.info("Loading summarization model...")
    summarizer = load_model_with_retry("facebook/bart-large-cnn", "summarization")
    logger.info("Summarization model loaded successfully")
except Exception as e:
    logger.error(f"Error loading summarization model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Failed to load summarization model: {str(e)}")

try:
    logger.info("Loading translation model...")
    translator = load_model_with_retry("facebook/mbart-large-50-many-to-many-mmt", "translation")
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