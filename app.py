from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import whisper
import tempfile
import os
from transformers import pipeline

app = FastAPI(title="TranscriptoCast AI (Demo)")

# Load models once at startup
whisper_model = whisper.load_model("base")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
translator = pipeline("translation", model="facebook/mbart-large-50-many-to-many-mmt")

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