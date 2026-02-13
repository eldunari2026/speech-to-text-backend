import os
import ssl
import tempfile
import certifi

# Fix SSL certificate issues
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import whisper
import anthropic
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Speech to Text API")

# CORS for frontend (allow Netlify and localhost)
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173,https://speechtotext07.netlify.app").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper model (use "base" for speed, "medium" or "large" for accuracy)
whisper_model = None


def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        whisper_model = whisper.load_model(os.getenv("WHISPER_MODEL", "base"))
    return whisper_model



# Claude client
def get_claude_client():

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY not configured")
    return anthropic.Anthropic(api_key=api_key)


class EnhanceRequest(BaseModel):
    text: str
    task: str = "cleanup"  # cleanup, summarize, action_items, format


@app.get("/")
async def root():
    return {"message": "Speech to Text API", "status": "running"}


@app.post("/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio file to text using Whisper."""
    if not audio.filename:
        raise HTTPException(status_code=400, detail="No audio file provided")

    # Save uploaded file temporarily
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        model = get_whisper_model()
        result = model.transcribe(tmp_path)
        return {
            "success": True,
            "text": result["text"],
            "language": result.get("language", "unknown"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.post("/enhance")
async def enhance_text(request: EnhanceRequest):
    """Enhance transcribed text using Claude."""
    prompts = {
        "cleanup": "Clean up this transcribed speech. Fix grammar, punctuation, and formatting while preserving the original meaning. Return only the cleaned text:",
        "summarize": "Summarize this transcribed speech concisely, capturing the key points:",
        "action_items": "Extract action items and key tasks from this transcribed speech. Format as a bullet list:",
        "format": "Format this transcribed speech into well-structured paragraphs with proper punctuation and headings where appropriate:",
    }

    prompt = prompts.get(request.task, prompts["cleanup"])

    try:
        client = get_claude_client()
        message = client.messages.create(
            model="claude-opus-4-5-20250514",
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\n{request.text}",
                }
            ],
        )
        return {
            "success": True,
            "original": request.text,
            "enhanced": message.content[0].text,
            "task": request.task,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Enhancement failed: {str(e)}")


@app.post("/transcribe-and-enhance")
async def transcribe_and_enhance(
    audio: UploadFile = File(...),
    task: str = "cleanup",
):
    """Transcribe audio and enhance with Claude in one step."""
    # First transcribe
    transcription = await transcribe_audio(audio)

    # Then enhance
    enhance_request = EnhanceRequest(text=transcription["text"], task=task)
    enhanced = await enhance_text(enhance_request)

    return {
        "success": True,
        "raw_transcription": transcription["text"],
        "enhanced_text": enhanced["enhanced"],
        "language": transcription["language"],
        "task": task,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
