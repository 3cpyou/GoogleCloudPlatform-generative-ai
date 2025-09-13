#!/usr/bin/env python3
import os
import sys
import json
import time
import mimetypes
from pathlib import Path
from typing import Tuple, List

from tqdm import tqdm
from deepgram import DeepgramClient, PrerecordedOptions
from openai import OpenAI, APIError, RateLimitError, APITimeoutError

# ================== Constants ==================
MAX_RETRIES = 3
MAX_SUMMARY_ITEMS = 6
MAX_FILE_SIZE_MB = 100  # Skip files larger than this
DEFAULT_MODEL = "nova-2"
GPT_MODEL = "gpt-4o-mini"
BACKOFF_BASE = 1.0

# ================== Config ==================
DG_API_KEY = os.getenv("DEEPGRAM_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DG_API_KEY:
    sys.exit("❌ Missing DEEPGRAM_API_KEY. Run: export DEEPGRAM_API_KEY=your_key_here")
if not OPENAI_API_KEY:
    sys.exit("❌ Missing OPENAI_API_KEY. Run: export OPENAI_API_KEY=your_key_here")

# Initialize clients
dg = DeepgramClient(DG_API_KEY)
oa = OpenAI(api_key=OPENAI_API_KEY)

# Directory configuration
BASE_DIR = Path(__file__).parent
AUDIO_DIR = BASE_DIR
OUT_DIR = BASE_DIR / "transcripts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Audio file extensions
AUDIO_EXTS = {
    ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".amr",  # Original formats
    ".aac", ".aiff", ".aif", ".wma", ".webm", ".m4b", ".3gp",  # Additional common formats
    ".au", ".ra", ".ape", ".ac3", ".dts", ".pcm", ".caf",       # Specialized formats
    ".m4p", ".m4r", ".m4v", ".avi", ".mov"                     # Container formats with audio
}

# ================== Helpers ==================
def guess_mime(file_path: Path) -> str:
    """Best-effort MIME guessing, including WhatsApp/Telegram quirks."""
    mime, _ = mimetypes.guess_type(file_path)
    if mime:
        return mime
    ext = file_path.suffix.lower()
    if ext == ".opus":
        # WhatsApp voice notes are often Opus in an OGG container
        return "audio/ogg"
    if ext == ".amr":
        return "audio/amr"
    return "audio/mpeg"  # safe fallback

def sanitize(text: str) -> str:
    return (text or "").strip().replace("\r\n", "\n").replace("\r", "\n")

def validate_file_size(file_path: Path) -> bool:
    """Check if file size is within acceptable limits."""
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        print(f"⚠️ Skipping {file_path.name}: too large ({file_size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB)")
        return False
    return True

def backoff_retry(fn, *, retries=MAX_RETRIES, base=BACKOFF_BASE, exc_types=(RateLimitError, APITimeoutError, APIError)):
    """Simple exponential backoff for API calls."""
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except exc_types as e:
            if attempt == retries:
                raise
            time.sleep(base * (2 ** (attempt - 1)))

# ================== Core Steps ==================
def transcribe_with_deepgram(file_path: Path) -> str:
    """Send file to Deepgram and return transcript."""
    if not validate_file_size(file_path):
        raise ValueError(f"File too large: {file_path.name}")

    mime = guess_mime(file_path)
    with open(file_path, "rb") as f:
        source = {"buffer": f, "mimetype": mime}
        options = PrerecordedOptions(
            model=DEFAULT_MODEL,
            smart_format=True,
            utterances=False,
        )
        response = dg.listen.rest.v("1").transcribe_file(source, options)

    try:
        transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
        if not transcript:
            raise ValueError("Empty transcript returned")
    except KeyError:
        raise RuntimeError(f"Invalid Deepgram response structure for {file_path.name}")
    except Exception as e:
        raise RuntimeError(f"Deepgram transcription failed for {file_path.name}: {e}")

    return sanitize(transcript)

def gpt_summarize_and_actions(transcript: str) -> Tuple[List[str], List[str]]:
    """
    Use GPT to produce:
      - up to MAX_SUMMARY_ITEMS bullet summary points
      - actionable tasks
    """
    if not transcript or len(transcript.strip()) < 10:
        return [], []

    sys_prompt = (
        "You turn transcripts into concise summaries and clear action items. "
        "Be specific, avoid fluff, and keep each item on a single line."
    )
    user_prompt = f"""
Return ONLY valid JSON with two fields: "summary" (array of strings, max {MAX_SUMMARY_ITEMS}) and "actions" (array of strings).
Each action should read like a task someone could do (e.g., "Call supplier to confirm lead time").
Transcript:
\"\"\"{transcript}\"\"\"
JSON schema:
{{"summary": ["..."], "actions": ["..."]}}
"""

    def _call():
        resp = oa.chat.completions.create(
            model=GPT_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
        )
        return resp

    try:
        resp = backoff_retry(_call)
        content = resp.choices[0].message.content
        data = json.loads(content)

        summary = [sanitize(s) for s in data.get("summary", []) if sanitize(s)]
        actions = [sanitize(a) for a in data.get("actions", []) if sanitize(a)]

        return summary[:MAX_SUMMARY_ITEMS], actions

    except json.JSONDecodeError:
        print("⚠️ GPT returned invalid JSON, returning empty results")
        return [], []
    except Exception as e:
        print(f"⚠️ GPT summarization failed: {e}")
        return [], []

def write_txt(file_path: Path, transcript: str, summary: List[str], actions: List[str]) -> Path:
    """
    Write a .txt with sections to OUT_DIR/<audio_stem>.txt
    """
    out_file = OUT_DIR / f"{file_path.stem}.txt"
    with open(out_file, "w", encoding="utf-8") as out:
        out.write("=== TRANSCRIPTION ===\n")
        out.write(transcript + "\n\n")
        out.write("=== SUMMARY ===\n")
        if summary:
            for s in summary:
                out.write(f"• {s}\n")
        else:
            out.write("(No summary points)\n")
        out.write("\n=== ACTION ITEMS ===\n")
        if actions:
            for i, a in enumerate(actions, 1):
                out.write(f"{i}. {a}\n")
        else:
            out.write("(No action items)\n")
    return out_file

# ================== Main ==================
def main():
    # Filter and validate audio files
    audio_files = []
    for p in AUDIO_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in AUDIO_EXTS:
            if validate_file_size(p):
                audio_files.append(p)

    audio_files.sort(key=lambda p: p.name.lower())

    if not audio_files:
        print("⚠️ No valid audio files found.")
        return

    print(f"🎧 Found {len(audio_files)} valid audio file(s) in {AUDIO_DIR}")
    print(f"🗂  Outputs will be saved to: {OUT_DIR}")

    success_count = 0
    error_count = 0

    for file in tqdm(audio_files, desc="Processing", unit="file"):
        try:
            transcript = transcribe_with_deepgram(file)
            if not transcript:
                raise ValueError("Empty transcript")

            summary, actions = gpt_summarize_and_actions(transcript)
            out_path = write_txt(file, transcript, summary, actions)

            tqdm.write(f"✅ {file.name} → {out_path.name} "
                       f"({len(summary)} summary • {len(actions)} actions)")
            success_count += 1

        except ValueError as e:
            tqdm.write(f"⚠️ Skipped {file.name}: {e}")
            error_count += 1
        except Exception as e:
            tqdm.write(f"❌ Failed {file.name}: {e}")
            error_count += 1

    print(f"✨ Processing complete! {success_count} successful, {error_count} errors")

if __name__ == "__main__":
    main()