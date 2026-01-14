"""
AI VIDEO GENERATOR WITH SPANISH SUPPORT - COMPLETE VERSION
============================================================
Features:
1. Multilingual Chatterbox TTS for Spanish Audio
2. T5 Translation for English Video Queries
3. Spanish Subtitles via AssemblyAI
4. Islamic Content Filtering
5. Dual Output: With & Without Subtitles
"""

import os
import subprocess
import sys
import re
import time
import random
import shutil
import json
import concurrent.futures
import requests
import gc
from pathlib import Path

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "chatterbox-tts",
        "torchaudio", 
        "assemblyai",
        "google-generativeai",
        "requests",
        "beautifulsoup4",
        "pydub",
        "numpy<2.0",
        "transformers",
        "pillow",
        "opencv-python",
        "sentencepiece",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, check=False)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio
import assemblyai as aai
import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========================================== 
# 2. CONFIGURATION
# ========================================== 

MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

# Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY")
PEXELS_KEYS = [k.strip() for k in os.environ.get("PEXELS_KEYS", "").split(",") if k.strip()]
PIXABAY_KEYS = [k.strip() for k in os.environ.get("PIXABAY_KEYS", "").split(",") if k.strip()]

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ========================================== 
# 3. LOAD AI MODELS
# ========================================== 

print("--- 🤖 Loading AI Models ---")

# T5 for Query Generation
print("Loading T5 Model...")
try:
    t5_tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
    T5_AVAILABLE = True
    print("✅ T5 Model loaded")
except Exception as e:
    print(f"⚠️ T5 Model failed: {e}")
    T5_AVAILABLE = False

# Translation Model
print("Loading Translation Model...")
try:
    translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    TRANSLATION_AVAILABLE = True
    print("✅ Translation Model loaded")
except Exception as e:
    print(f"⚠️ Translation failed: {e}")
    TRANSLATION_AVAILABLE = False

# ========================================== 
# 4. CONTENT FILTERS
# ========================================== 

EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic xxx', 'nsfw','lgbtq','war','pork',
    'bikini','swim','violence','drugs','terror','gun','gambling'
]

RELIGIOUS_HOLY_TERMS = [
    'jesus', 'christ', 'god', 'lord', 'bible', 'gospel', 'church worship',
    'crucifix', 'crucifixion', 'virgin mary', 'holy spirit', 'baptism',
    'yahweh', 'jehovah', 'torah', 'talmud', 'synagogue', 'rabbi', 'kosher',
    'hanukkah', 'yom kippur', 'passover',
    'krishna', 'rama', 'shiva', 'vishnu', 'brahma', 'ganesh', 'hindu temple',
    'vedas', 'bhagavad gita', 'diwali',
    'buddha', 'buddhist temple', 'nirvana', 'dharma', 'meditation buddha',
    'tibetan monk', 'dalai lama',
    'holy book', 'scripture', 'religious ceremony', 'worship service',
    'religious ritual', 'sacred text', 'divine revelation'
]

def is_content_appropriate(text):
    """Content filter with word boundary matching"""
    text_lower = text.lower()
    
    for term in EXPLICIT_CONTENT_BLACKLIST:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Inappropriate - '{term}'")
            return False
    
    for term in RELIGIOUS_HOLY_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious - '{term}'")
            return False
    
    return True

# ========================================== 
# 5. TRANSLATION
# ========================================== 

def translate_spanish_to_english(spanish_text):
    """Translate Spanish to English for video queries"""
    if not TRANSLATION_AVAILABLE:
        return spanish_text
    
    try:
        inputs = translation_tokenizer(spanish_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            translated = translation_model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)
        english_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        print(f"    🌐 Translated: '{spanish_text[:50]}...' -> '{english_text[:50]}...'")
        return english_text
    except Exception as e:
        print(f"    ⚠️ Translation error: {e}")
        return spanish_text

# ========================================== 
# 6. SMART QUERY GENERATION
# ========================================== 

def generate_smart_query_t5(spanish_script_text):
    """Generate search queries using T5"""
    english_text = translate_spanish_to_english(spanish_script_text)
    
    if not T5_AVAILABLE:
        words = re.findall(r'\b\w{5,}\b', english_text.lower())
        return words[0] if words else "background"
    
    try:
        inputs = t5_tokenizer([english_text], max_length=512, truncation=True, return_tensors="pt")
        output = t5_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True, no_repeat_ngram_size=2)
        decoded_output = t5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        for tag in tags:
            if is_content_appropriate(tag):
                return tag
        return "background"
    except Exception as e:
        print(f"    T5 Error: {e}")
        words = re.findall(r'\b\w{5,}\b', english_text.lower())
        return words[0] if words else "background"

# ========================================== 
# 7. STATUS UPDATES
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Update status for frontend"""
    timestamp = time.strftime("%H:%M:%S")
    log_entry = f"[{timestamp}] {message}"
    print(f"--- {progress}% | {message} ---")
    
    LOG_BUFFER.append(log_entry)
    if len(LOG_BUFFER) > 30:
        LOG_BUFFER.pop(0)
    
    repo = os.environ.get('GITHUB_REPOSITORY')
    token = os.environ.get('GITHUB_TOKEN')
    
    if not repo or not token:
        return
    
    path = f"status/status_{JOB_ID}.json"
    url = f"https://api.github.com/repos/{repo}/contents/{path}"
    
    data = {
        "progress": progress,
        "message": message,
        "status": status,
        "logs": "\n".join(LOG_BUFFER),
        "timestamp": time.time()
    }
    
    if file_url:
        data["file_io_url"] = file_url
    
    import base64
    content_json = json.dumps(data)
    content_b64 = base64.b64encode(content_json.encode('utf-8')).decode('utf-8')
    
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    try:
        get_req = requests.get(url, headers=headers)
        sha = get_req.json().get("sha") if get_req.status_code == 200 else None
        
        payload = {
            "message": f"Update {progress}%",
            "content": content_b64,
            "branch": "main"
        }
        if sha:
            payload["sha"] = sha
        
        requests.put(url, headers=headers, json=payload)
    except:
        pass

def download_asset(path, local):
    """Download asset from GitHub"""
    try:
        repo = os.environ.get('GITHUB_REPOSITORY')
        token = os.environ.get('GITHUB_TOKEN')
        url = f"https://api.github.com/repos/{repo}/contents/{path}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3.raw"}
        r = requests.get(url, headers=headers)
        if r.status_code == 200:
            with open(local, "wb") as f:
                f.write(r.content)
            return True
    except:
        pass
    return False

# ========================================== 
# 8. SCRIPT GENERATION
# ========================================== 

def generate_script(topic, minutes):
    """Generate Spanish script using Gemini"""
    words = int(minutes * 180)
    print(f"Generating Spanish Script (~{words} words)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
INSTRUCCIONES CRÍTICAS:
- Escribe SOLO texto de narración hablada en ESPAÑOL
- NO incluyas direcciones de escenario, efectos de sonido o [corchetes]
- Comienza directamente con el contenido
- Directrices de contenido islámico: No menciones alcohol, relaciones inapropiadas, apuestas o cerdo
- Tono educativo y apropiado para toda la familia
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Comenzar'
            prompt = f"{base_instructions}\nEscribe la Parte {i+1}/{chunks} sobre '{topic}'. Contexto: {context}. Longitud: 700 palabras."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nEscribe un guión documental en español sobre '{topic}'. {words} palabras."
        script = call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    return script.strip()

def call_gemini(prompt):
    """Call Gemini API"""
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except:
            continue
    return "Script generation failed."

# ========================================== 
# 9. CHATTERBOX TTS
# ========================================== 

def clone_voice_multilingual(text, ref_audio, out_path):
    """Synthesize Spanish audio using Chatterbox"""
    print("🎤 Synthesizing Spanish Audio...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    try:
        from chatterbox.mtl_tts import ChatterboxMultilingualTTS
        model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        all_wavs = []
        
        for i, chunk in enumerate(sentences):
            if i % 10 == 0:
                update_status(20 + int((i/len(sentences))*30), f"Voice {i}/{len(sentences)}")
            
            try:
                with torch.no_grad():
                    wav = model.generate(
                        text=chunk.replace('"', ''),
                        audio_prompt_path=str(ref_audio),
                        language_id="es",
                        exaggeration=0.5
                    )
                    all_wavs.append(wav.cpu())
                
                if i % 20 == 0:
                    torch.cuda.empty_cache()
            except Exception as e:
                print(f"    ⚠️ Chunk {i} failed: {e}")
                continue
        
        if all_wavs:
            full_audio = torch.cat(all_wavs, dim=1)
            silence = torch.zeros((full_audio.shape[0], int(2.0 * 24000)))
            full_audio_padded = torch.cat([full_audio, silence], dim=1)
            torchaudio.save(out_path, full_audio_padded, 24000)
            print("✅ Spanish audio generated")
            return True
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
    return False

# ========================================== 
# 10. ASSEMBLYAI SUBTITLES
# ========================================== 

def generate_subtitles(audio_path):
    """Generate Spanish subtitles using AssemblyAI"""
    if not ASSEMBLY_KEY:
        print("⚠️ AssemblyAI key missing")
        return []
    
    print("📝 Generating Spanish subtitles...")
    aai.settings.api_key = ASSEMBLY_KEY
    
    try:
        config = aai.TranscriptionConfig(
            language_code="es",
            speech_model=aai.SpeechModel.best
        )
        
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(str(audio_path), config=config)
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"❌ Transcription error: {transcript.error}")
            return []
        
        sentences = []
        for utterance in transcript.utterances:
            sentences.append({
                'text': utterance.text,
                'start': utterance.start / 1000.0,
                'end': utterance.end / 1000.0
            })
        
        print(f"✅ Generated {len(sentences)} subtitle segments")
        return sentences
        
    except Exception as e:
        print(f"❌ Subtitle generation failed: {e}")
        return []

# ========================================== 
# 11. SUBTITLE STYLES
# ========================================== 

SUBTITLE_STYLES = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",
        "outline_colour": "&H00000000",
        "bold": -1,
        "outline": 4,
        "shadow": 3,
        "margin_v": 45
    },
    "finance_blue": {
        "name": "Finance Blue",
        "fontname": "Arial",
        "fontsize": 80,
        "primary_colour": "&H00FFFFFF",
        "outline_colour": "&H00FF9900",
        "bold": -1,
        "outline": 2,
        "shadow": 3,
        "margin_v": 50
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\n\n")
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},&H00000000,{style['bold']},0,0,0,100,100,1.5,0,1,{style['outline']},{style['shadow']},2,25,25,{style['margin_v']},1\n\n")
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start = format_ass_time(s['start'])
            end = format_ass_time(s['end'])
            text = s['text'].strip().upper()
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

def format_ass_time(seconds):
    """Format timestamp for ASS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ========================================== 
# 12. VIDEO SEARCH
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_smart(spanish_text, index):
    """Search videos with translation"""
    query = generate_smart_query_t5(spanish_text)
    return search_videos_by_query(query, index)

def search_videos_by_query(query, index, page=None):
    """Search Pexels and Pixabay"""
    if page is None:
        page = random.randint(1, 3)
    
    results = []
    
    # Pexels
    if PEXELS_KEYS:
        try:
            key = random.choice(PEXELS_KEYS)
            response = requests.get(
                "https://api.pexels.com/videos/search",
                headers={"Authorization": key},
                params={"query": query, "per_page": 20, "page": page, "orientation": "landscape"},
                timeout=15
            )
            if response.status_code == 200:
                for video in response.json().get('videos', []):
                    files = video.get('video_files', [])
                    hd = [f for f in files if f.get('quality') == 'hd'] or files
                    if hd and is_content_appropriate(query):
                        url = hd[0]['link']
                        if url not in USED_VIDEO_URLS:
                            results.append({'url': url, 'service': 'pexels'})
        except:
            pass
    
    # Pixabay
    if PIXABAY_KEYS:
        try:
            key = random.choice(PIXABAY_KEYS)
            response = requests.get(
                "https://pixabay.com/api/videos/",
                params={"key": key, "q": query, "per_page": 20, "page": page},
                timeout=15
            )
            if response.status_code == 200:
                for video in response.json().get('hits', []):
                    videos = video.get('videos', {})
                    url = videos.get('large', {}).get('url') or videos.get('medium', {}).get('url')
                    if url and is_content_appropriate(query):
                        if url not in USED_VIDEO_URLS:
                            results.append({'url': url, 'service': 'pixabay'})
        except:
            pass
    
    return results

def download_video(result, duration, index):
    """Download and process video"""
    try:
        raw = TEMP_DIR / f"raw_{index}.mp4"
        response = requests.get(result['url'], timeout=30, stream=True)
        
        with open(raw, "wb") as f:
            for chunk in response.iter_content(8192):
                if chunk:
                    f.write(chunk)
        
        output = TEMP_DIR / f"clip_{index}.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", str(raw), "-t", str(duration),
            "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-an", str(output)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        os.remove(raw)
        
        if output.exists():
            USED_VIDEO_URLS.add(result['url'])
            return str(output)
    except:
        pass
    return None

# ========================================== 
# 13. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_drive(file_path):
    """Upload to Google Drive"""
    if not os.path.exists(file_path):
        return None
    
    print(f"☁️ Uploading {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        return None
    
    # Get token
    try:
        r = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        access_token = r.json()['access_token']
    except:
        return None
    
    # Upload
    metadata = {"name": os.path.basename(file_path), "mimeType": "video/mp4"}
    if folder_id:
        metadata["parents"] = [folder_id]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(os.path.getsize(file_path))
    }
    
    response = requests.post("https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable", 
                           headers=headers, json=metadata)
    
    if response.status_code != 200:
        return None
    
    session_uri = response.headers.get("Location")
    
    with open(file_path, "rb") as f:
        upload_resp = requests.put(session_uri, headers={"Content-Length": str(os.path.getsize(file_path))}, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_id = upload_resp.json().get('id')
        requests.post(
            f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
            headers={"Authorization": f"Bearer {access_token}"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        return f"https://drive.google.com/file/d/{file_id}/view"
    
    return None

# ========================================== 
# 14. MAIN EXECUTION
# ========================================== 

def main():
    """Main video generation pipeline"""
    try:
        update_status(0, "🎬 Starting video generation...")
        
        # 1. Get Script
        if MODE == "generate":
            update_status(5, "✍️ Generating Spanish script...")
            script = generate_script(TOPIC, DURATION_MINS)
        else:
            script = SCRIPT_TEXT
        
        update_status(15, "✅ Script ready")
        
        # 2. Download Voice Reference
        voice_file = TEMP_DIR / "voice_ref.wav"
        if VOICE_PATH.startswith("http"):
            update_status(17, "📥 Downloading voice reference...")
            download_asset(VOICE_PATH, voice_file)
        else:
            shutil.copy(VOICE_PATH, voice_file)
        
        # 3. Generate Audio
        update_status(20, "🎤 Generating Spanish audio...")
        audio_file = TEMP_DIR / "narration.wav"
        if not clone_voice_multilingual(script, voice_file, audio_file):
            raise Exception("Audio generation failed")
        
        # 4. Generate Subtitles
        update_status(50, "📝 Generating Spanish subtitles...")
        sentences = generate_subtitles(audio_file)
        
        if sentences:
            ass_file = TEMP_DIR / "subtitles.ass"
            create_ass_file(sentences, ass_file)
        
        # 5. Download Videos
        update_status(60, "🎥 Downloading background videos...")
        
        # Split script into segments
        script_parts = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script) if s.strip()]
        num_clips = min(len(script_parts), 20)
        
        video_clips = []
        audio_duration = torchaudio.info(str(audio_file)).num_frames / 24000
        clip_duration = audio_duration / num_clips
        
        for i in range(num_clips):
            update_status(60 + int((i/num_clips)*20), f"Video {i+1}/{num_clips}")
            part = script_parts[i] if i < len(script_parts) else script_parts[-1]
            results = search_videos_smart(part, i)
            
            for result in results[:3]:
                clip = download_video(result, clip_duration, i)
                if clip:
                    video_clips.append(clip)
                    break
        
        if not video_clips:
            raise Exception("No videos downloaded")
        
        # 6. Assemble Final Video
        update_status(85, "🎬 Assembling final video...")
        
        # Concatenate videos
        concat_list = TEMP_DIR / "concat.txt"
        with open(concat_list, "w") as f:
            for clip in video_clips:
                f.write(f"file '{clip}'\n")
        
        video_concat = TEMP_DIR / "video_concat.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat_list),
            "-c", "copy", str(video_concat)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Add audio (no subs)
        final_no_subs = OUTPUT_DIR / f"{JOB_ID}_no_subs.mp4"
        subprocess.run([
            "ffmpeg", "-y", "-i", str(video_concat), "-i", str(audio_file),
            "-c:v", "copy", "-c:a", "aac", "-shortest", str(final_no_subs)
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Add subtitles
        final_with_subs = OUTPUT_DIR / f"{JOB_ID}_with_subs.mp4"
        if sentences and ass_file.exists():
            subprocess.run([
                "ffmpeg", "-y", "-i", str(final_no_subs), "-vf", f"ass={ass_file}",
                "-c:a", "copy", str(final_with_subs)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # 7. Upload to Google Drive
        update_status(95, "☁️ Uploading to Google Drive...")
        
        links = {}
        if final_no_subs.exists():
            links['no_subs'] = upload_to_drive(final_no_subs)
        if final_with_subs.exists():
            links['with_subs'] = upload_to_drive(final_with_subs)
        
        update_status(100, "✅ Video generation complete!", status="completed", 
                     file_url=json.dumps(links))
        
        print("\n" + "="*60)
        print("🎉 VIDEO GENERATION COMPLETE!")
        print("="*60)
        if links.get('no_subs'):
            print(f"📹 No Subtitles: {links['no_subs']}")
        if links.get('with_subs'):
            print(f"📹 With Subtitles: {links['with_subs']}")
        print("="*60)
        
        return links
        
    except Exception as e:
        error_msg = f"❌ Error: {str(e)}"
        print(error_msg)
        update_status(0, error_msg, status="failed")
        raise

# ========================================== 
# 15. RUN THE PIPELINE
# ========================================== 

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🎬 AI VIDEO GENERATOR WITH SPANISH SUPPORT")
    print("="*60)
    print(f"Mode: {MODE}")
    print(f"Topic: {TOPIC}")
    print(f"Duration: {DURATION_MINS} minutes")
    print(f"Job ID: {JOB_ID}")
    print("="*60 + "\n")
    
    try:
        result = main()
        print("\n✅ Pipeline completed successfully!")
    except KeyboardInterrupt:
        print("\n⚠️ Process interrupted by user")
        update_status(0, "Process cancelled", status="cancelled")
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
