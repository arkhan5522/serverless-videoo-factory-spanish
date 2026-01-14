"""
AI VIDEO GENERATOR WITH SPANISH SUPPORT
========================================
ENHANCED VERSION:
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
        "numpy",
        "transformers",
        "pillow",
        "opencv-python",
        "sentencepiece",  # Required for T5 translation
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
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
PEXELS_KEYS = os.environ.get("PEXELS_KEYS", "").split(",")
PIXABAY_KEYS = os.environ.get("PIXABAY_KEYS", "").split(",")

# Paths
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ========================================== 
# 3. LOAD AI MODELS (T5 + Translation)
# ========================================== 

print("--- 🤖 Loading AI Models ---")

# T5 for Query Generation
print("Loading T5 Model for Query Generation...")
try:
    t5_tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
    T5_AVAILABLE = True
    print("✅ T5 Model loaded")
except Exception as e:
    print(f"⚠️ T5 Model failed to load: {e}")
    T5_AVAILABLE = False

# T5 for Spanish to English Translation
print("Loading T5 Translation Model (Spanish -> English)...")
try:
    translation_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    translation_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-es-en")
    TRANSLATION_AVAILABLE = True
    print("✅ Translation Model loaded")
except Exception as e:
    print(f"⚠️ Translation Model failed to load: {e}")
    TRANSLATION_AVAILABLE = False

# ========================================== 
# 4. CONTENT FILTERS
# ========================================== 

EXPLICIT_CONTENT_BLACKLIST = [
    'nude', 'nudity', 'naked', 'pornography', 'explicit sexual',
    'xxx', 'adult xxx', 'erotic xxx', 'nsfw','lgbtq','LGBTQ','war','pork','bikini','swim','violence','drugs','terror','gun','gambling'
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
            print(f"      🚫 BLOCKED: Inappropriate content - '{term}'")
            return False
    
    for term in RELIGIOUS_HOLY_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious content - '{term}'")
            return False
    
    return True

# ========================================== 
# 5. SPANISH TO ENGLISH TRANSLATION
# ========================================== 

def translate_spanish_to_english(spanish_text):
    """Translate Spanish text to English for video search queries"""
    if not TRANSLATION_AVAILABLE:
        print("    ⚠️ Translation model not available, using original text")
        return spanish_text
    
    try:
        # Prepare input
        inputs = translation_tokenizer(spanish_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Translate
        with torch.no_grad():
            translated = translation_model.generate(**inputs, max_length=100, num_beams=4, early_stopping=True)
        
        # Decode
        english_text = translation_tokenizer.decode(translated[0], skip_special_tokens=True)
        
        print(f"    🌐 Translated: '{spanish_text[:50]}...' -> '{english_text[:50]}...'")
        return english_text
        
    except Exception as e:
        print(f"    ⚠️ Translation error: {e}")
        return spanish_text

# ========================================== 
# 6. T5 SMART QUERY GENERATION (WITH TRANSLATION)
# ========================================== 

def generate_smart_query_t5(spanish_script_text):
    """Generate intelligent search queries using T5 transformer with translation"""
    
    # Step 1: Translate Spanish to English
    english_text = translate_spanish_to_english(spanish_script_text)
    
    if not T5_AVAILABLE:
        # Fallback to keyword extraction
        words = re.findall(r'\b\w{5,}\b', english_text.lower())
        return words[0] if words else "background"
    
    try:
        # Step 2: Generate tags using T5 on English text
        inputs = t5_tokenizer([english_text], max_length=512, truncation=True, return_tensors="pt")
        
        output = t5_model.generate(
            **inputs,
            max_length=50,
            num_beams=5,
            early_stopping=True,
            no_repeat_ngram_size=2
        )
        
        decoded_output = t5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        # Return first appropriate tag
        for tag in tags:
            if is_content_appropriate(tag):
                return tag
        
        return "background"
        
    except Exception as e:
        print(f"    T5 Error: {e}")
        words = re.findall(r'\b\w{5,}\b', english_text.lower())
        return words[0] if words else "background"

# ========================================== 
# 7. SUBTITLE STYLES
# ========================================== 

SUBTITLE_STYLES = {
    "mrbeast_yellow": {
        "name": "MrBeast Yellow (3D Pop)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 4,
        "shadow": 3,
        "margin_v": 45,
        "alignment": 2,
        "spacing": 1.5
    },
    "hormozi_green": {
        "name": "Hormozi Green (High Contrast)",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FF00",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 5,
        "shadow": 0,
        "margin_v": 55,
        "alignment": 2,
        "spacing": 0.5
    },
    "finance_blue": {
        "name": "Finance Blue (Neon Glow)",
        "fontname": "Arial",
        "fontsize": 80,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00FF9900",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 3,
        "margin_v": 50,
        "alignment": 2,
        "spacing": 2
    },
    "netflix_box": {
        "name": "Netflix Modern",
        "fontname": "Roboto",
        "fontsize": 80,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H90000000",
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 3,
        "outline": 0,
        "shadow": 0,
        "margin_v": 35,
        "alignment": 2,
        "spacing": 0.5
    },
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with proper format"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Using Subtitle Style: {style['name']}")
    
    with open(ass_file, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n")
        f.write("WrapStyle: 2\n")
        f.write("ScaledBorderAndShadow: yes\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        
        f.write(f"Style: Default,{style['fontname']},{style['fontsize']},{style['primary_colour']},&H000000FF,{style['outline_colour']},{style['back_colour']},{style['bold']},{style['italic']},0,0,100,100,{style['spacing']},0,{style['border_style']},{style['outline']},{style['shadow']},{style['alignment']},25,25,{style['margin_v']},1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start_time = format_ass_time(s['start'])
            end_time = format_ass_time(s['end'])
            
            text = s['text'].strip()
            text = text.replace('\\', '\\\\').replace('\n', ' ')
            
            if text.endswith('.'):
                text = text[:-1]
            if text.endswith(','):
                text = text[:-1]
            
            if "mrbeast" in style_key or "hormozi" in style_key:
                text = text.upper()
            
            MAX_CHARS = 35
            words = text.split()
            lines = []
            current_line = []
            current_length = 0
            
            for word in words:
                word_length = len(word) + 1
                if current_length + word_length > MAX_CHARS and current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = word_length
                else:
                    current_line.append(word)
                    current_length += word_length
            
            if current_line:
                lines.append(' '.join(current_line))
            
            formatted_text = '\\N'.join(lines)
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{formatted_text}\n")

def format_ass_time(seconds):
    """Format seconds to ASS timestamp"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

# ========================================== 
# 8. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Upload file to Google Drive"""
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return None
    
    print(f"☁️ Uploading {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Missing OAuth credentials")
        return None
    
    # Get access token
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "grant_type": "refresh_token"
    }
    
    try:
        r = requests.post(token_url, data=data)
        r.raise_for_status()
        access_token = r.json()['access_token']
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        return None
    
    # Upload
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
    
    metadata = {"name": filename, "mimeType": "video/mp4"}
    if folder_id:
        metadata["parents"] = [folder_id]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size)
    }
    
    response = requests.post(upload_url, headers=headers, json=metadata)
    if response.status_code != 200:
        print(f"❌ Init failed: {response.text}")
        return None
    
    session_uri = response.headers.get("Location")
    
    with open(file_path, "rb") as f:
        upload_headers = {"Content-Length": str(file_size)}
        upload_resp = requests.put(session_uri, headers=upload_headers, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_data = upload_resp.json()
        file_id = file_data.get('id')
        
        # Make public
        perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
        requests.post(
            perm_url,
            headers={"Authorization": f"Bearer {access_token}", "Content-Type": "application/json"},
            json={'role': 'reader', 'type': 'anyone'}
        )
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"✅ Uploaded: {link}")
        return link
    else:
        print(f"❌ Upload failed: {upload_resp.text}")
        return None

# ========================================== 
# 9. VIDEO SEARCH WITH TRANSLATION
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_smart(spanish_script_text, sentence_index):
    """Search videos using translation + T5 query generation"""
    
    # Generate English query from Spanish text
    if TRANSLATION_AVAILABLE and T5_AVAILABLE:
        primary_query = generate_smart_query_t5(spanish_script_text)
        print(f"    🧠 Translated Query: '{primary_query}'")
    else:
        # Fallback: translate and extract keywords
        english_text = translate_spanish_to_english(spanish_script_text)
        words = re.findall(r'\b\w{5,}\b', english_text.lower())
        primary_query = words[0] if words else "background"
        primary_query += " 4k cinematic"
        print(f"    📝 Fallback Query: '{primary_query}'")
    
    return search_videos_by_query(primary_query, sentence_index)

def download_and_rank_videos(results, script_text, target_duration, clip_index):
    """Download videos and use the first valid one"""
    
    for i, result in enumerate(results[:5]):
        try:
            raw_path = TEMP_DIR / f"raw_{clip_index}_{i}.mp4"
            response = requests.get(result['url'], timeout=30, stream=True)
            
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                output_path = TEMP_DIR / f"clip_{clip_index}.mp4"
                
                cmd = [
                    "ffmpeg", "-y", "-hwaccel", "cuda",
                    "-i", str(raw_path),
                    "-t", str(target_duration),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",
                    "-b:v", "8M",
                    "-an",
                    str(output_path)
                ]
                
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                try:
                    os.remove(raw_path)
                except:
                    pass
                
                if os.path.exists(output_path):
                    USED_VIDEO_URLS.add(result['url'])
                    print(f"    ✓ {result['service']} video downloaded")
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Download error: {str(e)[:60]}")
            continue
    
    return None

def search_videos_by_query(query, sentence_index, page=None):
    """Direct search with a specific query"""
    if page is None:
        page = random.randint(1, 3)
    
    all_results = []
    
    # Pexels
    if PEXELS_KEYS and PEXELS_KEYS[0]:
        try:
            key = random.choice([k for k in PEXELS_KEYS if k])
            url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": key}
            params = {
                "query": query,
                "per_page": 20,
                "page": page,
                "orientation": "landscape"
            }
            
            response = requests.get(url, headers=headers, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('videos', []):
                    video_files = video.get('video_files', [])
                    if video_files:
                        hd_files = [f for f in video_files if f.get('quality') == 'hd']
                        if not hd_files:
                            hd_files = [f for f in video_files if f.get('quality') == 'large']
                        if not hd_files:
                            hd_files = video_files
                        
                        if hd_files:
                            best_file = random.choice(hd_files)
                            video_url = best_file['link']
                            
                            video_title = video.get('user', {}).get('name', '')
                            if not is_content_appropriate(video_title + " " + query):
                                continue
                            
                            if video_url not in USED_VIDEO_URLS:
                                all_results.append({
                                    'url': video_url,
                                    'service': 'pexels',
                                    'duration': video.get('duration', 0)
                                })
        except Exception as e:
            print(f"    Pexels error: {str(e)[:50]}")
    
    # Pixabay
    if PIXABAY_KEYS and PIXABAY_KEYS[0]:
        try:
            key = random.choice([k for k in PIXABAY_KEYS if k])
            url = "https://pixabay.com/api/videos/"
            params = {
                "key": key,
                "q": query,
                "per_page": 20,
                "page": page,
                "orientation": "horizontal"
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    video_url = None
                    for quality in ['large', 'medium', 'small', 'tiny']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    
                    if video_url:
                        video_tags = video.get('tags', '')
                        if not is_content_appropriate(video_tags + " " + query):
                            continue
                        
                        if video_url not in USED_VIDEO_URLS:
                            all_results.append({
                                'url': video_url,
                                'service': 'pixabay',
                                'duration': video.get('duration', 0)
                            })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

# ========================================== 
# 10. STATUS UPDATES
# ========================================== 

LOG_BUFFER = []

def update_status(progress, message, status="processing", file_url=None):
    """Update status for HTML frontend"""
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
# 11. SCRIPT GENERATION (SPANISH)
# ========================================== 

def generate_script(topic, minutes):
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
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            return model.generate_content(prompt).text.replace("*","").replace("#","").strip()
        except:
            continue
    return "Script generation failed."

# ========================================== 
# 12. MULTILINGUAL CHATTERBOX TTS (SPANISH)
# ========================================== 

def clone_voice_multilingual(text, ref_audio, out_path):
    """Synthesize Spanish audio using Chatterbox Multilingual"""
    print("🎤 Synthesizing Spanish Audio with Chatterbox Multilingual...")
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
                    # Generate Spanish audio with language_id="es"
                    wav = model.generate(
                        text=chunk.replace('"', ''),
                        audio_prompt_path=str(ref_audio),
                        language_id="es",  # Spanish language
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
            print("✅ Spanish audio generated successfully")
            return True
    except Exception as e:
        print(f"❌ Audio generation failed: {e}")
    return False
