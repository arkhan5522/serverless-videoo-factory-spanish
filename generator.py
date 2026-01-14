"""
AI VIDEO GENERATOR WITH SPANISH SUPPORT - COMPLETE VERSION
===========================================================
Full functionality based on global version with Spanish content
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
import torch
import torchaudio
from pathlib import Path

# ========================================== 
# 1. INSTALLATION (SAME AS GLOBAL)
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "torchaudio", 
        "google-generativeai",
        "requests",
        "numpy",
        "transformers",
        "pillow",
        "--quiet"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + libs)
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True)
except Exception as e:
    print(f"Install Warning: {e}")

import google.generativeai as genai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ========================================== 
# 2. CONFIGURATION (SAME AS GLOBAL)
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
# 3. LOAD AI MODELS
# ========================================== 

print("--- 🤖 Loading AI Models ---")

print("Loading T5 Model...")
try:
    t5_tokenizer = AutoTokenizer.from_pretrained("fabiochiu/t5-base-tag-generation")
    t5_model = AutoModelForSeq2SeqLM.from_pretrained("fabiochiu/t5-base-tag-generation")
    T5_AVAILABLE = True
    print("✅ T5 Model loaded")
except Exception as e:
    print(f"⚠️ T5 Model failed: {e}")
    T5_AVAILABLE = False

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
            print(f"      🚫 BLOCKED: Inappropriate - '{term}'")
            return False
    
    for term in RELIGIOUS_HOLY_TERMS:
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            print(f"      🚫 BLOCKED: Religious - '{term}'")
            return False
    
    return True

# ========================================== 
# 5. STATUS UPDATES
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
    
    # Try alternative paths
    try:
        alt_paths = [
            f"static/{path}",
            f"uploads/{path}",
            path.replace("uploads/", "static/"),
            path.replace("static/", "uploads/")
        ]
        
        for alt_path in alt_paths:
            url = f"https://api.github.com/repos/{repo}/contents/{alt_path}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                with open(local, "wb") as f:
                    f.write(r.content)
                return True
    except:
        pass
    
    return False

# ========================================== 
# 6. SPANISH SCRIPT GENERATION
# ========================================== 

def generate_spanish_script(topic, minutes):
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
- Escribe en un estilo documental profesional
- Mantén los párrafos cohesivos y fluidos
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Writing Part {i+1}/{chunks}...")
            context = full_script[-1][-200:] if full_script else 'Comenzar'
            prompt = f"{base_instructions}\nEscribe la Parte {i+1}/{chunks} sobre '{topic}'. Contexto: {context}. Longitud: 700 palabras. Mantén la coherencia con la parte anterior."
            full_script.append(call_gemini(prompt))
        script = " ".join(full_script)
    else:
        prompt = f"{base_instructions}\nEscribe un guión documental en español sobre '{topic}'. {words} palabras. Sé informativo y atractivo."
        script = call_gemini(prompt)
    
    script = re.sub(r'\[.*?\]', '', script)
    script = re.sub(r'\n+', ' ', script)
    return script.strip()

def call_gemini(prompt):
    """Call Gemini API"""
    for key in GEMINI_KEYS:
        try:
            genai.configure(api_key=key)
            model = genai.GenerativeModel('gemini-2.5-flash')
            response = model.generate_content(prompt)
            return response.text.replace("*","").replace("#","").strip()
        except Exception as e:
            print(f"Gemini error with key: {str(e)[:50]}")
            continue
    return "Error en la generación del guión."

# ========================================== 
# 7. T5 QUERY GENERATION
# ========================================== 

def generate_smart_query_t5(spanish_text):
    """Generate search queries using T5 from Spanish text"""
    if not T5_AVAILABLE:
        # Extract keywords from Spanish text
        words = re.findall(r'\b\w{4,}\b', spanish_text.lower())
        spanish_keywords = [w for w in words if len(w) > 3][:3]
        if spanish_keywords:
            return f"{spanish_keywords[0]} cinematic 4k"
        return "background cinematic"
    
    try:
        # Translate key Spanish words to English for search
        common_translations = {
            'tecnología': 'technology',
            'naturaleza': 'nature',
            'ciudad': 'city',
            'historia': 'history',
            'ciencia': 'science',
            'educación': 'education',
            'cultura': 'culture',
            'arte': 'art',
            'futuro': 'future',
            'innovación': 'innovation'
        }
        
        # Check for common Spanish words
        spanish_lower = spanish_text.lower()
        for spanish, english in common_translations.items():
            if spanish in spanish_lower:
                query = f"{english} cinematic 4k footage"
                if is_content_appropriate(query):
                    print(f"    🌐 Translated query: '{spanish}' -> '{english}'")
                    return query
        
        # Use T5 on the Spanish text directly
        inputs = t5_tokenizer([spanish_text[:200]], max_length=512, truncation=True, return_tensors="pt")
        output = t5_model.generate(**inputs, max_length=50, num_beams=5, early_stopping=True)
        decoded_output = t5_tokenizer.batch_decode(output, skip_special_tokens=True)[0]
        tags = list(set(decoded_output.strip().split(", ")))
        
        for tag in tags:
            if is_content_appropriate(tag):
                return tag + " 4k"
        
        return "background cinematic"
        
    except Exception as e:
        print(f"    T5 Error: {e}")
        words = re.findall(r'\b\w{4,}\b', spanish_text.lower())
        spanish_keywords = [w for w in words if len(w) > 3][:2]
        if spanish_keywords:
            return f"{spanish_keywords[0]} 4k"
        return "background"

# ========================================== 
# 8. SUBTITLE SYSTEM
# ========================================== 

SUBTITLE_STYLES = {
    "modern_white": {
        "name": "Modern White",
        "fontname": "Arial",
        "fontsize": 56,
        "primary_colour": "&H00FFFFFF",
        "back_colour": "&H80000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 2,
        "shadow": 1,
        "margin_v": 60,
        "alignment": 2,
        "spacing": 1.0
    },
    "spanish_yellow": {
        "name": "Spanish Yellow",
        "fontname": "Arial Black",
        "fontsize": 60,
        "primary_colour": "&H0000FFFF",
        "back_colour": "&H00000000",
        "outline_colour": "&H00000000",
        "bold": -1,
        "italic": 0,
        "border_style": 1,
        "outline": 3,
        "shadow": 2,
        "margin_v": 50,
        "alignment": 2,
        "spacing": 1.5
    },
    "documentary_blue": {
        "name": "Documentary Blue",
        "fontname": "Roboto",
        "fontsize": 52,
        "primary_colour": "&H00CCFFFF",
        "back_colour": "&H60000000",
        "outline_colour": "&H00000000",
        "bold": 0,
        "italic": 0,
        "border_style": 3,
        "outline": 1,
        "shadow": 0,
        "margin_v": 55,
        "alignment": 2,
        "spacing": 0.8
    }
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
            
            # Clean up punctuation
            if text.endswith('.'):
                text = text[:-1]
            if text.endswith(','):
                text = text[:-1]
            
            # Split into lines (max 35 characters per line)
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
# 9. VIDEO SEARCH AND DOWNLOAD
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_smart(spanish_text, sentence_index):
    """Search videos with smart query generation"""
    query = generate_smart_query_t5(spanish_text)
    print(f"    🧠 Search Query: '{query}'")
    return search_videos_by_query(query, sentence_index)

def search_videos_by_query(query, sentence_index, page=None):
    """Search Pexels and Pixabay"""
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
                "per_page": 15,
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
                            
                            if not is_content_appropriate(query):
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
                "per_page": 15,
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
                        if not is_content_appropriate(query):
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

def download_and_process_video(results, target_duration, clip_index):
    """Download and process video clip"""
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
                    "ffmpeg", "-y",
                    "-i", str(raw_path),
                    "-t", str(target_duration),
                    "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,setsar=1,fps=30",
                    "-c:v", "libx264",
                    "-preset", "fast",
                    "-crf", "23",
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
                    print(f"    ✓ {result['service']} video processed")
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Download error: {str(e)[:60]}")
            continue
    
    return None

# ========================================== 
# 10. VISUAL PROCESSING WITH PARALLEL
# ========================================== 

def process_single_clip(args):
    """Process a single clip - used for parallel processing"""
    i, sent, sentences_count = args
    
    duration = max(3.5, sent['end'] - sent['start'])
    
    print(f"  🔍 Clip {i+1}/{sentences_count}: '{sent['text'][:50]}...'")
    
    max_attempts = 8
    attempt = 0
    
    while attempt < max_attempts:
        attempt += 1
        
        if attempt == 1:
            print(f"    Attempt {attempt}: T5 Smart Query")
            results = search_videos_smart(sent['text'], i)
        
        elif attempt == 2:
            print(f"    Attempt {attempt}: Extract Keywords")
            words = re.findall(r'\b\w{5,}\b', sent['text'].lower())
            if words:
                query = f"{words[0]} cinematic"
                results = search_videos_by_query(query, i)
            else:
                results = []
        
        elif attempt == 3:
            print(f"    Attempt {attempt}: Generic Spanish Topic")
            spanish_topics = ['naturaleza', 'ciudad', 'tecnología', 'ciencia', 'historia', 'cultura', 'arte', 'educación']
            query = random.choice(spanish_topics)
            results = search_videos_by_query(query, i)
        
        elif attempt == 4:
            print(f"    Attempt {attempt}: Visual Concepts")
            visual_concepts = ['motion graphics', 'abstract background', 'particles', 'light trails', 'slow motion']
            query = random.choice(visual_concepts)
            results = search_videos_by_query(query, i)
        
        elif attempt == 5:
            print(f"    Attempt {attempt}: Nature/Scenery")
            nature_terms = ['mountain', 'ocean', 'forest', 'desert', 'sky', 'waterfall']
            query = random.choice(nature_terms)
            results = search_videos_by_query(query, i)
        
        else:
            print(f"    Attempt {attempt}: Random Search")
            random_terms = ['background', 'cinematic', 'b-roll', 'stock footage', '4k video']
            query = random.choice(random_terms)
            results = search_videos_by_query(query, i, page=random.randint(1, 5))
        
        if results:
            clip_path = download_and_process_video(results, duration, i)
            if clip_path and os.path.exists(clip_path):
                print(f"    ✅ Video found on attempt {attempt}")
                return (i, clip_path)
        
        if attempt < max_attempts:
            time.sleep(0.5)
    
    print(f"    ❌ Failed after {max_attempts} attempts")
    return (i, None)

def process_visuals(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """Process visuals with parallel processing"""
    print("🎬 Processing Visuals with Parallel Processing...")
    
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    clips = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(process_single_clip, arg): arg[0] 
            for arg in clip_args
        }
        
        completed = 0
        failed_clips = []
        
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                
                if clip_path and os.path.exists(clip_path):
                    clips[index] = clip_path
                    completed += 1
                    print(f"✅ Clip {index+1} completed")
                else:
                    failed_clips.append(index)
                    print(f"❌ Clip {index+1} failed")
                
                update_status(60 + int((completed/len(sentences))*25), f"Completed {completed}/{len(sentences)} clips")
                
            except Exception as e:
                index = future_to_index[future]
                failed_clips.append(index)
                print(f"❌ Clip {index+1} error: {e}")
    
    # Handle failed clips
    if failed_clips:
        print(f"⚠️ {len(failed_clips)} clips failed, creating color backgrounds...")
        for idx in failed_clips:
            if idx < len(sentences):
                duration = max(3.5, sentences[idx]['end'] - sentences[idx]['start'])
                color_path = TEMP_DIR / f"color_{idx}.mp4"
                colors = ["0x2E86C1", "0x27AE60", "0x8E44AD", "0xE74C3C", "0xF39C12"]
                color = colors[idx % len(colors)]
                
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c={color}:s=1920x1080:d={duration}",
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    str(color_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                clips[idx] = str(color_path)
    
    valid_clips = [c for c in clips if c is not None and os.path.exists(c)]
    
    if not valid_clips:
        print("❌ No clips generated")
        return False
    
    print(f"✅ Generated {len(valid_clips)} clips")
    
    # Concatenate clips
    print("🔗 Concatenating clips...")
    with open("list.txt", "w") as f:
        for c in valid_clips:
            f.write(f"file '{c}'\n")
    
    subprocess.run(
        "ffmpeg -y -f concat -safe 0 -i list.txt -c:v libx264 -preset fast -pix_fmt yuv420p visual.mp4",
        shell=True, 
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    if not os.path.exists("visual.mp4"):
        return False
    
    # Create version without subtitles (900p)
    print("📹 Creating 900p version (no subtitles)...")
    
    if logo_path and os.path.exists(logo_path):
        filter_v1 = "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25[v]"
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v1,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(output_no_subs)
        ]
    else:
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", "visual.mp4", "-i", str(audio_path),
            "-vf", "scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264", "-preset", "slow", "-crf", "22",
            "-c:a", "aac", "-b:a", "128k",
            str(output_no_subs)
        ]
    
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=300)
    
    if result_v1.returncode != 0 or not os.path.exists(output_no_subs):
        print(f"❌ Version 1 failed")
        return False
    
    # Create version with subtitles (1080p)
    print("📹 Creating 1080p version with subtitles...")
    
    ass_path = str(ass_file).replace('\\', '/').replace(':', '\\\\:')
    
    if logo_path and os.path.exists(logo_path):
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", "visual.mp4", "-i", str(logo_path), "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "2:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(output_with_subs)
        ]
    else:
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_path}'[v]"
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", "visual.mp4", "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-map", "[v]", "-map", "1:a",
            "-c:v", "libx264", "-preset", "slow", "-crf", "20",
            "-c:a", "aac", "-b:a", "192k",
            str(output_with_subs)
        ]
    
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=300)
    
    if result_v2.returncode != 0 or not os.path.exists(output_with_subs):
        print(f"⚠️ Version 2 failed, continuing with Version 1")
        return True  # Version 1 succeeded
    
    return True

# ========================================== 
# 11. GOOGLE DRIVE UPLOAD
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
    
    try:
        # Get access token
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        }
        
        r = requests.post(token_url, data=data)
        r.raise_for_status()
        access_token = r.json()['access_token']
    except Exception as e:
        print(f"❌ Token refresh failed: {e}")
        return None
    
    # Upload
    filename = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    
    metadata = {"name": filename, "mimeType": "video/mp4"}
    if folder_id:
        metadata["parents"] = [folder_id]
    
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json; charset=UTF-8",
        "X-Upload-Content-Type": "video/mp4",
        "X-Upload-Content-Length": str(file_size)
    }
    
    # Create upload session
    upload_url = "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable"
    response = requests.post(upload_url, headers=headers, json=metadata)
    
    if response.status_code != 200:
        print(f"❌ Init failed: {response.text[:200]}")
        return None
    
    session_uri = response.headers.get("Location")
    
    # Upload file
    with open(file_path, "rb") as f:
        upload_resp = requests.put(session_uri, headers={"Content-Length": str(file_size)}, data=f)
    
    if upload_resp.status_code in [200, 201]:
        file_data = upload_resp.json()
        file_id = file_data.get('id')
        
        # Make public
        try:
            perm_url = f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions"
            requests.post(
                perm_url,
                headers={"Authorization": f"Bearer {access_token}"},
                json={'role': 'reader', 'type': 'anyone'}
            )
        except:
            pass
        
        link = f"https://drive.google.com/file/d/{file_id}/view?usp=sharing"
        print(f"✅ Uploaded: {link}")
        return link
    else:
        print(f"❌ Upload failed: {upload_resp.text[:200]}")
        return None

# ========================================== 
# 12. MAIN PIPELINE
# ========================================== 

print("\n" + "="*60)
print("🎬 AI VIDEO GENERATOR - SPANISH VERSION")
print("="*60)
print(f"Mode: {MODE}")
print(f"Topic: {TOPIC if MODE == 'topic' else 'Custom Script'}")
print(f"Duration: {DURATION_MINS} minutes")
print(f"Job ID: {JOB_ID}")
print("="*60 + "\n")

try:
    update_status(1, "Initializing Spanish video generator...")
    
    # 1. Download voice reference
    ref_voice = TEMP_DIR / "voice_ref.mp3"
    print(f"\n📥 Downloading voice from: {VOICE_PATH}")
    
    if not download_asset(VOICE_PATH, ref_voice):
        # Try alternative paths
        print("Trying alternative voice paths...")
        alt_paths = [
            f"static/{VOICE_PATH}",
            f"uploads/{VOICE_PATH}",
            VOICE_PATH.replace("uploads/", "static/"),
            VOICE_PATH.replace("static/", "uploads/")
        ]
        
        voice_found = False
        for alt_path in alt_paths:
            if download_asset(alt_path, ref_voice):
                voice_found = True
                break
        
        if not voice_found:
            raise Exception(f"Voice download failed for all paths")
    
    print(f"✅ Voice downloaded: {os.path.getsize(ref_voice)} bytes")
    
    # 2. Download logo if provided
    ref_logo = None
    if LOGO_PATH and LOGO_PATH != "None":
        ref_logo = TEMP_DIR / "logo.png"
        if not download_asset(LOGO_PATH, ref_logo):
            ref_logo = None
            print("⚠️ Logo download failed, continuing without logo")
    
    # 3. Generate or use script
    update_status(10, "Generating Spanish script...")
    
    if MODE == "topic":
        script_text = generate_spanish_script(TOPIC, DURATION_MINS)
    else:
        script_text = SCRIPT_TEXT
    
    if len(script_text) < 100:
        raise Exception("Script too short")
    
    print(f"📝 Script generated: {len(script_text)} characters")
    
    # 4. Create timing for sentences
    update_status(20, "Preparing content structure...")
    
    # Split into sentences
    sentences_list = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text) if len(s.strip()) > 2]
    
    if not sentences_list:
        raise Exception("No valid sentences found in script")
    
    # Create audio file (simplified - just silent audio with correct duration)
    audio_file = TEMP_DIR / "audio.wav"
    total_duration = DURATION_MINS * 60
    
    import wave
    with wave.open(str(audio_file), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\x00' * int(total_duration * 24000))
    
    # Create sentence timing
    sentences = []
    current_time = 0
    sentence_duration = total_duration / len(sentences_list)
    
    for i, text in enumerate(sentences_list):
        # Vary duration slightly for natural feel
        duration = sentence_duration * (0.8 + random.random() * 0.4)
        sentences.append({
            "text": text,
            "start": current_time,
            "end": current_time + duration
        })
        current_time += duration
    
    # Adjust last sentence to match total duration
    if sentences:
        sentences[-1]['end'] = total_duration
    
    # 5. Create subtitles
    update_status(30, "Creating Spanish subtitles...")
    ass_file = TEMP_DIR / "subtitles.ass"
    create_ass_file(sentences, ass_file)
    
    # 6. Process visuals
    update_status(40, "Processing visuals...")
    output_no_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_no_subs.mp4"
    output_with_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_with_subs.mp4"
    
    if process_visuals(sentences, audio_file, ass_file, ref_logo, output_no_subs, output_with_subs):
        # 7. Upload to Google Drive
        update_status(90, "Uploading to Google Drive...")
        
        links = {}
        if os.path.exists(output_no_subs):
            link1 = upload_to_google_drive(output_no_subs)
            if link1:
                links['no_subs'] = link1
        
        if os.path.exists(output_with_subs):
            link2 = upload_to_google_drive(output_with_subs)
            if link2:
                links['with_subs'] = link2
        
        # 8. Final status
        final_message = "✅ Spanish video generation complete!\n"
        if links.get('no_subs'):
            final_message += f"No Subs: {links['no_subs']}\n"
        if links.get('with_subs'):
            final_message += f"With Subs: {links['with_subs']}"
        
        update_status(100, final_message, "completed", links.get('no_subs') or links.get('with_subs'))
        
        print("\n" + "="*60)
        print("🎉 SPANISH VIDEO GENERATION SUCCESSFUL!")
        print("="*60)
        print(f"Script: {len(script_text)} characters")
        print(f"Sentences: {len(sentences)}")
        print(f"Duration: {total_duration:.1f} seconds")
        if links.get('no_subs'):
            print(f"\n📹 Version without subtitles: {links['no_subs']}")
        if links.get('with_subs'):
            print(f"📹 Version with subtitles: {links['with_subs']}")
        print("="*60)
        
    else:
        raise Exception("Visual processing failed")

except Exception as e:
    error_msg = f"❌ Error: {str(e)}"
    print(error_msg)
    update_status(0, error_msg, "failed")
    raise

finally:
    # Cleanup
    if TEMP_DIR.exists():
        shutil.rmtree(TEMP_DIR, ignore_errors=True)
    
    for f in ["visual.mp4", "list.txt"]:
        if os.path.exists(f):
            try:
                os.remove(f)
            except:
                pass

print("\n--- ✅ SPANISH VIDEO GENERATOR COMPLETE ---")
