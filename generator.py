"""
AI VIDEO GENERATOR - SPANISH VERSION (COMPLETE FIX)
===================================================
✅ Chatterbox Multilingual TTS for Spanish audio (language_id="es")
✅ AssemblyAI for accurate subtitle timing
✅ Better video clip distribution for long audio
✅ GPU acceleration
✅ Pure nature videos
✅ Google Drive upload
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
import wave
import math
from pathlib import Path

# ========================================== 
# 1. INSTALLATION
# ========================================== 

print("--- 🔧 Installing Dependencies ---")
try:
    libs = [
        "torch",
        "torchaudio", 
        "google-generativeai",
        "requests",
        "numpy",
        "pillow",
        "chatterbox-tts",
        "assemblyai"
    ]
    
    for lib in libs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
            print(f"✅ {lib}")
        except Exception as e:
            print(f"⚠️  {lib}: {str(e)[:50]}")
    
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", shell=True, check=False)
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio as ta
import google.generativeai as genai
import assemblyai as aai

# Import Chatterbox
TTS_MODEL = None
TTS_AVAILABLE = False
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS
    TTS_AVAILABLE = True
    print("✅ Chatterbox Multilingual TTS imported successfully")
except ImportError as e:
    print(f"⚠️ Chatterbox import failed: {e}")

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

# API Keys
raw_gemini = os.environ.get("GEMINI_API_KEY", "")
GEMINI_KEYS = [k.strip() for k in raw_gemini.split(",") if k.strip()]
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")
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

# Load Chatterbox Multilingual TTS ONLY
if TTS_AVAILABLE:
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading Chatterbox on {device}...")
        TTS_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("✅ Chatterbox Multilingual TTS loaded successfully")
    except Exception as e:
        print(f"⚠️ Chatterbox loading failed: {e}")
        TTS_AVAILABLE = False

print("🌲 All videos will show pure nature scenes")

# ========================================== 
# 4. NATURE VIDEO QUERIES (HARDCODED)
# ========================================== 

NATURE_QUERIES = [
    "forest trees cinematic 4k",
    "mountain landscape nature 4k",
    "waterfall nature cinematic",
    "green forest wilderness 4k",
    "river flowing nature 4k",
    "rainforest jungle cinematic",
    "pine forest trees 4k",
    "meadow grass flowers nature",
    "autumn forest leaves 4k",
    "spring forest green 4k",
    "misty forest morning 4k",
    "lake reflection nature 4k",
    "valley landscape cinematic",
    "hills greenery nature 4k",
    "woodland forest cinematic",
    "nature sunrise trees 4k",
    "sunset mountain landscape",
    "clouds sky nature 4k",
    "birds flying forest 4k",
    "deer forest wildlife 4k",
    "butterfly flowers nature",
    "leaves wind forest 4k",
    "rain forest nature 4k",
    "snow mountain landscape",
    "canyon nature cinematic",
    "tropical rainforest 4k",
    "alpine meadow flowers",
    "bamboo forest green",
    "redwood trees forest",
    "oak tree nature 4k"
]

def get_nature_query():
    """ALWAYS return random nature query"""
    query = random.choice(NATURE_QUERIES)
    return query

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
    return False

# ========================================== 
# 6. SPANISH SCRIPT GENERATION
# ========================================== 

def generate_spanish_script(topic, minutes):
    """Generate Spanish script using Gemini"""
    words = int(minutes * 180)
    print(f"Generando guión en español (~{words} palabras)...")
    random.shuffle(GEMINI_KEYS)
    
    base_instructions = """
INSTRUCCIONES CRÍTICAS:
- Escribe SOLO texto de narración hablada en ESPAÑOL
- NO incluyas direcciones de escenario, efectos de sonido o [corchetes]
- Comienza directamente con el contenido
- Tono educativo y apropiado para toda la familia
- Escribe en un estilo documental profesional
- Mantén los párrafos cohesivos y fluidos
- Incluye pausas naturales entre ideas
"""
    
    if minutes > 15:
        chunks = int(minutes / 5)
        full_script = []
        for i in range(chunks):
            update_status(5+i, f"Escribiendo Parte {i+1}/{chunks}...")
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
            model = genai.GenerativeModel('gemini-2.0-flash-exp')
            response = model.generate_content(prompt)
            return response.text.replace("*","").replace("#","").strip()
        except Exception as e:
            print(f"Gemini error: {str(e)[:50]}")
            continue
    return "Error en la generación del guión."

# ========================================== 
# 7. CHATTERBOX TTS AUDIO GENERATION (SPANISH)
# ========================================== 

def generate_spanish_audio_chatterbox(text, output_path, audio_prompt_path=None):
    """Generate Spanish TTS audio using Chatterbox"""
    if not TTS_AVAILABLE or TTS_MODEL is None:
        print("⚠️ Chatterbox TTS not available, creating silent audio")
        return create_silent_audio(text, output_path)
    
    print("🎙️ Generando audio español con Chatterbox TTS (language_id='es')...")
    
    try:
        # Split into manageable chunks
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
        
        all_audio_segments = []
        sample_rate = TTS_MODEL.sr
        
        for i, sent in enumerate(sentences):
            if i % 10 == 0:
                update_status(25 + int((i/len(sentences))*15), 
                            f"Generando audio {i+1}/{len(sentences)}...")
            
            try:
                if audio_prompt_path and os.path.exists(audio_prompt_path):
                    wav_audio = TTS_MODEL.generate(
                        sent, 
                        language_id="es",
                        audio_prompt_path=audio_prompt_path
                    )
                else:
                    wav_audio = TTS_MODEL.generate(sent, language_id="es")
                
                if wav_audio.dim() == 1:
                    wav_audio = wav_audio.unsqueeze(0)
                
                # Add pause between sentences
                silence_samples = int(0.3 * sample_rate)
                silence = torch.zeros((wav_audio.shape[0], silence_samples))
                segment_with_pause = torch.cat([wav_audio, silence], dim=-1)
                all_audio_segments.append(segment_with_pause)
                
            except Exception as e:
                print(f"⚠️ TTS error en oración {i}: {e}")
                continue
        
        if all_audio_segments:
            full_audio = torch.cat(all_audio_segments, dim=-1)
            
            # Ensure mono audio
            if full_audio.shape[0] > 1:
                full_audio = torch.mean(full_audio, dim=0, keepdim=True)
            
            # Save audio
            ta.save(output_path, full_audio, sample_rate)
            
            # Verify duration
            with wave.open(str(output_path), 'rb') as wav_file:
                duration = wav_file.getnframes() / wav_file.getframerate()
                print(f"✅ Audio generado: {duration:.1f}s, {len(sentences)} oraciones")
            
            return True
        else:
            return create_silent_audio(text, output_path)
        
    except Exception as e:
        print(f"❌ TTS failed: {e}")
        return create_silent_audio(text, output_path)

def create_silent_audio(text, output_path, duration_seconds=60):
    """Create silent audio as fallback"""
    sample_rate = 44100
    channels = 1
    with wave.open(str(output_path), 'wb') as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * int(duration_seconds * sample_rate))
    
    return True

# ========================================== 
# 8. IMPROVED ASSEMBLYAI TRANSCRIPTION
# ========================================== 

def transcribe_with_assemblyai_improved(audio_path, audio_duration):
    """Transcribe audio using AssemblyAI with better sentence splitting"""
    if not ASSEMBLY_KEY:
        print("⚠️ AssemblyAI key not found")
        return None
    
    print("🔤 Transcribiendo audio con AssemblyAI...")
    
    try:
        aai.settings.api_key = ASSEMBLY_KEY
        transcriber = aai.Transcriber()
        
        update_status(45, "Transcribiendo audio...")
        transcript = transcriber.transcribe(str(audio_path))
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"❌ AssemblyAI error: {transcript.error}")
            return None
        
        # Get words for better splitting
        words = []
        if hasattr(transcript, 'words') and transcript.words:
            words = transcript.words
        
        # Calculate optimal sentence count
        optimal_sentences = max(40, int(audio_duration / 4))  # 1 sentence per 4 seconds, min 40
        
        print(f"🎯 Audio: {audio_duration:.1f}s, Objetivo: {optimal_sentences} oraciones")
        
        if words:
            # Use word-level timing for better splitting
            sentences = split_words_into_sentences(words, optimal_sentences, audio_duration)
        else:
            # Fallback to sentence-level
            sentences = []
            for sentence in transcript.get_sentences():
                sentences.append({
                    "text": sentence.text,
                    "start": sentence.start / 1000,
                    "end": sentence.end / 1000
                })
            
            # If too few sentences, split them
            if len(sentences) < optimal_sentences * 0.5:
                sentences = split_long_sentences(sentences, optimal_sentences, audio_duration)
        
        print(f"✅ Transcripción: {len(sentences)} oraciones")
        return sentences
        
    except Exception as e:
        print(f"⚠️ AssemblyAI failed: {e}")
        return None

def split_words_into_sentences(words, target_count, audio_duration):
    """Split words into optimal sentences"""
    if not words:
        return []
    
    sentences = []
    current_text = []
    current_start = words[0].start / 1000 if hasattr(words[0], 'start') else 0
    words_per_sentence = max(8, len(words) // target_count)
    
    for i, word in enumerate(words):
        current_text.append(word.text)
        
        # Check if we should end sentence
        should_end = False
        
        # Word count check
        if len(current_text) >= words_per_sentence:
            should_end = True
            
            # Check for natural endings
            word_text = word.text.lower()
            if word_text.endswith(('.', '?', '!', ';')):
                should_end = True
            elif i < len(words) - 1:
                next_word = words[i + 1]
                if hasattr(word, 'end') and hasattr(next_word, 'start'):
                    pause = (next_word.start - word.end) / 1000
                    if pause > 0.25:  # 250ms pause
                        should_end = True
        
        # Last word
        if i == len(words) - 1:
            should_end = True
        
        if should_end and current_text:
            # Get end time
            end_time = word.end / 1000 if hasattr(word, 'end') else current_start + 3.0
            
            sentences.append({
                "text": ' '.join(current_text),
                "start": current_start,
                "end": end_time
            })
            
            # Reset for next sentence
            current_text = []
            if i < len(words) - 1:
                next_word = words[i + 1]
                current_start = next_word.start / 1000 if hasattr(next_word, 'start') else end_time
    
    # Adjust to match audio duration
    if sentences:
        adjust_sentence_timing(sentences, audio_duration)
    
    return sentences

def split_long_sentences(sentences, target_count, audio_duration):
    """Split long sentences into smaller ones"""
    if len(sentences) >= target_count:
        return sentences
    
    new_sentences = []
    
    for sent in sentences:
        text = sent['text']
        duration = sent['end'] - sent['start']
        
        # If sentence is too long (>7 seconds), split it
        if duration > 7.0:
            words = text.split()
            if len(words) > 15:
                # Split by commas, conjunctions, etc.
                parts = re.split(r'(?<=[,;:])\s+', text)
                if len(parts) > 1:
                    part_duration = duration / len(parts)
                    for i, part in enumerate(parts):
                        new_sentences.append({
                            "text": part.strip(),
                            "start": sent['start'] + (i * part_duration),
                            "end": sent['start'] + ((i + 1) * part_duration)
                        })
                    continue
        
        new_sentences.append(sent)
    
    # If still not enough, do word-based splitting
    if len(new_sentences) < target_count * 0.7:
        all_words = ' '.join([s['text'] for s in new_sentences]).split()
        words_per_sentence = max(8, len(all_words) // target_count)
        
        new_sentences = []
        current_time = 0
        time_per_sentence = audio_duration / (len(all_words) // words_per_sentence)
        
        for i in range(0, len(all_words), words_per_sentence):
            sentence_words = all_words[i:i + words_per_sentence]
            if sentence_words:
                new_sentences.append({
                    "text": ' '.join(sentence_words),
                    "start": current_time,
                    "end": current_time + time_per_sentence
                })
                current_time += time_per_sentence
    
    adjust_sentence_timing(new_sentences, audio_duration)
    return new_sentences

def adjust_sentence_timing(sentences, audio_duration):
    """Adjust sentence timing to match audio duration"""
    if not sentences:
        return
    
    total_time = sentences[-1]['end']
    if total_time < audio_duration:
        # Add time to last sentence
        sentences[-1]['end'] = audio_duration
    elif total_time > audio_duration * 1.1:
        # Scale down
        scale = audio_duration / total_time
        for s in sentences:
            s['start'] *= scale
            s['end'] *= scale

def create_sentences_fallback(text, audio_duration):
    """Create sentences when AssemblyAI is not available"""
    print("📝 Creando oraciones (método alternativo)...")
    
    # Split text into sentences
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
    
    # Calculate optimal count
    target_sentences = max(40, int(audio_duration / 4))
    
    # If too few sentences, split further
    if len(raw_sentences) < target_sentences * 0.7:
        all_words = text.split()
        words_per_sentence = max(8, len(all_words) // target_sentences)
        
        raw_sentences = []
        for i in range(0, len(all_words), words_per_sentence):
            sentence_words = all_words[i:i + words_per_sentence]
            if sentence_words:
                raw_sentences.append(' '.join(sentence_words))
    
    # Create timed sentences
    sentences = []
    total_words = len(text.split())
    words_per_second = total_words / audio_duration if audio_duration > 0 else 3
    
    current_time = 0
    for sent in raw_sentences:
        word_count = len(sent.split())
        duration = max(2.0, min(7.0, word_count / words_per_second))
        
        sentences.append({
            "text": sent,
            "start": current_time,
            "end": current_time + duration
        })
        
        current_time += duration + 0.3  # Add pause
    
    adjust_sentence_timing(sentences, audio_duration)
    print(f"✅ {len(sentences)} oraciones creadas")
    return sentences

# ========================================== 
# 9. SUBTITLE SYSTEM
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
    }
}

def create_ass_file(sentences, ass_file):
    """Create ASS subtitle file with proper format"""
    style_key = random.choice(list(SUBTITLE_STYLES.keys()))
    style = SUBTITLE_STYLES[style_key]
    
    print(f"✨ Usando estilo de subtítulos: {style['name']}")
    
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
            if text.endswith('.') or text.endswith(','):
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
# 10. VIDEO SEARCH & PROCESSING
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_nature_only():
    """Search for nature videos"""
    query = get_nature_query()
    return search_videos_by_query(query)

def search_videos_by_query(query, page=None):
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
                            hd_files = video_files
                        
                        if hd_files:
                            best_file = random.choice(hd_files)
                            video_url = best_file['link']
                            
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
                "page": page
            }
            
            response = requests.get(url, params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for video in data.get('hits', []):
                    videos_dict = video.get('videos', {})
                    
                    video_url = None
                    for quality in ['large', 'medium', 'small']:
                        if quality in videos_dict:
                            video_url = videos_dict[quality]['url']
                            break
                    
                    if video_url and video_url not in USED_VIDEO_URLS:
                        all_results.append({
                            'url': video_url,
                            'service': 'pixabay',
                            'duration': video.get('duration', 0)
                        })
        except Exception as e:
            print(f"    Pixabay error: {str(e)[:50]}")
    
    return all_results

def download_and_process_video(results, target_duration, clip_index):
    """Download and process video"""
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
                
                # Check for GPU
                gpu_available = False
                try:
                    result_gpu = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    gpu_available = (result_gpu.returncode == 0)
                except:
                    pass
                
                if gpu_available:
                    cmd = [
                        "ffmpeg", "-y",
                        "-hwaccel", "cuda",
                        "-i", str(raw_path),
                        "-t", str(target_duration),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
                        "-c:v", "h264_nvenc",
                        "-preset", "p4",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
                        "-an",
                        str(output_path)
                    ]
                else:
                    cmd = [
                        "ffmpeg", "-y",
                        "-i", str(raw_path),
                        "-t", str(target_duration),
                        "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "23",
                        "-pix_fmt", "yuv420p",
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
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:60]}")
            continue
    
    return None

# ========================================== 
# 11. IMPROVED VISUAL PROCESSING
# ========================================== 

def calculate_video_clips_needed(audio_duration, sentences):
    """Calculate how many video clips we need based on audio duration"""
    # For long audio, we need more video clips than sentences
    # Target: 1 video clip per 8-12 seconds of audio
    target_clip_count = max(30, int(audio_duration / 10))
    
    # But ensure we have at least as many clips as sentences
    target_clip_count = max(target_clip_count, len(sentences))
    
    # Cap at reasonable number
    target_clip_count = min(150, target_clip_count)
    
    print(f"🎬 Audio: {audio_duration:.1f}s, Objetivo: {target_clip_count} clips de video")
    
    # Calculate clip durations
    if target_clip_count > len(sentences):
        # We need more clips than sentences
        # Create clip segments independent of sentences
        clip_durations = []
        remaining_time = audio_duration
        
        while remaining_time > 0:
            # Variable clip duration: 5-12 seconds
            clip_duration = random.uniform(5.0, 12.0)
            if clip_duration > remaining_time:
                clip_duration = remaining_time
            
            clip_durations.append(clip_duration)
            remaining_time -= clip_duration
        
        # Adjust to match exact audio duration
        total_clip_time = sum(clip_durations)
        if total_clip_time != audio_duration:
            scale = audio_duration / total_clip_time
            clip_durations = [d * scale for d in clip_durations]
        
        return clip_durations
    
    else:
        # Use sentence durations
        clip_durations = []
        for sent in sentences:
            duration = sent['end'] - sent['start']
            # If sentence is too long, split it for video purposes
            if duration > 15:
                # Split into 2-3 clips
                num_parts = min(3, int(duration / 7) + 1)
                part_duration = duration / num_parts
                for _ in range(num_parts):
                    clip_durations.append(part_duration)
            else:
                clip_durations.append(duration)
        
        return clip_durations

def process_video_clips(clip_durations):
    """Process all video clips in parallel"""
    print(f"🌲 Procesando {len(clip_durations)} clips de naturaleza...")
    
    clip_args = [(i, duration, len(clip_durations)) for i, duration in enumerate(clip_durations)]
    clips = [None] * len(clip_durations)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(process_single_video_clip, arg): arg[0] 
            for arg in clip_args
        }
        
        completed = 0
        
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                
                if clip_path:
                    clips[index] = clip_path
                    completed += 1
                
                progress = 60 + int((completed/len(clip_durations))*25)
                update_status(progress, f"Completados {completed}/{len(clip_durations)} clips")
                
            except Exception as e:
                index = future_to_index[future]
                print(f"❌ Error en clip {index}: {e}")
    
    return clips

def process_single_video_clip(args):
    """Process single video clip"""
    i, duration, total_clips = args
    
    print(f"  📹 Clip {i+1}/{total_clips}: {duration:.1f}s")
    
    for attempt in range(1, 5):
        results = search_videos_nature_only()
        
        if results:
            clip_path = download_and_process_video(results, duration, i)
            if clip_path:
                print(f"    ✅ Éxito (intento {attempt})")
                return (i, clip_path)
        
        time.sleep(0.5)
    
    print(f"    ❌ Falló después de 4 intentos")
    return (i, None)

def create_fallback_clips(clip_durations, failed_indices):
    """Create fallback clips for failed downloads"""
    for idx in failed_indices:
        if idx < len(clip_durations):
            duration = clip_durations[idx]
            color_path = TEMP_DIR / f"color_{idx}.mp4"
            colors = ["0x2E7D32", "0x388E3C", "0x43A047"]
            
            subprocess.run([
                "ffmpeg", "-y", "-f", "lavfi",
                "-i", f"color=c={colors[idx % 3]}:s=1920x1080:d={duration}",
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(color_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            if os.path.exists(color_path):
                yield str(color_path)

def concatenate_videos(video_files, output_path):
    """Concatenate video files"""
    print("🔗 Concatenando videos...")
    
    list_file = TEMP_DIR / "concat_list.txt"
    with open(list_file, "w", encoding="utf-8") as f:
        for vfile in video_files:
            if vfile and os.path.exists(vfile):
                escaped_path = str(Path(vfile).absolute()).replace("\\", "/")
                f.write(f"file '{escaped_path}'\n")
    
    # Check for GPU
    gpu_available = False
    try:
        result = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        gpu_available = (result.returncode == 0)
    except:
        pass
    
    if gpu_available:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "h264_nvenc",
            "-preset", "p4",
            "-cq", "23",
            str(output_path)
        ]
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(output_path)
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Concatenation failed: {result.stderr[-500:]}")
        return False
    
    if not os.path.exists(output_path):
        print("❌ Output file not created")
        return False
    
    file_size = os.path.getsize(output_path)
    print(f"✅ Concatenación completa: {file_size / (1024*1024):.1f}MB")
    
    # Cleanup
    if list_file.exists():
        list_file.unlink()
    
    return True

# ========================================== 
# 12. FINAL VIDEO RENDERING
# ========================================== 

def render_final_videos(visual_path, audio_path, ass_file, logo_path, 
                       output_no_subs, output_with_subs):
    """Render final videos"""
    print("🎬 Renderizando videos finales...")
    
    # VERSION 1: 900p NO SUBTITLES
    print("\n📹 Versión 1: 900p (Sin subtítulos)")
    update_status(85, "Renderizando versión 900p...")
    
    if logo_path and os.path.exists(logo_path):
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", "[0:v]scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(output_no_subs)
        ]
    else:
        cmd_v1 = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(audio_path),
            "-vf", "scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            "-c:a", "aac",
            "-b:a", "128k",
            "-shortest",
            str(output_no_subs)
        ]
    
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True, timeout=300)
    
    if result_v1.returncode != 0:
        print(f"❌ Versión 1 falló: {result_v1.stderr[-300:]}")
        return False
    
    if not os.path.exists(output_no_subs) or os.path.getsize(output_no_subs) < 100000:
        print("❌ Salida de versión 1 inválida")
        return False
    
    file_size_v1 = os.path.getsize(output_no_subs) / (1024*1024)
    print(f"✅ Versión 1: {file_size_v1:.1f}MB")
    
    # VERSION 2: 1080p WITH SUBTITLES
    print("\n📹 Versión 2: 1080p (Con subtítulos)")
    update_status(90, "Renderizando versión 1080p con subtítulos...")
    
    # Escape ASS path
    ass_path = str(ass_file.absolute()).replace("\\", "/").replace(":", "\\\\:")
    
    if logo_path and os.path.exists(logo_path):
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'"
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(logo_path),
            "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(output_with_subs)
        ]
    else:
        filter_v2 = f"[0:v]scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2[bg];[bg]subtitles='{ass_path}'"
        cmd_v2 = [
            "ffmpeg", "-y",
            "-i", str(visual_path),
            "-i", str(audio_path),
            "-filter_complex", filter_v2,
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-c:a", "aac",
            "-b:a", "256k",
            "-shortest",
            str(output_with_subs)
        ]
    
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True, timeout=300)
    
    if result_v2.returncode != 0:
        print(f"⚠️ Versión 2 falló: {result_v2.stderr[-300:]}")
        print("Continuando solo con versión 1...")
        return True
    
    if not os.path.exists(output_with_subs) or os.path.getsize(output_with_subs) < 100000:
        print("⚠️ Salida de versión 2 inválida")
        return True
    
    file_size_v2 = os.path.getsize(output_with_subs) / (1024*1024)
    print(f"✅ Versión 2: {file_size_v2:.1f}MB")
    
    return True

# ========================================== 
# 13. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Upload to Google Drive"""
    if not os.path.exists(file_path):
        print(f"❌ Archivo no encontrado: {file_path}")
        return None
    
    print(f"☁️ Subiendo {os.path.basename(file_path)}...")
    
    client_id = os.environ.get("OAUTH_CLIENT_ID")
    client_secret = os.environ.get("OAUTH_CLIENT_SECRET")
    refresh_token = os.environ.get("OAUTH_REFRESH_TOKEN")
    folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
    
    if not all([client_id, client_secret, refresh_token]):
        print("❌ Credenciales OAuth faltantes")
        return None
    
    try:
        # Get access token
        r = requests.post("https://oauth2.googleapis.com/token", data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token"
        })
        r.raise_for_status()
        access_token = r.json()['access_token']
        
        # Prepare metadata
        filename = os.path.basename(file_path)
        file_size = os.path.getsize(file_path)
        
        metadata = {"name": filename, "mimeType": "video/mp4"}
        if folder_id:
            metadata["parents"] = [folder_id]
        
        # Initialize resumable upload
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            "X-Upload-Content-Type": "video/mp4",
            "X-Upload-Content-Length": str(file_size)
        }
        
        response = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=resumable",
            headers=headers,
            json=metadata
        )
        
        if response.status_code != 200:
            print(f"❌ Inicio fallido: {response.text}")
            return None
        
        session_uri = response.headers.get("Location")
        
        # Upload file
        with open(file_path, "rb") as f:
            upload_resp = requests.put(session_uri, data=f)
        
        if upload_resp.status_code in [200, 201]:
            file_id = upload_resp.json().get('id')
            
            # Make public
            requests.post(
                f"https://www.googleapis.com/drive/v3/files/{file_id}/permissions",
                headers={"Authorization": f"Bearer {access_token}"},
                json={'role': 'reader', 'type': 'anyone'}
            )
            
            link = f"https://drive.google.com/file/d/{file_id}/view"
            print(f"✅ Subido: {link}")
            return link
        else:
            print(f"❌ Subida fallida: {upload_resp.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error de subida: {e}")
        return None

# ========================================== 
# 14. MAIN EXECUTION
# ========================================== 

print("\n" + "="*60)
print("🎬 GENERADOR DE VIDEOS EN ESPAÑOL")
print("✅ Chatterbox TTS Español (language_id='es')")
print("🔤 AssemblyAI para subtítulos")
print("🌲 Videos de naturaleza pura")
print("="*60)

try:
    update_status(1, "Inicializando...")
    
    # Download assets
    ref_voice = TEMP_DIR / "voice_ref.mp3"
    if not download_asset(VOICE_PATH, ref_voice):
        print("⚠️ Descarga de voz fallida, usando voz por defecto")
        ref_voice = None
    
    ref_logo = None
    if LOGO_PATH and LOGO_PATH != "None":
        ref_logo = TEMP_DIR / "logo.png"
        if not download_asset(LOGO_PATH, ref_logo):
            ref_logo = None
    
    # Generate script
    update_status(10, "Generando guión...")
    
    if MODE == "topic":
        script_text = generate_spanish_script(TOPIC, DURATION_MINS)
    else:
        script_text = SCRIPT_TEXT
    
    print(f"📝 Guión: {len(script_text)} caracteres")
    
    # Generate audio
    update_status(20, "Generando audio español...")
    audio_file = TEMP_DIR / "audio.wav"
    
    if generate_spanish_audio_chatterbox(script_text, audio_file, ref_voice):
        # Get audio duration
        with wave.open(str(audio_file), 'rb') as wav_file:
            audio_duration = wav_file.getnframes() / wav_file.getframerate()
        
        print(f"🎵 Duración del audio: {audio_duration:.1f}s")
        
        # Create subtitles
        update_status(45, "Creando subtítulos...")
        
        sentences = None
        
        # Try AssemblyAI first
        if ASSEMBLY_KEY:
            sentences = transcribe_with_assemblyai_improved(audio_file, audio_duration)
        
        # Fallback if AssemblyAI fails
        if not sentences:
            sentences = create_sentences_fallback(script_text, audio_duration)
        
        if not sentences or len(sentences) < 10:
            raise Exception("No se pudieron crear suficientes subtítulos")
        
        print(f"✅ {len(sentences)} oraciones para subtítulos")
        
        # Create ASS subtitle file
        ass_file = TEMP_DIR / "subs.ass"
        create_ass_file(sentences, ass_file)
        
        # Calculate video clips needed
        update_status(55, "Calculando clips de video...")
        clip_durations = calculate_video_clips_needed(audio_duration, sentences)
        
        # Process video clips
        update_status(60, "🌲 Procesando videos de naturaleza...")
        clips = process_video_clips(clip_durations)
        
        # Filter valid clips
        valid_clips = [c for c in clips if c and os.path.exists(c)]
        
        # Create fallback for failed clips
        failed_indices = [i for i, c in enumerate(clips) if c is None]
        if failed_indices:
            print(f"⚠️ Creando clips de respaldo para {len(failed_indices)} fallos")
            fallback_clips = list(create_fallback_clips(clip_durations, failed_indices))
            valid_clips.extend(fallback_clips)
        
        if not valid_clips:
            raise Exception("No se generaron clips de video")
        
        print(f"✅ {len(valid_clips)} clips de video listos")
        
        # Concatenate videos
        visual_output = TEMP_DIR / "visual.mp4"
        if not concatenate_videos(valid_clips, visual_output):
            raise Exception("Error al concatenar videos")
        
        # Render final videos
        update_status(85, "Renderizando videos finales...")
        output_no_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_no_subs.mp4"
        output_with_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_with_subs.mp4"
        
        if not render_final_videos(visual_output, audio_file, ass_file, ref_logo, 
                                 output_no_subs, output_with_subs):
            raise Exception("Error al renderizar videos finales")
        
        # Upload to Google Drive
        update_status(95, "Subiendo a Google Drive...")
        
        links = {}
        if os.path.exists(output_no_subs):
            links['no_subs'] = upload_to_google_drive(output_no_subs)
        if os.path.exists(output_with_subs):
            links['with_subs'] = upload_to_google_drive(output_with_subs)
        
        # Final status
        final_msg = "✅ ¡Video en Español Completado!\n"
        final_msg += f"🎵 Duración: {audio_duration:.1f}s\n"
        final_msg += f"📝 Oraciones: {len(sentences)}\n"
        final_msg += f"🎬 Clips: {len(valid_clips)}\n"
        if links.get('no_subs'):
            final_msg += f"📹 Sin Subs: {links['no_subs']}\n"
        if links.get('with_subs'):
            final_msg += f"📹 Con Subs: {links['with_subs']}"
        
        update_status(100, final_msg, "completed", links.get('no_subs') or links.get('with_subs'))
        print(f"\n🎉 {final_msg}")
        
    else:
        raise Exception("Error en la generación de audio")

except Exception as e:
    error_msg = f"❌ Error: {str(e)}"
    print(error_msg)
    import traceback
    traceback.print_exc()
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

print("\n✅ COMPLETO")
