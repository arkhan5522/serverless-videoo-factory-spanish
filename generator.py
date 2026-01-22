"""
AI VIDEO GENERATOR - SPANISH VERSION WITH GPU & ASSEMBLYAI
==========================================================
✅ Chatterbox Multilingual TTS for Spanish audio (language_id="es")
✅ AssemblyAI for accurate subtitle timing
✅ GPU acceleration for video processing
✅ Pure natural greenery scenes
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
from pathlib import Path

# ========================================== 
# 1. INSTALLATION & GPU DETECTION
# ========================================== 

print("--- 🔧 Installing Dependencies & Detecting GPU ---")

def detect_gpu():
    """Detect available GPU hardware"""
    gpu_info = {
        "available": False,
        "type": None,
        "driver": None
    }
    
    try:
        # Check NVIDIA GPU
        result = subprocess.run(["nvidia-smi"], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0:
            gpu_info["available"] = True
            gpu_info["type"] = "nvidia"
            
            # Extract driver info
            for line in result.stdout.split('\n'):
                if "Driver Version" in line:
                    gpu_info["driver"] = line.strip()
                    break
            
            print(f"✅ NVIDIA GPU detected: {gpu_info['driver']}")
            return gpu_info
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # Check AMD GPU (ROCm)
        result = subprocess.run(["rocminfo"], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode == 0 and "GPU" in result.stdout:
            gpu_info["available"] = True
            gpu_info["type"] = "amd"
            gpu_info["driver"] = "ROCm"
            print("✅ AMD GPU detected (ROCm)")
            return gpu_info
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    try:
        # Check Intel GPU
        result = subprocess.run(["intel_gpu_top", "-h"], 
                              capture_output=True, 
                              text=True,
                              timeout=5)
        if result.returncode in [0, 1]:  # Some tools return 1 for help
            gpu_info["available"] = True
            gpu_info["type"] = "intel"
            gpu_info["driver"] = "Intel"
            print("✅ Intel GPU detected")
            return gpu_info
            
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    
    # Check via PyTorch
    try:
        import torch
        if torch.cuda.is_available():
            gpu_info["available"] = True
            gpu_info["type"] = "cuda"
            gpu_info["driver"] = f"CUDA {torch.version.cuda}"
            print(f"✅ CUDA GPU detected via PyTorch: {torch.cuda.get_device_name(0)}")
            return gpu_info
    except ImportError:
        pass
    
    print("❌ No GPU detected, using CPU")
    return gpu_info

# Detect GPU early
GPU_INFO = detect_gpu()

# Install dependencies
try:
    # Base requirements
    libs = [
        "torch",
        "torchaudio", 
        "google-generativeai",
        "requests",
        "numpy",
        "pillow",
        "chatterbox-tts",
        "assemblyai"  # AssemblyAI for subtitle timing
    ]
    
    for lib in libs:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib, "--quiet"])
            print(f"✅ {lib}")
        except Exception as e:
            print(f"⚠️  {lib}: {str(e)[:50]}")
    
    # Install FFmpeg
    subprocess.run("apt-get update -qq && apt-get install -qq -y ffmpeg", 
                   shell=True, check=False)
    
except Exception as e:
    print(f"Install Warning: {e}")

import torch
import torchaudio as ta
import google.generativeai as genai
import assemblyai as aai  # AssemblyAI for transcription

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
ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")  # AssemblyAI key
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
# 3. GPU-OPTIMIZED FFMPEG FUNCTIONS
# ========================================== 

def get_ffmpeg_gpu_params():
    """Return GPU-optimized FFmpeg parameters"""
    if GPU_INFO["type"] == "nvidia":
        return {
            "hwaccel": "cuda",
            "video_codec": "h264_nvenc",
            "preset": "p4",
            "quality_param": ["-cq", "23"]
        }
    elif GPU_INFO["type"] == "amd":
        return {
            "hwaccel": "vaapi",
            "video_codec": "h264_vaapi",
            "preset": "fast",
            "quality_param": ["-qp", "23"]
        }
    elif GPU_INFO["type"] == "intel":
        return {
            "hwaccel": "qsv",
            "video_codec": "h264_qsv",
            "preset": "fast",
            "quality_param": ["-q", "23"]
        }
    else:
        return {
            "hwaccel": None,
            "video_codec": "libx264",
            "preset": "fast",
            "quality_param": ["-crf", "23"]
        }

def build_ffmpeg_gpu_command(input_files, output_file, duration=None, 
                           scale_filter=None, has_audio=True, **kwargs):
    """Build FFmpeg command with GPU optimization"""
    cmd = ["ffmpeg", "-y"]
    
    # Add GPU acceleration if available
    gpu_params = get_ffmpeg_gpu_params()
    if gpu_params["hwaccel"]:
        cmd.extend(["-hwaccel", gpu_params["hwaccel"]])
    
    # Add input files
    for input_file in input_files:
        cmd.extend(["-i", str(input_file)])
    
    # Add duration limit
    if duration:
        cmd.extend(["-t", str(duration)])
    
    # Build filter complex if needed
    filter_parts = []
    if scale_filter:
        filter_parts.append(scale_filter)
    
    if filter_parts:
        cmd.extend(["-vf", ",".join(filter_parts)])
    
    # Video encoding with GPU
    cmd.extend(["-c:v", gpu_params["video_codec"]])
    cmd.extend(["-preset", gpu_params["preset"]])
    cmd.extend(gpu_params["quality_param"])
    
    # Audio handling
    if has_audio:
        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
    else:
        cmd.extend(["-an"])
    
    # Output
    cmd.append(str(output_file))
    
    return cmd

# ========================================== 
# 4. LOAD AI MODELS
# ========================================== 

print("--- 🤖 Loading AI Models ---")

# Load Chatterbox Multilingual TTS
if TTS_AVAILABLE:
    try:
        device = "cuda" if GPU_INFO["available"] and GPU_INFO["type"] == "nvidia" else "cpu"
        print(f"Loading Chatterbox on {device}...")
        TTS_MODEL = ChatterboxMultilingualTTS.from_pretrained(device=device)
        print("✅ Chatterbox Multilingual TTS loaded successfully")
    except Exception as e:
        print(f"⚠️ Chatterbox loading failed: {e}")
        TTS_AVAILABLE = False

print("🌲 All videos will show pure nature scenes")

# ========================================== 
# 5. NATURE VIDEO QUERIES (HARDCODED)
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
    print(f"    🌲 Using Nature Query: '{query}'")
    return query

# ========================================== 
# 6. STATUS UPDATES
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
    
    alt_paths = [
        f"static/{path}",
        f"uploads/{path}",
        path.replace("uploads/", "static/"),
        path.replace("static/", "uploads/")
    ]
    
    for alt_path in alt_paths:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{alt_path}"
            r = requests.get(url, headers=headers)
            if r.status_code == 200:
                with open(local, "wb") as f:
                    f.write(r.content)
                return True
        except:
            continue
    
    return False

# ========================================== 
# 7. SPANISH SCRIPT GENERATION
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
- Tono educativo y apropiado para toda la familia
- Escribe en un estilo documental profesional
- Mantén los párrafos cohesivos y fluidos
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
# 8. CHATTERBOX TTS AUDIO GENERATION (SPANISH)
# ========================================== 

def generate_spanish_audio_chatterbox(text, output_path, audio_prompt_path=None):
    """Generate Spanish TTS audio using Chatterbox"""
    if not TTS_AVAILABLE or TTS_MODEL is None:
        print("⚠️ Chatterbox TTS not available, creating silent audio")
        return create_silent_audio(text, output_path)
    
    print("🎙️ Generating Spanish Audio with Chatterbox TTS (language_id='es')...")
    
    try:
        # Split text into sentences for better processing
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
                
                # Add small pause between sentences
                if wav_audio.dim() == 1:
                    wav_audio = wav_audio.unsqueeze(0)
                
                silence_samples = int(0.2 * sample_rate)
                silence = torch.zeros((wav_audio.shape[0], silence_samples))
                segment_with_pause = torch.cat([wav_audio, silence], dim=-1)
                all_audio_segments.append(segment_with_pause)
                
            except Exception as e:
                print(f"⚠️ TTS error on sentence {i}: {e}")
                continue
        
        if all_audio_segments:
            # Concatenate all segments
            full_audio = torch.cat(all_audio_segments, dim=-1)
            
            # Ensure mono audio
            if full_audio.shape[0] > 1:
                full_audio = torch.mean(full_audio, dim=0, keepdim=True)
            
            # Save with proper format
            temp_path = str(output_path).replace('.wav', '_temp.wav')
            ta.save(temp_path, full_audio, sample_rate)
            
            # Convert to proper WAV format
            subprocess.run([
                "ffmpeg", "-y",
                "-i", temp_path,
                "-ar", "44100",
                "-ac", "1",
                "-acodec", "pcm_s16le",
                str(output_path)
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Verify audio
            if os.path.exists(output_path):
                with wave.open(str(output_path), 'rb') as wav_file:
                    duration = wav_file.getnframes() / wav_file.getframerate()
                    print(f"✅ Audio generado: {duration:.1f}s")
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
    
    print(f"⚠️ Silent audio created: {duration_seconds}s")
    return True

# ========================================== 
# 9. ASSEMBLYAI TRANSCRIPTION FOR SUBTITLES
# ========================================== 

def transcribe_with_assemblyai(audio_path):
    """Transcribe audio using AssemblyAI for accurate timing"""
    if not ASSEMBLY_KEY:
        print("⚠️ AssemblyAI key not found, using simple timing")
        return None
    
    print("🔤 Transcribing audio with AssemblyAI...")
    
    try:
        # Configure AssemblyAI
        aai.settings.api_key = ASSEMBLY_KEY
        
        # Create transcriber
        transcriber = aai.Transcriber()
        
        # Transcribe audio
        update_status(45, "Transcribiendo audio...")
        transcript = transcriber.transcribe(str(audio_path))
        
        if transcript.status == aai.TranscriptStatus.error:
            print(f"❌ AssemblyAI transcription error: {transcript.error}")
            return None
        
        # Extract sentences with timing
        sentences = []
        for sentence in transcript.get_sentences():
            sentences.append({
                "text": sentence.text,
                "start": sentence.start / 1000,  # Convert to seconds
                "end": sentence.end / 1000
            })
        
        print(f"✅ Transcription complete: {len(sentences)} sentences")
        return sentences
        
    except Exception as e:
        print(f"⚠️ AssemblyAI transcription failed: {e}")
        return None

def create_sentences_from_text(text, audio_duration):
    """Fallback method to create sentences when AssemblyAI is not available"""
    print("📝 Creating sentences from text (fallback method)...")
    
    # Split text into sentences
    raw_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 2]
    
    # Calculate approximate timing
    total_words = len(text.split())
    words_per_second = total_words / audio_duration if audio_duration > 0 else 3
    
    sentences = []
    current_time = 0
    
    for sent in raw_sentences:
        word_count = len(sent.split())
        duration = word_count / words_per_second
        
        # Ensure minimum duration
        duration = max(2.0, min(duration, 8.0))
        
        sentences.append({
            "text": sent,
            "start": current_time,
            "end": current_time + duration
        })
        
        current_time += duration
    
    # Adjust last sentence to match audio duration
    if sentences:
        total_sentence_time = sentences[-1]['end']
        if total_sentence_time < audio_duration:
            sentences[-1]['end'] = audio_duration
        elif total_sentence_time > audio_duration:
            scale_factor = audio_duration / total_sentence_time
            for s in sentences:
                s['start'] *= scale_factor
                s['end'] *= scale_factor
    
    print(f"✅ Created {len(sentences)} sentences")
    return sentences

# ========================================== 
# 10. SUBTITLE SYSTEM
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
# 11. VIDEO SEARCH (NATURE ONLY)
# ========================================== 

USED_VIDEO_URLS = set()

def search_videos_nature_only(clip_index):
    """ALWAYS use nature queries - IGNORE Spanish text completely"""
    query = get_nature_query()
    return search_videos_by_query(query, clip_index)

def search_videos_by_query(query, clip_index, page=None):
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

def download_and_process_video_gpu(results, target_duration, clip_index):
    """Download and process video with GPU acceleration"""
    for i, result in enumerate(results[:5]):
        try:
            raw_path = TEMP_DIR / f"raw_{clip_index}_{i}.mp4"
            
            # Download video
            response = requests.get(result['url'], timeout=30, stream=True)
            with open(raw_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            if os.path.exists(raw_path) and os.path.getsize(raw_path) > 0:
                output_path = TEMP_DIR / f"clip_{clip_index}.mp4"
                
                # Process with GPU acceleration
                cmd = build_ffmpeg_gpu_command(
                    input_files=[raw_path],
                    output_file=output_path,
                    duration=target_duration,
                    scale_filter="scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080,fps=30",
                    has_audio=False
                )
                
                # Run FFmpeg
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # Cleanup
                try:
                    os.remove(raw_path)
                except:
                    pass
                
                if os.path.exists(output_path):
                    USED_VIDEO_URLS.add(result['url'])
                    print(f"    ✓ {result['service']} video (GPU processed)")
                    return str(output_path)
                    
        except Exception as e:
            print(f"    ✗ Error: {str(e)[:60]}")
            continue
    
    return None

# ========================================== 
# 12. VIDEO PROCESSING WITH GPU
# ========================================== 

def process_single_clip_gpu(args):
    """Process single clip with GPU acceleration"""
    i, sent, sentences_count = args
    
    duration = max(3.5, sent['end'] - sent['start'])
    
    print(f"  🌲 Clip {i+1}/{sentences_count}: Nature Scene")
    
    for attempt in range(1, 7):
        print(f"    Attempt {attempt}: Nature Query")
        
        # ALWAYS use nature queries
        results = search_videos_nature_only(i)
        
        if results:
            clip_path = download_and_process_video_gpu(results, duration, i)
            if clip_path:
                print(f"    ✅ Success")
                return (i, clip_path)
        
        time.sleep(0.5)
    
    print(f"    ❌ Failed")
    return (i, None)

def process_visuals_gpu(sentences, audio_path, ass_file, logo_path, output_no_subs, output_with_subs):
    """Process visuals with GPU acceleration"""
    print("🎬 Processing Visuals with GPU...")
    
    clip_args = [(i, sent, len(sentences)) for i, sent in enumerate(sentences)]
    clips = [None] * len(sentences)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_index = {
            executor.submit(process_single_clip_gpu, arg): arg[0] 
            for arg in clip_args
        }
        
        completed = 0
        failed_clips = []
        
        for future in concurrent.futures.as_completed(future_to_index):
            try:
                index, clip_path = future.result()
                
                if clip_path:
                    clips[index] = clip_path
                    completed += 1
                else:
                    failed_clips.append(index)
                
                update_status(60 + int((completed/len(sentences))*25), 
                            f"Completed {completed}/{len(sentences)}")
                
            except Exception as e:
                index = future_to_index[future]
                failed_clips.append(index)
    
    # Create green backgrounds for failed clips
    if failed_clips:
        print(f"⚠️ Creating green backgrounds for {len(failed_clips)} clips")
        for idx in failed_clips:
            if idx < len(sentences):
                duration = max(3.5, sentences[idx]['end'] - sentences[idx]['start'])
                color_path = TEMP_DIR / f"color_{idx}.mp4"
                colors = ["0x2E7D32", "0x388E3C", "0x43A047"]
                
                subprocess.run([
                    "ffmpeg", "-y", "-f", "lavfi",
                    "-i", f"color=c={colors[idx % 3]}:s=1920x1080:d={duration}",
                    "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(color_path)
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                if os.path.exists(color_path):
                    clips[idx] = str(color_path)
    
    # Filter valid clips
    valid_clips = []
    for c in clips:
        if c and os.path.exists(c) and os.path.getsize(c) > 1000:
            valid_clips.append(c)
    
    if not valid_clips:
        print("❌ No valid clips generated")
        return False
    
    print(f"✅ Valid clips: {len(valid_clips)}/{len(sentences)}")
    
    # Concatenate clips with GPU
    print("⚡ Concatenating clips with GPU...")
    list_file = Path("list.txt")
    
    with open(list_file, "w", encoding="utf-8") as f:
        for c in valid_clips:
            escaped_path = str(Path(c).absolute()).replace("\\", "/")
            f.write(f"file '{escaped_path}'\n")
    
    visual_output = Path("visual.mp4")
    
    # Build GPU concatenation command
    gpu_params = get_ffmpeg_gpu_params()
    
    if GPU_INFO["available"]:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", gpu_params["video_codec"],
            "-preset", gpu_params["preset"],
        ]
        cmd.extend(gpu_params["quality_param"])
        cmd.append(str(visual_output))
    else:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "23",
            str(visual_output)
        ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Concatenation failed: {result.stderr[-500:]}")
        return False
    
    if not os.path.exists(visual_output):
        print("❌ visual.mp4 not created")
        return False
    
    file_size = os.path.getsize(visual_output)
    print(f"✅ Concatenation complete: {file_size / (1024*1024):.1f}MB")
    
    # Render final videos with GPU
    print("📹 Creating final videos with GPU...")
    
    # Escape ASS file path properly
    ass_path = str(ass_file.absolute()).replace("\\", "/").replace(":", "\\\\:")
    
    # VERSION 1: 900p NO SUBTITLES
    print("\n📹 Version 1: 900p (No Subtitles)")
    update_status(85, "Rendering 900p version...")
    
    scale_filter_900 = "scale=1600:900:force_original_aspect_ratio=decrease,pad=1600:900:(ow-iw)/2:(oh-ih)/2"
    
    if logo_path and os.path.exists(logo_path):
        filter_complex = f"[0:v]{scale_filter_900}[bg];[1:v]scale=200:-1[logo];[bg][logo]overlay=25:25[v]"
        inputs = [visual_output, logo_path, audio_path]
        map_args = ["-map", "[v]", "-map", "2:a"]
    else:
        filter_complex = scale_filter_900
        inputs = [visual_output, audio_path]
        map_args = ["-map", "0:v", "-map", "1:a"]
    
    cmd_v1 = build_ffmpeg_gpu_command(
        input_files=inputs,
        output_file=output_no_subs,
        scale_filter=filter_complex,
        has_audio=True
    )
    
    # Insert map arguments before output
    cmd_v1 = cmd_v1[:-1] + map_args + [cmd_v1[-1]]
    
    result_v1 = subprocess.run(cmd_v1, capture_output=True, text=True)
    
    if result_v1.returncode != 0:
        print(f"❌ Version 1 failed: {result_v1.stderr[-300:]}")
        return False
    
    if not os.path.exists(output_no_subs) or os.path.getsize(output_no_subs) < 100000:
        print("❌ Version 1 output invalid")
        return False
    
    file_size_v1 = os.path.getsize(output_no_subs) / (1024*1024)
    print(f"✅ Version 1: {file_size_v1:.1f}MB")
    
    # VERSION 2: 1080p WITH SUBTITLES
    print("\n📹 Version 2: 1080p (With Subtitles)")
    update_status(90, "Rendering 1080p with subtitles...")
    
    scale_filter_1080 = "scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2"
    
    if logo_path and os.path.exists(logo_path):
        filter_complex_v2 = f"[0:v]{scale_filter_1080}[bg];[1:v]scale=230:-1[logo];[bg][logo]overlay=30:30[withlogo];[withlogo]subtitles='{ass_path}'[v]"
        inputs_v2 = [visual_output, logo_path, audio_path]
        map_args_v2 = ["-map", "[v]", "-map", "2:a"]
    else:
        filter_complex_v2 = f"[0:v]{scale_filter_1080}[bg];[bg]subtitles='{ass_path}'[v]"
        inputs_v2 = [visual_output, audio_path]
        map_args_v2 = ["-map", "[v]", "-map", "1:a"]
    
    cmd_v2 = build_ffmpeg_gpu_command(
        input_files=inputs_v2,
        output_file=output_with_subs,
        scale_filter=filter_complex_v2,
        has_audio=True
    )
    
    # Replace audio bitrate for higher quality
    if "-b:a" in cmd_v2:
        idx = cmd_v2.index("-b:a")
        cmd_v2[idx + 1] = "256k"
    
    # Insert map arguments before output
    cmd_v2 = cmd_v2[:-1] + map_args_v2 + [cmd_v2[-1]]
    
    result_v2 = subprocess.run(cmd_v2, capture_output=True, text=True)
    
    if result_v2.returncode != 0:
        print(f"⚠️ Version 2 failed: {result_v2.stderr[-300:]}")
        print("Continuing with Version 1 only...")
        return True
    
    if not os.path.exists(output_with_subs) or os.path.getsize(output_with_subs) < 100000:
        print("⚠️ Version 2 output invalid")
        return True
    
    file_size_v2 = os.path.getsize(output_with_subs) / (1024*1024)
    print(f"✅ Version 2: {file_size_v2:.1f}MB")
    
    return True

# ========================================== 
# 13. GOOGLE DRIVE UPLOAD
# ========================================== 

def upload_to_google_drive(file_path):
    """Upload to Google Drive"""
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
            print(f"❌ Init failed: {response.text}")
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
            print(f"✅ Uploaded: {link}")
            return link
        else:
            print(f"❌ Upload failed: {upload_resp.text}")
            return None
            
    except Exception as e:
        print(f"❌ Upload error: {e}")
        return None

# ========================================== 
# 14. MAIN EXECUTION
# ========================================== 

print("\n" + "="*60)
print("🎬 SPANISH VIDEO GENERATOR - GPU OPTIMIZED")
print(f"💻 GPU: {'✅ ' + GPU_INFO['type'].upper() if GPU_INFO['available'] else '❌ CPU'}")
print("✅ Chatterbox Multilingual TTS (language_id='es')")
print("🔤 AssemblyAI for subtitle timing")
print("🌲 Pure Nature Videos")
print("="*60)

try:
    update_status(1, "Inicializando...")
    
    # Download voice reference
    ref_voice = TEMP_DIR / "voice_ref.mp3"
    if not download_asset(VOICE_PATH, ref_voice):
        print("⚠️ Voice download failed, will use default voice")
        ref_voice = None
    else:
        print(f"✅ Voice: {os.path.getsize(ref_voice)} bytes")
    
    # Download logo
    ref_logo = None
    if LOGO_PATH and LOGO_PATH != "None":
        ref_logo = TEMP_DIR / "logo.png"
        if not download_asset(LOGO_PATH, ref_logo):
            ref_logo = None
    
    # Generate script
    update_status(10, "Generando guión en español...")
    
    if MODE == "topic":
        script_text = generate_spanish_script(TOPIC, DURATION_MINS)
    else:
        script_text = SCRIPT_TEXT
    
    print(f"📝 Guión: {len(script_text)} caracteres")
    
    # Generate Spanish audio with Chatterbox
    update_status(20, "🎙️ Generando audio español con Chatterbox...")
    
    audio_file = TEMP_DIR / "audio.wav"
    
    if generate_spanish_audio_chatterbox(script_text, audio_file, ref_voice):
        # Get audio duration
        with wave.open(str(audio_file), 'rb') as wav_file:
            audio_duration = wav_file.getnframes() / wav_file.getframerate()
        
        print(f"🎵 Duración del audio: {audio_duration:.1f}s")
        
        # Create subtitles using AssemblyAI for accurate timing
        update_status(45, "🔤 Creando subtítulos con AssemblyAI...")
        
        sentences = None
        
        # Try AssemblyAI first
        if ASSEMBLY_KEY:
            sentences = transcribe_with_assemblyai(audio_file)
        
        # Fallback to text-based timing if AssemblyAI fails
        if not sentences:
            print("⚠️ Usando temporización simple (fallback)...")
            sentences = create_sentences_from_text(script_text, audio_duration)
        
        if not sentences:
            raise Exception("No se pudieron crear subtítulos")
        
        # Create ASS subtitle file
        ass_file = TEMP_DIR / "subs.ass"
        create_ass_file(sentences, ass_file)
        
        # Process visuals with GPU acceleration
        update_status(55, "🌲 Procesando videos de naturaleza con GPU...")
        output_no_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_no_subs.mp4"
        output_with_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_with_subs.mp4"
        
        if process_visuals_gpu(sentences, audio_file, ass_file, ref_logo, output_no_subs, output_with_subs):
            
            # Upload to Google Drive
            update_status(95, "☁️ Subiendo a Google Drive...")
            
            links = {}
            if os.path.exists(output_no_subs):
                links['no_subs'] = upload_to_google_drive(output_no_subs)
            if os.path.exists(output_with_subs):
                links['with_subs'] = upload_to_google_drive(output_with_subs)
            
            # Final status
            final_msg = "✅ ¡Video en Español Completado!\n"
            final_msg += f"💻 Procesado con: {'GPU ' + GPU_INFO['type'].upper() if GPU_INFO['available'] else 'CPU'}\n"
            final_msg += "🎙️ Chatterbox TTS Español\n"
            final_msg += "🔤 Subtítulos con AssemblyAI\n"
            final_msg += "🌲 Videos de Naturaleza Pura\n"
            if links.get('no_subs'):
                final_msg += f"📹 Sin Subs: {links['no_subs']}\n"
            if links.get('with_subs'):
                final_msg += f"📹 Con Subs: {links['with_subs']}"
            
            update_status(100, final_msg, "completed", links.get('no_subs') or links.get('with_subs'))
            
            print("\n🎉 ¡ÉXITO!")
            print(f"Guion: {len(script_text)} caracteres")
            print(f"Oraciones: {len(sentences)}")
            print(f"Duración: {audio_duration:.1f}s")
            print(f"GPU utilizado: {'✅ Sí' if GPU_INFO['available'] else '❌ No'}")
            if links:
                print("Enlaces:", links)
            
        else:
            raise Exception("Error en el procesamiento visual")
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

print("\n✅ COMPLETe")
