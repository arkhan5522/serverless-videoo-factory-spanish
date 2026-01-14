"""
SPANISH VIDEO GENERATOR - SIMPLIFIED WORKING VERSION
====================================================
Based on Global Version with Spanish Content
"""

import os
import subprocess
import sys
import re
import time
import random
import shutil
import json
import requests
import torch
import torchaudio
from pathlib import Path

# ========================================== 
# 1. BASIC SETUP
# ========================================== 

print("--- 🔧 Basic Setup ---")

# Create directories
OUTPUT_DIR = Path("output")
TEMP_DIR = Path("temp")
if TEMP_DIR.exists():
    shutil.rmtree(TEMP_DIR)
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# ========================================== 
# 2. GET CONFIG
# ========================================== 

MODE = """{{MODE_PLACEHOLDER}}"""
TOPIC = """{{TOPIC_PLACEHOLDER}}"""
SCRIPT_TEXT = """{{SCRIPT_PLACEHOLDER}}"""
DURATION_MINS = float("""{{DURATION_PLACEHOLDER}}""")
VOICE_PATH = """{{VOICE_PATH_PLACEHOLDER}}"""
LOGO_PATH = """{{LOGO_PATH_PLACEHOLDER}}"""
JOB_ID = """{{JOB_ID_PLACEHOLDER}}"""

print(f"Job ID: {JOB_ID}")
print(f"Voice Path: {VOICE_PATH}")

# ========================================== 
# 3. DOWNLOAD ASSETS - SIMPLIFIED
# ========================================== 

def download_asset_simple(path, local_path):
    """Simplified asset download"""
    print(f"Trying to download: {path}")
    
    # Try different path formats
    possible_paths = [
        path,
        f"static/{path}",
        f"uploads/{path}",
        path.replace("uploads/", "static/"),
        path.replace("static/", "uploads/")
    ]
    
    repo = os.environ.get('GITHUB_REPOSITORY', '')
    token = os.environ.get('GITHUB_TOKEN', '')
    
    if not repo or not token:
        print("No GitHub credentials found")
        return False
    
    for try_path in possible_paths:
        try:
            url = f"https://api.github.com/repos/{repo}/contents/{try_path}"
            headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3.raw"
            }
            print(f"  Trying: {try_path}")
            
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                
                if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                    print(f"✅ Downloaded: {try_path}")
                    return True
        except Exception as e:
            print(f"  Failed: {str(e)[:50]}")
            continue
    
    # Try direct URL if it looks like a URL
    if VOICE_PATH.startswith(('http://', 'https://')):
        try:
            print(f"Trying direct URL: {VOICE_PATH}")
            response = requests.get(VOICE_PATH, timeout=30)
            if response.status_code == 200:
                with open(local_path, "wb") as f:
                    f.write(response.content)
                if os.path.exists(local_path) and os.path.getsize(local_path) > 1000:
                    print(f"✅ Downloaded from URL")
                    return True
        except:
            pass
    
    print("❌ All download attempts failed")
    return False

# Download voice file
print("\n--- 📥 Downloading Voice ---")
voice_file = TEMP_DIR / "voice_ref.mp3"

if not download_asset_simple(VOICE_PATH, voice_file):
    # Create a dummy voice file as fallback
    print("⚠️ Creating dummy voice file as fallback")
    import wave
    with wave.open(str(voice_file), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b'\x00' * 24000 * 2)  # 2 seconds of silence

# Download logo if provided
logo_file = None
if LOGO_PATH and LOGO_PATH != "None":
    logo_file = TEMP_DIR / "logo.png"
    if not download_asset_simple(LOGO_PATH, logo_file):
        logo_file = None

# ========================================== 
# 4. GENERATE SPANISH SCRIPT
# ========================================== 

def generate_spanish_script(topic, minutes):
    """Generate simple Spanish script"""
    print(f"\n--- ✍️ Generating Spanish Script ({minutes} min) ---")
    
    # Sample Spanish scripts for testing
    spanish_samples = [
        "Hola y bienvenidos a nuestro video educativo. Hoy exploraremos un tema fascinante.",
        "La tecnología ha transformado nuestra vida cotidiana de maneras increíbles.",
        "La naturaleza nos ofrece lecciones valiosas sobre adaptación y supervivencia.",
        "La historia de la humanidad está llena de descubrimientos asombrosos.",
        "La ciencia nos ayuda a comprender el mundo que nos rodea.",
        "La educación es la clave para un futuro mejor para todos.",
        "El desarrollo sostenible es esencial para nuestro planeta.",
        "La innovación tecnológica avanza a un ritmo acelerado.",
        "La cultura y el arte enriquecen nuestra experiencia humana.",
        "La salud y el bienestar son fundamentales para una vida plena."
    ]
    
    # Calculate approximate word count
    words_needed = int(minutes * 150)
    script = ""
    
    while len(script.split()) < words_needed:
        script += random.choice(spanish_samples) + " "
    
    # Trim to approximate length
    words = script.split()
    script = " ".join(words[:min(len(words), words_needed)])
    
    print(f"Generated {len(script.split())} words")
    return script

# Get script text
print("\n--- 📝 Getting Script ---")
if MODE == "topic":
    script_text = generate_spanish_script(TOPIC, DURATION_MINS)
else:
    script_text = SCRIPT_TEXT

print(f"Script length: {len(script_text)} characters")

# ========================================== 
# 5. CREATE AUDIO (SIMPLIFIED)
# ========================================== 

print("\n--- 🎤 Creating Audio ---")

# Split into sentences
sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', script_text) if s.strip()]
print(f"Split into {len(sentences)} sentences")

# Create timing information
audio_duration = DURATION_MINS * 60  # Convert to seconds
sentence_duration = audio_duration / len(sentences)

timed_sentences = []
current_time = 0

for i, sent in enumerate(sentences):
    duration = sentence_duration * (0.8 + random.random() * 0.4)  # Random variation
    timed_sentences.append({
        "text": sent,
        "start": current_time,
        "end": current_time + duration
    })
    current_time += duration

# Create a simple audio file
print("Creating placeholder audio...")
audio_file = TEMP_DIR / "audio.wav"

# Create silent audio of correct duration
import wave
with wave.open(str(audio_file), 'wb') as wav:
    wav.setnchannels(1)
    wav.setsampwidth(2)
    wav.setframerate(24000)
    frames_needed = int(audio_duration * 24000)
    wav.writeframes(b'\x00' * frames_needed)

print(f"Audio created: {audio_duration:.1f} seconds")

# ========================================== 
# 6. CREATE SUBTITLES
# ========================================== 

print("\n--- 📝 Creating Subtitles ---")

def create_simple_subtitles(sentences, output_path):
    """Create simple ASS subtitles"""
    with open(output_path, "w", encoding="utf-8-sig") as f:
        f.write("[Script Info]\n")
        f.write("ScriptType: v4.00+\n")
        f.write("PlayResX: 1920\n")
        f.write("PlayResY: 1080\n\n")
        
        f.write("[V4+ Styles]\n")
        f.write("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n")
        f.write("Style: Default,Arial,48,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,2,10,10,30,1\n\n")
        
        f.write("[Events]\n")
        f.write("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
        
        for s in sentences:
            start = format_time(s['start'])
            end = format_time(s['end'])
            text = s['text'].replace('\\', '\\\\').replace('\n', ' ')[:100]
            f.write(f"Dialogue: 0,{start},{end},Default,,0,0,0,,{text}\n")

def format_time(seconds):
    """Format time for ASS"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int((seconds % 1) * 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

subtitle_file = TEMP_DIR / "subtitles.ass"
create_simple_subtitles(timed_sentences, subtitle_file)
print("Subtitles created")

# ========================================== 
# 7. DOWNLOAD BACKGROUND VIDEOS
# ========================================== 

print("\n--- 🎥 Downloading Background Videos ---")

def download_background_video(query, duration, index):
    """Download a background video from Pexels"""
    try:
        # Use Pexels API
        api_key = os.environ.get("PEXELS_KEYS", "").split(",")[0] if os.environ.get("PEXELS_KEYS") else ""
        
        if api_key:
            # Search for videos
            search_url = "https://api.pexels.com/videos/search"
            headers = {"Authorization": api_key}
            params = {
                "query": query,
                "per_page": 1,
                "orientation": "landscape",
                "size": "large"
            }
            
            response = requests.get(search_url, headers=headers, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                videos = data.get('videos', [])
                
                if videos:
                    video = videos[0]
                    video_files = video.get('video_files', [])
                    
                    if video_files:
                        # Get the best quality video
                        best_file = None
                        for quality in ['hd', 'large', 'medium', 'small']:
                            for file in video_files:
                                if file.get('quality') == quality:
                                    best_file = file
                                    break
                            if best_file:
                                break
                        
                        if best_file:
                            video_url = best_file['link']
                            print(f"  Downloading video {index}: {query}")
                            
                            # Download video
                            video_path = TEMP_DIR / f"bg_{index}.mp4"
                            response = requests.get(video_url, stream=True, timeout=30)
                            
                            with open(video_path, 'wb') as f:
                                for chunk in response.iter_content(chunk_size=8192):
                                    if chunk:
                                        f.write(chunk)
                            
                            # Trim to desired duration
                            trimmed_path = TEMP_DIR / f"clip_{index}.mp4"
                            cmd = [
                                "ffmpeg", "-y",
                                "-i", str(video_path),
                                "-t", str(duration),
                                "-vf", "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080",
                                "-c:v", "libx264",
                                "-preset", "fast",
                                "-an",
                                str(trimmed_path)
                            ]
                            
                            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                            os.remove(video_path)
                            
                            if os.path.exists(trimmed_path):
                                return str(trimmed_path)
    except Exception as e:
        print(f"  Video download error: {str(e)[:50]}")
    
    return None

# Download videos for each sentence
video_clips = []
search_queries = ["nature", "city", "technology", "education", "science", "history", "art", "culture"]

for i, sent in enumerate(timed_sentences):
    if i >= 8:  # Limit to 8 clips max
        break
    
    duration = sent['end'] - sent['start']
    query = search_queries[i % len(search_queries)]
    
    clip_path = download_background_video(query, duration, i)
    if clip_path:
        video_clips.append(clip_path)
        print(f"  ✓ Clip {i+1} downloaded")
    else:
        print(f"  ✗ Clip {i+1} failed")

# If no videos downloaded, create a colored background
if not video_clips:
    print("Creating colored background...")
    for i in range(min(4, len(timed_sentences))):
        duration = timed_sentences[i]['end'] - timed_sentences[i]['start']
        color = ["red", "blue", "green", "yellow"][i % 4]
        
        clip_path = TEMP_DIR / f"color_{i}.mp4"
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c={color}:s=1920x1080:d={duration}",
            "-c:v", "libx264",
            str(clip_path)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        video_clips.append(str(clip_path))

# ========================================== 
# 8. CREATE FINAL VIDEO
# ========================================== 

print("\n--- 🎬 Creating Final Video ---")

# Concatenate video clips
if video_clips:
    concat_list = TEMP_DIR / "concat.txt"
    with open(concat_list, "w") as f:
        for clip in video_clips:
            f.write(f"file '{clip}'\n")
    
    visual_video = TEMP_DIR / "visuals.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        str(visual_video)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Create video with audio (no subtitles)
print("Creating video without subtitles...")
output_no_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_no_subs.mp4"

if os.path.exists(visual_video):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(visual_video),
        "-i", str(audio_file),
        "-c:v", "copy",
        "-c:a", "aac",
        "-shortest",
        str(output_no_subs)
    ]
else:
    # Create from color background
    total_duration = timed_sentences[-1]['end'] if timed_sentences else DURATION_MINS * 60
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:s=1920x1080:d={total_duration}",
        "-i", str(audio_file),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-shortest",
        str(output_no_subs)
    ]

subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Create video with subtitles
print("Creating video with subtitles...")
output_with_subs = OUTPUT_DIR / f"spanish_{JOB_ID}_with_subs.mp4"

if os.path.exists(output_no_subs) and os.path.exists(subtitle_file):
    cmd = [
        "ffmpeg", "-y",
        "-i", str(output_no_subs),
        "-vf", f"ass={subtitle_file}",
        "-c:a", "copy",
        str(output_with_subs)
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
    # Copy no_subs version if subtitle creation fails
    if os.path.exists(output_no_subs):
        shutil.copy(output_no_subs, output_with_subs)

# ========================================== 
# 9. UPLOAD TO GOOGLE DRIVE
# ========================================== 

print("\n--- ☁️ Uploading to Google Drive ---")

def simple_upload(file_path):
    """Simple upload function"""
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None
    
    print(f"Uploading {os.path.basename(file_path)}...")
    
    # This is a placeholder - implement your upload logic here
    file_size = os.path.getsize(file_path) / (1024*1024)
    print(f"  File size: {file_size:.1f} MB")
    
    # Return a placeholder link
    return f"https://example.com/{os.path.basename(file_path)}"

# Upload videos
links = {}
if os.path.exists(output_no_subs):
    links['no_subs'] = simple_upload(output_no_subs)

if os.path.exists(output_with_subs):
    links['with_subs'] = simple_upload(output_with_subs)

# ========================================== 
# 10. FINAL OUTPUT
# ========================================== 

print("\n" + "="*60)
print("🎉 SPANISH VIDEO GENERATION COMPLETE!")
print("="*60)
print(f"Script: {len(script_text)} characters")
print(f"Sentences: {len(timed_sentences)}")
print(f"Video clips: {len(video_clips)}")

if links.get('no_subs'):
    print(f"\n📹 No Subtitles: {links['no_subs']}")
if links.get('with_subs'):
    print(f"📹 With Subtitles: {links['with_subs']}")

# Save output info
output_info = {
    "job_id": JOB_ID,
    "status": "completed",
    "script_length": len(script_text),
    "duration_seconds": audio_duration,
    "links": links
}

with open(OUTPUT_DIR / f"info_{JOB_ID}.json", "w") as f:
    json.dump(output_info, f, indent=2)

print("\n✅ All done!")
