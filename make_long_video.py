import os
import json
import time
import asyncio
import textwrap
import numpy as np
from google import genai
from google.genai import types
import edge_tts
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
    ImageClip,
    AudioFileClip,
    VideoClip,
    CompositeAudioClip,
    concatenate_videoclips,
)
from moviepy.audio.AudioClip import AudioArrayClip
from moviepy.audio.fx import AudioLoop, MultiplyVolume
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# 1. Generate a 5-Round Escalating Mental-Math Quiz via Gemini
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=120000),
)

prompt = """Create a 5-round escalating mental-math challenge video for 'Math Minute Marvels'.
Rules:
- All 5 riddles must be solvable in your head without pen or paper, each featuring a clever twist or trick.
- Escalate the difficulty across 5 rounds with these exact timer_seconds:
  * Round 1: "WARM-UP" -> timer_seconds: 15
  * Round 2: "EASY" -> timer_seconds: 20
  * Round 3: "MEDIUM" -> timer_seconds: 30
  * Round 4: "HARD" -> timer_seconds: 45
  * Round 5: "GENIUS BOSS" -> timer_seconds: 60
- Keep each "steps_text" to strictly 3 short, punchy lines (max 9 words per line).
Return JSON with these exact keys:
- "video_title": catchy YouTube title under 65 chars (NO #Shorts tag)
- "intro_hook": bold on-screen hook (e.g., "CAN YOU BEAT ALL 5 LEVELS?")
- "intro_voice": energetic 8-second welcome explaining the 5 escalating rounds
- "rounds": list of 5 objects, each with:
  * "round_num": 1 to 5
  * "difficulty": "WARM-UP", "EASY", "MEDIUM", "HARD", or "GENIUS BOSS"
  * "timer_seconds": 15, 20, 30, 45, or 60
  * "question_text": clean math riddle text for screen
  * "question_voice": spoken reading of the round number, riddle, and timer duration
  * "steps_text": 3 short lines explaining the step-by-step trick
  * "answer_text": bold final answer (e.g., "Answer: 42")
  * "solution_voice": spoken reveal of the trick and final answer
- "outro_voice": energetic 8-second outro asking viewers to comment their score out of 5 and subscribe"""

models_to_try = ["gemini-3.8-flash", "gemini-3.5-flash-lite"]
response = None

for model_name in models_to_try:
    for attempt in range(3):
        try:
            print(f"Trying {model_name} (attempt {attempt + 1})...", flush=True)
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            if response and response.text:
                print(f"Success with {model_name}!", flush=True)
                break
        except Exception as e:
            print(f"{model_name} failed ({type(e).__name__}): {e}. Retrying in 5s...", flush=True)
            time.sleep(5)
    if response and response.text:
        break

if not response or not response.text:
    raise RuntimeError("All Gemini models failed or timed out.")

data = json.loads(response.text)
# 2. Sequential Voiceover Generator with gTTS Fallback
async def create_audio(text, filename):
    clean_text = str(text).strip() or "Let's solve this math challenge!"
    for voice in ["en-US-GuyNeural", "en-US-ChristopherNeural", "en-US-JennyNeural"]:
        try:
            communicate = edge_tts.Communicate(clean_text, voice)
            await communicate.save(filename)
            if os.path.exists(filename) and os.path.getsize(filename) > 0:
                print(f"Generated {filename} with {voice}", flush=True)
                return
        except Exception as e:
            print(f"Edge-TTS ({voice}) failed: {e}", flush=True)
            await asyncio.sleep(1.5)

    print(f"Using gTTS fallback for {filename}...", flush=True)
    tts = gTTS(text=clean_text, lang="en", slow=False)
    tts.save(filename)

async def generate_all_audio():
    await create_audio(data["intro_voice"], "long_intro.mp3")
    await asyncio.sleep(1)
    for idx, rnd in enumerate(data["rounds"], start=1):
        await create_audio(rnd["question_voice"], f"r{idx}_q.mp3")
        await asyncio.sleep(1)
        await create_audio(rnd["solution_voice"], f"r{idx}_sol.mp3")
        await asyncio.sleep(1)
    await create_audio(data["outro_voice"], "long_outro.mp3")

asyncio.run(generate_all_audio())

# Helper: 1.0s Clock Tick
def get_tick_clip():
    if os.path.exists("tick.mp3"):
        tick = AudioFileClip("tick.mp3")
        return tick.subclipped(0, min(1.0, tick.duration))
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    envelope = np.exp(-t * 45)
    wave = 0.35 * envelope * (np.sin(2 * np.pi * 800 * t) + 0.5 * np.sin(2 * np.pi * 1600 * t))
    return AudioArrayClip(np.column_stack((wave, wave)), fps=sr)

tick_sound = get_tick_clip()

# 3. Build Widescreen 1920x1080 Base Slides
def build_wide_slide(output_name, badge, main_text, highlight="", badge_color="#FFD700", compact=False):
    if isinstance(main_text, list):
        main_text = "\n".join(str(item) for item in main_text)
    if isinstance(highlight, list):
        highlight = "\n".join(str(item) for item in highlight)
    main_text = str(main_text)
    highlight = str(highlight)

    img = Image.open("bg_long.png").convert("RGBA").resize((1920, 1080))
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    body_size = 44 if compact else 54
    wrap_width = 36 if compact else 30
    line_spacing = 16 if compact else 22

    if os.path.exists(font_path):
        font_badge = ImageFont.truetype(font_path, 46)
        font_body = ImageFont.truetype(font_path, body_size)
        font_hl = ImageFont.truetype(font_path, 64)
    else:
        font_badge = ImageFont.load_default(size=46)
        font_body = ImageFont.load_default(size=body_size)
        font_hl = ImageFont.load_default(size=64)

    board_center_x = 720
    draw.multiline_text((board_center_x, 220), badge, fill=badge_color, font=font_badge, anchor="ma", align="center", spacing=10)

    wrapped_body = "\n".join([textwrap.fill(line, width=wrap_width) for line in main_text.split("\n")])
    draw.multiline_text((board_center_x, 340), wrapped_body, fill="white", font=font_body, anchor="ma", align="center", spacing=line_spacing)

    if highlight:
        wrapped_hl = "\n".join(textwrap.wrap(highlight, width=26))
        draw.multiline_text((board_center_x, 790), wrapped_hl, fill="#00FFCC", font=font_hl, anchor="ma", align="center", spacing=12)

    img.save(output_name)
# 4. Widescreen Lip-Sync Mascot Overlay (Bottom-Right Corner)
char_closed = Image.open("char_closed.png").convert("RGBA").resize((480, 480))
char_open = Image.open("char_open.png").convert("RGBA").resize((480, 480))
char_think = Image.open("char_think.png").convert("RGBA").resize((480, 480)) if os.path.exists("char_think.png") else char_closed

def make_talking_scene(slide_path, audio_path):
    base_img = Image.open(slide_path).convert("RGBA")
    audio = AudioFileClip(audio_path)
    fps = 24

    samples = audio.to_soundarray(fps=22050)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.abs(samples)
    threshold = np.percentile(samples, 55) if len(samples) > 0 else 0.02

    def frame_func(t):
        idx = int(t * 22050)
        chunk = samples[idx : idx + 900]
        is_talking = (np.mean(chunk) > threshold) if len(chunk) > 0 else False

        frame = base_img.copy()
        sprite = char_open if is_talking else char_closed
        y_offset = 530 if is_talking else 540
        frame.paste(sprite, (1360, y_offset), sprite)
        return np.array(frame.convert("RGB"))

    return VideoClip(frame_func, duration=audio.duration).with_audio(audio).with_fps(fps)

def make_thinking_scene(slide_path, duration=1.0):
    base_img = Image.open(slide_path).convert("RGBA")
    base_img.paste(char_think, (1360, 540), char_think)
    temp_path = f"think_{slide_path}"
    base_img.convert("RGB").save(temp_path)
    return ImageClip(temp_path).with_duration(duration).with_audio(tick_sound)

# 5. Assemble Full 5-Round Video + Track Chapter Timestamps
clips = []
chapters = ["0:00 Intro & Rules"]
elapsed = 0.0

build_wide_slide("l_intro.png", "5-ROUND MENTAL MATH GAUNTLET", data["intro_hook"], "CAN YOU SCORE 5/5?", "#FFD700")
intro_clip = make_talking_scene("l_intro.png", "long_intro.mp3")
clips.append(intro_clip)
elapsed += intro_clip.duration

valid_timers = [15, 20, 30, 45, 60]

for idx, rnd in enumerate(data["rounds"], start=1):
    mins = int(elapsed // 60)
    secs = int(elapsed % 60)
    diff = rnd.get("difficulty", f"ROUND {idx}")
    chapters.append(f"{mins}:{secs:02d} Round {idx}: {diff}")

    t_sec = int(rnd.get("timer_seconds", valid_timers[idx - 1]))
    if t_sec not in valid_timers:
        t_sec = valid_timers[idx - 1]

    q_slide = f"l_r{idx}_q.png"
    build_wide_slide(q_slide, f"ROUND {idx} OF 5 • {diff}", rnd["question_text"], "", "#38BDF8")
    q_clip = make_talking_scene(q_slide, f"r{idx}_q.mp3")
    clips.append(q_clip)
    elapsed += q_clip.duration

    for sec in range(t_sec, 0, -1):
        t_slide = f"l_r{idx}_t{sec}.png"
        t_color = "#FF4D4D" if sec <= 5 else "#FF9F43"
        build_wide_slide(t_slide, f"ROUND {idx} • {diff}", rnd["question_text"], f"TIME LEFT: {sec}s", t_color)
        t_clip = make_thinking_scene(t_slide, 1.0)
        clips.append(t_clip)
        elapsed += 1.0

    sol_slide = f"l_r{idx}_sol.png"
    build_wide_slide(sol_slide, f"ROUND {idx} SOLUTION", rnd["steps_text"], rnd["answer_text"], "#4ADE80", compact=True)
    sol_clip = make_talking_scene(sol_slide, f"r{idx}_sol.mp3")
    clips.append(sol_clip)
    elapsed += sol_clip.duration

build_wide_slide("l_outro.png", "CHALLENGE COMPLETE!", "What was your score out of 5?\nComment below & subscribe for weekly quizzes!", "MATH MINUTE MARVELS", "#F472B6")
outro_clip = make_talking_scene("l_outro.png", "long_outro.mp3")
clips.append(outro_clip)
# 6. Render Widescreen Video Fast (preset="ultrafast")
final_video = concatenate_videoclips(clips, method="compose")

if os.path.exists("bgm.mp3"):
    bgm = AudioFileClip("bgm.mp3")
    bgm_looped = bgm.with_effects([
        AudioLoop(duration=final_video.duration),
        MultiplyVolume(0.10),
    ])
    final_audio = CompositeAudioClip([final_video.audio, bgm_looped])
    final_video = final_video.with_audio(final_audio)

final_video.write_videofile("long_video.mp4", fps=24, codec="libx264", audio_codec="aac", preset="ultrafast", threads=4)

# 7. Upload Publicly to YouTube with Auto-Generated Chapters
chapter_text = "\n".join(chapters)
description = (
    f"Test your brainpower with 5 escalating mental math riddles on Math Minute Marvels!\n\n"
    f"Chapters:\n{chapter_text}\n\n"
    f"How many did you solve before the timer ran out? Drop your score (0/5 to 5/5) in the comments!\n"
    f"#mentalmath #mathriddles #brainteasers #mathquiz"
)

creds_data = json.loads(os.environ["YOUTUBE_TOKEN_JSON"])
creds = Credentials.from_authorized_user_info(creds_data)
if creds.expired and creds.refresh_token:
    creds.refresh(Request())

youtube = build("youtube", "v3", credentials=creds)
request = youtube.videos().insert(
    part="snippet,status",
    body={
        "snippet": {
            "title": data["video_title"],
            "description": description,
            "categoryId": "27",
        },
        "status": {"privacyStatus": "public"},
    },
    media_body=MediaFileUpload("long_video.mp4"),
)
request.execute()
print("Uploaded PUBLIC 5-Round Long Video successfully!", flush=True)
