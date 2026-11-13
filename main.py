import os
import json
import time
import math
import asyncio
import textwrap
import numpy as np
from google import genai
from google.genai import types
import edge_tts
from gtts import gTTS
from PIL import Image, ImageDraw, ImageFont
from moviepy import (
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

# 1. Generate Twisty Mental-Math Riddle via Gemini (with Automatic Fallback)
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=90000),
)

prompt = """Create a twisty mental-math puzzle for 'Math Minute Marvels'.
Rules:
- Solvable in your head without pen/paper, with a clever trick or twist.
- Choose timer_seconds strictly from 30, 45, or 60 based on difficulty:
  * 30 = EASY (1-2 quick mental steps)
  * 45 = MEDIUM (2-3 mental steps)
  * 60 = HARD (3-4 mental steps)
Return JSON with these exact keys:
- "difficulty": "EASY", "MEDIUM", or "HARD"
- "timer_seconds": 30, 45, or 60
- "title": catchy YouTube title under 55 chars
- "intro_text": bold 4-word hook on screen
- "intro_voice": energetic 3-second spoken challenge to stop scrolling
- "question_text": clean, short text of the math riddle for screen
- "question_voice": spoken reading of the riddle + telling them the exact seconds on the clock
- "steps_text": 3-4 short lines explaining the step-by-step solution clearly
- "answer_text": bold final answer
- "solution_voice": spoken step-by-step reveal of the trick and final answer
- "outro_voice": fast 3-second outro asking them to comment their time and subscribe"""

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

timer_seconds = int(data.get("timer_seconds", 30))
if timer_seconds not in (30, 45, 60):
    timer_seconds = 30

# 2. Generate Voiceovers Sequentially (with Automatic gTTS Backup)
async def create_audio(text, filename):
    clean_text = str(text).strip() or "Let's solve this math puzzle!"
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
    await create_audio(data["intro_voice"], "intro.mp3")
    await asyncio.sleep(1)
    await create_audio(data["question_voice"], "question.mp3")
    await asyncio.sleep(1)
    await create_audio(data["solution_voice"], "solution.mp3")
    await asyncio.sleep(1)
    await create_audio(data["outro_voice"], "outro.mp3")

asyncio.run(generate_all_audio())

# Helper: Load Ticking Sound or Synthesize a 1.0s Clock Tick
def get_tick_clip():
    if os.path.exists("tick.mp3"):
        tick = AudioFileClip("tick.mp3")
        return tick.subclipped(0, min(1.0, tick.duration))
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    envelope = np.exp(-t * 45)
    wave = 0.35 * envelope * (np.sin(2 * np.pi * 800 * t) + 0.5 * np.sin(2 * np.pi * 1600 * t))
    stereo_wave = np.column_stack((wave, wave))
    return AudioArrayClip(stereo_wave, fps=sr)

tick_sound = get_tick_clip()

# 3. Build Clean Base Slides (1080x1920)
def build_base_slide(output_name, badge, main_text, highlight="", badge_color="#FFD700"):
    if isinstance(main_text, list):
        main_text = "\n".join(str(item) for item in main_text)
    if isinstance(highlight, list):
        highlight = "\n".join(str(item) for item in highlight)
    main_text = str(main_text)
    highlight = str(highlight)

    img = Image.open("bg.png").convert("RGBA").resize((1080, 1920))
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    # Slightly tighter body font when showing multi-line steps
    is_long_text = len(main_text) > 85 or main_text.count("\n") >= 3
    body_size = 54 if is_long_text else 64

    if os.path.exists(font_path):
        font_badge = ImageFont.truetype(font_path, 48)
        font_body = ImageFont.truetype(font_path, body_size)
        font_hl = ImageFont.truetype(font_path, 82)
    else:
        font_badge = ImageFont.load_default(size=48)
        font_body = ImageFont.load_default(size=body_size)
        font_hl = ImageFont.load_default(size=82)

    # 1. Place badge cleanly below the "MARVELS" logo text
    wrapped_badge = "\n".join(textwrap.wrap(badge, width=28))
    draw.multiline_text((W / 2, 485), wrapped_badge, fill=badge_color, font=font_badge, anchor="ma", align="center", spacing=10)

    # 2. Place main question or steps below the badge
    wrap_width = 26 if is_long_text else 22
    line_spacing = 18 if is_long_text else 24
    wrapped_body = "\n".join([textwrap.fill(line, width=wrap_width) for line in main_text.split("\n")])
    body_y = 630
    draw.multiline_text((W / 2, body_y), wrapped_body, fill="white", font=font_body, anchor="ma", align="center", spacing=line_spacing)

    # 3. Dynamically measure body height so highlight/answer NEVER overlaps
    if highlight:
        bbox = draw.multiline_textbbox((W / 2, body_y), wrapped_body, font=font_body, anchor="ma", align="center", spacing=line_spacing)
        dynamic_hl_y = max(1160, bbox[3] + 55)
        wrapped_hl = "\n".join(textwrap.wrap(highlight, width=20))
        draw.multiline_text((W / 2, dynamic_hl_y), wrapped_hl, fill="#00FFCC", font=font_hl, anchor="ma", align="center", spacing=14)

    img.save(output_name)


# 4. 5-Pose Smooth Lip-Sync, Blinking & Floating Character Engine
def load_pose(filename, fallback_img=None):
    if os.path.exists(filename):
        return Image.open(filename).convert("RGBA").resize((460, 460), Image.Resampling.LANCZOS)
    return fallback_img

char_closed = load_pose("char_closed.png")
if char_closed is None:
    raise FileNotFoundError("Missing char_closed.png in the repository root!")

char_open = load_pose("char_open.png", char_closed)
char_half = load_pose("char_half.png", char_open)
char_think = load_pose("char_think.png", char_closed)
char_blink = load_pose("char_blink.png", char_closed)

def is_blinking(t):
    cycle = t % 3.4
    return 3.15 <= cycle <= 3.30

def apply_squash_stretch(sprite, stretch_factor):
    if abs(stretch_factor - 1.0) < 0.002:
        return sprite, 0, 0
    w, h = sprite.size
    new_w = int(w / math.sqrt(stretch_factor))
    new_h = int(h * stretch_factor)
    resized = sprite.resize((new_w, new_h), Image.Resampling.BILINEAR)
    dx = (w - new_w) // 2
    dy = h - new_h
    return resized, dx, dy

def make_talking_scene(slide_path, audio_path, time_offset=0.0):
    base_img = Image.open(slide_path).convert("RGBA")
    audio = AudioFileClip(audio_path)
    fps = 24

    samples = audio.to_soundarray(fps=22050)
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    samples = np.abs(samples)
    window = np.ones(1900) / 1900
    smoothed = np.convolve(samples, window, mode="same") if len(samples) > 0 else samples

    active = smoothed[smoothed > 0.005]
    if len(active) > 0:
        low_thresh = float(np.percentile(active, 25))
        high_thresh = float(np.percentile(active, 62))
    else:
        low_thresh, high_thresh = 0.012, 0.030

    def frame_func(t):
        global_t = t + time_offset
        idx = min(int(t * 22050), max(0, len(smoothed) - 1))
        vol = float(smoothed[idx]) if len(smoothed) > 0 else 0.0

        if vol >= high_thresh:
            sprite = char_open
            stretch = 1.025
        elif vol >= low_thresh:
            sprite = char_half
            stretch = 1.012
        else:
            sprite = char_blink if is_blinking(global_t) else char_closed
            stretch = 1.0

        bob_y = math.sin(global_t * 4.2) * 8.0
        styled_sprite, dx, dy = apply_squash_stretch(sprite, stretch)
        x_pos = int(520 + dx)
        y_pos = int(1378 + bob_y + dy)

        frame = base_img.copy()
        frame.paste(styled_sprite, (x_pos, y_pos), styled_sprite)
        return np.array(frame.convert("RGB"))

    return VideoClip(frame_func, duration=audio.duration).with_audio(audio).with_fps(fps)

def make_thinking_scene(slide_path, duration=1.0, time_offset=0.0):
    base_img = Image.open(slide_path).convert("RGBA")
    fps = 24

    def frame_func(t):
        global_t = t + time_offset
        bob_y = math.sin(global_t * 3.4) * 10.0
        breath = 1.0 + 0.01 * math.sin(global_t * 3.4)
        styled_sprite, dx, dy = apply_squash_stretch(char_think, breath)

        x_pos = int(520 + dx)
        y_pos = int(1380 + bob_y + dy)

        frame = base_img.copy()
        frame.paste(styled_sprite, (x_pos, y_pos), styled_sprite)
        return np.array(frame.convert("RGB"))

    return VideoClip(frame_func, duration=duration).with_audio(tick_sound).with_fps(fps)

# 5. Assemble All 5 Scenes with Continuous Timeline Offset
clips = []
current_time = 0.0

build_base_slide("s_intro.png", f"MATH MINUTE MARVELS • {data['difficulty']}", data["intro_text"], f"{timer_seconds}-SECOND CHALLENGE!", "#FFD700")
c_intro = make_talking_scene("s_intro.png", "intro.mp3", current_time)
clips.append(c_intro)
current_time += c_intro.duration

build_base_slide("s_q.png", "SOLVE IN YOUR HEAD", data["question_text"], "", "#38BDF8")
c_q = make_talking_scene("s_q.png", "question.mp3", current_time)
clips.append(c_q)
current_time += c_q.duration

for sec in range(timer_seconds, 0, -1):
    t_slide = f"s_timer_{sec}.png"
    timer_color = "#FF4D4D" if sec <= 5 else "#FF9F43"
    build_base_slide(t_slide, f"DIFFICULTY: {data['difficulty']}", data["question_text"], f"TIME LEFT: {sec}s", timer_color)
    c_tick = make_thinking_scene(t_slide, 1.0, current_time)
    clips.append(c_tick)
    current_time += 1.0

build_base_slide("s_sol.png", "STEP-BY-STEP SOLUTION", data["steps_text"], data["answer_text"], "#4ADE80")
c_sol = make_talking_scene("s_sol.png", "solution.mp3", current_time)
clips.append(c_sol)
current_time += c_sol.duration

build_base_slide("s_outro.png", "BEAT THE CLOCK?", "Comment how many seconds it took!\nSubscribe for daily brainteasers!", "MATH MINUTE MARVELS", "#F472B6")
c_outro = make_talking_scene("s_outro.png", "outro.mp3", current_time)
clips.append(c_outro)

# 6. Combine Video & Layer Looped Background Music
final_video = concatenate_videoclips(clips, method="compose")

if os.path.exists("bgm.mp3"):
    bgm = AudioFileClip("bgm.mp3")
    bgm_looped = bgm.with_effects([
        AudioLoop(duration=final_video.duration),
        MultiplyVolume(0.12),
    ])
    final_audio = CompositeAudioClip([final_video.audio, bgm_looped])
    final_video = final_video.with_audio(final_audio)

final_video.write_videofile("short.mp4", fps=24, codec="libx264", audio_codec="aac")

# 7. Upload Publicly to YouTube Automatically
creds_data = json.loads(os.environ["YOUTUBE_TOKEN_JSON"])
creds = Credentials.from_authorized_user_info(creds_data)
if creds.expired and creds.refresh_token:
    creds.refresh(Request())

youtube = build("youtube", "v3", credentials=creds)
request = youtube.videos().insert(
    part="snippet,status",
    body={
        "snippet": {
            "title": f"{data['title']} #Shorts",
            "description": f"{data['question_text']}\n\nDifficulty: {data['difficulty']} ({timer_seconds}s timer). Did you beat the clock?\n#mathriddles #mentalmath #shorts",
            "categoryId": "27",
        },
        "status": {"privacyStatus": "public"},
    },
    media_body=MediaFileUpload("short.mp4"),
)
request.execute()
print(f"Uploaded PUBLIC {timer_seconds}s {data['difficulty']} Short successfully!", flush=True)
