import os
import json
import asyncio
import textwrap
import numpy as np
from google import genai
from google.genai import types
import edge_tts
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

# 1. Generate Mental-Math Riddle via Gemini 3.8 Flash (Strict JSON Mode)
import time

client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=90000),  # 90-second wait so 3.8-flash finishes cleanly
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
    for attempt in range(2):
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
            print(f"{model_name} failed ({type(e).__name__}): {e}. Retrying in 3s...", flush=True)
            time.sleep(3)
    if response and response.text:
        break

if not response or not response.text:
    raise RuntimeError("All Gemini models failed or timed out.")

data = json.loads(response.text)


timer_seconds = int(data.get("timer_seconds", 30))
if timer_seconds not in (30, 45, 60):
    timer_seconds = 30

# 2. Generate Voiceovers via Edge-TTS 7.3.1
from gtts import gTTS

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

    # Fallback to Google TTS if Microsoft blocks the GitHub server IP
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

    if os.path.exists("font.ttf"):
        font_badge = ImageFont.truetype("font.ttf", 64)
        font_body = ImageFont.truetype("font.ttf", 60)
        font_hl = ImageFont.truetype("font.ttf", 80)
    else:
        font_badge = ImageFont.load_default(size=58)
        font_body = ImageFont.load_default(size=50)
        font_hl = ImageFont.load_default(size=68)

    wrapped_badge = "\n".join(textwrap.wrap(badge, width=24))
    draw.multiline_text((W / 2, 330), wrapped_badge, fill=badge_color, font=font_badge, anchor="ma", align="center", spacing=12)

    wrapped_body = "\n".join([textwrap.fill(line, width=25) for line in main_text.split("\n")])
    draw.multiline_text((W / 2, 600), wrapped_body, fill="white", font=font_body, anchor="ma", align="center", spacing=22)

    if highlight:
        wrapped_hl = "\n".join(textwrap.wrap(highlight, width=20))
        draw.multiline_text((W / 2, 1140), wrapped_hl, fill="#00FFCC", font=font_hl, anchor="ma", align="center", spacing=12)

    img.save(output_name)

# 4. Lip-Sync Character Overlay Engine (MoviePy 2.2.1)
char_closed = Image.open("char_closed.png").convert("RGBA").resize((460, 460))
char_open = Image.open("char_open.png").convert("RGBA").resize((460, 460))
char_think = Image.open("char_think.png").convert("RGBA").resize((460, 460)) if os.path.exists("char_think.png") else char_closed

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
        y_offset = 1390 if is_talking else 1400
        frame.paste(sprite, (580, y_offset), sprite)
        return np.array(frame.convert("RGB"))

    return VideoClip(frame_func, duration=audio.duration).with_audio(audio).with_fps(fps)

def make_thinking_scene(slide_path, duration=1.0):
    base_img = Image.open(slide_path).convert("RGBA")
    base_img.paste(char_think, (580, 1400), char_think)
    temp_path = f"think_{slide_path}"
    base_img.convert("RGB").save(temp_path)
    return ImageClip(temp_path).with_duration(duration).with_audio(tick_sound)

# 5. Assemble All 5 Scenes
clips = []

build_base_slide("s_intro.png", f"MATH MINUTE MARVELS • {data['difficulty']}", data["intro_text"], f"{timer_seconds}-SECOND CHALLENGE!", "#FFD700")
clips.append(make_talking_scene("s_intro.png", "intro.mp3"))

build_base_slide("s_q.png", "SOLVE IN YOUR HEAD", data["question_text"], "", "#38BDF8")
clips.append(make_talking_scene("s_q.png", "question.mp3"))

for sec in range(timer_seconds, 0, -1):
    t_slide = f"s_timer_{sec}.png"
    timer_color = "#FF4D4D" if sec <= 5 else "#FF9F43"
    build_base_slide(t_slide, f"DIFFICULTY: {data['difficulty']}", data["question_text"], f"⏱️ {sec}s", timer_color)
    clips.append(make_thinking_scene(t_slide, 1.0))

build_base_slide("s_sol.png", "STEP-BY-STEP SOLUTION", data["steps_text"], data["answer_text"], "#4ADE80")
clips.append(make_talking_scene("s_sol.png", "solution.mp3"))

build_base_slide("s_outro.png", "BEAT THE CLOCK?", "Comment how many seconds it took!\nSubscribe for daily brainteasers!", "MATH MINUTE MARVELS", "#F472B6")
clips.append(make_talking_scene("s_outro.png", "outro.mp3"))

# 6. Combine Video & Layer Looped Background Music (MoviePy 2.2.1 Effects)
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

# 7. Upload to YouTube Automatically
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
        "status": {"privacyStatus": "private"},
    },
    media_body=MediaFileUpload("short.mp4"),
)
request.execute()
print(f"Uploaded {timer_seconds}s {data['difficulty']} Short successfully!")
