import os
import json
import time
import random
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

# Load past questions so Gemini never repeats already-uploaded puzzles
history_file = "history.json"
past_questions = []
if os.path.exists(history_file):
    try:
        with open(history_file, "r", encoding="utf-8") as f:
            past_questions = json.load(f)
    except Exception:
        past_questions = []

recent_avoid = "\n".join(f"- {q}" for q in past_questions[-80:]) or "- None yet"

# 1. Generate a Non-Repeating 5-Round Mental-Math Quiz via Gemini
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=120000),
)

round_pools = [
    "Missing operator or missing number gap puzzle using +, -, ×, ÷, =, and [ ? ] (e.g., 14 + [ ? ] × 3 = 35)",
    "Percentage swap shortcut using = and [ ? ] (e.g., 36% of 25 + 24% of 25 = [ ? ])",
    "Fast multiplication near 100 using × and = (e.g., 98 × 96 = [ ? ])",
    "Order-of-operations trap using ÷, ×, +, -, and ( ) (e.g., 48 ÷ 2 × (9 + 3) = [ ? ])",
    "Difference of two squares shortcut using - and = (e.g., 84² - 16² = [ ? ])",
    "Clever fraction or division cancellation using ÷, ×, and [ ? ] (e.g., (72 × 35) ÷ 18 = [ ? ])",
    "Two-step missing gap equation (e.g., ( [ ? ] ÷ 4 ) + 19 = 31)",
    "Squaring numbers ending in 5 or near 50 using × and = (e.g., 65 × 65 = [ ? ])",
    "Sequence pattern gap puzzle using +, ×, and [ ? ] (e.g., 3, 7, 15, 31, [ ? ])",
]

selected_topics = random.sample(round_pools, 5)
seed_number = random.randint(1000, 9999)

prompt = f"""Create a 5-round escalating mental-math challenge video for 'Math Minute Marvels' (Seed #{seed_number}).
Rules:
- All 5 riddles must be solvable in your head without pen or paper, each featuring a clever mental shortcut.
- All 5 rounds MUST be completely different from each other. Assign these exact 5 distinct topics:
  * Round 1 ("WARM-UP", timer_seconds: 15): {selected_topics}
  * Round 2 ("EASY", timer_seconds: 20): {selected_topics}
  * Round 3 ("MEDIUM", timer_seconds: 30): {selected_topics}
  * Round 4 ("HARD", timer_seconds: 45): {selected_topics}
  * Round 5 ("GENIUS BOSS", timer_seconds: 60): {selected_topics}
- CRITICAL UNIQUENESS RULE: Do NOT repeat, rephrase, or use the same numbers/structure as ANY of these already-uploaded questions:
{recent_avoid}
- Visual Formatting Rule for "question_text" and "steps_text":
  * Use clean math symbols (+, -, ×, ÷, =) and use "[ ? ]" as the blank gap for the unknown value!
  * Keep "question_text" crisp and visual (e.g., "Solve for [ ? ]:\\n\\n( 64 × 25 ) ÷ 16 = [ ? ]").
  * Keep "steps_text" to strictly 3 short equation lines using +, -, ×, ÷, = (max 8 words per line).
Return JSON with these exact keys:
- "video_title": catchy YouTube title under 65 chars (NO #Shorts tag)
- "thumb_hook": punchy 3-word thumbnail hook in ALL CAPS (e.g., "99% FAIL LEVEL 5")
- "thumb_math": short 1-line equation teaser with [ ? ] (e.g., "64 × 25 ÷ 16 = [ ? ]")
- "intro_hook": bold on-screen hook (e.g., "CAN YOU SOLVE ALL 5?")
- "intro_voice": energetic 7-second welcome explaining the 5 escalating rounds
- "rounds": list of 5 objects, each with:
  * "round_num": 1 to 5
  * "difficulty": "WARM-UP", "EASY", "MEDIUM", "HARD", or "GENIUS BOSS"
  * "timer_seconds": 15, 20, 30, 45, or 60
  * "question_text": clean visual equation with +, -, ×, ÷, = and [ ? ]
  * "question_voice": natural spoken reading of the round number, equation, and timer seconds
  * "steps_text": 3 short equation lines showing the mental shortcut
  * "answer_text": bold final answer (e.g., "[ ? ] = 100")
  * "solution_voice": spoken reveal of the mental trick and final answer
- "outro_voice": energetic 7-second outro asking viewers to comment their score out of 5 and subscribe"""

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
    clean_text = str(text).replace("[ ? ]", "what number").replace("[?]", "what number").strip()
    if not clean_text:
        clean_text = "Let's solve this math challenge!"
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

def format_math_symbols(text):
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    return str(text).replace(" * ", " × ").replace(" / ", " ÷ ")

# 3. Build Widescreen 1920x1080 Base Slides (Centered Inside Glowing Cyan Box)
def build_wide_slide(output_name, badge, main_text, highlight="", badge_color="#FFD700", compact=False):
    main_text = format_math_symbols(main_text)
    highlight = format_math_symbols(highlight)

    img = Image.open("bg_long.png").convert("RGBA").resize((1920, 1080))
    draw = ImageDraw.Draw(img)

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    body_size = 44 if compact else 56
    wrap_width = 32 if compact else 26
    line_spacing = 18 if compact else 24

    if os.path.exists(font_path):
        font_badge = ImageFont.truetype(font_path, 42)
        font_body = ImageFont.truetype(font_path, body_size)
        font_hl = ImageFont.truetype(font_path, 64)
    else:
        font_badge = ImageFont.load_default(size=42)
        font_body = ImageFont.load_default(size=body_size)
        font_hl = ImageFont.load_default(size=64)

    board_center_x = 615
    draw.multiline_text((board_center_x, 305), badge, fill=badge_color, font=font_badge, anchor="ma", align="center", spacing=10)

    wrapped_body = "\n".join([textwrap.fill(line, width=wrap_width) for line in main_text.split("\n")])
    draw.multiline_text((board_center_x, 420), wrapped_body, fill="white", font=font_body, anchor="ma", align="center", spacing=line_spacing)

    if highlight:
        wrapped_hl = "\n".join(textwrap.wrap(highlight, width=24))
        draw.multiline_text((board_center_x, 795), wrapped_hl, fill="#00FFCC", font=font_hl, anchor="ma", align="center", spacing=12)

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

valid_timers =

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

# Helper: Generate High-CTR Widescreen Thumbnail (1280x720)
def create_long_thumbnail(hook_line, teaser_math, output_path="thumb_long.jpg"):
    thumb = Image.open("bg_long.png").convert("RGBA").resize((1280, 720))
    draw = ImageDraw.Draw(thumb)

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    font_top = ImageFont.truetype(font_path, 56) if os.path.exists(font_path) else ImageFont.load_default(size=56)
    font_big = ImageFont.truetype(font_path, 78) if os.path.exists(font_path) else ImageFont.load_default(size=78)

    mascot_img = Image.open("char_think.png" if os.path.exists("char_think.png") else "char_closed.png").convert("RGBA").resize((430, 430))
    thumb.paste(mascot_img, (830, 260), mascot_img)

    wrapped_hook = "\n".join(textwrap.wrap(str(hook_line).upper(), width=15))
    draw.multiline_text((410, 220), wrapped_hook, fill="#FFD700", font=font_top, anchor="ma", align="center", spacing=10)

    clean_teaser = format_math_symbols(teaser_math)
    wrapped_math = "\n".join(textwrap.wrap(clean_teaser, width=15))
    draw.multiline_text((410, 415), wrapped_math, fill="#00FFCC", font=font_big, anchor="ma", align="center", spacing=12)

    thumb.convert("RGB").save(output_path, "JPEG", quality=92)
    return output_path

# 7. Upload Publicly to YouTube, Attach Thumbnail & Save Question History
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
video_response = request.execute()
video_id = video_response["id"]
print(f"Uploaded PUBLIC 5-Round Long Video ({video_id})!", flush=True)

try:
    t_hook = data.get("thumb_hook", "99% FAIL LEVEL 5")
    t_math = data.get("thumb_math", "CAN YOU SOLVE ALL 5?")
    thumb_file = create_long_thumbnail(t_hook, t_math)
    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(thumb_file, mimetype="image/jpeg")
    ).execute()
    print(f"Uploaded custom Long Video thumbnail for {video_id}!", flush=True)
except Exception as e:
    print(f"Long Video thumbnail skipped: {e}", flush=True)

for rnd in data["rounds"]:
    past_questions.append(str(rnd["question_text"]).strip())
with open(history_file, "w", encoding="utf-8") as f:
    json.dump(past_questions[-150:], f, indent=2)
