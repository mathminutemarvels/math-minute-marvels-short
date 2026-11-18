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

# 1. Generate a Fresh, Non-Repeating Mental-Math Riddle via Gemini
client = genai.Client(
    api_key=os.environ["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=90000),
)

topics = [
    "Missing operator or missing number gap puzzle using +, -, ×, ÷, =, and [ ? ]",
    "Percentage swap shortcut using = and [ ? ] (e.g., 36% of 25 + 24% of 25 = [ ? ])",
    "Multiplying two numbers close to 100 using × and = (e.g., 97 × 94 = [ ? ])",
    "Squaring a 2-digit number ending in 5 or close to 50 using × and =",
    "Fast mental division or multiplication by 5, 25, or 125 using ÷, ×, and [ ? ]",
    "Order-of-operations viral trap using ÷, ×, +, -, and ( )",
    "Difference of two squares shortcut using - and = (e.g., 84² - 16² = [ ? ])",
    "Sequence pattern gap puzzle using +, ×, and [ ? ] (e.g., 3, 7, 15, 31, [ ? ])",
]

title_styles = [
    "Start title with 'Only 1% Can Solve:'",
    "Start title with 'Brain Age Test:'",
    "Start title with 'Harvard Interview Math:'",
    "Start title with 'Stop Using a Calculator:'",
    "Start title with 'Beat The Clock:'",
    "Start title with 'Genius Math Shortcut:'",
]

chosen_topic = random.choice(topics)
chosen_style = random.choice(title_styles)
seed_number = random.randint(1000, 9999)

prompt = f"""Create a twisty mental-math puzzle for 'Math Minute Marvels' (Seed #{seed_number}).
Rules:
- Solvable in your head without pen/paper, with a clever trick or twist.
- Focus specifically on this topic: {chosen_topic}
- Title rule: {chosen_style} (keep under 55 chars, NO #Shorts tag in the JSON title).
- Use clean math symbols (+, -, ×, ÷, =) and use "[ ? ]" as the blank gap for the unknown value.
- CRITICAL UNIQUENESS RULE: Do NOT repeat, rephrase, or use the same numbers/structure as ANY of these already-uploaded questions:
{recent_avoid}
- Choose timer_seconds strictly from 30, 45, or 60 based on difficulty:
  * 30 = EASY (1-2 quick mental steps)
  * 45 = MEDIUM (2-3 mental steps)
  * 60 = HARD (3-4 mental steps)
- Keep "steps_text" to strictly 3 short, punchy equation lines (max 8 words per line).
Return JSON with these exact keys:
- "difficulty": "EASY", "MEDIUM", or "HARD"
- "timer_seconds": 30, 45, or 60
- "title": catchy YouTube title under 55 chars
- "intro_text": bold 4-word hook on screen
- "intro_voice": energetic 3-second spoken challenge to stop scrolling
- "question_text": clean visual equation with +, -, ×, ÷, = and [ ? ]
- "question_voice": spoken reading of the equation + telling them the exact seconds on the clock
- "steps_text": 3 short equation lines showing the mental shortcut
- "answer_text": bold final answer (e.g., "[ ? ] = 72")
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
    clean_text = str(text).replace("[ ? ]", "what number").replace("[?]", "what number").strip()
    if not clean_text:
        clean_text = "Let's solve this math puzzle!"
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
    return AudioArrayClip(np.column_stack((wave, wave)), fps=sr)

tick_sound = get_tick_clip()

def format_math_symbols(text):
    if isinstance(text, list):
        text = "\n".join(str(item) for item in text)
    return str(text).replace(" * ", " × ").replace(" / ", " ÷ ")

# 3. Build Clean Base Slides (1080x1920) - Mobile Safe-Zone Aligned
def build_base_slide(output_name, badge, main_text, highlight="", badge_color="#FFD700", compact=False):
    main_text = format_math_symbols(main_text)
    highlight = format_math_symbols(highlight)

    img = Image.open("bg.png").convert("RGBA").resize((1080, 1920))
    draw = ImageDraw.Draw(img)
    W, H = img.size

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    body_size = 46 if compact else 60
    wrap_width = 28 if compact else 22
    line_spacing = 16 if compact else 24

    if os.path.exists(font_path):
        font_badge = ImageFont.truetype(font_path, 48)
        font_body = ImageFont.truetype(font_path, body_size)
        font_hl = ImageFont.truetype(font_path, 68)
    else:
        font_badge = ImageFont.load_default(size=48)
        font_body = ImageFont.load_default(size=body_size)
        font_hl = ImageFont.load_default(size=68)

    wrapped_badge = "\n".join(textwrap.wrap(badge, width=26))
    draw.multiline_text((W / 2, 515), wrapped_badge, fill=badge_color, font=font_badge, anchor="ma", align="center", spacing=10)

    wrapped_body = "\n".join([textwrap.fill(line, width=wrap_width) for line in main_text.split("\n")])
    draw.multiline_text((W / 2, 650), wrapped_body, fill="white", font=font_body, anchor="ma", align="center", spacing=line_spacing)

    if highlight:
        wrapped_hl = "\n".join(textwrap.wrap(highlight, width=18))
        draw.multiline_text((W / 2, 1110), wrapped_hl, fill="#00FFCC", font=font_hl, anchor="ma", align="center", spacing=12)

    img.save(output_name)

# 4. Lip-Sync Character Overlay Engine (Positioned Above YouTube Bottom UI)
char_closed = Image.open("char_closed.png").convert("RGBA").resize((420, 420))
char_open = Image.open("char_open.png").convert("RGBA").resize((420, 420))
char_think = Image.open("char_think.png").convert("RGBA").resize((420, 420)) if os.path.exists("char_think.png") else char_closed

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
        y_offset = 1230 if is_talking else 1240
        frame.paste(sprite, (510, y_offset), sprite)
        return np.array(frame.convert("RGB"))

    return VideoClip(frame_func, duration=audio.duration).with_audio(audio).with_fps(fps)

def make_thinking_scene(slide_path, duration=1.0):
    base_img = Image.open(slide_path).convert("RGBA")
    base_img.paste(char_think, (510, 1240), char_think)
    temp_path = f"think_{slide_path}"
    base_img.convert("RGB").save(temp_path)
    return ImageClip(temp_path).with_duration(duration).with_audio(tick_sound)

# 5. Assemble All 5 Scenes
clips = []

build_base_slide("s_intro.png", f"DIFFICULTY: {data['difficulty']}", data["intro_text"], f"{timer_seconds}-SEC CHALLENGE!", "#FFD700")
clips.append(make_talking_scene("s_intro.png", "intro.mp3"))

build_base_slide("s_q.png", "SOLVE IN YOUR HEAD", data["question_text"], "", "#38BDF8")
clips.append(make_talking_scene("s_q.png", "question.mp3"))

for sec in range(timer_seconds, 0, -1):
    t_slide = f"s_timer_{sec}.png"
    timer_color = "#FF4D4D" if sec <= 5 else "#FF9F43"
    build_base_slide(t_slide, f"DIFFICULTY: {data['difficulty']}", data["question_text"], f"TIME LEFT: {sec}s", timer_color)
    clips.append(make_thinking_scene(t_slide, 1.0))

build_base_slide("s_sol.png", "STEP-BY-STEP SOLUTION", data["steps_text"], data["answer_text"], "#4ADE80", compact=True)
clips.append(make_talking_scene("s_sol.png", "solution.mp3"))

build_base_slide("s_outro.png", "BEAT THE CLOCK?", "Comment your time below!\nSubscribe for daily puzzles!", "MATH MINUTE MARVELS", "#F472B6")
clips.append(make_talking_scene("s_outro.png", "outro.mp3"))

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

# Helper: Generate High-CTR Vertical Shorts Thumbnail (1080x1920)
def create_short_thumbnail(hook_line, teaser_math, output_path="thumb_short.jpg"):
    thumb = Image.open("bg.png").convert("RGBA").resize((1080, 1920))
    draw = ImageDraw.Draw(thumb)

    font_path = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
    if not os.path.exists(font_path):
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

    font_hook = ImageFont.truetype(font_path, 68) if os.path.exists(font_path) else ImageFont.load_default(size=68)
    font_math = ImageFont.truetype(font_path, 76) if os.path.exists(font_path) else ImageFont.load_default(size=76)

    mascot_img = Image.open("char_think.png" if os.path.exists("char_think.png") else "char_closed.png").convert("RGBA").resize((520, 520))
    thumb.paste(mascot_img, (480, 1180), mascot_img)

    wrapped_hook = "\n".join(textwrap.wrap(str(hook_line).upper(), width=16))
    draw.multiline_text((540, 530), wrapped_hook, fill="#FFD700", font=font_hook, anchor="ma", align="center", spacing=12)

    clean_teaser = format_math_symbols(teaser_math)
    wrapped_math = "\n".join(textwrap.wrap(clean_teaser, width=18))
    draw.multiline_text((540, 780), wrapped_math, fill="#00FFCC", font=font_math, anchor="ma", align="center", spacing=14)

    thumb.convert("RGB").save(output_path, "JPEG", quality=92)
    return output_path

# 7. Upload Publicly to YouTube, Attach Thumbnail & Save Question History
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
video_response = request.execute()
video_id = video_response["id"]
print(f"Uploaded PUBLIC {timer_seconds}s {data['difficulty']} Short ({video_id})!", flush=True)

try:
    short_q = data["question_text"][:45] + "..." if len(data["question_text"]) > 45 else data["question_text"]
    thumb_file = create_short_thumbnail(data["intro_text"], short_q)
    youtube.thumbnails().set(
        videoId=video_id,
        media_body=MediaFileUpload(thumb_file, mimetype="image/jpeg")
    ).execute()
    print(f"Uploaded custom Shorts thumbnail for {video_id}!", flush=True)
except Exception as e:
    print(f"Shorts thumbnail skipped: {e}", flush=True)

past_questions.append(str(data["question_text"]).strip())
with open(history_file, "w", encoding="utf-8") as f:
    json.dump(past_questions[-150:], f, indent=2)

