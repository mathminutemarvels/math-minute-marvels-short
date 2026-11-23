# ==============================================================================
# PART 1 OF 20: STANDARD IMPORTS & CLI ARGUMENTS
# ==============================================================================
import os
import json
import time
import math
import wave
import asyncio
import argparse
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont
from google import genai
from google.genai import types


def parse_args():
    parser = argparse.ArgumentParser(description="MathMinuteMarvels Unified Pipeline")
    parser.add_argument(
        "--mode",
        choices=["short", "long"],
        default="short",
        help="Render a 9:16 Short (1 puzzle) or 16:9 Long video (5 puzzles)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Render video locally without uploading to YouTube",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately instead of scheduling for the next IST upload slot",
    )
    return parser.parse_args()

# ==============================================================================
# PART 2 OF 20: EXACT-SLOT IST -> UTC SCHEDULE CALCULATOR
# ==============================================================================
SHORT_SLOTS_IST = [2, 8, 14, 20]  # 02:00, 08:00, 14:00, 20:00 IST (4 Shorts/day)
LONG_SLOTS_IST = [19]             # 19:00 IST on Wed & Sat (2 Long videos/week)


def get_next_publish_at_utc(mode: str) -> str:
    """
    Finds the next upcoming IST upload slot (~45-60 mins after cron triggers)
    and returns an ISO-8601 UTC string for YouTube's status.publishAt field.
    """
    ist = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(ist)
    slots = SHORT_SLOTS_IST if mode == "short" else LONG_SLOTS_IST

    candidates = []
    for day_offset in (0, 1):
        base_date = (now_ist + timedelta(days=day_offset)).date()
        for hour in slots:
            slot_dt = datetime(base_date.year, base_date.month, base_date.day, hour, 0, 0, tzinfo=ist)
            if slot_dt > now_ist + timedelta(minutes=15):
                candidates.append(slot_dt)

    next_slot_ist = min(candidates)
    next_slot_utc = next_slot_ist.astimezone(timezone.utc)
    print(f"[Scheduler] Target IST: {next_slot_ist.strftime('%Y-%m-%d %I:%M %p IST')} -> UTC: {next_slot_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    return next_slot_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

# ==============================================================================
# PART 3 OF 20: MONETIZATION-SAFE ROTATING COLOR THEMES
# ==============================================================================
COLOR_THEMES = [
    {
        "name": "Cyber Cyan",
        "bg": (11, 17, 32),
        "card": (21, 32, 58),
        "accent": (0, 245, 212),
        "highlight": (255, 214, 10),
        "text": (255, 255, 255),
        "muted": (160, 180, 210),
    },
    {
        "name": "Electric Violet",
        "bg": (19, 11, 35),
        "card": (36, 21, 66),
        "accent": (185, 131, 255),
        "highlight": (72, 245, 168),
        "text": (255, 255, 255),
        "muted": (185, 168, 215),
    },
    {
        "name": "Solar Sky",
        "bg": (13, 23, 38),
        "card": (24, 42, 68),
        "accent": (56, 189, 248),
        "highlight": (251, 146, 60),
        "text": (255, 255, 255),
        "muted": (165, 190, 218),
    },
    {
        "name": "Emerald Coral",
        "bg": (10, 26, 24),
        "card": (19, 46, 43),
        "accent": (52, 211, 153),
        "highlight": (251, 113, 133),
        "text": (255, 255, 255),
        "muted": (162, 204, 196),
    },
]


def pick_theme(history_data: list) -> dict:
    """Rotates visual themes deterministically based on history length."""
    chosen = COLOR_THEMES[len(history_data) % len(COLOR_THEMES)]
    print(f"[Theme] Selected visual palette: {chosen['name']}")
    return chosen

# ==============================================================================
# PART 4 OF 20: ENRICHED HISTORY & WEEKLY INSIGHTS MEMORY
# ==============================================================================
HISTORY_FILE = "history.json"
INSIGHTS_FILE = "weekly_insights.json"


def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def get_past_questions(history_data: list, limit: int = 150) -> set:
    """Extracts normalized past questions from both legacy strings and new metadata dicts."""
    past = set()
    for entry in history_data[-limit:]:
        if isinstance(entry, str):
            past.add(entry.strip().lower())
        elif isinstance(entry, dict) and entry.get("question"):
            past.add(str(entry["question"]).strip().lower())
    return past


def save_history_entries(new_entries: list):
    history_data = load_history()
    history_data.extend(new_entries)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data[-600:], f, indent=2, ensure_ascii=False)


def load_weekly_insights() -> dict:
    default_insights = {
        "best_topics": ["Mental Multiplication", "Percentage Shortcuts", "Square Tricks", "Number Patterns"],
        "preferred_difficulty": "MEDIUM",
    }
    if not os.path.exists(INSIGHTS_FILE):
        return default_insights
    try:
        with open(INSIGHTS_FILE, "r", encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, dict):
                default_insights.update(loaded)
            return default_insights
    except Exception:
        return default_insights

# ==============================================================================
# PART 5 OF 20: GEMINI 2.5 FLASH PUZZLE GENERATOR WITH VALIDATION & RETRIES
# ==============================================================================
REQUIRED_PUZZLE_KEYS = {
    "hook",
    "question",
    "topic",
    "difficulty",
    "timer_seconds",
    "answer",
    "explanation",
    "outro",
}

TOPIC_POOL = [
    "Fast Mental Multiplication",
    "Square & Cube Shortcuts",
    "Number Sequence & Pattern Logic",
    "Fraction & Ratio Tricks",
    "Algebraic Speed Tricks",
    "Mental Division & Divisibility Hacks",
    "Percentage Shortcuts",
    "Tricky Math & Logic Riddles",
]

DIFFICULTY_TIMERS = {
    "EASY": 15,
    "MEDIUM": 25,
    "HARD": 35,
}


def _pick_forced_topics(history_data: list, count: int) -> list:
    """Selects distinct math topics that were NOT used in the last 6 runs."""
    recent_topics = [
        str(item.get("topic", "")).strip().lower()
        for item in history_data[-6:]
        if isinstance(item, dict) and item.get("topic")
    ]
    fresh = [t for t in TOPIC_POOL if t.lower() not in recent_topics]
    if len(fresh) < count:
        fresh = TOPIC_POOL[:]
    random.shuffle(fresh)
    return fresh[:count]


def _pick_random_difficulty(history_data: list) -> tuple:
    """Picks a random difficulty (EASY, MEDIUM, or HARD) different from the last run."""
    last_diff = next(
        (
            str(item.get("difficulty", "")).strip().upper()
            for item in reversed(history_data)
            if isinstance(item, dict) and item.get("difficulty")
        ),
        None,
    )
    choices = [d for d in DIFFICULTY_TIMERS if d != last_diff] or list(DIFFICULTY_TIMERS)
    chosen = random.choice(choices)
    return chosen, DIFFICULTY_TIMERS.get(chosen, 25)

def generate_puzzles(mode: str = "short", max_retries: int = 4) -> list:
    """
    Generates 1 puzzle (short) or 5 progressive puzzles (long) via gemini-2.5-flash,
    enforcing schema completeness, duplicate prevention, and automatic retries.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )
    history_data = load_history()
    past_questions = get_past_questions(history_data)
    insights = load_weekly_insights()

    count = 1 if mode == "short" else 5
    best_topics = ", ".join(insights.get("best_topics", [])[:4])
    pref_diff = insights.get("preferred_difficulty", "MEDIUM")

    assigned_topics = _pick_forced_topics(history_data, count)
    single_topic = next(iter(assigned_topics), "Fast Mental Multiplication")

    if mode == "short":
        chosen_diff, chosen_timer = _pick_random_difficulty(history_data)
        topic_instruction = (
            f'You MUST create a puzzle strictly from this exact topic: "{single_topic}". '
            f"Do NOT use Percentage Shortcuts unless explicitly assigned above."
        )
        difficulty_rule = (
            f'You MUST set "difficulty" to "{chosen_diff}" and "timer_seconds" to {chosen_timer}.'
        )
    else:
        topic_instruction = (
            f"Create {count} puzzles covering these exact {count} distinct topics in order: {json.dumps(assigned_topics)}."
        )
        difficulty_rule = (
            "Progress smoothly across the 5 puzzles: Puzzle 1-2 EASY (15s), Puzzle 3-4 MEDIUM (25s), Puzzle 5 HARD (35s)."
        )

    prompt = f"""
You are an energetic mental-math creator making {count} original math puzzle(s) for Math Minute Marvels ({mode} format).
{topic_instruction}
{difficulty_rule}

Strict rules:
1. Return ONLY a valid JSON array containing exactly {count} object(s). No markdown fences.
2. Each object MUST include these exact keys:
   - "hook": high-energy 10-14 word spoken intro + watch-till-the-end challenge (e.g., "Welcome to Math Minute Marvels! Only 3% solve this—stay till the end to test your speed!")
   - "question": clean math expression using %, ×, ÷, =, - WITHOUT brackets (e.g., "18 × 35 = ?" or "88² - 12² = ?")
   - "topic": exact assigned topic name
   - "difficulty": "EASY", "MEDIUM", or "HARD"
   - "timer_seconds": integer (15 for EASY, 25 for MEDIUM, 35 for HARD)
   - "answer": exact final numerical or percentage answer
   - "explanation": 2 to 3 short equation-style steps with completed calculations (never leave "?" inside explanation steps)
   - "outro": energetic call-to-action asking viewers to comment their solve time, hit Like if they beat the clock, and Subscribe to Math Minute Marvels!
3. NEVER repeat or closely paraphrase any of these recent questions:
{json.dumps(list(past_questions)[-45:], ensure_ascii=False)}
"""

    FALLBACK_MODELS = [
        "gemini-3.5-flash-lite",
        "gemini-3.5-flash",
        "gemini-3.8-flash",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite",
    ]

    max_retries = len(FALLBACK_MODELS)

    for attempt in range(1, max_retries + 1):
        model_name = FALLBACK_MODELS[attempt - 1]
        try:
            print(f"[Gemini Attempt {attempt}/{max_retries}] Trying model: {model_name}")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.85,
                ),
            )
            raw_text = (response.text or "").strip()
            puzzles = json.loads(raw_text)

            if not isinstance(puzzles, list) or len(puzzles) != count:
                raise ValueError(f"Expected JSON list of {count} item(s), got {type(puzzles).__name__}.")

            validated = []
            seen_in_batch = set()
            for p in puzzles:
                if not isinstance(p, dict) or not REQUIRED_PUZZLE_KEYS.issubset(p.keys()):
                    missing = REQUIRED_PUZZLE_KEYS - set(p.keys() if isinstance(p, dict) else [])
                    raise ValueError(f"Missing required keys: {missing}")

                q_clean = str(p["question"]).strip()
                q_norm = q_clean.lower()
                if not q_clean or q_norm in past_questions or q_norm in seen_in_batch:
                    raise ValueError(f"Duplicate or empty question detected: {q_clean}")

                seen_in_batch.add(q_norm)
                diff = str(p["difficulty"]).strip().upper()
                p["difficulty"] = diff if diff in {"EASY", "MEDIUM", "HARD"} else "MEDIUM"
                p["timer_seconds"] = max(12, min(38, int(p["timer_seconds"])))
                validated.append(p)

            print(f"[Gemini] Successfully generated {len(validated)} unique {mode} puzzle(s) using {model_name}.")
            return validated

        except Exception as err:
            print(f"[Gemini Attempt {attempt}/{max_retries}] {model_name} failed: {err}")
            if attempt == max_retries:
                raise RuntimeError("Failed to generate valid puzzles after trying all fallback models.") from err
            time.sleep(3 * attempt)
    

# ==============================================================================
# PART 6 OF 20: EDGE-TTS VOICE ROTATION & DURATION MEASURER
# ==============================================================================
try:
    from moviepy import AudioFileClip
except ImportError:
    from moviepy.editor import AudioFileClip

TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-RyanNeural",
]


def pick_voice(history_data: list) -> str:
    """Rotates neural voices across uploads so videos never sound monotonous."""
    chosen = TTS_VOICES[len(history_data) % len(TTS_VOICES)]
    print(f"[Audio] Selected TTS voice: {chosen}")
    return chosen


async def _synthesize_edge_tts(text: str, output_path: str, voice: str, rate: str = "+6%"):
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate)
    await communicate.save(output_path)


def create_voice_clip(text: str, output_path: str, voice: str, rate: str = "+6%") -> float:
    """Generates an MP3 via Edge-TTS and returns its exact duration in seconds."""
    asyncio.run(_synthesize_edge_tts(text.strip(), output_path, voice, rate))
    clip = AudioFileClip(output_path)
    duration = float(clip.duration)
    clip.close()
    return duration

# ==============================================================================
# PART 7 OF 20: PROCEDURAL CLOCK TICK & REVEAL CHIME SFX GENERATOR
# ==============================================================================
def ensure_sfx_assets(sfx_dir: str = "assets") -> tuple:
    """Synthesizes 1.0s clock tick and 0.6s reveal chime WAVs if not already present."""
    os.makedirs(sfx_dir, exist_ok=True)
    tick_path = os.path.join(sfx_dir, "tick_1s.wav")
    chime_path = os.path.join(sfx_dir, "reveal_chime.wav")
    sample_rate = 44100

    if not os.path.exists(tick_path):
        samples = np.zeros(sample_rate, dtype=np.float32)
        click_len = int(0.035 * sample_rate)
        t = np.linspace(0, 0.035, click_len, endpoint=False)
        env = np.exp(-t * 110.0)
        samples[:click_len] = 0.45 * env * np.sin(2 * np.pi * 1150 * t)
        with wave.open(tick_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((samples * 32767).astype(np.int16).tobytes())

    if not os.path.exists(chime_path):
        dur = 0.6
        t = np.linspace(0, dur, int(dur * sample_rate), endpoint=False)
        env = np.exp(-t * 6.5)
        wave_data = 0.35 * env * (np.sin(2 * np.pi * 880 * t) + 0.6 * np.sin(2 * np.pi * 1318.5 * t))
        with wave.open(chime_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((wave_data * 32767).astype(np.int16).tobytes())

    return tick_path, chime_path

# ==============================================================================
# PART 8 OF 20: PUZZLE AUDIO BUNDLE & DYNAMIC <58s TIMER SCALING
# ==============================================================================
import re

def clean_math_for_tts(text: str) -> str:
    """Converts visual math symbols (%, =, ×, ÷, -) into natural spoken English."""
    s = str(text)
    s = re.sub(r"=\s*\[?\s*\?\s*\]?", " equals what?", s)
    s = re.sub(r"\[?\s*\?\s*\]?", "", s)
    s = re.sub(r"(\d)\s*%", r"\1 percent", s)
    s = s.replace("%", " percent ")
    s = s.replace("×", " times ").replace("÷", " divided by ").replace("=", " equals ")
    s = re.sub(r"(?<=\w)\s*-\s*(?=\w)", " minus ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def prepare_puzzle_audio(
    puzzle: dict,
    idx: int,
    mode: str,
    voice: str,
    work_dir: str,
    include_outro: bool = True,
) -> dict:
    """Synthesizes hook/intro, question, countdown motivation, explanation, and outro."""
    os.makedirs(work_dir, exist_ok=True)
    hook_mp3 = os.path.join(work_dir, f"p{idx}_hook.mp3")
    q_mp3 = os.path.join(work_dir, f"p{idx}_question.mp3")
    challenge_mp3 = os.path.join(work_dir, f"p{idx}_challenge.mp3")
    exp_mp3 = os.path.join(work_dir, f"p{idx}_explanation.mp3")
    outro_mp3 = os.path.join(work_dir, f"p{idx}_outro.mp3")

    hook_dur = create_voice_clip(clean_math_for_tts(puzzle["hook"]), hook_mp3, voice, rate="+8%")
    q_dur = create_voice_clip(clean_math_for_tts(puzzle["question"]), q_mp3, voice, rate="+4%")
    challenge_dur = create_voice_clip(
        "Your clock starts now! Comment your answer before time runs out!",
        challenge_mp3,
        voice,
        rate="+9%",
    )
    exp_text = clean_math_for_tts(f"The answer is {puzzle['answer']}. {puzzle['explanation']}")
    exp_dur = create_voice_clip(exp_text, exp_mp3, voice, rate="+5%")

    outro_dur = 0.0
    if include_outro:
        outro_dur = create_voice_clip(clean_math_for_tts(puzzle["outro"]), outro_mp3, voice, rate="+7%")

    buffer_sec = 1.4
    speech_total = hook_dur + q_dur + exp_dur + outro_dur + buffer_sec
    requested_timer = int(puzzle["timer_seconds"])

    if mode == "short":
        max_allowed_timer = max(10, int(math.floor(57.5 - speech_total)))
        final_timer = min(requested_timer, max_allowed_timer)
    else:
        final_timer = requested_timer

    return {
        "hook_mp3": hook_mp3,
        "hook_dur": hook_dur,
        "question_mp3": q_mp3,
        "question_dur": q_dur,
        "challenge_mp3": challenge_mp3,
        "challenge_dur": challenge_dur,
        "explanation_mp3": exp_mp3,
        "explanation_dur": exp_dur,
        "outro_mp3": outro_mp3 if include_outro else None,
        "outro_dur": outro_dur,
        "final_timer": final_timer,
        "estimated_total_sec": round(speech_total + final_timer, 2),
    }

# ==============================================================================
# PART 9 & 10 OF 20: BACKGROUND LOADER, FLOATING SYMBOLS & 430PX LIP-SYNC MASCOT
# ==============================================================================
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "assets/font_bold.ttf",
    "arialbd.ttf",
]

MATH_SYMBOLS = ["%", "+", "=", "×", "÷", "!"]

BG_CANDIDATES = [
    "bg.png", "bg.jpg", "bg.jpeg",
    "background.png", "background.jpg", "background.jpeg",
    "assets/bg.png", "assets/bg.jpg", "assets/bg.jpeg",
    "assets/background.png", "assets/background.jpg", "assets/background.jpeg",
]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def get_canvas_spec(mode: str) -> dict:
    """Returns resolution, larger mascot size (430px Short), and responsive fonts."""
    if mode == "short":
        return {
            "width": 1080,
            "height": 1920,
            "pad": 56,
            "font_badge": load_font(34),
            "font_hook": load_font(58),
            "font_question": load_font(76),
            "font_timer": load_font(120),
            "font_answer": load_font(80),
            "font_body": load_font(38),
            "font_symbol_bg": load_font(64),
            "font_symbol_orbit": load_font(48),
            "mascot_size": 430,
        }
    return {
        "width": 1920,
        "height": 1080,
        "pad": 80,
        "font_badge": load_font(32),
        "font_hook": load_font(50),
        "font_question": load_font(68),
        "font_timer": load_font(108),
        "font_answer": load_font(72),
        "font_body": load_font(36),
        "font_symbol_bg": load_font(58),
        "font_symbol_orbit": load_font(44),
        "mascot_size": 350,
    }


def prepare_base_background(mode: str, spec: dict, theme: dict) -> Image.Image:
    """
    Loads bg.png for Shorts or bg_long.png for Long videos, scales/crops to fit,
    and blends with a subtle dark theme tint so text and neon cards stay crisp.
    """
    w, h = spec["width"], spec["height"]
    tint_layer = Image.new("RGB", (w, h), theme["bg"])

    primary = "bg.png" if mode == "short" else "bg_long.png"
    fallback = "bg_long.png" if mode == "short" else "bg.png"
    found_bg = next((p for p in [primary, fallback] if os.path.exists(p)), None)

    if not found_bg:
        print("[Background] No bg.png / bg_long.png found; using solid theme background.")
        return tint_layer

    print(f"[Background] Loaded {found_bg} for mode={mode}")
    raw = Image.open(found_bg).convert("RGB")
    scale = max(w / raw.width, h / raw.height)
    new_w, new_h = int(raw.width * scale), int(raw.height * scale)
    resized = raw.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    cropped = resized.crop((left, top, left + w, top + h))

    # Blend 85% custom background & logo + 15% subtle dark tint
    return Image.blend(cropped, tint_layer, alpha=0.15)

def _blend_rgb(fg: tuple, bg: tuple, alpha: float) -> tuple:
    r0, g0, b0 = fg
    r1, g1, b1 = bg
    return (
        int(r0 * alpha + r1 * (1.0 - alpha)),
        int(g0 * alpha + g1 * (1.0 - alpha)),
        int(b0 * alpha + b1 * (1.0 - alpha)),
    )


def draw_floating_math_symbols(canvas: Image.Image, spec: dict, theme: dict, t_sec: float):
    """Renders drifting %, +, =, ×, ÷, ! operators across the background."""
    w, h = spec["width"], spec["height"]
    draw = ImageDraw.Draw(canvas)

    count = 16
    for i in range(count):
        sym = MATH_SYMBOLS[i % len(MATH_SYMBOLS)]
        col_rgb = theme["accent"] if i % 2 == 0 else theme["highlight"]
        x_base = int(((i * 0.6180339) % 1.0) * (w - 100)) + 50
        x_wobble = int(math.sin(t_sec * 1.6 + i * 1.3) * 28)
        speed = 42 + (i % 4) * 14
        y_pos = int((h - ((t_sec * speed + i * (h / count)) % (h + 120))) - 60)

        alpha = 0.22 + 0.10 * math.sin(t_sec * 2.5 + i)
        blended_col = _blend_rgb(col_rgb, theme["bg"], alpha)
        draw.text((x_base + x_wobble, y_pos), sym, font=spec["font_symbol_bg"], fill=blended_col)

def load_mascot_sprites(target_size: int) -> dict:
    """Loads all 5 uploaded character sprites: open, half, closed, blink, think."""
    sprites = {}
    files = {
        "open": "char_open.png",
        "half": "char_half.png",
        "closed": "char_closed.png",
        "blink": "char_blink.png",
        "think": "char_think.png",
    }
    for key, filename in files.items():
        if os.path.exists(filename):
            img = Image.open(filename).convert("RGBA")
            img.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
            sprites[key] = img
        else:
            sprites[key] = None
    return sprites


def paste_mascot_with_symbols(
    canvas: Image.Image,
    sprites: dict,
    stage: str,
    is_speaking_now: bool,
    cx: int,
    cy: int,
    t_sec: float,
    spec: dict,
    theme: dict,
):
    """
    Composites the 430px mascot using all 5 sprites:
    - Smooth 3-frame talking cycle (open -> half -> closed) ONLY when voice is active
    - Natural eye-blink (char_blink.png) every ~3.5 seconds during pauses
    - Thinking pose (char_think.png) during countdown
    - Orbiting %, +, =, ×, ÷, ! operators
    """
    if stage == "countdown":
        pose = "think"
        bounce_y = int(math.sin(t_sec * 3.2) * 8)
    elif is_speaking_now:
        cycle = int(t_sec * 10.0) % 4
        if cycle == 0:
            pose = "open"
        elif cycle in (1, 3):
            pose = "half" if sprites.get("half") is not None else "open"
        else:
            pose = "closed"
        bounce_y = -int(abs(math.sin(t_sec * 8.0)) * 10)
    else:
        # Closed mouth during pauses, with a quick natural blink every 3.5 seconds
        blink_phase = t_sec % 3.5
        if blink_phase < 0.16 and sprites.get("blink") is not None:
            pose = "blink"
        else:
            pose = "closed"
        bounce_y = int(math.sin(t_sec * 2.4) * 4)

    draw = ImageDraw.Draw(canvas)
    orbit_radius_x = int(spec["mascot_size"] * 0.66)
    orbit_radius_y = int(spec["mascot_size"] * 0.20)
    for idx, sym in enumerate(MATH_SYMBOLS):
        angle = t_sec * 1.9 + (idx * (2 * math.pi / len(MATH_SYMBOLS)))
        sx = int(cx + math.cos(angle) * orbit_radius_x)
        sy = int(cy + math.sin(angle) * orbit_radius_y + bounce_y // 2)
        sym_color = theme["highlight"] if idx % 2 == 0 else theme["accent"]
        draw.text((sx - 14, sy - 20), sym, font=spec["font_symbol_orbit"], fill=sym_color)

    sprite = sprites.get(pose) or sprites.get("closed") or sprites.get("open")
    if sprite is not None:
        px = int(cx - sprite.width // 2)
        py = int(cy - sprite.height // 2 + bounce_y)
        canvas.paste(sprite, (px, py), sprite)
    else:
        r = 105
        draw.ellipse([cx - r, cy - r + bounce_y, cx + r, cy + r + bounce_y], outline=theme["accent"], width=6)

# ==============================================================================
# PART 11 OF 20: ROUNDED CARD & CENTERED MULTI-LINE TEXT RENDERER
# ==============================================================================
def draw_rounded_card(
    canvas_or_draw,
    box: tuple,
    radius: int,
    fill: tuple,
    outline: tuple,
    width: int = 3,
    fill_alpha: int = 170,
):
    """Draws a translucent glass card so background artwork & logo remain visible."""
    if isinstance(canvas_or_draw, Image.Image):
        x0, y0, x1, y1 = [int(v) for v in box]
        cw, ch = max(1, x1 - x0), max(1, y1 - y0)
        overlay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle(
            (0, 0, cw - 1, ch - 1),
            radius=radius,
            fill=(fill, fill, fill, fill_alpha),
            outline=(outline, outline, outline, 255),
            width=width,
        )
        canvas_or_draw.paste(overlay, (x0, y0), overlay)
    else:
        canvas_or_draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)

def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: int,
    start_y: int,
    max_width: int,
    fill: tuple,
    line_spacing: int = 14,
) -> int:
    """Wraps text cleanly, draws each line centered at center_x, and returns final bottom Y."""
    words = str(text).split()
    lines, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        x0, y0, x1, y1 = draw.textbbox((0, 0), candidate, font=font)
        if (x1 - x0) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))

    y = start_y
    for line in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=font)
        line_w = x1 - x0
        line_h = y1 - y0
        draw.text((center_x - line_w // 2, y), line, font=font, fill=fill)
        y += line_h + line_spacing
    return y

# ==============================================================================
# PART 12 OF 20: SHRINKING NEON COUNTDOWN PROGRESS BAR
# ==============================================================================
def draw_countdown_bar(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    top_y: int,
    bar_width: int,
    bar_height: int,
    remaining_sec: float,
    total_sec: int,
    theme: dict,
):
    """Renders large countdown digits plus a smooth shrinking neon progress bar."""
    ratio = max(0.0, min(1.0, remaining_sec / max(1, total_sec)))
    active_color = theme["highlight"] if remaining_sec <= 4.0 else theme["accent"]

    left = center_x - bar_width // 2
    right = center_x + bar_width // 2
    bottom = top_y + bar_height

    # Track background
    draw.rounded_rectangle((left, top_y, right, bottom), radius=bar_height // 2, fill=theme["bg"], outline=theme["muted"], width=2)

    # Active shrinking fill
    fill_w = int(bar_width * ratio)
    if fill_w > bar_height:
        draw.rounded_rectangle(
            (left, top_y, left + fill_w, bottom),
            radius=bar_height // 2,
            fill=active_color,
        )

# ==============================================================================
# PART 13 OF 20: SAFE-ZONE COMPOSITOR (BG IMAGE, AUTO-SIZED STEPS & LIP-SYNC)
# ==============================================================================
def _split_explanation_steps(explanation: str) -> list:
    """Splits explanation text cleanly on ., !, or ? into 2-4 crisp steps."""
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(explanation).strip()) if s.strip()]
    return parts if parts else [str(explanation).strip()]


def _measure_wrapped_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    words = str(text).split()
    lines, current = [], []
    for word in words:
        candidate = " ".join(current + [word])
        x0, y0, x1, y1 = draw.textbbox((0, 0), candidate, font=font)
        if (x1 - x0) <= max_width or not current:
            current.append(word)
        else:
            lines.append(" ".join(current))
            current = [word]
    if current:
        lines.append(" ".join(current))
    return lines


def render_puzzle_frame(
    t_sec: float,
    puzzle: dict,
    audio_info: dict,
    puzzle_num: int,
    total_puzzles: int,
    mode: str,
    spec: dict,
    theme: dict,
    sprites: dict,
    base_bg: Image.Image = None,
    is_speaking_now: bool = False,
) -> np.ndarray:
    """Builds one RGB frame with bg.png/bg_long.png, Shorts safe zones & auto-sized steps."""
    w, h, pad = spec["width"], spec["height"], spec["pad"]
    canvas = base_bg.copy() if base_bg is not None else Image.new("RGB", (w, h), theme["bg"])

    draw_floating_math_symbols(canvas, spec, theme, t_sec)
    draw = ImageDraw.Draw(canvas)

    # YouTube Shorts UI safe margins (clears top icons and bottom title/channel bar)
    top_margin = 470 if mode == "short" else 235
    bottom_margin = 250 if mode == "short" else 50

    t_hook_end = audio_info["hook_dur"] + 0.35
    t_q_end = t_hook_end + audio_info["question_dur"] + 0.35
    t_timer_end = t_q_end + audio_info["final_timer"]
    t_exp_end = t_timer_end + audio_info["explanation_dur"] + 0.4

    if t_sec < t_hook_end:
        stage = "hook"
        stage_elapsed = t_sec
    elif t_sec < t_q_end:
        stage = "question"
        stage_elapsed = t_sec - t_hook_end
    elif t_sec < t_timer_end:
        stage = "countdown"
        stage_elapsed = t_sec - t_q_end
    elif t_sec < t_exp_end:
        stage = "reveal"
        stage_elapsed = t_sec - t_timer_end
    else:
        stage = "outro"
        stage_elapsed = t_sec - t_exp_end

    ease = min(1.0, stage_elapsed / 0.25)
    slide_y = int(((1.0 - ease) ** 3) * 28)

    # Top header pill (positioned safely below YouTube's top Shorts icons)
    label_prefix = f"PUZZLE {puzzle_num}/{total_puzzles}  •  " if total_puzzles > 1 else ""
    header_text = f"{label_prefix}{puzzle['topic'].upper()}  •  {puzzle['difficulty']}"
    draw_rounded_card(canvas, (pad, card_top, w - pad, card_bottom), radius=34, fill=theme["card"], outline=theme["accent"], width=3, fill_alpha=170)
    draw_centered_multiline(draw, header_text, spec["font_badge"], w // 2, top_margin + 16, w - pad * 2 - 40, theme["accent"])

    # Main content card (leaves generous space above bottom safe zone for 430px mascot)
    mascot_band_h = 370 if mode == "short" else 280
    card_top = top_margin + 88 + slide_y
    card_bottom = h - bottom_margin - mascot_band_h
    draw_rounded_card(canvas, (pad, card_top, w - pad, card_bottom), radius=34, fill=theme["card"], outline=theme["accent"], width=3, fill_alpha=175)

    inner_w = w - pad * 2 - 64
    cx = w // 2

    if stage == "hook":
        draw_centered_multiline(draw, "⚡ QUICK BRAIN CHECK ⚡", spec["font_badge"], cx, card_top + 70, inner_w, theme["highlight"])
        draw_centered_multiline(draw, puzzle["hook"], spec["font_hook"], cx, card_top + 165, inner_w, theme["text"])
        draw_centered_multiline(draw, "%   +   =   ×   ÷   !", spec["font_hook"], cx, card_bottom - 110, inner_w, theme["accent"])

    else:
        # Auto-sized Question Box
        q_lines = _measure_wrapped_lines(draw, puzzle["question"], spec["font_question"], inner_w - 36)
        q_box_h = max(120, len(q_lines) * 80 + 44)
        q_box_top = card_top + 22
        q_box_bottom = q_box_top + q_box_h
        draw_rounded_card(draw, (pad + 22, q_box_top, w - pad - 22, q_box_bottom), radius=22, fill=theme["bg"], outline=theme["muted"], width=2)
        draw_centered_multiline(draw, puzzle["question"], spec["font_question"], cx, q_box_top + 22, inner_w - 36, theme["text"])

        if stage == "countdown":
            rem = max(0.0, t_timer_end - t_sec)
            total_t = max(1, audio_info["final_timer"])
            ratio = rem / total_t
            urgent = rem <= 5.0
            ring_color = theme["highlight"] if urgent else theme["accent"]

            ring_cy = (q_box_bottom + card_bottom - 70) // 2
            ring_r = 135 if mode == "short" else 105
            pulse_r = ring_r + (int(abs(math.sin(t_sec * 6.0)) * 10) if urgent else 0)
            bbox = (cx - pulse_r, ring_cy - pulse_r, cx + pulse_r, ring_cy + pulse_r)
            draw.ellipse(bbox, outline=theme["bg"], width=15)
            draw.arc(bbox, start=-90, end=-90 + int(360 * ratio), fill=ring_color, width=15)

            draw_centered_multiline(draw, f"{int(math.ceil(rem))}s", spec["font_timer"], cx, ring_cy - 62, inner_w, ring_color)
            draw_countdown_bar(draw, cx, card_bottom - 64, int(inner_w * 0.85), 24, rem, total_t, theme)

        elif stage in ("reveal", "outro"):
            ans_top = q_box_bottom + 18
            ans_bottom = ans_top + (106 if mode == "short" else 88)
            draw_rounded_card(draw, (pad + 22, ans_top, w - pad - 22, ans_bottom), radius=22, fill=theme["bg"], outline=theme["accent"], width=3)
            draw_centered_multiline(draw, f"ANSWER: {puzzle['answer']}", spec["font_answer"], cx, ans_top + 14, inner_w - 36, theme["accent"])

            steps = _split_explanation_steps(puzzle["explanation"])
            exp_progress = min(0.999, max(0.0, stage_elapsed / max(0.5, audio_info["explanation_dur"])))
            visible_count = len(steps) if stage == "outro" else min(len(steps), int(exp_progress * len(steps)) + 1)

            step_font = load_font(35 if mode == "short" else 33)
            step_y = ans_bottom + 18
            outro_reserve = 72 if (stage == "outro" and puzzle.get("outro")) else 16

            for s_idx in range(visible_count):
                step_label = f"Step {s_idx + 1}: {steps[s_idx]}"
                wrapped = _measure_wrapped_lines(draw, step_label, step_font, inner_w - 52)
                dyn_box_h = len(wrapped) * 44 + 32

                if step_y + dyn_box_h > card_bottom - outro_reserve:
                    break

                is_active = (stage == "reveal") and (s_idx == visible_count - 1)
                step_bg = theme["bg"] if is_active else theme["card"]
                step_border = theme["highlight"] if is_active else theme["muted"]
                step_text_col = theme["highlight"] if is_active else theme["text"]

                draw_rounded_card(draw, (pad + 22, step_y, w - pad - 22, step_y + dyn_box_h), radius=18, fill=step_bg, outline=step_border, width=2)
                draw_centered_multiline(draw, step_label, step_font, cx, step_y + 15, inner_w - 52, step_text_col, line_spacing=9)
                step_y += dyn_box_h + 12

            if stage == "outro" and puzzle.get("outro"):
                draw_centered_multiline(draw, puzzle["outro"], spec["font_badge"], cx, card_bottom - 54, inner_w, theme["highlight"])

    # Large 430px Mascot & handle placed safely above YouTube's bottom Shorts bar
    mascot_y = card_bottom + (175 if mode == "short" else 135)
    paste_mascot_with_symbols(canvas, sprites, stage, is_speaking_now, cx, mascot_y, t_sec, spec, theme)
    draw_centered_multiline(draw, "@MathMinuteMarvels", spec["font_badge"], cx, card_bottom + mascot_band_h - 38, inner_w, theme["muted"])

    return np.array(canvas)

# ==============================================================================
# PART 14 OF 20: AUDIO-REACTIVE LIP-SYNC & 24 FPS VIDEOCLIP BUILDER
# ==============================================================================
try:
    from moviepy import VideoClip
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import VideoClip
    MOVIEPY_V2 = False


def extract_speech_activity(audio_info: dict, total_dur: float, fps: int = 24) -> np.ndarray:
    """
    Returns a per-frame boolean array that is True ONLY when the narrator's voice
    is actively speaking (closes mascot mouth during every silent pause).
    """
    n_frames = max(1, int(math.ceil(total_dur * fps)) + 2)
    activity = np.zeros(n_frames, dtype=bool)

    t_hook_start = 0.0
    t_q_start = audio_info["hook_dur"] + 0.35
    t_timer_start = t_q_start + audio_info["question_dur"] + 0.35
    t_exp_start = t_timer_start + audio_info["final_timer"] + 0.15
    t_outro_start = t_exp_start + audio_info["explanation_dur"] + 0.25

    segments = [
        (audio_info["hook_mp3"], t_hook_start),
        (audio_info["question_mp3"], t_q_start),
        (audio_info["explanation_mp3"], t_exp_start),
    ]
    if audio_info.get("challenge_mp3"):
        segments.append((audio_info["challenge_mp3"], t_timer_start))
    if audio_info.get("outro_mp3"):
        segments.append((audio_info["outro_mp3"], t_outro_start))

    for mp3_path, start_t in segments:
        if not mp3_path or not os.path.exists(mp3_path):
            continue
        clip = AudioFileClip(mp3_path)
        try:
            arr = clip.to_soundarray(fps=2400)
            arr = np.mean(np.abs(arr), axis=1) if arr.ndim > 1 else np.abs(arr)
            samples_per_frame = max(1, int(2400 / fps))
            seg_frames = len(arr) // samples_per_frame
            for f_idx in range(seg_frames):
                chunk = arr[f_idx * samples_per_frame : (f_idx + 1) * samples_per_frame]
                if float(np.mean(chunk)) > 0.018:
                    global_f = int(round((start_t * fps) + f_idx))
                    if 0 <= global_f < n_frames:
                        activity[global_f] = True
        finally:
            clip.close()

    return activity


def build_puzzle_video_clip(
    puzzle: dict,
    audio_info: dict,
    puzzle_num: int,
    total_puzzles: int,
    mode: str,
    spec: dict,
    theme: dict,
    sprites: dict,
    base_bg: Image.Image = None,
) -> VideoClip:
    """Creates a 24 FPS VideoClip with custom background and audio-synced lip movement."""
    fps = 24
    total_dur = (
        audio_info["hook_dur"]
        + 0.35
        + audio_info["question_dur"]
        + 0.35
        + audio_info["final_timer"]
        + audio_info["explanation_dur"]
        + 0.40
        + (audio_info["outro_dur"] + 0.30 if audio_info["outro_mp3"] else 0.0)
    )
    speech_activity = extract_speech_activity(audio_info, total_dur, fps=fps)

    def make_frame(t):
        f_idx = min(len(speech_activity) - 1, max(0, int(t * fps)))
        is_speaking_now = bool(speech_activity[f_idx])
        return render_puzzle_frame(
            t_sec=t,
            puzzle=puzzle,
            audio_info=audio_info,
            puzzle_num=puzzle_num,
            total_puzzles=total_puzzles,
            mode=mode,
            spec=spec,
            theme=theme,
            sprites=sprites,
            base_bg=base_bg,
            is_speaking_now=is_speaking_now,
        )

    if MOVIEPY_V2:
        return VideoClip(frame_function=make_frame, duration=total_dur).with_fps(fps)
    return VideoClip(make_frame=make_frame, duration=total_dur).set_fps(fps)

# ==============================================================================
# PART 15 OF 20: SYNCHRONIZED VOICEOVER, TICK & CHIME AUDIO MIXER
# ==============================================================================
try:
    from moviepy import CompositeAudioClip
except ImportError:
    from moviepy.editor import CompositeAudioClip


def _with_start(clip, start_t: float):
    return clip.with_start(start_t) if MOVIEPY_V2 else clip.set_start(start_t)


def build_puzzle_audio_track(audio_info: dict, tick_path: str, chime_path: str) -> tuple:
    """
    Combines hook, question, per-second countdown ticks, reveal chime, explanation,
    and optional outro into one CompositeAudioClip. Returns (composite_clip, opened_clips).
    """
    tracks = []
    opened = []

    t_cursor = 0.0
    hook_clip = AudioFileClip(audio_info["hook_mp3"])
    opened.append(hook_clip)
    tracks.append(_with_start(hook_clip, t_cursor))
    t_cursor += audio_info["hook_dur"] + 0.35

    q_clip = AudioFileClip(audio_info["question_mp3"])
    opened.append(q_clip)
    tracks.append(_with_start(q_clip, t_cursor))
    t_cursor += audio_info["question_dur"] + 0.35

    if audio_info.get("challenge_mp3") and os.path.exists(audio_info["challenge_mp3"]):
        ch_clip = AudioFileClip(audio_info["challenge_mp3"])
        ch_clip = ch_clip.with_start(t_cursor) if MOVIEPY_V2 else ch_clip.set_start(t_cursor)
        audio_layers.append(ch_clip)

    # Schedule 1 tick sound per second of countdown
    timer_secs = int(audio_info["final_timer"])
    for sec_idx in range(timer_secs):
        tick_clip = AudioFileClip(tick_path)
        opened.append(tick_clip)
        tracks.append(_with_start(tick_clip, t_cursor + sec_idx))
    t_cursor += timer_secs

    # Play chime at exact reveal timestamp
    chime_clip = AudioFileClip(chime_path)
    opened.append(chime_clip)
    tracks.append(_with_start(chime_clip, t_cursor))

    exp_clip = AudioFileClip(audio_info["explanation_mp3"])
    opened.append(exp_clip)
    tracks.append(_with_start(exp_clip, t_cursor + 0.15))
    t_cursor += audio_info["explanation_dur"] + 0.40

    if audio_info.get("outro_mp3"):
        outro_clip = AudioFileClip(audio_info["outro_mp3"])
        opened.append(outro_clip)
        tracks.append(_with_start(outro_clip, t_cursor))

    return CompositeAudioClip(tracks), opened

# ==============================================================================
# PART 16 OF 20: FULL VIDEO STITCHER & H.264 MP4 EXPORTER
# ==============================================================================
import shutil

try:
    from moviepy import concatenate_videoclips
except ImportError:
    from moviepy.editor import concatenate_videoclips


def _set_clip_audio(vclip, aclip):
    return vclip.with_audio(aclip) if MOVIEPY_V2 else vclip.set_audio(aclip)


def render_complete_video(puzzles: list, mode: str, theme: dict, voice: str) -> tuple:
    """
    Renders all puzzle clips, mixes audio + optional low-volume bgm.mp3,
    exports output MP4, and returns (output_mp4_path, enriched_metadata_list).
    """
    work_dir = f"build_{mode}"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    spec = get_canvas_spec(mode)
    sprites = load_mascot_sprites(spec["mascot_size"])
    base_bg = prepare_base_background(mode, spec, theme)
    tick_path, chime_path = ensure_sfx_assets("assets")

    segment_clips = []
    all_opened_audio = []
    enriched_records = []
    total_puzzles = len(puzzles)

    for idx, puzzle in enumerate(puzzles):
        is_last = idx == (total_puzzles - 1)
        audio_info = prepare_puzzle_audio(
            puzzle=puzzle,
            idx=idx,
            mode=mode,
            voice=voice,
            work_dir=work_dir,
            include_outro=is_last,
        )

        vclip = build_puzzle_video_clip(
            puzzle=puzzle,
            audio_info=audio_info,
            puzzle_num=idx + 1,
            total_puzzles=total_puzzles,
            mode=mode,
            spec=spec,
            theme=theme,
            sprites=sprites,
            base_bg=base_bg,
        )

        aclip, opened_handles = build_puzzle_audio_track(audio_info, tick_path, chime_path)
        all_opened_audio.extend(opened_handles)
        segment_clips.append(_set_clip_audio(vclip, aclip))

        enriched_records.append({
            "question": puzzle["question"],
            "topic": puzzle["topic"],
            "difficulty": puzzle["difficulty"],
            "timer": audio_info["final_timer"],
            "theme": theme["name"],
            "voice": voice,
            "mode": mode,
            "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

    final_video = segment_clips[0] if len(segment_clips) == 1 else concatenate_videoclips(segment_clips, method="compose")
    output_mp4 = "short.mp4" if mode == "short" else "long_video.mp4"

    final_video.write_videofile(
        output_mp4,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    # Cleanly release all file handles
    final_video.close()
    for sc in segment_clips:
        sc.close()
    for ah in all_opened_audio:
        try:
            ah.close()
        except Exception:
            pass

    print(f"[Render] Exported {output_mp4} successfully.")
    return output_mp4, enriched_records

# ==============================================================================
# PART 17 OF 20: YOUTUBE DATA API V3 OAUTH2 CLIENT BUILDER
# ==============================================================================
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

def get_youtube_service():
    """Builds an authenticated YouTube Data API v3 client from YOUTUBE_TOKEN_JSON."""
    token_raw = os.environ.get("YOUTUBE_TOKEN_JSON")
    if not token_raw:
        raise EnvironmentError("YOUTUBE_TOKEN_JSON secret is missing.")

    token_info = json.loads(token_raw)
    creds = Credentials.from_authorized_user_info(token_info)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build("youtube", "v3", credentials=creds)

# ==============================================================================
# PART 18 OF 20: SEO METADATA & EXACT-SLOT PUBLISHAT YOUTUBE UPLOADER
# ==============================================================================
def upload_to_youtube(
    youtube,
    video_path: str,
    puzzles: list,
    records: list,
    mode: str,
    immediate: bool = False,
) -> str:
    """Uploads video to YouTube and schedules exact-slot publishAt (or publishes immediately)."""
    first = puzzles[0]
    first_rec = records[0]

    if mode == "short":
        title = f"{first['hook']} | Mental Math Trick #Shorts"[:98]
        description = (
            f"{first['question']}\n\n"
            f"Topic: {first['topic']}\n"
            f"Difficulty: {first['difficulty']} ({first_rec['timer']}s timer)\n\n"
            "Comment your exact solve time below! Subscribe to @MathMinuteMarvels for daily brain workouts.\n"
            "#Shorts #MentalMath #MathTricks #BrainTeaser #MathMinuteMarvels"
        )
        tags = ["Shorts", "mental math", "math tricks", first["topic"], "brain teaser", "fast math"]
    else:
        topics_list = ", ".join(dict.fromkeys(p["topic"] for p in puzzles))
        title = f"5 Mental Math Puzzles That Will Upgrade Your Brain ({topics_list})"[:98]
        puzzle_lines = "\n".join(
            f"Puzzle {i + 1} [{p['difficulty']} - {r['timer']}s]: {p['topic']}"
            for i, (p, r) in enumerate(zip(puzzles, records))
        )
        description = (
            "Can you solve all 5 progressive mental math challenges before the timer runs out?\n\n"
            f"{puzzle_lines}\n\n"
            "Drop your final score (out of 5) in the comments! Subscribe to @MathMinuteMarvels.\n"
            "#MentalMath #MathPuzzles #MathTricks #BrainTraining #MathMinuteMarvels"
        )
        tags = ["mental math", "math puzzles", "math tricks", "brain training", "math shortcuts"]

    status_body = {
        "selfDeclaredMadeForKids": False,
    }
    if immediate:
        status_body["privacyStatus"] = "public"
    else:
        status_body["privacyStatus"] = "private"
        status_body["publishAt"] = get_next_publish_at_utc(mode)

    body = {
        "snippet": {
            "title": title,
            "description": description,
            "tags": tags,
            "categoryId": "27",  # Education
            "defaultLanguage": "en",
        },
        "status": status_body,
    }

    media = MediaFileUpload(video_path, chunksize=5 * 1024 * 1024, resumable=True, mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body, media_body=media)

    response = None
    while response is None:
        _, response = request.next_chunk()

    video_id = response["id"]
    print(f"[YouTube] Uploaded video ID: {video_id} | Status: {status_body}")
    return video_id

# ==============================================================================
# PART 19 OF 20: AUTOMATIC FIRST-COMMENT ENGAGEMENT SEEDER
# ==============================================================================
def post_starter_comment(youtube, video_id: str, puzzles: list, mode: str):
    """Posts an engaging first comment to spark viewer replies (non-fatal if scope is upload-only)."""
    if mode == "short":
        comment_text = (
            f"Did you solve it before the {puzzles[0]['difficulty']} timer hit 0? "
            "Drop your time in seconds and tell us which shortcut you used!"
        )
    else:
        comment_text = (
            "How many out of 5 did you solve mentally before the timer ended? "
            "Comment your score (e.g., 4/5) below!"
        )

    try:
        youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {
                        "snippet": {"textOriginal": comment_text}
                    },
                }
            },
        ).execute()
        print(f"[YouTube] Posted starter engagement comment on {video_id}.")
    except Exception as err:
        print(f"[YouTube] Skipping starter comment (non-fatal): {err}")

# ==============================================================================
# PART 20 OF 20: MAIN PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    args = parse_args()
    print(f"=== MathMinuteMarvels Pipeline | mode={args.mode} | test={args.test} | immediate={args.immediate} ===")

    history_data = load_history()
    theme = pick_theme(history_data)
    voice = pick_voice(history_data)

    puzzles = generate_puzzles(mode=args.mode)
    video_path, enriched_records = render_complete_video(
        puzzles=puzzles,
        mode=args.mode,
        theme=theme,
        voice=voice,
    )

    if args.test:
        print(f"[Test Mode] Skipping YouTube upload & history commit. Local output ready at: {video_path}")
        return

    youtube = get_youtube_service()
    video_id = upload_to_youtube(
        youtube=youtube,
        video_path=video_path,
        puzzles=puzzles,
        records=enriched_records,
        mode=args.mode,
        immediate=args.immediate,
    )
    post_starter_comment(youtube, video_id, puzzles, args.mode)

    for rec in enriched_records:
        rec["video_id"] = video_id
    save_history_entries(enriched_records)
    print(f"[Done] Saved {len(enriched_records)} record(s) to {HISTORY_FILE} for video {video_id}.")


if __name__ == "__main__":
    main()
