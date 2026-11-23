# ==============================================================================
# PART 1 OF 30: STANDARD IMPORTS, OPTIONAL DEPENDENCIES & CLI ARGUMENTS
# ==============================================================================
import os
import re
import sys
import json
import time
import math
import wave
import random
import shutil
import asyncio
import argparse
import subprocess
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

try:
    import sympy as sp
    SYMPY_AVAILABLE = True
except ImportError:
    sp = None
    SYMPY_AVAILABLE = False

try:
    from moviepy import AudioFileClip, CompositeAudioClip, VideoClip, concatenate_videoclips
    MOVIEPY_V2 = True
except ImportError:
    from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoClip, concatenate_videoclips
    MOVIEPY_V2 = False


def parse_args():
    parser = argparse.ArgumentParser(description="MathMinuteMarvels 30-Part Autonomous Engine")
    parser.add_argument(
        "--mode",
        choices=["short", "long", "tune"],
        default="short",
        help="Render a 9:16 Short, a 16:9 Long video, or run Sunday analytics tuning",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Render locally without uploading to YouTube or writing history",
    )
    parser.add_argument(
        "--immediate",
        action="store_true",
        help="Publish immediately instead of scheduling for the next IST slot",
    )
    return parser.parse_args()

# ==============================================================================
# PART 2 OF 30: CHANNEL CONSTANTS, ASSET PATHS & SAFE SUBPROCESS RUNNER
# ==============================================================================
CHANNEL_NAME = "Math Minute Marvels"
CHANNEL_HANDLE = "@MathMinuteMarvels"

HISTORY_FILE = "history.json"
INSIGHTS_FILE = "weekly_insights.json"
ASSETS_DIR = "assets"

SHORT_SLOTS_IST = [2, 8, 14, 20]  # 02:00, 08:00, 14:00, 20:00 IST
LONG_SLOTS_IST = [19]             # 19:00 IST (Wed & Sat)

MASCOT_FILES = {
    "open": "char_open.png",
    "half": "char_half.png",
    "closed": "char_closed.png",
    "blink": "char_blink.png",
    "think": "char_think.png",
}


def run_cmd_safe(cmd: list, timeout_sec: int = 180) -> tuple:
    """Executes a subprocess command with timeout protection and captures output."""
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_sec,
            check=False,
        )
        return proc.returncode == 0, proc.stdout.strip(), proc.stderr.strip()
    except subprocess.TimeoutExpired:
        return False, "", f"Command timed out after {timeout_sec}s"
    except Exception as exc:
        return False, "", str(exc)

# ==============================================================================
# PART 3 OF 30: PERSISTENT STATE MEMORY (HISTORY & WEEKLY INSIGHTS)
# ==============================================================================
def load_history() -> list:
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history_entries(new_entries: list):
    history_data = load_history()
    history_data.extend(new_entries)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history_data[-600:], f, indent=2, ensure_ascii=False)


def get_past_questions(history_data: list, limit: int = 180) -> set:
    """Extracts normalized past question strings to prevent duplicate puzzles."""
    past = set()
    for entry in history_data[-limit:]:
        if isinstance(entry, str):
            past.add(entry.strip().lower())
        elif isinstance(entry, dict) and entry.get("question"):
            past.add(str(entry["question"]).strip().lower())
    return past


def load_weekly_insights() -> dict:
    default_insights = {
        "best_topics": [
            "Fast Mental Multiplication",
            "Square & Cube Shortcuts",
            "Number Sequence & Pattern Logic",
            "Algebraic Speed Tricks",
        ],
        "peak_hours_ist": [8, 14, 19, 20],
        "winning_hook_style": "challenge",
        "shorts_per_day": 4,
        "longs_per_week": 2,
        "last_tuned_utc": None,
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


def save_weekly_insights(insights: dict):
    with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

# ==============================================================================
# PART 4 OF 30: MULTI-ARMED BANDIT IST -> UTC SLOT SCHEDULER
# ==============================================================================
def get_next_publish_at_utc(mode: str, insights: dict = None) -> str:
    """
    Computes the next upcoming IST upload slot (~45-60 mins after cron triggers),
    applying 80% exploitation of proven peak hours and 20% exploration.
    """
    ist = ZoneInfo("Asia/Kolkata")
    now_ist = datetime.now(ist)
    insights = insights or load_weekly_insights()

    base_slots = list(SHORT_SLOTS_IST if mode == "short" else LONG_SLOTS_IST)
    peak_hours = [
        int(h) for h in insights.get("peak_hours_ist", [])
        if isinstance(h, int) and 0 <= h <= 23
    ]

    if mode == "short" and peak_hours and random.random() < 0.80:
        active_slots = sorted(set(base_slots) & set(peak_hours)) or base_slots
    else:
        active_slots = base_slots

    candidates = []
    for day_offset in (0, 1):
        base_date = (now_ist + timedelta(days=day_offset)).date()
        for hour in active_slots:
            slot_dt = datetime(base_date.year, base_date.month, base_date.day, hour, 0, 0, tzinfo=ist)
            if slot_dt > now_ist + timedelta(minutes=15):
                candidates.append(slot_dt)

    if not candidates:
        fallback_dt = now_ist + timedelta(hours=1)
        candidates.append(fallback_dt.replace(minute=0, second=0, microsecond=0))

    next_slot_ist = min(candidates)
    next_slot_utc = next_slot_ist.astimezone(timezone.utc)
    print(
        f"[Scheduler] Target IST: {next_slot_ist.strftime('%Y-%m-%d %I:%M %p IST')} "
        f"-> UTC: {next_slot_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"
    )
    return next_slot_utc.strftime("%Y-%m-%dT%H:%M:%SZ")

# ==============================================================================
# PART 5 OF 30: MONETIZATION-SAFE COLOR THEMES & NEURAL VOICE SELECTOR
# ==============================================================================
COLOR_THEMES = [
    {
        "name": "Cyber Cyan",
        "bg": (11, 17, 32),
        "card": (21, 32, 58),
        "accent": (0, 245, 212),
        "highlight": (255, 214, 10),
        "gold": (255, 209, 102),
        "text": (255, 255, 255),
        "muted": (160, 180, 210),
    },
    {
        "name": "Electric Violet",
        "bg": (19, 11, 35),
        "card": (36, 21, 66),
        "accent": (185, 131, 255),
        "highlight": (72, 245, 168),
        "gold": (255, 209, 102),
        "text": (255, 255, 255),
        "muted": (185, 168, 215),
    },
    {
        "name": "Solar Sky",
        "bg": (13, 23, 38),
        "card": (24, 42, 68),
        "accent": (56, 189, 248),
        "highlight": (251, 146, 60),
        "gold": (255, 214, 10),
        "text": (255, 255, 255),
        "muted": (165, 190, 218),
    },
    {
        "name": "Emerald Coral",
        "bg": (10, 26, 24),
        "card": (19, 46, 43),
        "accent": (52, 211, 153),
        "highlight": (251, 113, 133),
        "gold": (255, 209, 102),
        "text": (255, 255, 255),
        "muted": (162, 204, 196),
    },
]

TTS_VOICES = [
    "en-US-ChristopherNeural",
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-GB-RyanNeural",
]


def pick_theme(history_data: list) -> dict:
    chosen = COLOR_THEMES[len(history_data) % len(COLOR_THEMES)]
    print(f"[Theme] Selected visual palette: {chosen['name']}")
    return chosen


def pick_voice(history_data: list) -> str:
    chosen = TTS_VOICES[len(history_data) % len(TTS_VOICES)]
    print(f"[Audio] Selected neural voice: {chosen}")
    return chosen

# ==============================================================================
# PART 6 OF 30: CURRICULUM TREE, TOPIC LOCKOUT & DIFFICULTY ROTATOR
# ==============================================================================
CURRICULUM_TOPICS = [
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


def pick_curriculum_topics(history_data: list, count: int) -> list:
    """Selects `count` distinct math topics NOT used in the last 6 runs."""
    recent_topics = [
        str(item.get("topic", "")).strip().lower()
        for item in history_data[-6:]
        if isinstance(item, dict) and item.get("topic")
    ]
    fresh = [t for t in CURRICULUM_TOPICS if t.lower() not in recent_topics]
    if len(fresh) < count:
        fresh = CURRICULUM_TOPICS[:]
    random.shuffle(fresh)
    return fresh[:count]


def pick_short_difficulty(history_data: list) -> tuple:
    """Rotates EASY (15s), MEDIUM (25s), and HARD (35s) randomly without repeating the last run."""
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
    return chosen, DIFFICULTY_TIMERS[chosen]

# ==============================================================================
# PART 7 OF 30: SYMPY MATHEMATICAL EXPRESSION NORMALIZER
# ==============================================================================
def normalize_math_expr(raw: str) -> str:
    """Converts human-readable math notation (×, ÷, ², %, 'of') into SymPy syntax."""
    s = str(raw).strip()
    s = re.sub(r"=\s*\[?\s*\?\s*\]?$", "", s).strip()
    s = s.rstrip("?").strip()

    # Convert "X% of Y" -> "((X)/100)*(Y)"
    s = re.sub(
        r"(\d+(?:\.\d+)?)\s*%\s*of\s*(\d+(?:\.\d+)?)",
        r"((\1)/100)*(\2)",
        s,
        flags=re.IGNORECASE,
    )
    # Convert standalone percentages like "25%" -> "(25/100)"
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"((\1)/100)", s)

    # Convert unicode operators and superscripts
    s = s.replace("×", "*").replace("÷", "/").replace("−", "-").replace("–", "-")
    s = s.replace("²", "**2").replace("³", "**3").replace("^", "**")
    s = re.sub(r"√\s*\(([^)]+)\)", r"sqrt(\1)", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"sqrt(\1)", s)

    # Insert implicit multiplication (e.g., 2(3+4) -> 2*(3+4), 5x -> 5*x)
    s = re.sub(r"(\d)\s*\(", r"\1*(", s)
    s = re.sub(r"\)\s*\(", r")*(", s)
    s = re.sub(r"(\d)\s*([a-zA-Z])", r"\1*\2", s)
    return s.strip()

# ==============================================================================
# PART 8 OF 30: SYMPY MATHEMATICAL VERIFICATION GATE
# ==============================================================================
def verify_puzzle_with_sympy(puzzle: dict) -> tuple:
    """
    Verifies pure arithmetic/algebraic expressions using SymPy when parseable.
    Returns (is_valid: bool, reason: str). Word/sequence riddles pass gracefully.
    """
    if not SYMPY_AVAILABLE:
        return True, "SymPy not installed; skipping symbolic check"

    q_raw = str(puzzle.get("question", "")).strip()
    a_raw = str(puzzle.get("answer", "")).strip()

    # Skip pure word problems or comma-separated number sequences
    if "," in q_raw or len(re.findall(r"[a-zA-Z]{4,}", re.sub(r"\bof\b", "", q_raw, flags=re.I))) > 0:
        return True, "Word/sequence puzzle; passed structural check"

    lhs_str = normalize_math_expr(q_raw)
    rhs_str = normalize_math_expr(a_raw)

    if not lhs_str or not rhs_str or "=" in lhs_str:
        return True, "Non-standard equation format; passed"

    try:
        lhs_val = sp.sympify(lhs_str)
        rhs_val = sp.sympify(rhs_str)
        diff = sp.simplify(lhs_val - rhs_val)
        if diff == 0:
            return True, f"Verified: {lhs_str} == {rhs_str}"
        return False, f"Math mismatch: {lhs_str} evaluated to {sp.simplify(lhs_val)}, not {rhs_str}"
    except Exception as exc:
        return True, f"Symbolic parse skipped ({exc})"

# ==============================================================================
# PART 9 OF 30: GEMINI STRUCTURED SCRIPTWRITER (SINGLE INTRO & SINGLE OUTRO)
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

GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.1-flash-lite",
]


def build_gemini_prompt(
    mode: str,
    count: int,
    assigned_topics: list,
    difficulty_rule: str,
    past_questions: set,
) -> str:
    single_topic = next(iter(assigned_topics), "Fast Mental Multiplication")
    if mode == "short":
        topic_rule = (
            f'Create 1 puzzle strictly from this topic: "{single_topic}". '
            f'Include a single energetic opening intro in "hook" starting with "Welcome to Math Minute Marvels!" '
            f'and a single closing CTA in "outro".'
        )
    else:
        topic_rule = (
            f"Create {count} puzzles covering these exact {count} distinct topics in order: {json.dumps(assigned_topics)}. "
            f'ONLY Puzzle 1 ("index 0") may have a "hook" starting with "Welcome to Math Minute Marvels!". '
            f'Puzzles 2 through {count} MUST set "hook" to "" (empty string) so there is NO repeated intro between questions. '
            f'ONLY Puzzle {count} (the final puzzle) may have a closing "outro"; Puzzles 1 through {count - 1} MUST set "outro" to "".'
        )

    return f"""
You are an elite mental-math creator for {CHANNEL_NAME} ({mode} format).
{topic_rule}
{difficulty_rule}

Strict rules:
1. Return ONLY a valid JSON array containing exactly {count} object(s). No markdown fences.
2. Each object MUST include these exact keys:
   - "hook": string (ONLY populated for Puzzle 1; "" for Puzzles 2..{count})
   - "question": clean math expression or short problem using %, ×, ÷, =, - WITHOUT brackets (e.g., "83 × 47 + 83 × 53 = ?" or "88² - 12² = ?")
   - "topic": exact assigned topic name
   - "difficulty": "EASY", "MEDIUM", or "HARD"
   - "timer_seconds": integer (15 for EASY, 25 for MEDIUM, 35 for HARD)
   - "answer": exact final numerical or percentage answer
   - "explanation": 2 to 3 short equation-style steps with completed calculations (never leave "?" inside steps)
   - "outro": string (ONLY populated for the final puzzle; "" for earlier puzzles)
3. Double-check all arithmetic so every step and final answer is 100% mathematically exact.
4. NEVER repeat or closely paraphrase any of these recent questions:
{json.dumps(list(past_questions)[-50:], ensure_ascii=False)}
"""

# ==============================================================================
# PART 10 OF 30: PUZZLE GENERATOR WITH SCHEMA & SYMPY RETRY LOOP
# ==============================================================================
def generate_puzzles(mode: str = "short") -> list:
    """Generates and mathematically verifies 1 puzzle (short) or 5 puzzles (long)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY environment variable is missing.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30000),
    )
    history_data = load_history()
    past_questions = get_past_questions(history_data)
    count = 1 if mode == "short" else 5

    assigned_topics = pick_curriculum_topics(history_data, count)
    if mode == "short":
        chosen_diff, chosen_timer = pick_short_difficulty(history_data)
        difficulty_rule = f'You MUST set "difficulty" to "{chosen_diff}" and "timer_seconds" to {chosen_timer}.'
    else:
        difficulty_rule = (
            "Progress smoothly across the 5 puzzles: Puzzle 1-2 EASY (15s), Puzzle 3-4 MEDIUM (25s), Puzzle 5 HARD (35s)."
        )

    prompt = build_gemini_prompt(mode, count, assigned_topics, difficulty_rule, past_questions)
    max_attempts = len(GEMINI_FALLBACK_MODELS) + 2

    for attempt in range(1, max_attempts + 1):
        model_name = GEMINI_FALLBACK_MODELS[(attempt - 1) % len(GEMINI_FALLBACK_MODELS)]
        try:
            print(f"[Gemini Attempt {attempt}/{max_attempts}] Generating via {model_name}...")
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.80,
                ),
            )
            puzzles = json.loads((response.text or "").strip())
            if not isinstance(puzzles, list) or len(puzzles) != count:
                raise ValueError(f"Expected {count} puzzle(s), got {type(puzzles).__name__}")

            validated = []
            seen_batch = set()
            for idx, p in enumerate(puzzles):
                if not isinstance(p, dict) or not REQUIRED_PUZZLE_KEYS.issubset(p.keys()):
                    missing = REQUIRED_PUZZLE_KEYS - set(p.keys() if isinstance(p, dict) else [])
                    raise ValueError(f"Missing required keys: {missing}")

                q_clean = str(p["question"]).strip()
                q_norm = q_clean.lower()
                if not q_clean or q_norm in past_questions or q_norm in seen_batch:
                    raise ValueError(f"Duplicate or empty question: {q_clean}")

                ok_math, math_msg = verify_puzzle_with_sympy(p)
                if not ok_math:
                    raise ValueError(f"SymPy rejected puzzle {idx + 1}: {math_msg}")
                print(f"[SymPy] Puzzle {idx + 1}: {math_msg}")

                # Enforce single intro (idx == 0) and single outro (idx == count - 1)
                p["hook"] = str(p["hook"]).strip() if idx == 0 else ""
                p["outro"] = str(p["outro"]).strip() if idx == (count - 1) else ""

                diff = str(p["difficulty"]).strip().upper()
                p["difficulty"] = diff if diff in DIFFICULTY_TIMERS else "MEDIUM"
                p["timer_seconds"] = max(12, min(38, int(p["timer_seconds"])))

                seen_batch.add(q_norm)
                validated.append(p)

            print(f"[Gemini] Verified {len(validated)} unique {mode} puzzle(s).")
            return validated

        except Exception as err:
            print(f"[Gemini Attempt {attempt}/{max_attempts}] Failed: {err}")
            if attempt == max_attempts:
                raise RuntimeError("Exhausted all retries generating verified puzzles.") from err
            time.sleep(2 * attempt)

# ==============================================================================
# PART 11 OF 30: NATURAL MATH-TO-SPEECH TEXT NORMALIZER
# ==============================================================================
def clean_math_for_tts(text: str) -> str:
    """Converts math symbols (%, =, ×, ÷, -, ², ³, √) into natural spoken English."""
    s = str(text)
    s = re.sub(r"=\s*\[?\s*\?\s*\]?", " equals what?", s)
    s = re.sub(r"\[?\s*\?\s*\]?", "", s)
    s = re.sub(r"(\d+(?:\.\d+)?)\s*%", r"\1 percent", s)
    s = s.replace("%", " percent ")
    s = s.replace("²", " squared").replace("³", " cubed")
    s = re.sub(r"√\s*\(([^)]+)\)", r"square root of \1", s)
    s = re.sub(r"√\s*(\d+(?:\.\d+)?)", r"square root of \1", s)
    s = s.replace("×", " times ").replace("÷", " divided by ").replace("=", " equals ")
    s = re.sub(r"(?<=\w)\s*[-−–]\s*(?=\w)", " minus ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ==============================================================================
# PART 12 OF 30: EDGE-TTS VOICE SYNTHESIZER & PROCEDURAL TICK/CHIME SFX
# ==============================================================================
async def _synthesize_edge_tts_async(text: str, output_path: str, voice: str, rate: str = "+6%"):
    last_err = None
    for candidate_voice in [voice] + [v for v in TTS_VOICES if v != voice]:
        try:
            communicate = edge_tts.Communicate(text=text, voice=candidate_voice, rate=rate)
            await communicate.save(output_path)
            return
        except Exception as err:
            last_err = err
    raise RuntimeError(f"All Edge-TTS voices failed: {last_err}")


def create_voice_clip(text: str, output_path: str, voice: str, rate: str = "+6%") -> float:
    """Synthesizes speech to MP3 and returns its exact duration in seconds."""
    asyncio.run(_synthesize_edge_tts_async(text.strip(), output_path, voice, rate))
    clip = AudioFileClip(output_path)
    duration = float(clip.duration)
    clip.close()
    return duration


def ensure_sfx_assets(sfx_dir: str = ASSETS_DIR) -> tuple:
    """Generates crisp 1.0s clock tick and 0.6s reveal chime WAV files if missing."""
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
# PART 13 OF 30: PUZZLE AUDIO BUNDLE (SINGLE INTRO, TIMER CTA & SINGLE OUTRO)
# ==============================================================================
RANDOM_TIMER_CTAS = [
    "Your clock is ticking! Drop your answer in the comments, hit Like, and Subscribe!",
    "Can you beat the timer? Comment your solve time below and hit Like and Subscribe!",
    "Think fast! Smash Like, Subscribe to Math Minute Marvels, and comment your answer!",
    "Time is running! Comment your answer before zero and hit Like and Subscribe!",
]


def prepare_puzzle_audio(
    puzzle: dict,
    idx: int,
    mode: str,
    voice: str,
    work_dir: str,
    include_hook: bool = False,
    include_timer_cta: bool = False,
    include_outro: bool = False,
) -> dict:
    """
    Synthesizes distinct MP3 files for each stage:
    - hook_mp3: ONLY when include_hook is True (start of video)
    - question_mp3: Always synthesized
    - timer_cta_mp3: Synthesized ONLY when include_timer_cta is True (1 random slot during timer)
    - explanation_mp3: Always synthesized
    - outro_mp3: ONLY when include_outro is True (end of video)
    """
    os.makedirs(work_dir, exist_ok=True)
    hook_mp3 = os.path.join(work_dir, f"p{idx}_hook.mp3")
    q_mp3 = os.path.join(work_dir, f"p{idx}_question.mp3")
    timer_cta_mp3 = os.path.join(work_dir, f"p{idx}_timer_cta.mp3")
    exp_mp3 = os.path.join(work_dir, f"p{idx}_explanation.mp3")
    outro_mp3 = os.path.join(work_dir, f"p{idx}_outro.mp3")

    hook_dur = 0.0
    if include_hook and puzzle.get("hook"):
        hook_dur = create_voice_clip(clean_math_for_tts(puzzle["hook"]), hook_mp3, voice, rate="+8%")

    q_dur = create_voice_clip(clean_math_for_tts(puzzle["question"]), q_mp3, voice, rate="+4%")

    timer_cta_dur = 0.0
    timer_cta_text = ""
    if include_timer_cta:
        timer_cta_text = random.choice(RANDOM_TIMER_CTAS)
        timer_cta_dur = create_voice_clip(timer_cta_text, timer_cta_mp3, voice, rate="+9%")

    exp_text = clean_math_for_tts(f"The answer is {puzzle['answer']}. {puzzle['explanation']}")
    exp_dur = create_voice_clip(exp_text, exp_mp3, voice, rate="+5%")

    outro_dur = 0.0
    if include_outro and puzzle.get("outro"):
        outro_dur = create_voice_clip(clean_math_for_tts(puzzle["outro"]), outro_mp3, voice, rate="+7%")

    # Ensure Shorts total length stays strictly <= 58.0s
    non_timer_sec = (
        ((hook_dur + 0.30) if hook_dur > 0 else 0.0)
        + (q_dur + 0.30)
        + (0.15 + exp_dur + 0.35)
        + ((outro_dur + 0.25) if outro_dur > 0 else 0.0)
    )
    requested_timer = max(int(puzzle["timer_seconds"]), int(math.ceil(timer_cta_dur + 2.0)))

    if mode == "short":
        max_allowed_timer = max(10, int(math.floor(58.0 - non_timer_sec)))
        final_timer = min(requested_timer, max_allowed_timer)
    else:
        final_timer = requested_timer

    return {
        "hook_mp3": hook_mp3 if hook_dur > 0 else None,
        "hook_dur": hook_dur,
        "hook_text": puzzle.get("hook", "") if hook_dur > 0 else "",
        "question_mp3": q_mp3,
        "question_dur": q_dur,
        "question_text": puzzle.get("question", ""),
        "timer_cta_mp3": timer_cta_mp3 if timer_cta_dur > 0 else None,
        "timer_cta_dur": timer_cta_dur,
        "timer_cta_text": timer_cta_text,
        "explanation_mp3": exp_mp3,
        "explanation_dur": exp_dur,
        "explanation_text": f"The answer is {puzzle['answer']}. {puzzle['explanation']}",
        "outro_mp3": outro_mp3 if outro_dur > 0 else None,
        "outro_dur": outro_dur,
        "outro_text": puzzle.get("outro", "") if outro_dur > 0 else "",
        "final_timer": final_timer,
    }

# ==============================================================================
# PART 14 OF 30: SINGLE MASTER TIMELINE BUILDER (ZERO AUDIO/VIDEO DRIFT)
# ==============================================================================
def build_puzzle_timeline(audio_info: dict) -> dict:
    """
    Computes ONE authoritative timestamp map used identically by the frame renderer,
    audio mixer, lip-sync engine, and ASS caption generator—eliminating any drift.
    """
    t = 0.0
    if audio_info["hook_dur"] > 0:
        hook_start = t
        hook_end = hook_start + audio_info["hook_dur"]
        t = hook_end + 0.30
    else:
        hook_start = hook_end = 0.0

    q_start = t
    q_end = q_start + audio_info["question_dur"]
    t = q_end + 0.30

    timer_start = t
    timer_end = timer_start + float(audio_info["final_timer"])
    # Randomize when the timer CTA plays inside the countdown window (at least 1.5s after timer starts)
    if audio_info["timer_cta_dur"] > 0:
        slack = max(0.5, float(audio_info["final_timer"]) - audio_info["timer_cta_dur"] - 1.5)
        cta_offset = round(random.uniform(1.0, min(4.0, slack)), 2)
        cta_start = timer_start + cta_offset
        cta_end = cta_start + audio_info["timer_cta_dur"]
    else:
        cta_start = cta_end = 0.0

    t = timer_end
    chime_start = t
    exp_start = chime_start + 0.15
    exp_end = exp_start + audio_info["explanation_dur"]
    t = exp_end + 0.35

    if audio_info["outro_dur"] > 0:
        outro_start = t
        outro_end = outro_start + audio_info["outro_dur"]
        t = outro_end + 0.25
    else:
        outro_start = outro_end = 0.0

    return {
        "hook_start": hook_start,
        "hook_end": hook_end,
        "q_start": q_start,
        "q_end": q_end,
        "timer_start": timer_start,
        "timer_end": timer_end,
        "cta_start": cta_start,
        "cta_end": cta_end,
        "chime_start": chime_start,
        "exp_start": exp_start,
        "exp_end": exp_end,
        "outro_start": outro_start,
        "outro_end": outro_end,
        "total_dur": round(t, 3),
    }

# ==============================================================================
# PART 15 OF 30: WORD-SYNCED KINETIC YELLOW .ASS CAPTION GENERATOR
# ==============================================================================
def _format_ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def build_ass_subtitles_file(
    segments_with_offsets: list,
    ass_path: str,
    mode: str,
) -> str:
    """
    Generates an Advanced SubStation Alpha (.ass) subtitle file with punchy
    kinetic yellow karaoke word highlighting synced to every spoken voice segment.
    """
    play_res_x = 1080 if mode == "short" else 1920
    play_res_y = 1920 if mode == "short" else 1080
    font_size = 54 if mode == "short" else 46
    margin_v = 250 if mode == "short" else 55

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {play_res_x}
PlayResY: {play_res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,{font_size},&H0000D6FF,&H00FFFFFF,&H00101010,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,60,60,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    for text, start_t, end_t in segments_with_offsets:
        clean = clean_math_for_tts(text)
        words = clean.split()
        if not words or end_t <= start_t:
            continue

        # Split into chunks of up to 6 words for clean readability
        chunk_size = 6
        chunks = [words[i : i + chunk_size] for i in range(0, len(words), chunk_size)]
        total_words = len(words)
        cursor = start_t

        for chunk in chunks:
            chunk_dur = (end_t - start_t) * (len(chunk) / total_words)
            cs_per_word = max(8, int(round((chunk_dur / len(chunk)) * 100)))
            karaoke_text = "".join(f"{{\\k{cs_per_word}}}{w} " for w in chunk).strip()
            events.append(
                f"Dialogue: 0,{_format_ass_time(cursor)},{_format_ass_time(cursor + chunk_dur)},Default,,0,0,0,,{karaoke_text}"
            )
            cursor += chunk_dur

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")
    return ass_path

# ==============================================================================
# PART 16 OF 30: FONT LOADER & EXACT BG.PNG / BG_LONG.PNG SAFE-ZONE COORDINATES
# ==============================================================================
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "assets/font_bold.ttf",
    "arialbd.ttf",
]

MATH_SYMBOLS = ["%", "+", "=", "×", "÷", "!"]


def load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def get_canvas_spec(mode: str) -> dict:
    """
    Locks exact safe-zone coordinates matching your uploaded bg.png (9:16) and
    bg_long.png (16:9) so the infinity-stopwatch logo and mascot never overlap.
    """
    if mode == "short":
        return {
            "width": 1080,
            "height": 1920,
            "pad": 56,
            # Top logo sits at y=0..560 -> header & card start cleanly at y=585
            "header_box": (56, 585, 1024, 653),
            "card_top": 670,
            "card_bottom": 1315,
            # Bottom mascot sits at y=1485, handle at y=1645 (above YouTube bottom UI)
            "mascot_cx": 540,
            "mascot_cy": 1485,
            "handle_y": 1642,
            "mascot_size": 430,
            "font_badge": load_font(33),
            "font_hook": load_font(54),
            "font_question": load_font(72),
            "font_timer": load_font(116),
            "font_answer": load_font(76),
            "font_step": load_font(34),
            "font_outro": load_font(26),
            "font_symbol_bg": load_font(62),
            "font_symbol_orbit": load_font(46),
        }
    return {
        "width": 1920,
        "height": 1080,
        "pad": 80,
        # Top-left logo sits at x=0..560, y=0..330 -> header pill sits to its right
        "header_box": (590, 95, 1840, 165),
        "card_top": 345,
        "card_bottom": 810,
        # Mascot sits cleanly in the lower center-right below card_bottom (y=925)
        "mascot_cx": 960,
        "mascot_cy": 925,
        "handle_y": 1038,
        "mascot_size": 250,
        "font_badge": load_font(31),
        "font_hook": load_font(48),
        "font_question": load_font(64),
        "font_timer": load_font(102),
        "font_answer": load_font(68),
        "font_step": load_font(32),
        "font_outro": load_font(26),
        "font_symbol_bg": load_font(56),
        "font_symbol_orbit": load_font(40),
    }

# ==============================================================================
# PART 17 OF 30: CUSTOM BACKGROUND LOADER (BG.PNG & BG_LONG.PNG)
# ==============================================================================
def prepare_base_background(mode: str, spec: dict, theme: dict) -> Image.Image:
    """
    Loads bg.png for Shorts (1080x1920) or bg_long.png for Long videos (1920x1080)
    and preserves full logo brightness with a minimal 10% theme blend.
    """
    w, h = spec["width"], spec["height"]
    tint_layer = Image.new("RGB", (w, h), theme["bg"])

    primary = "bg.png" if mode == "short" else "bg_long.png"
    fallback = "bg_long.png" if mode == "short" else "bg.png"
    found_bg = next((p for p in [primary, fallback] if os.path.exists(p)), None)

    if not found_bg:
        print("[Background] Custom bg not found; using solid theme canvas.")
        return tint_layer

    print(f"[Background] Loaded {found_bg} for mode={mode}")
    raw = Image.open(found_bg).convert("RGB")
    scale = max(w / raw.width, h / raw.height)
    new_w, new_h = int(raw.width * scale), int(raw.height * scale)
    resized = raw.resize((new_w, new_h), Image.Resampling.LANCZOS)

    left = (new_w - w) // 2
    top = (new_h - h) // 2
    cropped = resized.crop((left, top, left + w, top + h))
    return Image.blend(cropped, tint_layer, alpha=0.10)

# ==============================================================================
# PART 18 OF 30: AMBIENT FLOATING MATH SYMBOLS LAYER
# ==============================================================================
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

    count = 14
    for i in range(count):
        sym = MATH_SYMBOLS[i % len(MATH_SYMBOLS)]
        col_rgb = theme["accent"] if i % 2 == 0 else theme["highlight"]
        x_base = int(((i * 0.6180339) % 1.0) * (w - 100)) + 50
        x_wobble = int(math.sin(t_sec * 1.6 + i * 1.3) * 24)
        speed = 38 + (i % 4) * 12
        y_pos = int((h - ((t_sec * speed + i * (h / count)) % (h + 120))) - 60)

        alpha = 0.20 + 0.08 * math.sin(t_sec * 2.4 + i)
        blended_col = _blend_rgb(col_rgb, theme["bg"], alpha)
        draw.text((x_base + x_wobble, y_pos), sym, font=spec["font_symbol_bg"], fill=blended_col)

# ==============================================================================
# PART 19 OF 30: PERFECTED 5-SPRITE MASCOT STATE MACHINE
# ==============================================================================
def load_mascot_sprites(target_size: int) -> dict:
    """Loads all 5 character sprites: open, half, closed, blink, think."""
    sprites = {}
    for key, filename in MASCOT_FILES.items():
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
    Exact mascot behavior rules:
    1. Countdown stage: ALWAYS holds `char_think.png` (no lip-sync) even when the
       random Like/Subscribe/Comment timer voiceover is playing.
    2. Non-countdown with active voiceover (`is_speaking_now`): Crisp 3-stage lip-sync
       (`open` -> `half` -> `closed`) driven strictly by audio energy.
    3. Non-countdown silent pauses (`not is_speaking_now`): Animated conversational
       gestures (subtle `half`/`closed`/`blink` chatter & lively bobbing) so the
       mascot stays expressive and never looks frozen.
    """
    if stage == "countdown":
        pose = "think"
        bounce_y = int(math.sin(t_sec * 3.0) * 7)
    elif is_speaking_now:
        cycle = int(t_sec * 11.0) % 4
        if cycle == 0:
            pose = "open"
        elif cycle in (1, 3):
            pose = "half" if sprites.get("half") is not None else "open"
        else:
            pose = "closed"
        bounce_y = -int(abs(math.sin(t_sec * 8.5)) * 9)
    else:
        # Expressive idle animation with periodic blinks and subtle mouth movement
        blink_phase = t_sec % 3.2
        chatter_phase = t_sec % 1.8
        if blink_phase < 0.16 and sprites.get("blink") is not None:
            pose = "blink"
        elif chatter_phase < 0.35 and sprites.get("half") is not None:
            pose = "half"
        else:
            pose = "closed"
        bounce_y = int(math.sin(t_sec * 3.6) * 6)

    draw = ImageDraw.Draw(canvas)
    orbit_radius_x = int(spec["mascot_size"] * 0.66)
    orbit_radius_y = int(spec["mascot_size"] * 0.20)
    for idx, sym in enumerate(MATH_SYMBOLS):
        angle = t_sec * 1.9 + (idx * (2 * math.pi / len(MATH_SYMBOLS)))
        sx = int(cx + math.cos(angle) * orbit_radius_x)
        sy = int(cy + math.sin(angle) * orbit_radius_y + bounce_y // 2)
        sym_color = theme["highlight"] if idx % 2 == 0 else theme["accent"]
        draw.text((sx - 12, sy - 18), sym, font=spec["font_symbol_orbit"], fill=sym_color)

    sprite = sprites.get(pose) or sprites.get("closed") or sprites.get("open")
    if sprite is not None:
        px = int(cx - sprite.width // 2)
        py = int(cy - sprite.height // 2 + bounce_y)
        canvas.paste(sprite, (px, py), sprite)

# ==============================================================================
# PART 20 OF 30: TRANSLUCENT GLASS CARD, TEXT WRAPPER & COUNTDOWN BAR
# ==============================================================================
def draw_rounded_card(
    canvas_or_draw,
    box: tuple,
    radius: int,
    fill: tuple,
    outline: tuple,
    width: int = 3,
    fill_alpha: int = 185,
):
    """Draws a translucent glass card when passed an Image, or solid when passed ImageDraw."""
    if isinstance(canvas_or_draw, Image.Image):
        x0, y0, x1, y1 = [int(v) for v in box]
        cw, ch = max(1, x1 - x0), max(1, y1 - y0)
        overlay = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.rounded_rectangle(
            (0, 0, cw - 1, ch - 1),
            radius=radius,
            fill=(*fill, fill_alpha),
            outline=(*outline, 255),
            width=width,
        )
        canvas_or_draw.paste(overlay, (x0, y0), overlay)
    else:
        canvas_or_draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def wrap_lines(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list:
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


def draw_centered_multiline(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.FreeTypeFont,
    center_x: int,
    start_y: int,
    max_width: int,
    fill: tuple,
    line_spacing: int = 12,
) -> int:
    lines = wrap_lines(draw, text, font, max_width)
    y = start_y
    for line in lines:
        x0, y0, x1, y1 = draw.textbbox((0, 0), line, font=font)
        draw.text((center_x - (x1 - x0) // 2, y), line, font=font, fill=fill)
        y += (y1 - y0) + line_spacing
    return y


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
    ratio = max(0.0, min(1.0, remaining_sec / max(1, total_sec)))
    active_color = theme["highlight"] if remaining_sec <= 4.0 else theme["accent"]

    left = center_x - bar_width // 2
    right = center_x + bar_width // 2
    bottom = top_y + bar_height

    draw.rounded_rectangle(
        (left, top_y, right, bottom),
        radius=bar_height // 2,
        fill=theme["bg"],
        outline=theme["muted"],
        width=2,
    )
    fill_w = int(bar_width * ratio)
    if fill_w > bar_height:
        draw.rounded_rectangle(
            (left, top_y, left + fill_w, bottom),
            radius=bar_height // 2,
            fill=active_color,
        )

# ==============================================================================
# PART 21 OF 30: SAFE-ZONE FRAME COMPOSITOR (SINGLE-TIMELINE DRIVEN)
# ==============================================================================
def _split_explanation_steps(explanation: str) -> list:
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", str(explanation).strip()) if s.strip()]
    return parts if parts else [str(explanation).strip()]


def render_puzzle_frame(
    t_sec: float,
    puzzle: dict,
    audio_info: dict,
    timeline: dict,
    puzzle_num: int,
    total_puzzles: int,
    mode: str,
    spec: dict,
    theme: dict,
    sprites: dict,
    base_bg: Image.Image = None,
    is_speaking_now: bool = False,
) -> np.ndarray:
    """Renders one RGB frame synced strictly to `timeline` with zero logo/mascot overlap."""
    w, h, pad = spec["width"], spec["height"], spec["pad"]
    canvas = base_bg.copy() if base_bg is not None else Image.new("RGB", (w, h), theme["bg"])

    draw_floating_math_symbols(canvas, spec, theme, t_sec)
    draw = ImageDraw.Draw(canvas)

    # Determine current visual stage from the single master timeline
    if timeline["hook_end"] > 0 and t_sec < timeline["q_start"]:
        stage = "hook"
        stage_elapsed = t_sec
    elif t_sec < timeline["timer_start"]:
        stage = "question"
        stage_elapsed = t_sec - timeline["q_start"]
    elif t_sec < timeline["timer_end"]:
        stage = "countdown"
        stage_elapsed = t_sec - timeline["timer_start"]
    elif timeline["outro_end"] > 0 and t_sec >= timeline["outro_start"]:
        stage = "outro"
        stage_elapsed = t_sec - timeline["outro_start"]
    else:
        stage = "reveal"
        stage_elapsed = t_sec - timeline["timer_end"]

    ease = min(1.0, max(0.0, stage_elapsed) / 0.25)
    slide_y = int(((1.0 - ease) ** 3) * 24)

    # Header pill positioned cleanly in the safe zone below/beside the logo
    hx0, hy0, hx1, hy1 = spec["header_box"]
    label_prefix = f"PUZZLE {puzzle_num}/{total_puzzles}  •  " if total_puzzles > 1 else ""
    header_text = f"{label_prefix}{puzzle['topic'].upper()}  •  {puzzle['difficulty']}"
    draw_rounded_card(canvas, (hx0, hy0, hx1, hy1), radius=24, fill=theme["card"], outline=theme["accent"], width=2, fill_alpha=200)
    draw_centered_multiline(draw, header_text, spec["font_badge"], (hx0 + hx1) // 2, hy0 + 16, (hx1 - hx0) - 36, theme["accent"])

    # Main translucent glass card
    card_top = spec["card_top"] + slide_y
    card_bottom = spec["card_bottom"]
    draw_rounded_card(canvas, (pad, card_top, w - pad, card_bottom), radius=34, fill=theme["card"], outline=theme["accent"], width=3, fill_alpha=185)

    inner_w = w - pad * 2 - 64
    cx = w // 2

    if stage == "hook":
        draw_centered_multiline(draw, "⚡ QUICK BRAIN CHECK ⚡", spec["font_badge"], cx, card_top + 60, inner_w, theme["highlight"])
        draw_centered_multiline(draw, puzzle["hook"], spec["font_hook"], cx, card_top + 150, inner_w, theme["text"])
        draw_centered_multiline(draw, "%   +   =   ×   ÷   !", spec["font_hook"], cx, card_bottom - 105, inner_w, theme["accent"])
    else:
        # Question box
        q_lines = wrap_lines(draw, puzzle["question"], spec["font_question"], inner_w - 36)
        q_box_h = max(110, len(q_lines) * 74 + 36)
        q_box_top = card_top + 20
        q_box_bottom = q_box_top + q_box_h
        draw_rounded_card(draw, (pad + 22, q_box_top, w - pad - 22, q_box_bottom), radius=22, fill=theme["bg"], outline=theme["muted"], width=2)
        draw_centered_multiline(draw, puzzle["question"], spec["font_question"], cx, q_box_top + 18, inner_w - 36, theme["text"])

        if stage == "countdown":
            rem = max(0.0, timeline["timer_end"] - t_sec)
            total_t = max(1, audio_info["final_timer"])
            ratio = rem / total_t
            urgent = rem <= 5.0
            ring_color = theme["highlight"] if urgent else theme["accent"]

            ring_cy = (q_box_bottom + card_bottom - 65) // 2
            ring_r = 128 if mode == "short" else 92
            pulse_r = ring_r + (int(abs(math.sin(t_sec * 6.0)) * 9) if urgent else 0)
            bbox = (cx - pulse_r, ring_cy - pulse_r, cx + pulse_r, ring_cy + pulse_r)
            draw.ellipse(bbox, outline=theme["bg"], width=14)
            draw.arc(bbox, start=-90, end=-90 + int(360 * ratio), fill=ring_color, width=14)

            draw_centered_multiline(draw, f"{int(math.ceil(rem))}s", spec["font_timer"], cx, ring_cy - 58, inner_w, ring_color)
            draw_countdown_bar(draw, cx, card_bottom - 58, int(inner_w * 0.85), 22, rem, total_t, theme)

        elif stage in ("reveal", "outro"):
            ans_top = q_box_bottom + 16
            ans_bottom = ans_top + (98 if mode == "short" else 82)
            draw_rounded_card(draw, (pad + 22, ans_top, w - pad - 22, ans_bottom), radius=20, fill=theme["bg"], outline=theme["accent"], width=3)
            draw_centered_multiline(draw, f"ANSWER: {puzzle['answer']}", spec["font_answer"], cx, ans_top + 12, inner_w - 36, theme["accent"])

            steps = _split_explanation_steps(puzzle["explanation"])
            exp_dur = max(0.5, audio_info["explanation_dur"])
            exp_progress = min(0.999, max(0.0, (t_sec - timeline["exp_start"]) / exp_dur))
            visible_count = len(steps) if stage == "outro" else min(len(steps), int(exp_progress * len(steps)) + 1)

            step_font = spec["font_step"]
            step_y = ans_bottom + 14
            outro_reserve = 108 if (stage == "outro" and puzzle.get("outro")) else 14

            for s_idx in range(visible_count):
                step_label = f"Step {s_idx + 1}: {steps[s_idx]}"
                wrapped = wrap_lines(draw, step_label, step_font, inner_w - 48)
                dyn_box_h = len(wrapped) * 40 + 26
                if step_y + dyn_box_h > card_bottom - outro_reserve:
                    break

                is_active = (stage == "reveal") and (s_idx == visible_count - 1)
                step_bg = theme["bg"] if is_active else theme["card"]
                step_border = theme["highlight"] if is_active else theme["muted"]
                step_text_col = theme["highlight"] if is_active else theme["text"]

                draw_rounded_card(draw, (pad + 22, step_y, w - pad - 22, step_y + dyn_box_h), radius=16, fill=step_bg, outline=step_border, width=2)
                draw_centered_multiline(draw, step_label, step_font, cx, step_y + 12, inner_w - 48, step_text_col, line_spacing=8)
                step_y += dyn_box_h + 10

            if stage == "outro" and puzzle.get("outro"):
                box_top = card_bottom - 98
                box_bot = card_bottom - 14
                draw_rounded_card(draw, (pad + 22, box_top, w - pad - 22, box_bot), radius=18, fill=theme["bg"], outline=theme["accent"], width=2)
                draw_centered_multiline(draw, puzzle["outro"], spec["font_outro"], cx, box_top + 12, inner_w - 56, fill=theme["gold"])

    # Mascot & channel handle placed in dedicated lower safe zone
    paste_mascot_with_symbols(
        canvas,
        sprites,
        stage,
        is_speaking_now,
        spec["mascot_cx"],
        spec["mascot_cy"],
        t_sec,
        spec,
        theme,
    )
    draw_centered_multiline(draw, CHANNEL_HANDLE, spec["font_badge"], spec["mascot_cx"], spec["handle_y"], inner_w, theme["muted"])

    return np.array(canvas)

# ==============================================================================
# PART 22 OF 30: PER-FRAME AUDIO ENERGY LIP-SYNC EXTRACTOR
# ==============================================================================
def extract_speech_activity(audio_info: dict, timeline: dict, fps: int = 24) -> np.ndarray:
    """
    Computes per-frame boolean speech activity strictly for non-countdown voiceovers
    (hook, question, explanation, outro). Countdown CTA audio intentionally does NOT
    trigger lip-sync so the mascot stays in `char_think.png` during the timer.
    """
    total_dur = timeline["total_dur"]
    n_frames = max(1, int(math.ceil(total_dur * fps)) + 2)
    activity = np.zeros(n_frames, dtype=bool)

    segments = [
        (audio_info.get("hook_mp3"), timeline["hook_start"]),
        (audio_info.get("question_mp3"), timeline["q_start"]),
        (audio_info.get("explanation_mp3"), timeline["exp_start"]),
        (audio_info.get("outro_mp3"), timeline["outro_start"]),
    ]

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

# ==============================================================================
# PART 23 OF 30: 24 FPS VIDEOCLIP BUILDER (SYNCED TO MASTER TIMELINE)
# ==============================================================================
def build_puzzle_video_clip(
    puzzle: dict,
    audio_info: dict,
    timeline: dict,
    puzzle_num: int,
    total_puzzles: int,
    mode: str,
    spec: dict,
    theme: dict,
    sprites: dict,
    base_bg: Image.Image = None,
) -> VideoClip:
    fps = 24
    total_dur = timeline["total_dur"]
    speech_activity = extract_speech_activity(audio_info, timeline, fps=fps)

    def make_frame(t):
        f_idx = min(len(speech_activity) - 1, max(0, int(t * fps)))
        is_speaking_now = bool(speech_activity[f_idx])
        return render_puzzle_frame(
            t_sec=t,
            puzzle=puzzle,
            audio_info=audio_info,
            timeline=timeline,
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
# PART 24 OF 30: MASTER-TIMELINE AUDIO MIXER
# ==============================================================================
def _with_start(clip, start_t: float):
    return clip.with_start(start_t) if MOVIEPY_V2 else clip.set_start(start_t)


def build_puzzle_audio_track(
    audio_info: dict,
    timeline: dict,
    tick_path: str,
    chime_path: str,
) -> tuple:
    """Places every voiceover, clock tick, and chime at its exact `timeline` timestamp."""
    tracks = []
    opened = []

    def _add(path: str, start_t: float):
        if path and os.path.exists(path):
            c = AudioFileClip(path)
            opened.append(c)
            tracks.append(_with_start(c, start_t))

    _add(audio_info.get("hook_mp3"), timeline["hook_start"])
    _add(audio_info.get("question_mp3"), timeline["q_start"])
    _add(audio_info.get("timer_cta_mp3"), timeline["cta_start"])

    # 1 clock tick per second across the exact countdown window
    timer_secs = int(audio_info["final_timer"])
    for sec_idx in range(timer_secs):
        _add(tick_path, timeline["timer_start"] + sec_idx)

    # Reveal chime + explanation at exact timer_end
    _add(chime_path, timeline["chime_start"])
    _add(audio_info.get("explanation_mp3"), timeline["exp_start"])
    _add(audio_info.get("outro_mp3"), timeline["outro_start"])

    return CompositeAudioClip(tracks), opened

# ==============================================================================
# PART 25 OF 30: FFMPEG KINETIC YELLOW .ASS SUBTITLE BURNER
# ==============================================================================
def burn_ass_subtitles_ffmpeg(raw_mp4: str, ass_path: str, final_mp4: str) -> str:
    """Burns word-synced yellow ASS captions onto the rendered video via FFmpeg."""
    if not os.path.exists(ass_path):
        shutil.copyfile(raw_mp4, final_mp4)
        return final_mp4

    escaped_ass = ass_path.replace("\\", "/").replace(":", "\\:")
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        raw_mp4,
        "-vf",
        f"ass={escaped_ass}",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "copy",
        "-movflags",
        "+faststart",
        final_mp4,
    ]
    ok, _, err = run_cmd_safe(cmd, timeout_sec=240)
    if ok and os.path.exists(final_mp4):
        print(f"[Captions] Burned kinetic yellow ASS subtitles -> {final_mp4}")
        return final_mp4

    print(f"[Captions] FFmpeg ASS burn fallback (using clean render): {err[:160]}")
    shutil.copyfile(raw_mp4, final_mp4)
    return final_mp4

# ==============================================================================
# PART 26 OF 30: COMPLETE VIDEO STITCHER, CHAPTER COLLECTOR & .ASS BURNER
# ==============================================================================
def _set_clip_audio(vclip, aclip):
    return vclip.with_audio(aclip) if MOVIEPY_V2 else vclip.set_audio(aclip)


def render_complete_video(puzzles: list, mode: str, theme: dict, voice: str) -> tuple:
    """
    Renders all puzzles using one authoritative timeline per segment:
    - Single intro at Puzzle 1 (`idx == 0`)
    - Single random Like/Subscribe/Comment promo during the timer (1 slot per video)
    - Single outro at the final puzzle (`idx == total_puzzles - 1`)
    - Collects exact 00:00 chapters and burns kinetic yellow ASS captions.
    Returns (final_mp4_path, enriched_records, chapter_markers).
    """
    work_dir = f"build_{mode}"
    if os.path.exists(work_dir):
        shutil.rmtree(work_dir)
    os.makedirs(work_dir, exist_ok=True)

    spec = get_canvas_spec(mode)
    sprites = load_mascot_sprites(spec["mascot_size"])
    base_bg = prepare_base_background(mode, spec, theme)
    tick_path, chime_path = ensure_sfx_assets(ASSETS_DIR)

    total_puzzles = len(puzzles)
    # Pick 1 random puzzle slot to play the Like/Subscribe/Comment CTA during its countdown
    cta_slot_idx = random.randrange(total_puzzles)

    segment_clips = []
    all_opened_audio = []
    enriched_records = []
    chapter_markers = []
    caption_segments = []
    global_offset = 0.0

    for idx, puzzle in enumerate(puzzles):
        is_first = idx == 0
        is_last = idx == (total_puzzles - 1)
        has_timer_cta = idx == cta_slot_idx

        audio_info = prepare_puzzle_audio(
            puzzle=puzzle,
            idx=idx,
            mode=mode,
            voice=voice,
            work_dir=work_dir,
            include_hook=is_first,
            include_timer_cta=has_timer_cta,
            include_outro=is_last,
        )
        timeline = build_puzzle_timeline(audio_info)

        # Record clickable chapter timestamp for Long-Form description
        mins = int(global_offset // 60)
        secs = int(global_offset % 60)
        chapter_markers.append(
            f"{mins:02d}:{secs:02d} Puzzle {idx + 1}: {puzzle['topic']} ({puzzle['difficulty']})"
        )

        # Collect spoken caption intervals shifted by global_offset
        if audio_info["hook_dur"] > 0:
            caption_segments.append(
                (audio_info["hook_text"], global_offset + timeline["hook_start"], global_offset + timeline["hook_end"])
            )
        caption_segments.append(
            (audio_info["question_text"], global_offset + timeline["q_start"], global_offset + timeline["q_end"])
        )
        if audio_info["timer_cta_dur"] > 0:
            caption_segments.append(
                (audio_info["timer_cta_text"], global_offset + timeline["cta_start"], global_offset + timeline["cta_end"])
            )
        caption_segments.append(
            (audio_info["explanation_text"], global_offset + timeline["exp_start"], global_offset + timeline["exp_end"])
        )
        if audio_info["outro_dur"] > 0:
            caption_segments.append(
                (audio_info["outro_text"], global_offset + timeline["outro_start"], global_offset + timeline["outro_end"])
            )

        vclip = build_puzzle_video_clip(
            puzzle=puzzle,
            audio_info=audio_info,
            timeline=timeline,
            puzzle_num=idx + 1,
            total_puzzles=total_puzzles,
            mode=mode,
            spec=spec,
            theme=theme,
            sprites=sprites,
            base_bg=base_bg,
        )
        aclip, opened_handles = build_puzzle_audio_track(audio_info, timeline, tick_path, chime_path)
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
        global_offset += timeline["total_dur"]

    final_video = (
        segment_clips[0]
        if len(segment_clips) == 1
        else concatenate_videoclips(segment_clips, method="compose")
    )

    raw_mp4 = os.path.join(work_dir, f"raw_{mode}.mp4")
    output_mp4 = "short.mp4" if mode == "short" else "long_video.mp4"

    final_video.write_videofile(
        raw_mp4,
        fps=30,
        codec="libx264",
        audio_codec="aac",
        audio_bitrate="192k",
        preset="medium",
        ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )

    final_video.close()
    for sc in segment_clips:
        sc.close()
    for ah in all_opened_audio:
        try:
            ah.close()
        except Exception:
            pass

    ass_path = os.path.join(work_dir, f"captions_{mode}.ass")
    build_ass_subtitles_file(caption_segments, ass_path, mode)
    burn_ass_subtitles_ffmpeg(raw_mp4, ass_path, output_mp4)

    print(f"[Render] Exported {output_mp4} ({round(global_offset, 2)}s) successfully.")
    return output_mp4, enriched_records, chapter_markers

# ==============================================================================
# PART 27 OF 30: HIGH-CTR 1280X720 LONG-FORM THUMBNAIL GENERATOR
# ==============================================================================
def generate_long_thumbnail(puzzles: list, theme: dict, thumb_path: str = "thumb_long.jpg") -> str:
    """
    Creates a high-contrast 1280x720 YouTube thumbnail from bg_long.png with a red
    top-left urgency badge, bold stroked unsolved equation, mascot, and gold border.
    """
    tw, th = 1280, 720
    if os.path.exists("bg_long.png"):
        base = Image.open("bg_long.png").convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        base = ImageEnhance.Contrast(base).enhance(1.18)
    else:
        base = Image.new("RGB", (tw, th), theme["bg"])

    # Darken lower half slightly for crisp text legibility
    grad = Image.new("RGBA", (tw, th), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grad)
    for y in range(th // 2, th):
        alpha = int(165 * ((y - th // 2) / (th // 2)))
        gdraw.line([(0, y), (tw, y)], fill=(8, 12, 24, alpha))
    base = Image.alpha_composite(base.convert("RGBA"), grad).convert("RGB")

    draw = ImageDraw.Draw(base)

    # Gold outer border
    draw.rectangle((6, 6, tw - 7, th - 7), outline=(255, 214, 10), width=8)

    # Red top-left urgency badge pill
    badge_font = load_font(36)
    draw.rounded_rectangle((36, 32, 430, 102), radius=20, fill=(220, 38, 38), outline=(255, 255, 255), width=3)
    draw.text((233, 67), "ONLY 1% SOLVE ALL 5!", font=badge_font, fill=(255, 255, 255), anchor="mm")

    # Feature the hardest puzzle equation as the main thumbnail hook
    featured_q = str(puzzles[-1].get("question", "88² - 12² = ?")).strip()
    eq_font = load_font(86 if len(featured_q) <= 18 else 66)
    draw.text(
        (520, 390),
        featured_q,
        font=eq_font,
        fill=(255, 255, 255),
        anchor="mm",
        stroke_width=6,
        stroke_fill=(10, 15, 30),
    )

    # Bold gold sub-headline
    sub_font = load_font(54)
    draw.text(
        (520, 520),
        "BEAT THE 15s CLOCK!",
        font=sub_font,
        fill=(255, 214, 10),
        anchor="mm",
        stroke_width=5,
        stroke_fill=(10, 15, 30),
    )

    # Paste mascot on right side if available
    mascot_file = "char_think.png" if os.path.exists("char_think.png") else "char_open.png"
    if os.path.exists(mascot_file):
        m_img = Image.open(mascot_file).convert("RGBA")
        m_img.thumbnail((360, 360), Image.Resampling.LANCZOS)
        base.paste(m_img, (tw - m_img.width - 36, th - m_img.height - 24), m_img)

    base.save(thumb_path, format="JPEG", quality=92)
    print(f"[Thumbnail] Generated 1280x720 thumbnail -> {thumb_path}")
    return thumb_path

# ==============================================================================
# PART 28 OF 30: YOUTUBE DATA API V3 PUBLISHER, THUMBNAIL & STARTER COMMENT
# ==============================================================================
def get_youtube_service():
    token_raw = os.environ.get("YOUTUBE_TOKEN_JSON")
    if not token_raw:
        raise EnvironmentError("YOUTUBE_TOKEN_JSON secret is missing.")

    token_info = json.loads(token_raw)
    creds = Credentials.from_authorized_user_info(token_info)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)


def upload_to_youtube(
    youtube,
    video_path: str,
    puzzles: list,
    records: list,
    chapters: list,
    mode: str,
    thumb_path: str = None,
    immediate: bool = False,
) -> str:
    first = puzzles[0]
    first_rec = records[0]

    if mode == "short":
        title = f"{first['question'].replace('= ?', '').strip()} = ?? | {first['topic']} #Shorts"[:98]
        description = (
            f"{first['question']}\n\n"
            f"Topic: {first['topic']} | Difficulty: {first['difficulty']} ({first_rec['timer']}s)\n\n"
            f"Comment your exact solve time below! Subscribe to {CHANNEL_HANDLE} for daily brain workouts.\n"
            "#Shorts #MentalMath #MathTricks #BrainTeaser #MathMinuteMarvels"
        )
        tags = ["Shorts", "mental math", "math tricks", first["topic"], "brain teaser", "fast math"]
    else:
        topics_list = ", ".join(dict.fromkeys(p["topic"] for p in puzzles))
        title = f"5 Mental Math Puzzles That Will Upgrade Your Brain ({topics_list})"[:98]
        chapters_block = "\n".join(chapters)
        description = (
            "Can you solve all 5 progressive mental math challenges before the timer hits zero?\n\n"
            "CHAPTERS:\n"
            f"{chapters_block}\n\n"
            f"Drop your final score (out of 5) in the comments! Subscribe to {CHANNEL_HANDLE}.\n"
            "#MentalMath #MathPuzzles #MathTricks #BrainTraining #MathMinuteMarvels"
        )
        tags = ["mental math", "math puzzles", "math tricks", "brain training", "math shortcuts"]

    status_body = {"selfDeclaredMadeForKids": False}
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
            "categoryId": "27",
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
    print(f"[YouTube] Uploaded video ID: {video_id}")

    if thumb_path and os.path.exists(thumb_path):
        try:
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=MediaFileUpload(thumb_path, mimetype="image/jpeg"),
            ).execute()
            print(f"[YouTube] Attached custom 1280x720 thumbnail to {video_id}.")
        except Exception as terr:
            print(f"[YouTube] Thumbnail upload skipped (non-fatal): {terr}")

    return video_id


def post_starter_comment(youtube, video_id: str, puzzles: list, mode: str):
    comment_text = (
        f"Did you solve it before the {puzzles[0]['difficulty']} timer hit 0? Drop your time in seconds below!"
        if mode == "short"
        else "How many out of 5 did you solve mentally before the timer ended? Comment your score (e.g., 4/5)!"
    )
    try:
        youtube.commentThreads().insert(
            part="snippet",
            body={
                "snippet": {
                    "videoId": video_id,
                    "topLevelComment": {"snippet": {"textOriginal": comment_text}},
                }
            },
        ).execute()
        print(f"[YouTube] Posted starter comment on {video_id}.")
    except Exception as err:
        print(f"[YouTube] Starter comment skipped (non-fatal): {err}")

# ==============================================================================
# PART 29 OF 30: SUNDAY WEEKLY AI BRAIN ANALYTICS SELF-TUNER (--mode tune)
# ==============================================================================
def run_weekly_analytics_tuning():
    """
    Queries YouTube Data API v3 statistics for recent uploads in history.json,
    ranks top-performing math topics and peak IST publish hours, and updates
    weekly_insights.json automatically every Sunday.
    """
    history_data = load_history()
    insights = load_weekly_insights()
    video_map = {}

    for rec in history_data[-80:]:
        if isinstance(rec, dict) and rec.get("video_id"):
            video_map.setdefault(rec["video_id"], []).append(rec)

    if not video_map:
        print("[AI Brain] No uploaded video IDs in history.json yet; keeping baseline insights.")
        return

    youtube = get_youtube_service()
    ids = list(video_map.keys())[:50]
    resp = youtube.videos().list(part="snippet,statistics", id=",".join(ids)).execute()

    topic_scores = {}
    hour_scores = {}
    ist = ZoneInfo("Asia/Kolkata")

    for item in resp.get("items", []):
        vid = item["id"]
        stats = item.get("statistics", {})
        views = int(stats.get("viewCount", 0))
        likes = int(stats.get("likeCount", 0))
        comments = int(stats.get("commentCount", 0))
        engagement_score = views + (likes * 12) + (comments * 25)

        pub_str = item.get("snippet", {}).get("publishedAt")
        if pub_str:
            try:
                pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00")).astimezone(ist)
                hour_scores[pub_dt.hour] = hour_scores.get(pub_dt.hour, 0) + engagement_score
            except Exception:
                pass

        for rec in video_map.get(vid, []):
            t_name = rec.get("topic")
            if t_name:
                topic_scores[t_name] = topic_scores.get(t_name, 0) + engagement_score

    if topic_scores:
        ranked_topics = sorted(topic_scores, key=topic_scores.get, reverse=True)
        insights["best_topics"] = ranked_topics[:5]

    if hour_scores:
        ranked_hours = sorted(hour_scores, key=hour_scores.get, reverse=True)
        insights["peak_hours_ist"] = ranked_hours[:4]

    insights["last_tuned_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    save_weekly_insights(insights)
    print(f"[AI Brain] Updated {INSIGHTS_FILE}: {json.dumps(insights)}")

# ==============================================================================
# PART 30 OF 30: MAIN AUTONOMOUS PIPELINE ORCHESTRATOR
# ==============================================================================
def main():
    args = parse_args()
    print(
        f"=== {CHANNEL_NAME} 30-Part Engine | mode={args.mode} | "
        f"test={args.test} | immediate={args.immediate} ==="
    )

    if args.mode == "tune":
        run_weekly_analytics_tuning()
        return

    history_data = load_history()
    theme = pick_theme(history_data)
    voice = pick_voice(history_data)

    puzzles = generate_puzzles(mode=args.mode)
    video_path, enriched_records, chapters = render_complete_video(
        puzzles=puzzles,
        mode=args.mode,
        theme=theme,
        voice=voice,
    )

    thumb_path = None
    if args.mode == "long":
        thumb_path = generate_long_thumbnail(puzzles, theme, thumb_path="thumb_long.jpg")

    if args.test:
        print(
            f"[Test Mode] Dry-run complete. Video: {video_path}"
            + (f" | Thumbnail: {thumb_path}" if thumb_path else "")
        )
        return

    youtube = get_youtube_service()
    video_id = upload_to_youtube(
        youtube=youtube,
        video_path=video_path,
        puzzles=puzzles,
        records=enriched_records,
        chapters=chapters,
        mode=args.mode,
        thumb_path=thumb_path,
        immediate=args.immediate,
    )
    post_starter_comment(youtube, video_id, puzzles, args.mode)

    for rec in enriched_records:
        rec["video_id"] = video_id
    save_history_entries(enriched_records)
    print(f"[Done] Saved {len(enriched_records)} record(s) to {HISTORY_FILE} for {video_id}.")


if __name__ == "__main__":
    main()
