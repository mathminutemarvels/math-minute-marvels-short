# ==============================================================================
# WEEKLY_REVIEW.PY: YOUTUBE ANALYTICS -> WEEKLY_INSIGHTS.JSON FEEDBACK LOOP
# ==============================================================================
import os
import json
from collections import defaultdict
from datetime import datetime, timezone

from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build

HISTORY_FILE = "history.json"
INSIGHTS_FILE = "weekly_insights.json"

DEFAULT_TOPICS = [
    "Mental Multiplication",
    "Percentage Shortcuts",
    "Square Tricks",
    "Number Patterns",
]


def get_youtube_service():
    token_raw = os.environ.get("YOUTUBE_TOKEN_JSON")
    if not token_raw:
        raise EnvironmentError("YOUTUBE_TOKEN_JSON secret is missing.")

    token_info = json.loads(token_raw)
    creds = Credentials.from_authorized_user_info(token_info)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
    return build("youtube", "v3", credentials=creds)

def load_video_records() -> dict:
    """Maps video_id -> {topic, difficulty, question} from recent history.json entries."""
    if not os.path.exists(HISTORY_FILE):
        return {}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except Exception:
        return {}

    mapping = {}
    for entry in entries[-100:]:
        if isinstance(entry, dict) and entry.get("video_id"):
            vid = entry["video_id"]
            if vid not in mapping:
                mapping[vid] = {
                    "topic": entry.get("topic", "Mental Math"),
                    "difficulty": entry.get("difficulty", "MEDIUM"),
                    "question": entry.get("question", ""),
                }
    return mapping


def fetch_video_stats(youtube, video_ids: list) -> dict:
    """Fetches views, likes, and comments in batches of 50 from YouTube Data API."""
    stats_by_id = {}
    for i in range(0, len(video_ids), 50):
        batch = video_ids[i : i + 50]
        resp = youtube.videos().list(part="statistics,snippet", id=",".join(batch)).execute()
        for item in resp.get("items", []):
            s = item.get("statistics", {})
            views = int(s.get("viewCount", 0))
            likes = int(s.get("likeCount", 0))
            comments = int(s.get("commentCount", 0))
            # Weighted score rewarding likes and comments alongside views
            score = views + (likes * 15) + (comments * 30)
            stats_by_id[item["id"]] = {
                "title": item.get("snippet", {}).get("title", ""),
                "views": views,
                "likes": likes,
                "comments": comments,
                "score": score,
            }
    return stats_by_id


def run_weekly_review():
    records = load_video_records()
    if not records:
        insights = {
            "best_topics": DEFAULT_TOPICS,
            "preferred_difficulty": "MEDIUM",
            "analyzed_videos": 0,
            "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)
        print("[WeeklyReview] No video_ids found yet; saved default insights.")
        return

    youtube = get_youtube_service()
    stats = fetch_video_stats(youtube, list(records.keys()))

    topic_scores = defaultdict(list)
    diff_scores = defaultdict(list)
    ranked_videos = []

    for vid, meta in records.items():
        if vid not in stats:
            continue
        st = stats[vid]
        topic_scores[meta["topic"]].append(st["score"])
        diff_scores[meta["difficulty"]].append(st["score"])
        ranked_videos.append({"video_id": vid, "topic": meta["topic"], **st})

    # Rank topics and difficulty by average engagement score
    sorted_topics = sorted(
        topic_scores.keys(),
        key=lambda t: sum(topic_scores[t]) / max(1, len(topic_scores[t])),
        reverse=True,
    )
    best_topics = (sorted_topics + DEFAULT_TOPICS)[:5]

    preferred_difficulty = "MEDIUM"
    if diff_scores:
        preferred_difficulty = max(
            diff_scores.keys(),
            key=lambda d: sum(diff_scores[d]) / max(1, len(diff_scores[d])),
        )

    ranked_videos.sort(key=lambda x: x["score"], reverse=True)

    insights = {
        "best_topics": best_topics,
        "preferred_difficulty": preferred_difficulty,
        "analyzed_videos": len(ranked_videos),
        "top_videos": ranked_videos[:5],
        "updated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    with open(INSIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(insights, f, indent=2, ensure_ascii=False)

    print(f"[WeeklyReview] Updated {INSIGHTS_FILE}: top topics={best_topics[:3]}, difficulty={preferred_difficulty}")


if __name__ == "__main__":
    run_weekly_review()
