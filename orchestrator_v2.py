import json, time, datetime, argparse, re, textwrap, requests
from pathlib import Path

# ---- Config ----
CONF = json.loads(Path("config.json").read_text(encoding="utf-8"))
API_BASE = CONF["API_BASE"].rstrip("/")
MODEL_ID = CONF["MODEL_ID"]
TIMEOUT  = int(CONF.get("TIMEOUT_SEC", 90))
TEMP     = float(CONF.get("TEMPERATURE", 0.2))
MAXTOK   = int(CONF.get("MAX_TOKENS", 512))
RETRY    = CONF.get("RETRY", {"tries": 3, "backoff_sec": 2.0})
TRIES    = int(RETRY.get("tries", 3))
BACKOFF  = float(RETRY.get("backoff_sec", 2.0))

LOG_DIR = Path("logs"); LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = Path("out");  OUT_DIR.mkdir(exist_ok=True)

# ---- Utilities / Guardrails ----
THINK_BLOCK = re.compile(r"<think>.*?</think>", flags=re.DOTALL|re.IGNORECASE)

def strip_think(text: str) -> str:
    """Remove chain-of-thought blocks like <think>...</think>."""
    return THINK_BLOCK.sub("", text).strip()

def chat(messages, temperature=TEMP, max_tokens=MAXTOK):
    """OpenAI-compatible /v1/chat/completions with simple retries."""
    url = f"{API_BASE}/chat/completions"
    payload = {"model": MODEL_ID, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    last_err = None
    for attempt in range(1, TRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:600]}")
            data = r.json()
            raw = data["choices"][0]["message"]["content"]
            return strip_think(raw), data
        except Exception as e:
            last_err = e
            if attempt < TRIES:
                time.sleep(BACKOFF * attempt)
            else:
                raise RuntimeError(f"API call failed after {TRIES} tries: {e}") from e

def save_json(stem, obj):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = LOG_DIR / f"{ts}-{stem}.json"
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LOG] {p.resolve()}")
    return p

def save_text(stem, text):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = OUT_DIR / f"{ts}-{stem}.txt"
    p.write_text(text, encoding="utf-8")
    print(f"[OUT] {p.resolve()}")
    return p

def split_posts(text: str):
    """
    Try to split into lines/posts. Accept either newline-separated or 1), 2), 3) patterns.
    Returns a list of non-empty trimmed posts.
    """
    # First try numbered bullets
    items = re.split(r"(?:^\s*\d+[\).\-]\s*|\n\s*\d+[\).\-]\s*)", text, flags=re.MULTILINE)
    candidates = [s.strip() for s in items if s and not s.strip().isdigit()]
    # If that failed, just split by lines
    if len(candidates) <= 1:
        candidates = [s.strip() for s in text.splitlines() if s.strip()]
    return [c for c in candidates if c]

def posts_within_limit(posts, limit=280):
    return all(len(p) <= limit for p in posts)

def enforce_limits(posts, limit=280):
    """Hard trim any over-limit posts while keeping words intact."""
    fixed = []
    for p in posts:
        if len(p) <= limit:
            fixed.append(p)
        else:
            # Trim with ellipsis if needed
            fixed.append(p[: max(0, limit-1)] + "…")
    return fixed

# ---- Roles ----
def role_planner(goal: str):
    msgs = [
        {"role":"system","content":(
            "You are the Planner. Output ONLY a numbered, concrete 3–5 step plan. "
            "Do NOT include chain-of-thought or explanations—just the plan."
        )},
        {"role":"user","content":(
            f"Goal: {goal}\n"
            "Constraints: single human, no paid APIs, local runtime, small model, produce a paste-ready output."
        )}
    ]
    return chat(msgs)

def role_worker(plan_text: str, goal: str):
    msgs = [
        {"role":"system","content":(
            "You are the Worker. Execute the plan into the final deliverable. "
            "Output ONLY the deliverable content—no preface, no commentary, no chain-of-thought."
        )},
        {"role":"user","content":(
            "Deliverable rules:\n"
            "• Exactly 3 social posts.\n"
            "• Each ≤ 280 characters.\n"
            "• Each ends with a clear CTA to join the mailing list (use placeholder link like https://trieboldinstitute.org/join if none provided).\n"
            "• Return as three lines, one post per line, no numbering.\n\n"
            f"Plan to execute:\n{plan_text}\n\nGoal:\n{goal}"
        )}
    ]
    return chat(msgs)

def role_critic(deliverable: str):
    posts = split_posts(deliverable)
    ok = (len(posts) == 3) and posts_within_limit(posts, 280) and all("join" in p.lower() and "mail" in p.lower() for p in posts)
    feedback = []
    if len(posts) != 3:
        feedback.append(f"Expected 3 posts; got {len(posts)}.")
    if not posts_within_limit(posts, 280):
        feedback.append("One or more posts exceed 280 chars.")
    if not all("join" in p.lower() and "mail" in p.lower() for p in posts):
        feedback.append("Each post must clearly include a CTA to join the mailing list.")
    return ok, feedback, posts

def role_reviser(posts, feedback):
    fixed = enforce_limits(posts, 280)
    # Ensure CTA present
    fixed2 = []
    for p in fixed:
        add_cta = "join" not in p.lower() or "mail" not in p.lower()
        if add_cta:
            suffix = " — Join our mailing list: https://trieboldinstitute.org/join"
            if len(p) + len(suffix) > 280:
                # trim to fit suffix
                p = p[: (280 - len(suffix))].rstrip()
            p = p + suffix
        fixed2.append(p)
    return "\n".join(fixed2)

# ---- Pipeline ----
def run_pipeline(goal: str):
    transcript = {"goal": goal, "model": MODEL_ID, "api_base": API_BASE, "steps": []}

    plan, raw1 = role_planner(goal)
    transcript["steps"].append({"role":"planner","response":plan,"raw":raw1})
    print("\n[PLANNER]\n", plan)

    work, raw2 = role_worker(plan, goal)
    work = strip_think(work)
    transcript["steps"].append({"role":"worker","response":work,"raw":raw2})
    print("\n[WORKER]\n", work)

    ok, feedback, posts = role_critic(work)
    transcript["steps"].append({"role":"critic","ok":ok,"feedback":feedback,"parsed_posts":posts})
    print("\n[CRITIC]", "PASS" if ok else "FAIL", ("| " + " | ".join(feedback) if feedback else ""))

    if not ok:
        revised = role_reviser(posts, feedback)
        transcript["steps"].append({"role":"reviser","response":revised})
        print("\n[REVISER]\n", revised)
        work = revised

    out_path = save_text("social-intro", work)
    log_path = save_json("planner-worker-critic", transcript)
    return out_path, log_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, default="Draft a 3-post intro for Triebold Institute (≤280 chars each) with a CTA to join the mailing list.")
    args = parser.parse_args()
    run_pipeline(args.goal)
