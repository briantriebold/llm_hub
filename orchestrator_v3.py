import json, time, datetime, argparse, re, requests
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

# ---- Guardrails ----
THINK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
META_RE  = re.compile(r"^\s*\[(?:PLANNER|WORKER|CRITIC|REVISER|LOG)\].*$", re.IGNORECASE | re.MULTILINE)
URL_RE   = re.compile(r"https?://\S+")

def sanitize(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = META_RE.sub("", text)
    return text.strip()

def chat(messages, temperature=TEMP, max_tokens=MAXTOK):
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
            return sanitize(raw), data
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
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        return lines
    parts = re.split(r"(?:^\s*\d+[\).\-]\s*|\n\s*\d+[\).\-]\s*)", text, flags=re.MULTILINE)
    posts = [s.strip() for s in parts if s and not s.strip().isdigit()]
    return [p for p in posts if p]

def enforce_three(posts):
    posts = [p for p in posts if p]
    if len(posts) > 3: posts = posts[:3]
    while len(posts) < 3:
        posts.append("Join our mailing list for updates: https://trieboldinstitute.org/join")
    return posts

def trim_to_limit(text, limit=280):
    if len(text) <= limit:
        return text
    # prefer trimming at last whitespace before limit
    cut = text.rfind(" ", 0, max(0, limit-1))
    if cut == -1:
        return (text[:limit-1] + "…").rstrip()
    return (text[:cut] + "…").rstrip()

def has_cta(p: str) -> bool:
    s = p.lower()
    if "trieboldinstitute.org/join" in s: return True
    if re.search(r"\bjoin\b.*\bmail(ing)?\s*list\b", s): return True
    if re.search(r"\b(sign\s*up|subscribe)\b", s): return True
    return False

def ensure_cta(posts):
    out = []
    for p in posts:
        if has_cta(p):
            out.append(p if len(p) <= 280 else trim_to_limit(p, 280))
            continue
        # If a URL already exists, don't add another one—just add wording.
        if URL_RE.search(p):
            suffix = " — Join our mailing list."
        else:
            suffix = " — Join our mailing list: https://trieboldinstitute.org/join"
        if len(p) + len(suffix) > 280:
            p = trim_to_limit(p, 280 - len(suffix))
        out.append((p + suffix) if len(p) + len(suffix) <= 280 else trim_to_limit(p + suffix, 280))
    return out

# ---- Roles ----
def role_planner(goal: str):
    msgs = [
        {"role":"system","content":"You are the Planner. Output ONLY a numbered, concrete 3–5 step plan. No explanations."},
        {"role":"user","content":f"Goal: {goal}\nConstraints: single human, no paid APIs, local runtime, small model, produce a paste-ready output."}
    ]
    return chat(msgs)

def role_worker(plan_text: str, goal: str):
    msgs = [
        {"role":"system","content":(
            "You are the Worker. Output ONLY the deliverable. "
            "Exactly three lines, one post per line. "
            "No numbering, no preface, no counters like '10/280'."
        )},
        {"role":"user","content":(
            "Deliverable rules:\n"
            "• Exactly 3 social posts.\n"
            "• Each ≤ 280 characters.\n"
            "• Each ends with a clear CTA to join the mailing list. Use https://trieboldinstitute.org/join if no link.\n"
            "• Return as three lines, one post per line, plain text only.\n\n"
            f"Plan:\n{plan_text}\n\nGoal:\n{goal}"
        )}
    ]
    return chat(msgs)

def role_critic(deliverable: str):
    posts = split_posts(deliverable)
    posts = enforce_three(posts)
    ok_len = all(len(p) <= 280 for p in posts)
    ok_cta = all(has_cta(p) for p in posts)
    ok_cnt = (len(posts) == 3)
    ok = ok_len and ok_cta and ok_cnt
    feedback = []
    if not ok_cnt: feedback.append(f"Expected 3 posts; got {len(posts)}.")
    if not ok_len: feedback.append("One or more posts exceed 280 chars.")
    if not ok_cta: feedback.append("Each post must include a clear CTA (e.g., 'Join our mailing list' OR 'Sign up/Subscribe' OR the /join link).")
    return ok, feedback, posts

def role_reviser(posts, feedback):
    posts = [trim_to_limit(p, 280) for p in posts]
    posts = ensure_cta(posts)
    posts = [trim_to_limit(p, 280) for p in posts]
    return "\n".join(posts)

# ---- Pipeline ----
def run_pipeline(goal: str):
    transcript = {"goal": goal, "model": MODEL_ID, "api_base": API_BASE, "steps": []}

    plan, raw1 = role_planner(goal)
    transcript["steps"].append({"role":"planner","response":plan,"raw":raw1})
    print("\n[PLANNER]\n", plan)

    work, raw2 = role_worker(plan, goal)
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

    out_path = save_text("social-intro", sanitize(work))
    log_path = save_json("planner-worker-critic", transcript)
    return out_path, log_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, default="Draft a 3-post intro for Triebold Institute (≤280 chars each) with a CTA to join the mailing list.")
    args = parser.parse_args()
    run_pipeline(args.goal)
