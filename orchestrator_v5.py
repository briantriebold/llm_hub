import json, time, datetime, argparse, re, requests, hashlib, subprocess, shlex, os
from pathlib import Path

# ---- Config ----
CONF = json.loads(Path("config.json").read_text(encoding="utf-8"))
API_BASE = CONF["API_BASE"].rstrip("/")
DEFAULT_MODEL = CONF.get("MODEL_ID")
PLANNER_MODEL = CONF.get("PLANNER_MODEL_ID", DEFAULT_MODEL)
WORKER_MODEL  = CONF.get("WORKER_MODEL_ID", DEFAULT_MODEL)
TIMEOUT  = int(CONF.get("TIMEOUT_SEC", 90))
TEMP     = float(CONF.get("TEMPERATURE", 0.2))
MAXTOK   = int(CONF.get("MAX_TOKENS", 512))
RETRY    = CONF.get("RETRY", {"tries": 3, "backoff_sec": 2.0})
TRIES    = int(RETRY.get("tries", 3))
BACKOFF  = float(RETRY.get("backoff_sec", 2.0))

ROOT    = Path.cwd()
LOG_DIR = ROOT / "logs"; LOG_DIR.mkdir(exist_ok=True)
OUT_DIR = ROOT / "out";  OUT_DIR.mkdir(exist_ok=True)
STATE_F = ROOT / "state.json"
TASKS_F = ROOT / "tasks.jsonl"

# ---- Guardrails ----
THINK_RE = re.compile(r"<\s*think\s*>.*?<\s*/\s*think\s*>", re.IGNORECASE | re.DOTALL)
META_RE  = re.compile(r"^\s*\[(?:PLANNER|WORKER|CRITIC|REVISER|LOG)\].*$", re.IGNORECASE | re.MULTILINE)
URL_RE   = re.compile(r"https?://\S+")

def sanitize(text: str) -> str:
    text = THINK_RE.sub("", text)
    text = META_RE.sub("", text)
    return text.strip()

def chat(messages, model_id, temperature=TEMP, max_tokens=MAXTOK):
    url = f"{API_BASE}/chat/completions"
    payload = {"model": model_id, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
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
    if len(text) <= limit: return text
    cut = text.rfind(" ", 0, max(0, limit-1))
    return (text[:cut] + "…").rstrip() if cut != -1 else (text[:limit-1] + "…").rstrip()

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
            out.append(p if len(p) <= 280 else trim_to_limit(p, 280)); continue
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
        {"role":"user","content":f"Goal: {goal}\nConstraints: single human, no paid APIs, local runtime, small model, paste-ready output."}
    ]
    return chat(msgs, PLANNER_MODEL)

def role_worker(plan_text: str, goal: str):
    msgs = [
        {"role":"system","content":"You are the Worker. Output ONLY the deliverable. Exactly three lines, one post per line. No numbering, no preface, no counters."},
        {"role":"user","content":(
            "Deliverable rules:\n"
            "• Exactly 3 social posts.\n"
            "• Each ≤ 280 characters.\n"
            "• Each ends with a clear CTA to join the mailing list. Use https://trieboldinstitute.org/join if no link.\n"
            "• Return exactly three lines, one post per line, plain text."
        ) + f"\n\nPlan:\n{plan_text}\n\nGoal:\n{goal}"}
    ]
    return chat(msgs, WORKER_MODEL)

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
    if not ok_cta: feedback.append("Each post must include a clear CTA (Join/Sign up/Subscribe or /join link).")
    return ok, feedback, posts

def role_reviser(posts, feedback):
    posts = [trim_to_limit(p, 280) for p in posts]
    posts = ensure_cta(posts)
    posts = [trim_to_limit(p, 280) for p in posts]
    return "\n".join(posts)

def run_pipeline(goal: str):
    transcript = {"goal": goal, "planner_model": PLANNER_MODEL, "worker_model": WORKER_MODEL, "api_base": API_BASE, "steps": []}
    plan, raw1 = role_planner(goal); transcript["steps"].append({"role":"planner","response":plan,"raw":raw1}); print("\n[PLANNER]\n", plan)
    work, raw2 = role_worker(plan, goal); transcript["steps"].append({"role":"worker","response":work,"raw":raw2}); print("\n[WORKER]\n", work)
    ok, feedback, posts = role_critic(work); transcript["steps"].append({"role":"critic","ok":ok,"feedback":feedback,"parsed_posts":posts}); print("\n[CRITIC]", "PASS" if ok else "FAIL", ("| " + " | ".join(feedback) if feedback else ""))
    if not ok:
        revised = role_reviser(posts, feedback); transcript["steps"].append({"role":"reviser","response":revised}); print("\n[REVISER]\n", revised); work = revised
    out_path = save_text("social-intro", sanitize(work))
    log_path = save_json("planner-worker-critic", transcript)
    return out_path, log_path

# ---- Git helpers ----
def git(args, cwd=ROOT):
    return subprocess.run(["git"] + args, cwd=str(cwd), text=True, capture_output=True)

def ensure_repo():
    if not (ROOT / ".git").exists():
        r = git(["init"])
        if r.returncode != 0: raise RuntimeError(f"git init failed: {r.stderr}")
    # ensure main branch exists
    r = git(["rev-parse", "--abbrev-ref", "HEAD"])
    if r.returncode != 0 or not r.stdout.strip():
        git(["checkout", "-b", "main"])

def task_id_for(goal: str) -> str:
    return hashlib.sha1(goal.encode("utf-8")).hexdigest()[:12]

def slugify(s: str, maxlen=40) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip().lower()).strip("-")
    return (s[:maxlen]).strip("-") or "task"

def checkout_task_branch(goal: str):
    ensure_repo()
    tid = task_id_for(goal)
    name = f"task/{tid}-{slugify(goal)}"
    # create or switch
    exists = git(["rev-parse", "--verify", name]).returncode == 0
    base = "main"
    if not exists:
        # ensure base exists
        if git(["rev-parse", "--verify", base]).returncode != 0:
            git(["checkout", "-b", base])
            git(["add", "."]); git(["commit", "-m", "chore: bootstrap repo"])
        r = git(["checkout", "-b", name, base])
    else:
        r = git(["checkout", name])
    if r.returncode != 0: raise RuntimeError(f"git checkout failed: {r.stderr}")
    return name

def commit_artifacts(goal: str, out_path: Path, log_path: Path):
    git(["add", str(out_path)])
    git(["add", str(log_path)])
    # Optionally track task/queue state
    if TASKS_F.exists(): git(["add", str(TASKS_F)])
    if STATE_F.exists(): git(["add", str(STATE_F)])
    msg = f"content: {goal}"
    r = git(["commit", "-m", msg])
    if r.returncode != 0:
        # If nothing to commit, ignore
        if "nothing to commit" not in (r.stdout + r.stderr).lower():
            raise RuntimeError(f"git commit failed: {r.stderr}")

# ---- Queue ----
def load_state():
    if STATE_F.exists():
        try: return json.loads(STATE_F.read_text(encoding="utf-8"))
        except Exception: pass
    return {"processed": []}

def save_state(state):
    STATE_F.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")

def run_queue():
    state = load_state(); processed = set(state.get("processed", []))
    if not TASKS_F.exists(): print(f"[QUEUE] No tasks file at {TASKS_F.resolve()}"); return
    with TASKS_F.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try: task = json.loads(line)
            except Exception: print(f"[QUEUE] Skip invalid JSON: {line[:120]}"); continue
            goal = (task.get("goal") or "").strip()
            if not goal: print("[QUEUE] Skip task without 'goal'"); continue
            tid = task.get("id") or task_id_for(goal)
            if tid in processed: print(f"[QUEUE] Skip already processed: {tid}"); continue
            print(f"[QUEUE] Running: {tid} -> {goal}")
            branch = checkout_task_branch(goal)
            out_path, log_path = run_pipeline(goal)
            commit_artifacts(goal, out_path, log_path)
            processed.add(tid)
    state["processed"] = sorted(processed); save_state(state)
    print("[QUEUE] Done.")

# ---- CLI ----
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", type=str, help="Run a single goal immediately and commit on a task branch.")
    parser.add_argument("--queue", action="store_true", help="Process tasks.jsonl; create branch+commit per task.")
    args = parser.parse_args()

    if args.goal:
        branch = checkout_task_branch(args.goal)
        out_path, log_path = run_pipeline(args.goal)
        commit_artifacts(args.goal, out_path, log_path)
        print(f"[GIT] Committed on branch: {branch}")
    elif args.queue:
        run_queue()
    else:
        default = "Draft a 3-post intro for Triebold Institute (≤280 chars each) with a CTA to join the mailing list."
        branch = checkout_task_branch(default)
        out_path, log_path = run_pipeline(default)
        commit_artifacts(default, out_path, log_path)
        print(f"[GIT] Committed on branch: {branch}")
