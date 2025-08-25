import json, time, datetime, requests
from pathlib import Path

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

def chat(messages, temperature=TEMP, max_tokens=MAXTOK):
    url = f"{API_BASE}/chat/completions"
    payload = {"model": MODEL_ID, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    last_err = None
    for attempt in range(1, TRIES + 1):
        try:
            r = requests.post(url, json=payload, timeout=TIMEOUT)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
            data = r.json()
            content = data["choices"][0]["message"]["content"]
            return content.strip(), data
        except Exception as e:
            last_err = e
            if attempt < TRIES:
                time.sleep(BACKOFF * attempt)
            else:
                raise RuntimeError(f"API call failed after {TRIES} tries: {e}") from e

def save_log(stem, transcript):
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    p = LOG_DIR / f"{ts}-{stem}.json"
    p.write_text(json.dumps(transcript, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[LOG] {p.resolve()}")

def run_pipeline(goal: str):
    transcript = {"goal": goal, "model": MODEL_ID, "api_base": API_BASE, "steps": []}
    planner_msgs = [
        {"role":"system","content":"You are the Planner. Create a numbered, concrete 3–5 step plan. Be terse and actionable."},
        {"role":"user","content":f"Goal: {goal}\nConstraints: single human, no paid APIs, local runtime, small model, produce a paste-ready output."}
    ]
    plan_text, raw1 = chat(planner_msgs)
    print("\n[PLANNER]\n", plan_text)
    transcript["steps"].append({"role":"planner","messages":planner_msgs,"response":plan_text,"raw":raw1})
    worker_msgs = [
        {"role":"system","content":"You are the Worker. Execute the plan into a compact deliverable. Output only the deliverable."},
        {"role":"user","content":f"Plan:\n{plan_text}\n\nDeliver exactly what the plan requires (no extra commentary)."}
    ]
    work_text, raw2 = chat(worker_msgs)
    print("\n[WORKER]\n", work_text)
    transcript["steps"].append({"role":"worker","messages":worker_msgs,"response":work_text,"raw":raw2})
    save_log("planner-worker", transcript)

if __name__ == "__main__":
    goal = "Draft a 3-post social intro for Triebold Institute (<=280 chars each), with a CTA to join the mailing list."
    run_pipeline(goal)
