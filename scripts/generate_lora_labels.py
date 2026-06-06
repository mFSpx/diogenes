#!/usr/bin/env python3
"""
Generate training labels for all 5 LoRA brains.
Brain 1 (Speed Demon): ABBA³ heuristic (local, no API)
Brains 2-5: GROQ qwen/qwen3-32b API calls with batching/retry
"""
import hashlib, json, os, re, sys, time, traceback, subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from books_batch_process import extract_text, abba3_heuristic

OUT_DIR = ROOT / "05_OUTPUTS" / "training_data" / "labels"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "qwen/qwen3-32b"
PASSAGE_WORDS = 500
BATCH_SIZE = 5          # passages per Groq call
SAVE_INTERVAL = 10      # save progress every N calls
MAX_RETRIES = 8
INITIAL_BACKOFF = 2.0   # seconds

# --- Book assignments ---
BRAIN_BOOKS = {
    "speed_demon": [
        "The Art of War -- Sun Tzu",
        "Common Sense -- Thomas Paine",
        "The Communist Manifesto -- Karl Marx and Friedrich Engels",
    ],
    "philosopher": [
        "The Republic -- Plato",
        "On Liberty -- John Stuart Mill",
        "The Social Contract -- Jean-Jacques Rousseau",
        "Discourse on the Method -- Rene Descartes",
    ],
    "narrative_mind": [
        "Gulliver's Travels -- Jonathan Swift",
        "Candide -- Voltaire",
        "Herland -- Charlotte Perkins Gilman",
        "A Big Boy Did It and Ran Away",
    ],
    "observer": [
        "Out of Darkness _ Essays on Corporate Power",
        "Blood in the Machine_ The Origins of the Rebellion",
        "The Small and the Mighty_ Twelve Unsung Americans",
    ],
    "pattern_seer": [
        "The Prince -- Niccolo Machiavelli",
        "A Vindication of the Rights of Woman -- Mary Wollstonecraft",
        "One Day, Everyone Will Have Always Been Against This",
        "A Death in Malta - An Assassination and a Family's Quest",
    ],
}

def find_book_file(title_hint: str) -> Path:
    """Find the book file matching a title hint."""
    for f in (ROOT / "BOOKS").iterdir():
        if f.is_file() and title_hint in str(f):
            return f
    raise FileNotFoundError(f"No book file matching: {title_hint}")

def get_passages(text: str) -> list[str]:
    """Split text into ~500-word passages."""
    words = text.split()
    passages = []
    for i in range(0, len(words), PASSAGE_WORDS):
        chunk = " ".join(words[i:i + PASSAGE_WORDS])
        if len(chunk) > 50:  # skip tiny scraps
            passages.append(chunk)
    return passages

def read_groq_key() -> str:
    """Read GROQ_API_KEY from secrets file."""
    secrets_path = Path.home() / ".config" / "lucidota" / "secrets.env"
    if not secrets_path.exists():
        raise RuntimeError(f"Secrets file not found: {secrets_path}")
    for line in secrets_path.read_text().splitlines():
        line = line.strip()
        # Handle both `export KEY=VALUE` and `KEY=VALUE` formats
        if "GROQ_API_KEY=" in line:
            val = line.split("=", 1)[1].strip().strip('"').strip("'")
            if val:
                return val
    raise RuntimeError("GROQ_API_KEY not found in secrets file")

# ---- Brain 1: ABBA³ heuristic (local) ----

CATEGORY_KEYWORDS = {
    0: ["attack", "defend", "battle", "war", "strategy", "tactical", "force", "enemy", "victory",
        "defeat", "army", "soldier", "fight", "combat", "weapon", "ambush", "march", "siege"],
    1: ["government", "state", "law", "rights", "citizen", "political", "sovereign", "democracy",
        "liberty", "freedom", "vote", "power", "authority", "constitution", "parliament"],
    2: ["truth", "knowledge", "being", "existence", "soul", "god", "reason", "idea", "form",
        "essence", "nature", "reality", "thought", "mind", "consciousness", "dialectic"],
    3: ["said", "replied", "asked", "cried", "exclaimed", "walked", "went", "came", "looked",
        "told", "story", "narrative", "once upon", "journey", "adventure"],
    4: ["beautiful", "bright", "dark", "large", "small", "house", "tree", "river", "mountain",
        "sky", "land", "city", "appearance", "described", "visible"],
}

def classify_category(text: str) -> int:
    """Classify passage into category based on keyword overlap."""
    lower = text.lower()
    scores = {}
    for cat_id, keywords in CATEGORY_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        scores[cat_id] = score
    # If no match, default to philosophical (2)
    if not any(scores.values()):
        return 2
    return max(scores, key=scores.get)

def compute_abba3_vector(text: str) -> list[float]:
    """Compute 4-dim ABBA³ heuristic vector from passage text."""
    heur = abba3_heuristic(text)
    ld = heur["lexical_density"]
    cr = heur["compression_ratio"]
    ent = heur["entropy"]

    # urgency: high compression ratio + high lexical density = text feels urgent/concise
    # Normalize: higher compression = more urgent (condensed meaning)
    urgency = min(1.0, cr / 10000.0)

    category_id = float(classify_category(text))

    # complexity: lower lexical density = more unique words per passage = more complex thought
    # Also incorporate abba3_score
    complexity = min(1.0, (1.0 - ld) * 2.0)

    # certainty: entropy (character diversity) + lexical density signal definitive claims
    certainty = min(1.0, ent * 0.6 + ld * 0.4)

    return [round(urgency, 4), round(category_id, 1), round(complexity, 4), round(certainty, 4)]

def safe_book_name(filepath: Path) -> str:
    """Derive a safe short name from a book filepath."""
    name = filepath.stem
    # Strip metadata
    for sep in [" -- Anna", " -- gutenberg", "-- Brookmyre"]:
        if sep in name:
            name = name.split(sep)[0]
    name = re.sub(r'[^\w\s-]', '', name).strip()
    name = re.sub(r'\s+', '_', name)
    return name[:60]

def run_brain1():
    """Speed Demon: local ABBA³ per-passage vectors."""
    brain_id = "speed_demon"
    out_file = OUT_DIR / brain_id / "abba3_labels.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    for title_hint in BRAIN_BOOKS[brain_id]:
        book_file = find_book_file(title_hint)
        bname = safe_book_name(book_file)
        text, ext = extract_text(book_file)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        passages = get_passages(text)
        print(f"\n  Brain1: {bname} -> {len(passages)} passages")

        for pi, passage in enumerate(passages):
            vec = compute_abba3_vector(passage)
            record = {
                "brain_id": brain_id,
                "book": bname,
                "passage_index": pi,
                "passage_preview": passage[:100],
                "label": {
                    "urgency": vec[0],
                    "category_id": vec[1],
                    "complexity": vec[2],
                    "certainty": vec[3],
                    "heuristic_name": "ABBA3",
                    "source": "local_heuristic",
                },
            }
            with open(out_file, "a") as f:
                f.write(json.dumps(record) + "\n")
            written += 1

            if (pi + 1) % 50 == 0:
                print(f"    -> {pi+1}/{len(passages)} passages written")

    print(f"\n  Brain1 done: {written} label records -> {out_file}")
    return written

# ---- Brains 2-5: Groq Qwen 3 32B ----

# Rate limiter for Groq API calls
_last_groq_call = 0.0
RATE_LIMIT_DELAY = 1.2  # seconds between calls

def _rate_limit():
    global _last_groq_call
    now = time.time()
    elapsed = now - _last_groq_call
    if elapsed < RATE_LIMIT_DELAY:
        time.sleep(RATE_LIMIT_DELAY - elapsed)
    _last_groq_call = time.time()

GROQ_PROMPTS = {
    "philosopher": """You are a structured argument extraction system. Given the following passage from {book}, extract the argument structure as a JSON object.

Analyze the passage for:
1. "claims": list of claim statements (each as a string)
2. "evidence": list of evidence items (each as a string)
3. "conclusions": list of conclusions drawn (each as a string)
4. "counterarguments": list of counterarguments considered (each as a string)

Passage:
{passage}

Respond with ONLY a valid JSON object in exactly this format:
{{"claims": [...], "evidence": [...], "conclusions": [...], "counterarguments": [...]}}
Make each item concise (10-30 words). If a field has no items, use an empty array.""" ,

    "narrative_mind": """You are a narrative analysis system. Given the following passage from {book}, extract the narrative graph as a JSON object.

Analyze the passage for:
1. "characters": array of objects with "name" (string), "arc" (string describing character development), "traits" (array of 2-4 trait strings)
2. "irony_distance": float 0-1 where 0 = completely earnest, 1 = deeply ironic/satirical
3. "subtext": array of strings describing thematic undercurrents not explicitly stated

Passage:
{passage}

Respond with ONLY a valid JSON object in exactly this format:
{{"characters": [{{"name": "...", "arc": "...", "traits": [...]}}], "irony_distance": 0.0, "subtext": [...]}}
If a field has no items, use an empty array. Be concise.""" ,

    "observer": """You are a visual embedding description system. Given the following passage from {book}, describe what a SigLIP (vision-language) encoder would "see" if the passage were a visual scene.

Produce a JSON object with:
1. "scene_composition": string describing the spatial layout and visual elements (who/what is where)
2. "lighting_tone": string describing the visual mood/lighting (e.g. "dim chiaroscuro", "bright documentary", "neutral academic")
3. "visual_embedding_vibe": string of 3-5 comma-separated aesthetic tags
4. "key_visual_saliencies": array of 2-4 strings describing the most visually prominent elements
5. "motion_quality": string describing implied movement or stasis

Passage:
{passage}

Respond with ONLY a valid JSON object in exactly this format:
{{"scene_composition": "...", "lighting_tone": "...", "visual_embedding_vibe": "...", "key_visual_saliencies": [...], "motion_quality": "..."}}""" ,

    "pattern_seer": """You are a style attribution and authorship analysis system. Given the following passage from {book}, extract author attribution markers and style signals as a JSON object.

Analyze for:
1. "likely_author": string (the inferred author name)
2. "era": string (estimated era/century of writing)
3. "style_markers": array of 3-6 strings describing distinctive stylistic features (sentence length, rhetorical devices, vocabulary level, tone, use of dialogue, etc.)
4. "confidence": float 0-1 for how confident the attribution is
5. "distinctive_phrases": array of 2-4 strings of characteristic phrasing patterns

Passage:
{passage}

Respond with ONLY a valid JSON object in exactly this format:
{{"likely_author": "...", "era": "...", "style_markers": [...], "confidence": 0.0, "distinctive_phrases": [...]}}""" ,
}

def groq_call(prompt: str, groq_key: str, retries: int = MAX_RETRIES) -> dict:
    """Call Groq API with exponential backoff on 429."""
    _rate_limit()

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 4096,
    })

    for attempt in range(retries):
        try:
            result = subprocess.run(
                ["curl", "-s", "-w", "\n%{http_code}", "-X", "POST", GROQ_URL,
                 "-H", f"Authorization: Bearer {groq_key}",
                 "-H", "Content-Type: application/json",
                 "-d", payload],
                capture_output=True, text=True, timeout=120
            )
            stdout = result.stdout.strip()
            # Split response body from HTTP code
            if "\n" in stdout:
                *body_lines, http_code = stdout.rsplit("\n", 1)
                body = "\n".join(body_lines)
            else:
                body = stdout
                http_code = "000"

            http_code = http_code.strip()

            if http_code == "200":
                data = json.loads(body)
                usage = data.get("usage", {})
                content = data["choices"][0]["message"]["content"]
                return {
                    "content": content,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                }
            elif http_code == "429":
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                jitter = backoff * 0.1 * (hash(str(time.time())) % 10) / 10
                wait = min(backoff + jitter, 120)
                print(f"      429 rate limit, retry {attempt+1}/{retries} after {wait:.1f}s")
                time.sleep(wait)
            elif http_code == "500" or http_code == "502" or http_code == "503":
                backoff = INITIAL_BACKOFF * (2 ** attempt)
                time.sleep(min(backoff, 60))
                print(f"      {http_code} server error, retry {attempt+1}/{retries}")
            else:
                print(f"      HTTP {http_code}: {body[:200]}")
                if attempt < retries - 1:
                    time.sleep(INITIAL_BACKOFF)
        except subprocess.TimeoutExpired:
            print(f"      Timeout, retry {attempt+1}/{retries}")
        except Exception as e:
            print(f"      Error: {e}, retry {attempt+1}/{retries}")
            time.sleep(INITIAL_BACKOFF)

    return None  # all retries exhausted

def run_brain_groq(brain_id: str):
    """Run Groq-based label generation for a brain."""
    out_file = OUT_DIR / brain_id / {
        "philosopher": "argument_maps.jsonl",
        "narrative_mind": "narrative_graphs.jsonl",
        "observer": "siglip_descriptions.jsonl",
        "pattern_seer": "style_markers.jsonl",
    }[brain_id]
    out_file.parent.mkdir(parents=True, exist_ok=True)

    groq_key = read_groq_key()
    prompt_template = GROQ_PROMPTS[brain_id]

    # Collect all books and passages
    all_passages = []  # list of (bname, passage_text, passage_index, book_file)
    for title_hint in BRAIN_BOOKS[brain_id]:
        book_file = find_book_file(title_hint)
        bname = safe_book_name(book_file)
        text, ext = extract_text(book_file)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        passages = get_passages(text)
        print(f"\n  {brain_id}: {bname} -> {len(passages)} passages")
        for pi, passage in enumerate(passages):
            all_passages.append((bname, passage, pi, book_file))

    total_passages = len(all_passages)
    total_calls = (total_passages + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"  Total: {total_passages} passages = ~{total_calls} Groq API calls")

    # Load existing progress
    existing_indices = set()
    if out_file.exists():
        with open(out_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rec = json.loads(line)
                        key = (rec["book"], rec["passage_index"])
                        existing_indices.add(key)
                    except json.JSONDecodeError:
                        pass
        print(f"  Found {len(existing_indices)} existing records, will skip")

    token_usage = {"input": 0, "output": 0}
    calls_made = 0
    written_new = 0
    skipped = 0

    # Process in batches
    batch_passages = []
    for idx, (bname, passage_text, pi, book_file) in enumerate(all_passages):
        key = (bname, pi)
        if key in existing_indices:
            skipped += 1
            continue

        batch_passages.append((bname, passage_text, pi, book_file))

        if len(batch_passages) >= BATCH_SIZE or idx == total_passages - 1:
            if not batch_passages:
                continue

            calls_made += 1

            # Build one prompt with all passages in batch
            batch_text = ""
            for bi, (bp_bname, bp_text, bp_pi, _) in enumerate(batch_passages):
                batch_text += f"\n--- PASSAGE {bi+1} (book: {bp_bname}, index: {bp_pi}) ---\n{bp_text}\n"

            prompt = prompt_template.replace("{book}", batch_passages[0][0])
            # Replace {passage} with the batch of passages and instructions to process each
            full_prompt = f"""Process each of the following {len(batch_passages)} passages separately. For EACH passage, output a JSON object on its own line.

{prompt_template.replace("{book}", "MULTIPLE BOOKS").replace("{passage}", "See passages below")}

PASSAGES TO PROCESS:
{batch_text}

Respond with EXACTLY {len(batch_passages)} lines of JSON, one per passage, in order. Each line must be a valid JSON object. Do NOT include markdown formatting or code blocks."""

            result = groq_call(full_prompt, groq_key)

            if result is None:
                print(f"  WARNING: Call failed for batch starting at passage {batch_passages[0][2]}, saving partial progress")
                # Save what we have and continue
                batch_passages = []
                continue

            token_usage["input"] += result["input_tokens"]
            token_usage["output"] += result["output_tokens"]
            content = result["content"]

            # Parse JSON lines from response
            # Try to extract JSON objects
            lines = content.strip().split("\n")
            json_objects = []
            for line in lines:
                line = line.strip()
                # Remove markdown code block markers
                line = re.sub(r'^```json\s*', '', line)
                line = re.sub(r'^```\s*', '', line)
                line = re.sub(r'\s*```$', '', line)
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    json_objects.append(obj)
                except json.JSONDecodeError:
                    # Try to find JSON within the line
                    m = re.search(r'\{.*\}', line, re.DOTALL)
                    if m:
                        try:
                            obj = json.loads(m.group(0))
                            json_objects.append(obj)
                        except json.JSONDecodeError:
                            pass

            # Write records
            for bi, (bp_bname, bp_text, bp_pi, _) in enumerate(batch_passages):
                label = json_objects[bi] if bi < len(json_objects) else {"error": "parse_failed", "raw": content[:200]}
                record = {
                    "brain_id": brain_id,
                    "book": bp_bname,
                    "passage_index": bp_pi,
                    "passage_preview": bp_text[:100],
                    "label": label,
                    "groq_model": GROQ_MODEL,
                }
                with open(out_file, "a") as f:
                    f.write(json.dumps(record) + "\n")
                written_new += 1

            batch_passages = []

            # Progress save
            if calls_made % SAVE_INTERVAL == 0:
                pct = (idx + 1) / total_passages * 100
                print(f"    Call {calls_made}/{total_calls} ({pct:.0f}%) | "
                      f"Tokens: {token_usage['input']:,} in / {token_usage['output']:,} out | "
                      f"{written_new} new + {skipped} skipped")

    print(f"\n  {brain_id} done: {written_new} new + {skipped} skipped records -> {out_file}")
    print(f"  Token usage: {token_usage['input']:,} input, {token_usage['output']:,} output")
    return written_new, skipped, token_usage


def main():
    os.environ["PYTHONIOENCODING"] = "utf-8"
    total_labels = 0
    total_api_calls = 0
    grand_tokens = {"input": 0, "output": 0}

    # Brain 1: Speed Demon (local ABBA³) -- already done
    print("=" * 70)
    print("BRAIN 1: Speed Demon (Mamba 1.4B) -- SKIP (already done)")
    print("=" * 70)

    # Brains 2-5: Groq Qwen 3 32B
    for brain_id in ["philosopher", "narrative_mind", "observer", "pattern_seer"]:
        print("\n" + "=" * 70)
        labels = {"philosopher": "Philosopher (Bonsai 8B Q2)",
                  "narrative_mind": "Narrative Mind (Bonsai 8B Q1)",
                  "observer": "Observer (BitVLA)",
                  "pattern_seer": "Pattern Seer (RWKV 500M)"}
        print(f"BRAIN: {labels[brain_id]} -- Groq {GROQ_MODEL}")
        print("=" * 70)
        n_new, n_skip, tokens = run_brain_groq(brain_id)
        total_labels += n_new
        grand_tokens["input"] += tokens["input"]
        grand_tokens["output"] += tokens["output"]

    print("\n" + "=" * 70)
    print("ALL DONE")
    print(f"  Total label records: {total_labels}")
    print(f"  Total Groq tokens: {grand_tokens['input']:,} in / {grand_tokens['output']:,} out")
    print(f"  Output: {OUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
