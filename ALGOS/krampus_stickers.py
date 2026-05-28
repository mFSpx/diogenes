#!/usr/bin/env python3
"""KORPUS/KRAMPUS cognitive-document sticker algorithms.

Reusable deterministic feature math lives here; runtime scripts only orchestrate.
These are document/communication telemetry metrics, not diagnosis. No LLM calls.
"""
from __future__ import annotations

import re
from typing import Any

try:
    import pronouncing  # type: ignore
except Exception:  # pragma: no cover
    pronouncing = None  # type: ignore

from ALGOS.krampus_chrono import parse_loose_datetime
from ALGOS.shannon_entropy import shannon_entropy

MAX_COMPONENT_TOKENS = 500


def normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def token_count(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def entropy_for_text(text: str) -> float:
    return float(shannon_entropy(list((text or "")[:10000]))) if text else 0.0


def links_from_text(text: str) -> list[dict[str, Any]]:
    links: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for m in re.finditer(r"\[([^\]]{0,240})\]\(([^)\s]{1,1000})\)", text or ""):
        target = m.group(2).strip()
        kind = "url" if target.lower().startswith(("http://", "https://")) else "markdown"
        key = (kind, target, m.group(1))
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": kind, "raw_target": target, "anchor_text": m.group(1), "source": "markdown_link"})
    for m in re.finditer(r"\[\[([^\]|#]{1,500})(?:#[^\]|]+)?(?:\|([^\]]{1,240}))?\]\]", text or ""):
        target = m.group(1).strip()
        anchor = (m.group(2) or target).strip()
        key = ("wikilink", target, anchor)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "wikilink", "raw_target": target, "anchor_text": anchor, "source": "wikilink"})
    for m in re.finditer(r'''\bhttps?://[^\s<>'")]+''', text or "", re.I):
        target = m.group(0).rstrip(".,;")
        key = ("url", target, target)
        if key not in seen:
            seen.add(key)
            links.append({"link_kind": "url", "raw_target": target, "anchor_text": target[:240], "source": "bare_url"})
    return links

VISCERAL_RE = re.compile(
    r"\b(?:vomit|shit|cum|bleed|blood|bone|bones|meatspace|meat|sewage|flood|burn|kill|organ|organs|"
    r"chew|chewing|digest|digested|spine|body|bodies|knife|knives|gut|guts|teeth|mouth|hunger|wound|scar|skin)\b",
    re.I,
)
HARD_TECH_RE = re.compile(
    r"\b(?:jsonl|yaml|yml|aes-256(?:-gcm)?|namespace|namespaces|endpoint|endpoints|pgvector|schema|schemas|"
    r"auth|absurd|postgres|postgresql|lora|embedding|embeddings|vector|vectors|sha256|sha-256|uuid|uuidv7|cas|"
    r"systemd|inotify|ast|regex|api|cli|kernel|graph|ontology|workflow|workflows|queue|bytewax|riverml|river|"
    r"ocr|exif|ffprobe|json|sql|table|index|fts)\b",
    re.I,
)
LEGAL_OSINT_RE = re.compile(
    r"\b(?:bcfsa|complaint|complaints|evidence|diligence|title|rezoning|liability|realtor|realtors|"
    r"tenant|tenancy|landlord|lease|court|registry|corporate|director|filing|filings|property|address|"
    r"alias|aliases|phone|permit|bylaw|flood|sewage|claim|claims|fraud|unmasked|osint|investigation)\b",
    re.I,
)
SELF_REF_RE = re.compile(r"\b(?:indy|operator|lucidota|masthead|system|northern\.?strike|ponyboy|diogenes|krampus)\b", re.I)
AUDIT_RE = re.compile(r"\b(?:audit|critique|drift|evaluate|evaluation|verdict|review|aar|after[- ]action|diagnose|inspect)\b", re.I)
DIRECTIVE_RE = re.compile(
    r"(?:^|\n|\.\s+|\!\s+|\?\s+|\-\s+|\*\s+|\d+\.\s+)"
    r"(?:Fire|Read|Check|Add|Change|Split|Do it|Walk away|Assert|Build|Wire|Run|Drop|Ingest|Hash|"
    r"Normalize|Title|Date|OCR|Extract|Route|Lock|Push|Generate|Create|Make|Fix|Patch|Audit|Search|Pivot|"
    r"Document|Verify|Test|Ship|Start|Stop|Enable|Disable|Move|Delete|Keep|Preserve|Map)\b",
    re.I,
)
HEDGE_RE = re.compile(r"\b(?:maybe|might|probably|seems|seem|dunno|perhaps|possibly|guess|unsure|unclear|kinda|sorta)\b", re.I)
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|.*\|\s*$", re.M)
CHECKBOX_RE = re.compile(r"\[[xX ]\]")
LIST_ITEM_RE = re.compile(r"^\s*(?:[-*+]|\d+\.)\s+", re.M)
PHONE_RE = re.compile(r"(?<!\d)(?:\+?1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}(?!\d)")
BC_CORP_RE = re.compile(r"\bBC\d{7}\b", re.I)
MONEY_RE = re.compile(r"\$\s?\d+(?:,\d{3})*(?:\.\d{2})?")
DISTRESS_RE = re.compile(
    r"\b(?:pain|kill|die|death|suicide|suicidal|trauma|abuse|alone|drown|drowning|hurt|grief|"
    r"collapse|breakdown|despair|panic|terror|shame|cry|crying|blood|bleed|wound|wounded)\b",
    re.I,
)
FORENSIC_TAG_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?(?:Analyst|Method|Evidence|Evidence base|Status|Generated|"
    r"INFERRED|OBSERVED|Critical limitation|Operational Status|Finding|Findings|Verdict|Confidence|Source):?(?:\*\*)?",
    re.I,
)
FIRST_PERSON_RE = re.compile(r"\b(?:i|me|my|mine|myself)\b", re.I)
THIRD_PERSON_SELF_RE = re.compile(r"\b(?:ponyboy|the operator|operator|northern\.?strike|he|him|his)\b", re.I)
BUREAUCRACY_RE = re.compile(r"\b(?:rtdrs|cra|bcfsa|tribunal|police|trust\s*&\s*safety|ombudsperson|bylaw|court|registry|commission|ministry)\b", re.I)
STRIKE_RE = re.compile(r"\b(?:file|dial|sue|report|serve|pull|send|escalate|complain|complaint|fuck them|destroy|enforce|breach|claim)\b", re.I)
EMPATHIC_TARGET_RE = re.compile(
    r"\b(?:protect|shield|safe|safety|fair|vulnerable|friends?|rowyn|kai|chance|wade|sex\s+workers?|"
    r"community|tenant|tenants|survivor|survivors|help|care|harm|exploitation|slumlord)\b",
    re.I,
)
PALADIN_NODE_RE = re.compile(
    r"\b(?:bcfsa|rtb|rtdrs|211|foi|injunction|tribunal|cra|police|bylaw|ombudsperson|court|registry|"
    r"ministry|tenancy|trust\s*&\s*safety|complaint|evidence|diligence)\b",
    re.I,
)
SCARCITY_RE = re.compile(
    r"\b(?:broke|homeless|evict(?:ed|ion)?|rent|debt|overdraft|no money|can't afford|cannot afford|"
    r"last\s+\$?\d+|\$\s?\d+\s+to my name|precarious|starving|exhausted|no sleep|sleep deprived|burned out)\b",
    re.I,
)
BODY_GRIEF_RE = re.compile(
    r"\b(?:wayne|pain|surgery|surgeries|hip|infection|hospital|medical|trauma|crying|grief|grieving|"
    r"mourning|dead|death|loss|lost|alone|hurt|injury|injured|disabled)\b",
    re.I,
)
SPRINT_ENTITY_RE = re.compile(
    r"\b(?:godot|17lands|percyphon|rowyn\s*v?\d*|miazahole|lora|dashboard|dashboards|agent|agents|"
    r"sub-agent|subagents|scraper|scrapers|svg|mtg|sprint|landing|landings|swarm|tickets?|codex)\b",
    re.I,
)
TIMEBOX_RE = re.compile(
    r"\b(?:afk\s*)?(?:\d+h(?:\d+m)?|\d+\s*(?:min|mins|minutes|hours?|hrs?))(?:\s+into\s+the\s+clock)?\b|\b(?:eta|sprint|clock)\b",
    re.I,
)
HYPOTHESIS_RE = re.compile(r"\b(?:conspiracy|astroturf|fever\s+dream|rabbit\s+hole|spud|tides|theory|theories|angle|angles|hypothesis|wild|crazy|hunch)\b", re.I)
VERIFICATION_RE = re.compile(r"\b(?:confirmed|verified|sourced|source|evidence|zero\s+criminal\s+evidence|foi|cra|irs|court|registry|records?|filings?|proof|logic|legally)\b", re.I)
DELIVERABLE_RE = re.compile(r"\b(?:website|hosted|keys|invoice|summary|package|deployment|domain|docs?|documentation|deliverable|handoff|client|fixed costs?|costs?)\b", re.I)
ABSURDITY_RE = re.compile(r"\b(?:fuck|fucking|lmao|lol|haha|hahaha|krampus\s+fee|pay what|you pick|cheerio|shit|goddamn|nice)\b|\b(?:150\.69|69|6900|6,900)\b", re.I)
CORPORATE_RE = re.compile(
    r"\b(?:roi|capital|equity|loi|revenue|amortized|amortization|bylaw|compliance|investor|investors|"
    r"series\s+a|pro[- ]rata|valuation|cap\s+table|term\s+sheet|partnership|sponsor|sponsorship|margin|profit|runway|market)\b",
    re.I,
)
GRIT_RE = re.compile(r"\b(?:shitty|raw|sweat|chaotic|feral|street|hustle|hustler|scrappy|grind|boots|weird|cute)\b", re.I)
COUNTDOWN_RE = re.compile(r"(?:-\d+\s*days?|\bdays?\s+to\b|\bcritical path\b|\bdeadline\b|\blaunch\b|\bq[1-4]\b|\bfifa\s+day\s+\d+\b|\beta\b|\bt-minus\b|\btimeline\b)", re.I)
FILE_EXT_RE = re.compile(r"\b(?:docx|xlsx|pptx|html|pdf|csv)\b", re.I)
PHASE_RE = re.compile(r"\b(?:phase\s+\d+|tier\s+\d+|deliverables?|milestone|track\s+\d+|workstream)\b", re.I)
BOLD_TAG_RE = re.compile(r"\*\*.*?\*\*", re.S)
COLLABORATIVE_RE = re.compile(
    r"\b(?:we|us|our|ours|together|let's|lets|thank you|good job|nice work|we got this|"
    r"let's go|lets go|not insane|not crazy|you are not alone|indy|agent|agents)\b",
    re.I,
)
COMMAND_SYNTAX_RE = re.compile(
    r"\b(?:generate|output|execute|ticket|fix|build|write|halt|route|dispatch|run|ingest|embed|"
    r"wire|deploy|parse|audit|patch|ship|create|make|start|stop|restart|enable|disable)\b",
    re.I,
)
PROTOCOL_TAG_RE = re.compile(
    r"(?:\*\*)?(?:Filed|Verdict|Status|Operator|Result|Assert|Break|Build|Seal|Cut|Decision|Protocol|"
    r"Evidence|Method|Generated):?(?:\*\*)?",
    re.I,
)
SEAL_REF_RE = re.compile(r"\b(?:cuts\.jsonl|sealed|unsealed|assert|break|build|subtle\s+knife|ponyboy\s+protocol|abba3|protocol)\b", re.I)
CAPS_CLUSTER_RE = re.compile(r"\b[A-Z]{4,}\b(?:\s+\b[A-Z]{4,}\b){2,}")
SWEAR_RE = re.compile(r"\b(?:fuck|fucking|shit|damn|cunt|hell|goddamn)\b", re.I)
INLINE_TIMESTAMP_RE = re.compile(
    r"\b20\d{2}[-_/]\d{2}[-_/]\d{2}(?:[T\s_]\d{2}:?\d{2}(?::?\d{2})?(?:Z|[+-]\d{2}:?\d{2})?)?\b"
)
RUNON_RE = re.compile(r"[^.!?\n]{240,}[.!?]?", re.S)
CONTENT_DATE_PATTERNS = [
    ("frontmatter_date", re.compile(r"(?im)^\s*(?:date|created|created_at|created at|generated|timestamp|time|filed|updated|modified)\s*[:=]\s*[\"']?((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)")),
    ("iso_inline", re.compile(r"\b((?:20|19)\d{2}[-_/]\d{1,2}[-_/]\d{1,2}(?:[T\s_]\d{1,2}:?\d{2}(?::?\d{2})?(?:\s?(?:Z|[+-]\d{2}:?\d{2}))?)?)\b")),
    ("compact_yyyymmdd", re.compile(r"\b((?:20|19)\d{2})(\d{2})(\d{2})(?:[_-]?(\d{2})(\d{2})(\d{2})?)?\b")),
]
MONTH_NAME_RE = re.compile(
    r"\b(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
    r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+([0-3]?\d)(?:st|nd|rd|th)?,?\s+((?:20|19)\d{2})\b",
    re.I,
)
def extract_operator_vibes(text: str) -> dict[str, float]:
    """Northern.Strike / Ponyboy stickers: deterministic cognition-control features."""
    words = re.findall(r"\b[\w'.-]+\b", text)
    total_words = max(1, len(words))
    text_lower = text.lower()
    visceral_terms = len(VISCERAL_RE.findall(text_lower))
    tech_terms = len(HARD_TECH_RE.findall(text_lower))
    legal_osint_terms = len(LEGAL_OSINT_RE.findall(text_lower))
    table_rows = len(TABLE_ROW_RE.findall(text))
    checkboxes = len(CHECKBOX_RE.findall(text))
    list_items = len(LIST_ITEM_RE.findall(text))
    self_reference_terms = len(SELF_REF_RE.findall(text_lower))
    audit_terms = len(AUDIT_RE.findall(text_lower))
    directives = len(DIRECTIVE_RE.findall(text))
    hedges = len(HEDGE_RE.findall(text_lower))
    phone_numbers = len(PHONE_RE.findall(text))
    corporate_ids = len(BC_CORP_RE.findall(text))
    money_mentions = len(MONEY_RE.findall(text))
    return {
        "operator_total_words": float(total_words),
        "operator_visceral_ratio": visceral_terms / total_words,
        "operator_tech_ratio": tech_terms / total_words,
        "operator_legal_osint_ratio": legal_osint_terms / total_words,
        "operator_ledger_density": (table_rows + checkboxes + list_items) / total_words,
        "operator_recursion_score": (self_reference_terms * audit_terms) / total_words,
        "operator_directive_ratio": (directives + 1) / (hedges + 1),
        "operator_target_density": (phone_numbers + corporate_ids + money_mentions) / total_words,
        "operator_visceral_terms": float(visceral_terms),
        "operator_tech_terms": float(tech_terms),
        "operator_legal_osint_terms": float(legal_osint_terms),
        "operator_table_rows": float(table_rows),
        "operator_checkboxes": float(checkboxes),
        "operator_list_items": float(list_items),
        "operator_self_reference_terms": float(self_reference_terms),
        "operator_audit_terms": float(audit_terms),
        "operator_directives": float(directives),
        "operator_hedges": float(hedges),
        "operator_phone_numbers": float(phone_numbers),
        "operator_corporate_ids": float(corporate_ids),
        "operator_money_mentions": float(money_mentions),
    }


def rhymeish(left: str, right: str) -> bool:
    if not left or not right or left == right:
        return False
    if pronouncing is not None:
        try:
            return left in set(pronouncing.rhymes(right)) or right in set(pronouncing.rhymes(left))
        except Exception as exc:
            _ = exc
    # Dependency-free fallback: compare vowel-tail/suffix. This is not poetry analysis;
    # it is a deterministic weak signal for free-associative rhyme/rhythm.
    def tail(word: str) -> str:
        m = re.search(r"[aeiouy][a-z]{0,4}$", word)
        return m.group(0) if m else word[-3:]
    return len(left) >= 3 and len(right) >= 3 and (left[-3:] == right[-3:] or tail(left) == tail(right))


def extract_psyche_vibes(text: str) -> dict[str, float]:
    """Lexical/format safety stickers; not diagnosis, only deterministic document signals."""
    words = re.findall(r"\b[\w'.-]+\b", text)
    total_words = max(1, len(words))
    text_lower = text.lower()
    distress_words = len(DISTRESS_RE.findall(text_lower))
    forensic_tags = len(FORENSIC_TAG_RE.findall(text))
    structural_tags = len(TABLE_ROW_RE.findall(text)) + len(CHECKBOX_RE.findall(text)) + len(LIST_ITEM_RE.findall(text))
    first_person = len(FIRST_PERSON_RE.findall(text_lower))
    third_person_self = len(THIRD_PERSON_SELF_RE.findall(text_lower))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    end_words: list[str] = []
    for line in lines:
        pieces = re.findall(r"[A-Za-z']+", line)
        if pieces:
            end_words.append(re.sub(r"[^a-z]", "", pieces[-1].lower()))
    rhyme_count = 0
    for left, right in zip(end_words, end_words[1:]):
        if rhymeish(left, right):
            rhyme_count += 1
    rhyme_density = rhyme_count / max(1, len(lines))
    markdown_density = min(1.0, structural_tags / max(1, len(lines)))
    bureaucracy = len(BUREAUCRACY_RE.findall(text_lower))
    strikes = len(STRIKE_RE.findall(text_lower))
    return {
        "psyche_distress_words": float(distress_words),
        "psyche_forensic_tags": float(forensic_tags),
        "psyche_structural_tags": float(structural_tags),
        "psyche_forensic_shield_ratio": (distress_words * max(forensic_tags, min(structural_tags, 12))) / total_words,
        "psyche_first_person": float(first_person),
        "psyche_third_person_self": float(third_person_self),
        "psyche_dissociative_index": (third_person_self + 1) / (first_person + 1),
        "psyche_rhyme_density": rhyme_density,
        "psyche_markdown_density": markdown_density,
        "psyche_poetic_entropy": rhyme_density * (1.0 - markdown_density),
        "psyche_bureaucracy_terms": float(bureaucracy),
        "psyche_strike_terms": float(strikes),
        "psyche_wrath_velocity": (bureaucracy * strikes) / total_words,
    }


def extract_resilience_vibes(text: str) -> dict[str, float]:
    """Final operational stickers: protector-mode, exhaustion, sprint, verification, humor-tax."""
    words = re.findall(r"\b[\w'.-]+\b", text)
    total_words = max(1, len(words))
    text_lower = text.lower()
    empathic_targets = len(EMPATHIC_TARGET_RE.findall(text_lower))
    paladin_nodes = len(PALADIN_NODE_RE.findall(text_lower))
    scarcity_markers = len(SCARCITY_RE.findall(text_lower))
    body_grief_markers = len(BODY_GRIEF_RE.findall(text_lower))
    sprint_mentions = SPRINT_ENTITY_RE.findall(text)
    sprint_entities = {normalize_ws(m).lower() for m in sprint_mentions}
    timeboxes = len(TIMEBOX_RE.findall(text_lower))
    hypothesis_terms = len(HYPOTHESIS_RE.findall(text_lower))
    verification_terms = len(VERIFICATION_RE.findall(text_lower))
    deliverable_terms = len(DELIVERABLE_RE.findall(text_lower))
    absurdity_terms = len(ABSURDITY_RE.findall(text_lower))
    return {
        "resilience_empathic_targets": float(empathic_targets),
        "resilience_bureaucratic_nodes": float(paladin_nodes),
        "resilience_bureaucratic_weaponization_index": (empathic_targets * paladin_nodes) / total_words,
        "resilience_scarcity_markers": float(scarcity_markers),
        "resilience_body_grief_markers": float(body_grief_markers),
        "resilience_resource_exhaustion_metric": (scarcity_markers * body_grief_markers) / total_words,
        "resilience_sprint_entity_mentions": float(len(sprint_mentions)),
        "resilience_distinct_sprint_entities": float(len(sprint_entities)),
        "resilience_timeboxes": float(timeboxes),
        "resilience_swarm_orchestration_density": (len(sprint_entities) + timeboxes) / total_words,
        "resilience_hypothesis_terms": float(hypothesis_terms),
        "resilience_verification_terms": float(verification_terms),
        "resilience_conspiracy_grounding_ratio": verification_terms / (hypothesis_terms + 1),
        "resilience_logic_crucifixion_index": (hypothesis_terms * verification_terms) / total_words,
        "resilience_deliverable_terms": float(deliverable_terms),
        "resilience_absurdity_terms": float(absurdity_terms),
        "resilience_chaotic_good_tax": (deliverable_terms * absurdity_terms) / total_words,
    }


def extract_rainmaker_vibes(text: str) -> dict[str, float]:
    """Capital/pitch/package-mode stickers: money traps, launch clocks, asset grids."""
    words = re.findall(r"\b[\w'.-]+\b", text)
    total_words = max(1, len(words))
    text_lower = text.lower()
    corporate_terms = len(CORPORATE_RE.findall(text_lower))
    grit_terms = len(GRIT_RE.findall(text_lower))
    countdown_markers = len(COUNTDOWN_RE.findall(text_lower))
    file_extensions = len(FILE_EXT_RE.findall(text_lower))
    phase_markers = len(PHASE_RE.findall(text_lower))
    bold_tags = len(BOLD_TAG_RE.findall(text))
    return {
        "rainmaker_corporate_terms": float(corporate_terms),
        "rainmaker_grit_terms": float(grit_terms),
        "rainmaker_corporate_grit_tension": (corporate_terms * grit_terms) / total_words,
        "rainmaker_countdown_markers": float(countdown_markers),
        "rainmaker_countdown_density": countdown_markers / total_words,
        "rainmaker_file_extensions": float(file_extensions),
        "rainmaker_phase_markers": float(phase_markers),
        "rainmaker_asset_structuring_weight": (file_extensions + phase_markers) / total_words,
        "rainmaker_bold_tags": float(bold_tags),
        "rainmaker_pitch_formatting_ratio": bold_tags / total_words,
    }


def extract_operator_telemetry(text: str) -> dict[str, float]:
    """Cybernetic/operator telemetry stickers: agent merge, protocol discipline, rapidthink.

    These are text-shape metrics only. They intentionally avoid mental-health labels.
    """
    words = re.findall(r"\b[\w'.-]+\b", text)
    total_words = max(1, len(words))
    text_lower = text.lower()
    collaborative = len(COLLABORATIVE_RE.findall(text_lower))
    command = len(COMMAND_SYNTAX_RE.findall(text_lower))
    formal_tags = len(PROTOCOL_TAG_RE.findall(text))
    seal_references = len(SEAL_REF_RE.findall(text_lower))
    tables = len(TABLE_ROW_RE.findall(text))
    checkboxes = len(CHECKBOX_RE.findall(text))
    list_items = len(LIST_ITEM_RE.findall(text))
    runons = len(RUNON_RE.findall(text))
    caps_clusters = len(CAPS_CLUSTER_RE.findall(text))
    swear_terms = len(SWEAR_RE.findall(text_lower))
    timestamp_markers = len(INLINE_TIMESTAMP_RE.findall(text))
    rapid_timestamp_pairs = 0
    parsed_times: list[dt.datetime] = []
    for raw in INLINE_TIMESTAMP_RE.findall(text)[:128]:
        parsed = parse_loose_datetime(raw)
        if parsed:
            parsed_times.append(parsed)
    parsed_times.sort()
    for left, right in zip(parsed_times, parsed_times[1:]):
        if 0 <= (right - left).total_seconds() <= 15 * 60:
            rapid_timestamp_pairs += 1
    structure = formal_tags + seal_references + tables + checkboxes + list_items
    unsealed_entropy = runons + max(0, caps_clusters - structure)
    return {
        "telemetry_collaborative_terms": float(collaborative),
        "telemetry_command_terms": float(command),
        "telemetry_agent_symmetry_ratio": (collaborative + 1) / (command + 1),
        "telemetry_formal_tags": float(formal_tags),
        "telemetry_seal_references": float(seal_references),
        "telemetry_protocol_structure": float(structure),
        "telemetry_unsealed_entropy": float(unsealed_entropy),
        "telemetry_protocol_discipline": structure / max(1.0, total_words / 100.0),
        "telemetry_caps_clusters": float(caps_clusters),
        "telemetry_swear_terms": float(swear_terms),
        "telemetry_timestamp_markers": float(timestamp_markers),
        "telemetry_rapid_timestamp_pairs": float(rapid_timestamp_pairs),
        "telemetry_manic_velocity": ((caps_clusters * (swear_terms + 1)) + rapid_timestamp_pairs + (timestamp_markers * 0.25)) / max(1.0, total_words / 100.0),
    }


def operator_cluster_hint(features: dict[str, float]) -> str:
    if features.get("telemetry_manic_velocity", 0.0) > 2.0:
        return "rapidthink_velocity_cluster"
    if features.get("telemetry_protocol_discipline", 0.0) > 4.0:
        return "subtle_knife_discipline_cluster"
    if 0.65 <= features.get("telemetry_agent_symmetry_ratio", 0.0) <= 1.6 and (
        features.get("telemetry_collaborative_terms", 0.0) + features.get("telemetry_command_terms", 0.0)
    ) >= 6:
        return "cybernetic_merge_cluster"
    if features.get("rainmaker_corporate_grit_tension", 0.0) > 0.02:
        return "rainmaker_corporate_grit_cluster"
    if features.get("rainmaker_countdown_density", 0.0) > 0.02 or features.get("rainmaker_asset_structuring_weight", 0.0) > 0.04:
        return "rainmaker_package_mode_cluster"
    if features.get("resilience_resource_exhaustion_metric", 0.0) > 0.02:
        return "resource_exhaustion_cluster"
    if features.get("resilience_bureaucratic_weaponization_index", 0.0) > 0.02:
        return "paladin_protocol_cluster"
    if features.get("resilience_swarm_orchestration_density", 0.0) > 0.04:
        return "god_mode_sprint_cluster"
    if features.get("resilience_logic_crucifixion_index", 0.0) > 0.02 or (
        features.get("resilience_hypothesis_terms", 0.0) >= 2 and features.get("resilience_verification_terms", 0.0) >= 2
    ):
        return "crucifixion_by_logic_cluster"
    if features.get("resilience_chaotic_good_tax", 0.0) > 0.02:
        return "chaotic_good_tax_cluster"
    ledger = features.get("operator_ledger_density", 0.0)
    recursion = features.get("operator_recursion_score", 0.0)
    directive = features.get("operator_directive_ratio", 1.0)
    visceral = features.get("operator_visceral_ratio", 0.0)
    tech = features.get("operator_tech_ratio", 0.0)
    legal = features.get("operator_legal_osint_ratio", 0.0)
    target = features.get("operator_target_density", 0.0)
    if (legal > 0.008 or target > 0.006) and directive >= 1.0:
        return "rr_aaron_investigation_cluster"
    if visceral > 0.008 and directive >= 1.5 and tech < 0.004:
        return "ponyboy_protocol_cluster"
    if ledger > 0.035:
        return "t2_grind_cluster"
    if recursion > 0.02 or (features.get("operator_self_reference_terms", 0) >= 2 and features.get("operator_audit_terms", 0) >= 1):
        return "abba3_recursion_cluster"
    if visceral > 0 and tech > 0:
        return "architecture_mutation_cluster"
    if features.get("operator_total_words", 0) > 200 and ledger < 0.01:
        return "reading_ingestion_cluster"
    return "mixed_operator_cluster"


def component_features(comp: dict[str, Any]) -> dict[str, float]:
    content = comp.get("content") or ""
    lines = content.splitlines()
    links = links_from_text(content)
    concepts = comp.get("concepts") or []
    entities = comp.get("entities") or []
    kind = comp.get("component_kind", "")
    features = {
        "char_count": float(len(content)),
        "line_count": float(len(lines)),
        "token_count": float(comp.get("token_count") or token_count(content)),
        "entropy": float(comp.get("entropy") or entropy_for_text(content)),
        "entity_count": float(len(entities)),
        "concept_count": float(len(concepts)),
        "link_count": float(len(links)),
        "code_signal": 1.0 if kind in {"python_function", "python_method", "python_class", "code_symbol", "code_chunk", "markdown_code_block"} else 0.0,
        "goal_signal": float(sum(1 for c in concepts if c.get("concept_kind") in {"goal", "todo"})),
        "friction_signal": float(sum(1 for c in concepts if c.get("concept_kind") == "bug")),
        "identity_signal": float(sum(1 for e in entities if e.get("entity_kind") in {"name", "alias", "phone", "ip", "email", "address"})),
    }
    feature_text = f"{comp.get('title') or ''}\n{comp.get('symbol') or ''}\n{content}"
    features.update(extract_operator_vibes(feature_text))
    features.update(extract_psyche_vibes(feature_text))
    features.update(extract_resilience_vibes(feature_text))
    features.update(extract_rainmaker_vibes(feature_text))
    features.update(extract_operator_telemetry(feature_text))
    return features


def hst_features(features: dict[str, float]) -> dict[str, float]:
    """Bound features to [0,1] because River HalfSpaceTrees assumes that range."""
    denominators = {
        "char_count": 10000.0,
        "line_count": 500.0,
        "token_count": float(MAX_COMPONENT_TOKENS),
        "entropy": 8.0,
        "entity_count": 50.0,
        "concept_count": 25.0,
        "link_count": 25.0,
        "goal_signal": 10.0,
        "friction_signal": 10.0,
        "identity_signal": 25.0,
        "operator_total_words": 500.0,
        "operator_directive_ratio": 10.0,
        "operator_visceral_terms": 25.0,
        "operator_tech_terms": 50.0,
        "operator_legal_osint_terms": 50.0,
        "operator_table_rows": 50.0,
        "operator_checkboxes": 50.0,
        "operator_list_items": 50.0,
        "operator_self_reference_terms": 50.0,
        "operator_audit_terms": 25.0,
        "operator_directives": 50.0,
        "operator_hedges": 25.0,
        "operator_phone_numbers": 25.0,
        "operator_corporate_ids": 25.0,
        "operator_money_mentions": 25.0,
        "psyche_distress_words": 25.0,
        "psyche_forensic_tags": 25.0,
        "psyche_structural_tags": 100.0,
        "psyche_first_person": 100.0,
        "psyche_third_person_self": 100.0,
        "psyche_dissociative_index": 10.0,
        "psyche_bureaucracy_terms": 25.0,
        "psyche_strike_terms": 25.0,
        "resilience_empathic_targets": 50.0,
        "resilience_bureaucratic_nodes": 50.0,
        "resilience_scarcity_markers": 25.0,
        "resilience_body_grief_markers": 25.0,
        "resilience_sprint_entity_mentions": 50.0,
        "resilience_distinct_sprint_entities": 25.0,
        "resilience_timeboxes": 25.0,
        "resilience_hypothesis_terms": 25.0,
        "resilience_verification_terms": 50.0,
        "resilience_conspiracy_grounding_ratio": 8.0,
        "resilience_deliverable_terms": 50.0,
        "resilience_absurdity_terms": 25.0,
        "rainmaker_corporate_terms": 50.0,
        "rainmaker_grit_terms": 25.0,
        "rainmaker_countdown_markers": 25.0,
        "rainmaker_file_extensions": 50.0,
        "rainmaker_phase_markers": 50.0,
        "rainmaker_bold_tags": 50.0,
        "telemetry_collaborative_terms": 50.0,
        "telemetry_command_terms": 50.0,
        "telemetry_agent_symmetry_ratio": 10.0,
        "telemetry_formal_tags": 50.0,
        "telemetry_seal_references": 50.0,
        "telemetry_protocol_structure": 100.0,
        "telemetry_unsealed_entropy": 50.0,
        "telemetry_protocol_discipline": 25.0,
        "telemetry_caps_clusters": 25.0,
        "telemetry_swear_terms": 50.0,
        "telemetry_timestamp_markers": 50.0,
        "telemetry_rapid_timestamp_pairs": 25.0,
        "telemetry_manic_velocity": 25.0,
    }
    bounded: dict[str, float] = {}
    for key, value in features.items():
        denom = denominators.get(key, 1.0)
        bounded[key] = max(0.0, min(1.0, float(value) / denom))
    return bounded


def dbstream_features(features: dict[str, float]) -> dict[str, float]:
    """Small bounded cognitive sticker vector for unsupervised River DBSTREAM."""
    raw = hst_features(features)
    keep = [
        "code_signal",
        "goal_signal",
        "friction_signal",
        "identity_signal",
        "operator_visceral_ratio",
        "operator_tech_ratio",
        "operator_legal_osint_ratio",
        "operator_ledger_density",
        "operator_recursion_score",
        "operator_directive_ratio",
        "operator_target_density",
        "psyche_forensic_shield_ratio",
        "psyche_dissociative_index",
        "psyche_poetic_entropy",
        "psyche_wrath_velocity",
        "resilience_bureaucratic_weaponization_index",
        "resilience_resource_exhaustion_metric",
        "resilience_swarm_orchestration_density",
        "resilience_logic_crucifixion_index",
        "resilience_conspiracy_grounding_ratio",
        "resilience_chaotic_good_tax",
        "rainmaker_corporate_grit_tension",
        "rainmaker_countdown_density",
        "rainmaker_asset_structuring_weight",
        "rainmaker_pitch_formatting_ratio",
        "telemetry_agent_symmetry_ratio",
        "telemetry_protocol_discipline",
        "telemetry_manic_velocity",
    ]
    scaled = {key: raw.get(key, 0.0) for key in keep}
    # Ratios are usually tiny. Expand them so DBSTREAM sees meaningful distance.
    for key, full_scale in {
        "operator_visceral_ratio": 0.04,
        "operator_tech_ratio": 0.08,
        "operator_legal_osint_ratio": 0.08,
        "operator_ledger_density": 0.08,
        "operator_recursion_score": 0.08,
        "operator_target_density": 0.04,
        "psyche_forensic_shield_ratio": 0.08,
        "psyche_poetic_entropy": 0.35,
        "psyche_wrath_velocity": 0.08,
        "resilience_bureaucratic_weaponization_index": 0.08,
        "resilience_resource_exhaustion_metric": 0.08,
        "resilience_swarm_orchestration_density": 0.10,
        "resilience_logic_crucifixion_index": 0.08,
        "resilience_chaotic_good_tax": 0.08,
        "rainmaker_corporate_grit_tension": 0.08,
        "rainmaker_countdown_density": 0.08,
        "rainmaker_asset_structuring_weight": 0.12,
        "rainmaker_pitch_formatting_ratio": 0.08,
        "telemetry_agent_symmetry_ratio": 2.0,
        "telemetry_protocol_discipline": 12.0,
        "telemetry_manic_velocity": 10.0,
    }.items():
        scaled[key] = max(0.0, min(1.0, float(features.get(key, 0.0)) / full_scale))
    return scaled



def heuristic_river_label(comp: dict[str, Any], features: dict[str, float]) -> str:
    kind = comp.get("component_kind", "")
    if features.get("telemetry_manic_velocity", 0.0) > 2.0:
        return "rapidthink_velocity"
    if features.get("telemetry_protocol_discipline", 0.0) > 4.0:
        return "subtle_knife_discipline"
    if 0.65 <= features.get("telemetry_agent_symmetry_ratio", 0.0) <= 1.6 and (
        features.get("telemetry_collaborative_terms", 0.0) + features.get("telemetry_command_terms", 0.0)
    ) >= 6:
        return "cybernetic_merge"
    if features.get("rainmaker_corporate_grit_tension", 0.0) > 0.02:
        return "rainmaker_corporate_grit"
    if features.get("rainmaker_countdown_density", 0.0) > 0.02 or features.get("rainmaker_asset_structuring_weight", 0.0) > 0.04:
        return "rainmaker_package_mode"
    if features.get("resilience_resource_exhaustion_metric", 0.0) > 0.02:
        return "resource_exhaustion_signal"
    if features.get("resilience_bureaucratic_weaponization_index", 0.0) > 0.02:
        return "paladin_protocol"
    if features.get("resilience_swarm_orchestration_density", 0.0) > 0.04:
        return "god_mode_sprint"
    if features.get("resilience_logic_crucifixion_index", 0.0) > 0.02:
        return "crucifixion_by_logic"
    if features.get("resilience_chaotic_good_tax", 0.0) > 0.02:
        return "chaotic_good_tax"
    if features.get("psyche_wrath_velocity", 0.0) > 0.01:
        return "tactical_wrath"
    if features.get("psyche_forensic_shield_ratio", 0.0) > 0.02:
        return "forensic_shield"
    if features.get("psyche_poetic_entropy", 0.0) > 0.08 and features.get("operator_ledger_density", 0.0) < 0.01:
        return "poetic_purge"
    if features.get("psyche_dissociative_index", 0.0) >= 2.0 and features.get("psyche_third_person_self", 0.0) >= 2:
        return "third_person_dissociation"
    hint = operator_cluster_hint(features)
    if hint != "mixed_operator_cluster":
        return hint
    if features.get("goal_signal", 0) > 0:
        return "goal_or_todo"
    if kind in {"python_function", "python_method", "python_class", "code_symbol", "markdown_code_block", "code_chunk"}:
        return "algorithm_or_tool"
    if features.get("friction_signal", 0) > 0:
        return "friction_or_bug"
    if features.get("identity_signal", 0) >= 2:
        return "identity_rich"
    if kind.startswith("markdown") or kind in {"text_chunk", "document_chunk"}:
        return "institutional_memory"
    return "archive_component"



def sticker_score(value: float, full_scale: float) -> int:
    return max(0, min(10000, int(round((float(value) / max(full_scale, 1e-9)) * 10000))))


def vibe_tags_for(comp: dict[str, Any]) -> list[tuple[str, int, str]]:
    tags: dict[str, tuple[int, str]] = {}
    kind = comp.get("component_kind", "")
    text = (comp.get("title") or "") + "\n" + (comp.get("symbol") or "") + "\n" + (comp.get("content") or "")[:5000]
    low = text.lower()
    sticker_features = {}
    sticker_features.update(extract_operator_vibes(text))
    sticker_features.update(extract_psyche_vibes(text))
    sticker_features.update(extract_resilience_vibes(text))
    sticker_features.update(extract_rainmaker_vibes(text))
    sticker_features.update(extract_operator_telemetry(text))
    if kind in {"python_function", "python_method", "code_symbol", "markdown_code_block", "code_chunk"}:
        tags["algorithm"] = (7000, "component_kind")
    if re.search(r"\b(goal|objective|mission|todo|action item|next)\b", low):
        tags["goal"] = (6500, "keyword")
    if re.search(r"\b(bug|broken|issue|risk|fail|error|blocker|debt)\b", low):
        tags["friction"] = (6500, "keyword")
    if re.search(r"\b(evidence|source|citation|proof|artifact|hash|sha256)\b", low):
        tags["evidence"] = (5500, "keyword")
    if re.search(r"\b(see also|depends on|uses|calls|imports|links to|->|=>)\b", low):
        tags["relationship"] = (5500, "keyword")
    if re.search(r"@|\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b|\b(?:\d{1,3}\.){3}\d{1,3}\b", low):
        tags["identity"] = (6000, "entity_signal")
    if kind.startswith("markdown") or re.search(r"\b(note|remember|institutional|history|legacy)\b", low):
        tags["memory"] = (5000, "memory_signal")
    if sticker_features.get("operator_visceral_ratio", 0.0) > 0.004 and sticker_features.get("operator_tech_ratio", 0.0) > 0.004:
        tags["visceral_tech"] = (max(sticker_score(sticker_features["operator_visceral_ratio"], 0.04), sticker_score(sticker_features["operator_tech_ratio"], 0.08)), "operator_sticker")
    if sticker_features.get("operator_ledger_density", 0.0) > 0.02:
        tags["ledger_density"] = (sticker_score(sticker_features["operator_ledger_density"], 0.08), "operator_sticker")
    if sticker_features.get("operator_recursion_score", 0.0) > 0.005 or (sticker_features.get("operator_self_reference_terms", 0) >= 2 and sticker_features.get("operator_audit_terms", 0) >= 1):
        tags["metacognitive_recursion"] = (max(5500, sticker_score(sticker_features["operator_recursion_score"], 0.08)), "operator_sticker")
    if sticker_features.get("operator_directive_ratio", 0.0) >= 2.0 or sticker_features.get("operator_directives", 0.0) >= 3:
        tags["directive_command"] = (sticker_score(sticker_features["operator_directive_ratio"], 8.0), "operator_sticker")
    if sticker_features.get("operator_legal_osint_ratio", 0.0) > 0.006 or sticker_features.get("operator_target_density", 0.0) > 0.003:
        tags["osint_targeting"] = (max(sticker_score(sticker_features["operator_legal_osint_ratio"], 0.08), sticker_score(sticker_features["operator_target_density"], 0.04)), "operator_sticker")
    if sticker_features.get("psyche_forensic_shield_ratio", 0.0) > 0.01:
        tags["forensic_shield"] = (sticker_score(sticker_features["psyche_forensic_shield_ratio"], 0.08), "psyche_sticker")
    if sticker_features.get("psyche_poetic_entropy", 0.0) > 0.05 and sticker_features.get("operator_ledger_density", 0.0) < 0.01:
        tags["poetic_purge"] = (sticker_score(sticker_features["psyche_poetic_entropy"], 0.35), "psyche_sticker")
    if sticker_features.get("psyche_dissociative_index", 0.0) >= 1.5 and sticker_features.get("psyche_third_person_self", 0.0) >= 2:
        tags["dissociative_index"] = (sticker_score(sticker_features["psyche_dissociative_index"], 8.0), "psyche_sticker")
    if sticker_features.get("psyche_wrath_velocity", 0.0) > 0.005:
        tags["tactical_wrath"] = (sticker_score(sticker_features["psyche_wrath_velocity"], 0.08), "psyche_sticker")
    if sticker_features.get("resilience_bureaucratic_weaponization_index", 0.0) > 0.01:
        tags["paladin_protocol"] = (sticker_score(sticker_features["resilience_bureaucratic_weaponization_index"], 0.08), "resilience_sticker")
    if sticker_features.get("resilience_resource_exhaustion_metric", 0.0) > 0.01:
        tags["resource_exhaustion"] = (sticker_score(sticker_features["resilience_resource_exhaustion_metric"], 0.08), "resilience_sticker")
    if sticker_features.get("resilience_swarm_orchestration_density", 0.0) > 0.025:
        tags["god_mode_sprint"] = (sticker_score(sticker_features["resilience_swarm_orchestration_density"], 0.10), "resilience_sticker")
    if sticker_features.get("resilience_logic_crucifixion_index", 0.0) > 0.01 or (
        sticker_features.get("resilience_hypothesis_terms", 0) >= 2 and sticker_features.get("resilience_verification_terms", 0) >= 2
    ):
        tags["conspiracy_grounding"] = (max(sticker_score(sticker_features["resilience_logic_crucifixion_index"], 0.08), sticker_score(sticker_features["resilience_conspiracy_grounding_ratio"], 8.0)), "resilience_sticker")
    if sticker_features.get("resilience_chaotic_good_tax", 0.0) > 0.01:
        tags["chaotic_good_tax"] = (sticker_score(sticker_features["resilience_chaotic_good_tax"], 0.08), "resilience_sticker")
    if sticker_features.get("rainmaker_corporate_grit_tension", 0.0) > 0.01:
        tags["rainmaker_corporate_grit"] = (sticker_score(sticker_features["rainmaker_corporate_grit_tension"], 0.08), "rainmaker_sticker")
    if sticker_features.get("rainmaker_countdown_density", 0.0) > 0.01:
        tags["capital_velocity"] = (sticker_score(sticker_features["rainmaker_countdown_density"], 0.08), "rainmaker_sticker")
    if sticker_features.get("rainmaker_asset_structuring_weight", 0.0) > 0.02:
        tags["deliverable_grid"] = (sticker_score(sticker_features["rainmaker_asset_structuring_weight"], 0.12), "rainmaker_sticker")
    if sticker_features.get("rainmaker_pitch_formatting_ratio", 0.0) > 0.01:
        tags["pitch_formatting"] = (sticker_score(sticker_features["rainmaker_pitch_formatting_ratio"], 0.08), "rainmaker_sticker")
    if 0.65 <= sticker_features.get("telemetry_agent_symmetry_ratio", 0.0) <= 1.6 and (
        sticker_features.get("telemetry_collaborative_terms", 0.0) + sticker_features.get("telemetry_command_terms", 0.0)
    ) >= 6:
        tags["cybernetic_merge"] = (max(5000, 10000 - sticker_score(abs(sticker_features["telemetry_agent_symmetry_ratio"] - 1.0), 1.0)), "telemetry_sticker")
    if sticker_features.get("telemetry_protocol_discipline", 0.0) > 2.0:
        tags["subtle_knife_discipline"] = (sticker_score(sticker_features["telemetry_protocol_discipline"], 12.0), "telemetry_sticker")
    if sticker_features.get("telemetry_manic_velocity", 0.0) > 1.0:
        tags["rapidthink_velocity"] = (sticker_score(sticker_features["telemetry_manic_velocity"], 10.0), "telemetry_sticker")
    return [(k, v[0], v[1]) for k, v in sorted(tags.items())]
