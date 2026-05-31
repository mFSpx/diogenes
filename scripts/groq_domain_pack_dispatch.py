#!/usr/bin/env python3
"""Groq dispatch: domain packs + Krampus Express port stubs."""
import json, urllib.request, os, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEY = os.environ["GROQ_API_KEY"]
URL = "https://api.groq.com/openai/v1/chat/completions"
HDR = {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json", "User-Agent": "groq-python/0.28.0"}

PACK_FORMAT = """ORNAMENT YAML pack format:
meta:
  id: <id>
  version: "1.0.0"
  name: "<name>"
  description: >
    <one paragraph>
signals:
  - id: <signal_id>
    display: "<display text>"
    default_dimensions: [dim1, dim2]
    default_weight: strong|moderate|weak
    ownership_hint: person_linked|company_linked|system_linked
    intentionality_hint: intentional_display|functional|accidental
clusters:
  - id: <cluster_id>
    name: "<name>"
    signal_ids: [sig1, sig2]
    min_signals: 2
hypotheses:
  - id: <hyp_id>
    statement: "<statement>"
    required_clusters: [cluster_id]
    min_cluster_strength: 0.5
    category: legal|financial|behavioral|operational
questions:
  - id: <q_id>
    target_hypothesis: <hyp_id>
    text: "<question>"
    rationale: "<why>"
    expected_information_gain: 0.85
"""

def groq_call(prompt, out_file, max_tokens=2500):
    body = json.dumps({"model": "llama-3.3-70b-versatile", "temperature": 0.15,
        "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(URL, data=body, headers=HDR)
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.load(r)
    code = d["choices"][0]["message"]["content"].strip()
    if "```" in code:
        code = code.split("```", 1)[1]
        if "\n" in code: code = code.split("\n", 1)[1]
        code = code.rsplit("```", 1)[0].strip()
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    Path(out_file).write_text(code)
    print(f"[groq] {out_file} ({len(code)}b)")

WO = [
    ("investigative_journalism",
     f"Write ORNAMENT YAML signal pack for investigative journalism.\n{PACK_FORMAT}\nid: investigative_journalist, name: Investigative Journalist Profile\n18+ signals: source protection, leaked documents, FOIA requests, whistleblower contact, document authentication, corporate registry research, offshore structures, encrypted comms, counter-surveillance, legal threat response, publication timing, financial paper trails, anonymous tips, source vetting, press freedom monitoring, defamation risk, shadow banning detection, metadata hygiene.\n5 clusters: source_protection, document_forensics, financial_investigation, publication_strategy, legal_defense\n5 hypotheses targeting investigative activity, source exposure risk, financial crime investigation, legal pressure, undercover operation\n8 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/profession/investigative_journalist_pack.yaml"),

    ("intelligence_osint",
     f"Write ORNAMENT YAML signal pack for intelligence analysis and OSINT.\n{PACK_FORMAT}\nid: intelligence_analyst, name: Intelligence Analyst OSINT Operator\n20+ signals: open source collection, social media pivot, corporate registry pivot, geolocation from imagery, alias detection, entity disambiguation, communication metadata, dark web monitoring, pattern of life analysis, financial network mapping, adversary profiling, counter-intelligence indicators, travel pattern analysis, data broker exposure, operational security gaps, network graph analysis, satellite imagery analysis, vessel/aircraft tracking.\n5 clusters: osint_collection, network_analysis, identity_resolution, opsec_gaps, adversary_profiling\n5 hypotheses. 8 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/profession/intelligence_analyst_pack.yaml"),

    ("creator_opsec",
     f"Write ORNAMENT YAML signal pack for content creator OPSEC — adult creators, fan platforms.\n{PACK_FORMAT}\nid: creator_opsec, name: Content Creator Fan Platform OPSEC\n18+ signals: real name exposure, home address leakage, payment processor trail, fan doxxing attempts, screenshot leaks, watermarking gaps, VPN usage, alternate identity, payment anonymization, DMCA patterns, platform account separation, device hygiene, metadata in content, geolocation exposure, chargeback fraud, subscription data exposure, stalker escalation, legal threat from fans.\n5 clusters: identity_protection, payment_privacy, content_security, fan_threat_detection, operational_separation\n5 hypotheses. 8 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/opsec/creator_opsec_pack.yaml"),

    ("activist_support",
     f"Write ORNAMENT YAML signal pack for activist and organizer support.\n{PACK_FORMAT}\nid: activist_profile, name: Activist Organizer Support Profile\n18+ signals: protest attendance, coalition affiliations, legal observer access, encrypted comms, police contact history, civil disobedience record, surveillance awareness, solidarity network, bail fund knowledge, infiltration detection, secure drop usage, press relationships, government watchlist indicators, doxxing exposure, device seizure risk, social media scraping exposure, funding source visibility, foreign agent designation risk.\n5 clusters: organizing_activity, security_culture, legal_preparedness, coalition_infrastructure, surveillance_exposure\n5 hypotheses. 8 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/profession/activist_organizer_pack.yaml"),

    ("publishing_house",
     f"Write ORNAMENT YAML signal pack for publishing house operations.\n{PACK_FORMAT}\nid: publishing_operations, name: Publishing House Operations Intelligence\n16+ signals: manuscript acquisition pipeline, author advance structure, foreign rights deals, distribution channel relationships, literary agent network, editorial calendar pressure, printing lead times, bookstore relationships, AI content detection policy, backlist revenue dependence, digital pivot signals, review copy strategy, award submission patterns, author platform requirements, self-publishing competition, subscription model adoption.\n4 clusters: acquisition_pipeline, author_relations, distribution_network, market_positioning\n4 hypotheses. 6 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/industry/publishing_house_pack.yaml"),

    ("therapy_support",
     f"Write ORNAMENT YAML signal pack for therapeutic support and mental health professional profiling.\n{PACK_FORMAT}\nid: therapy_profile, name: Therapeutic Relationship and Mental Health Support\n16+ signals: therapeutic alliance signals, trauma disclosure patterns, crisis escalation indicators, medication adherence, social support network, self-harm risk factors, substance use patterns, help-seeking behavior, stigma internalization, treatment compliance, care gap indicators, family system stress, economic stressor exposure, sleep disruption signals, occupational functioning, digital mental health tool usage.\n4 clusters: crisis_indicators, therapeutic_engagement, social_support, functional_assessment\n4 hypotheses. 6 questions. Output ONLY the YAML.",
     "BOOKS/ontology_packs/ornament/mental_health/therapy_support_pack.yaml"),

    ("lever_ledger",
     """Write scripts/lever_ledger.py for LUCIDOTA. No fences.
Reads staging_packet from lucidota_storage (psycopg2, LUCIDOTA_GO_STORAGE_DSN env default postgresql:///lucidota_storage).
Computes per-entity-cluster leverage score from 6 levers:
1. RTB_VIOLATION: count GRIP+SNARE terms * 200
2. UNLICENSED_OP: count LIE terms * 250
3. FINANCIAL_EXPOSURE: sum confidence_bps for EVIDENCE terms / 100
4. DISPLACEMENT_PATTERN: count SNARE terms * 150
5. REGULATORY_ATTENTION: count rows where parser_name contains 'regulatory' or 'ornament' * 100
6. NETWORK_CENTRALITY: count distinct source_id prefix segments * 50
Entity cluster = group by first 40 chars of source_id.
Risk bands: Watch<25, Investigate<50, High<75, Priority>=75.
Writes 05_OUTPUTS/risk/lever_ledger_<ts>.json with full results.
Prints summary table. mutation_class: read_only + receipt_only.
sys.path.insert for ROOT. Uses ROOT/05_OUTPUTS. Output ONLY the script.""",
     "scripts/lever_ledger.py"),

    ("pivot_search",
     """Write scripts/pivot_search.py for LUCIDOTA. No fences.
CLI: --entity TEXT --depth 1|2 (default 1). psycopg2, LUCIDOTA_GO_STORAGE_DSN.
1. Find matching staging_packet rows WHERE claim ILIKE %entity% OR raw_anchor ILIKE %entity%
2. For each match, find all staging_packets sharing same source_id[:40] prefix (co-document entities)
3. Depth 2: repeat for each co-document entity
4. Build network dict: {entity_term: {co_occurring_terms: [list], degree: int, source_ids: [list]}}
5. Sort by degree descending
6. Write 05_OUTPUTS/network/pivot_<slug>_<ts>.json
7. Print top-15 network table
mutation_class: read_only + receipt_only. Output ONLY the script.""",
     "scripts/pivot_search.py"),

    ("pressure_dossier",
     """Write scripts/pressure_dossier.py for LUCIDOTA. No fences.
CLI: --entity TEXT. psycopg2, LUCIDOTA_GO_STORAGE_DSN.
1. Collect all staging_packet rows WHERE claim ILIKE %entity% OR raw_anchor ILIKE %entity%
2. Group by proposed_term: GRIP, SNARE, LIE, CLAIM, EVIDENCE, GROUP, ENTITY
3. Build dossier: {entity, collected_at, total_signals, risk_band, evidence_groups: {}, top_claims: [top5 by confidence_bps], source_diversity: count(distinct source prefixes), recommended_actions: []}
4. recommended_actions: GRIP>3 -> 'File regulatory complaint'; LIE>2 -> 'Request disclosure'; SNARE>2 -> 'Document displacement pattern'; source_diversity>5 -> 'Expand network pivot'
5. risk_band from total_signals: <5=Watch, <15=Investigate, <30=High, >=30=Priority
6. Write 05_OUTPUTS/dossiers/dossier_<slug>_<ts>.json. Print summary.
mutation_class: read_only + receipt_only. Output ONLY the script.""",
     "scripts/pressure_dossier.py"),
]

results = {}
threads = []

def run(name, prompt, out_file):
    try:
        groq_call(prompt, out_file)
        results[name] = "ok"
    except Exception as e:
        results[name] = f"ERR: {e}"

for name, prompt, out_file in WO:
    t = threading.Thread(target=run, args=(name, prompt, out_file))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

for k, v in sorted(results.items()):
    print(f"{k}: {v}")
