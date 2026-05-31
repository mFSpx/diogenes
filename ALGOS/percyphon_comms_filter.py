#!/usr/bin/env python3
"""percyphon_comms_filter.py — deterministic comms/identity/proxy/burn filter.

Mutation class: read_only
No DB writes. No model calls. No randomness.
Pure keyword-set classification over text snippets.

Wires to percyphon_village for villager scaffold enrichment (read-only).
"""
from __future__ import annotations

import json
import os
from typing import Any


# ---------------------------------------------------------------------------
# Confidence basis points scale
# Allowed values: {0, 2, 4, 6, 10, 50, 69, 150}
# ---------------------------------------------------------------------------

_BPS_NONE = 0
_BPS_NOISE = 2
_BPS_WEAK = 4
_BPS_LOW = 6
_BPS_MID = 10
_BPS_SOLID = 50
_BPS_STRONG = 69
_BPS_CONFIRMED = 150

_VALID_BPS = {_BPS_NONE, _BPS_NOISE, _BPS_WEAK, _BPS_LOW, _BPS_MID,
              _BPS_SOLID, _BPS_STRONG, _BPS_CONFIRMED}

# Pattern class identifiers
PATTERN_CLASSES = ["vpn", "proxy", "burn_phone", "identity_change",
                   "anonymous_relay", "none"]


# ---------------------------------------------------------------------------
# Keyword sets — each entry is a (signal_label, keyword_tuple) pair.
# Keywords are matched case-insensitively against the lowercased text.
# More specific multi-word signals are listed first within each class.
# ---------------------------------------------------------------------------

_VPN_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("vpn_protocol",       ("wireguard", "openvpn", "ikev2", "ipsec tunnel", "l2tp")),
    ("vpn_service_named",  ("nordvpn", "expressvpn", "mullvad", "protonvpn",
                            "surfshark", "cyberghost", "pia vpn", "private internet access")),
    ("vpn_generic",        ("vpn", "virtual private network")),
    ("vpn_exit_node",      ("exit node", "vpn exit", "vpn gateway", "vpn server")),
    ("vpn_tunnel",         ("vpn tunnel", "tunneled", "tunnel endpoint")),
    ("kill_switch",        ("kill switch", "vpn kill")),
    ("split_tunneling",    ("split tunneling", "split-tunnel")),
]

_PROXY_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("socks5_proxy",       ("socks5", "socks 5")),
    ("socks_proxy",        ("socks proxy", "socks4")),
    ("http_proxy",         ("http_proxy", "https_proxy", "http proxy", "https proxy",
                            "proxy server", "proxy port")),
    ("reverse_proxy",      ("reverse proxy", "nginx proxy", "haproxy")),
    ("transparent_proxy",  ("transparent proxy", "tproxy")),
    ("squid_proxy",        ("squid", "proxy_pass")),
    ("proxy_generic",      ("proxy", "proxied", "proxying")),
    ("rotating_proxy",     ("rotating proxy", "ip rotation", "proxy pool")),
]

_BURN_PHONE_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("burner_explicit",    ("burner phone", "burn phone", "throwaway phone",
                            "disposable phone", "prepaid burner")),
    ("prepaid_sim",        ("prepaid sim", "prepaid card", "prepaid number",
                            "pay-as-you-go", "pay as you go", "no-contract phone")),
    ("temporary_number",   ("temp number", "temporary number", "disposable number",
                            "virtual number", "one-time number")),
    ("second_number_app",  ("google voice", "textnow", "burner app", "hushed",
                            "talkatone", "line2", "grasshopper")),
    ("sim_swap",           ("sim swap", "sim replacement", "sim change")),
    ("imei_change",        ("imei", "device identifier", "phone swap")),
]

_IDENTITY_CHANGE_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("alias_creation",     ("new alias", "create alias", "alternate identity",
                            "fake name", "false name", "assumed name")),
    ("fake_id",            ("fake id", "forged id", "counterfeit id",
                            "false document", "forged document")),
    ("name_change",        ("name change", "changed name", "new name", "rename")),
    ("new_identity",       ("new identity", "fresh identity", "clean identity",
                            "identity change", "identity switch")),
    ("persona_switch",     ("persona", "sock puppet", "sockpuppet", "alt account",
                            "alternate account", "throwaway account")),
    ("dox_evasion",        ("doxxing evasion", "dox proof", "anti-dox", "go dark")),
    ("legend_building",    ("legend", "backstory", "cover story", "cover identity")),
]

_ANON_RELAY_SIGNALS: list[tuple[str, tuple[str, ...]]] = [
    ("tor_network",        ("tor browser", "tor network", "onion routing",
                            ".onion", "tails os", "tails linux")),
    ("i2p_network",        ("i2p", "invisible internet project")),
    ("mixnet",             ("mixnet", "mix network", "nym network", "loopix")),
    ("email_relay",        ("anonymous email", "temp email", "throwaway email",
                            "guerrillamail", "mailinator", "10minutemail",
                            "email relay", "email alias")),
    ("anonymous_remailer", ("remailer", "anon mailer", "cypherpunk remailer")),
    ("zero_knowledge",     ("zero knowledge", "zero-knowledge", "zk proof")),
    ("encrypted_messaging",("signal app", "wickr", "briar", "session messenger",
                            "element.io", "matrix protocol")),
    ("relay_generic",      ("anonymous relay", "anon relay", "relay node")),
]


# ---------------------------------------------------------------------------
# Internal: score a text against one class's signal list
# ---------------------------------------------------------------------------

def _match_signals(text_lower: str,
                   signal_defs: list[tuple[str, tuple[str, ...]]],
                   ) -> list[str]:
    """Return list of signal labels that fired."""
    matched: list[str] = []
    for label, keywords in signal_defs:
        for kw in keywords:
            if kw in text_lower:
                matched.append(label)
                break  # one match per label sufficient
    return matched


def _bps_from_hit_count(n: int) -> int:
    """Map number of signal hits to a confidence_bps value."""
    if n == 0:
        return _BPS_NONE
    if n == 1:
        return _BPS_MID       # 10 bps — single weak signal
    if n == 2:
        return _BPS_SOLID     # 50 bps — two corroborating signals
    if n == 3:
        return _BPS_STRONG    # 69 bps — three signals
    return _BPS_CONFIRMED     # 150 bps — four or more signals


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_comms_pattern(text: str) -> dict[str, Any]:
    """Classify a text snippet for comms infrastructure patterns.

    Returns:
        pattern_class: one of PATTERN_CLASSES
        confidence_bps: int from {0, 2, 4, 6, 10, 50, 69, 150}
        signals: list of detected signal strings
    """
    t = text.lower()

    scores: dict[str, list[str]] = {
        "vpn":              _match_signals(t, _VPN_SIGNALS),
        "proxy":            _match_signals(t, _PROXY_SIGNALS),
        "burn_phone":       _match_signals(t, _BURN_PHONE_SIGNALS),
        "identity_change":  _match_signals(t, _IDENTITY_CHANGE_SIGNALS),
        "anonymous_relay":  _match_signals(t, _ANON_RELAY_SIGNALS),
    }

    # Pick the class with the most signal hits; ties broken by definition order
    best_class = "none"
    best_hits: list[str] = []
    for cls in ["vpn", "proxy", "burn_phone", "identity_change", "anonymous_relay"]:
        if len(scores[cls]) > len(best_hits):
            best_class = cls
            best_hits = scores[cls]

    confidence_bps = _bps_from_hit_count(len(best_hits))

    return {
        "pattern_class": best_class,
        "confidence_bps": confidence_bps,
        "signals": best_hits,
    }


def classify_batch(texts: list[str]) -> list[dict[str, Any]]:
    """Batch version of classify_comms_pattern."""
    return [classify_comms_pattern(t) for t in texts]


# ---------------------------------------------------------------------------
# Villager enrichment (read-only DB read)
# ---------------------------------------------------------------------------

def enrich_villager(vuuid: str, text: str, dsn: str) -> dict[str, Any]:
    """Read a villager scaffold from percyphon_village, classify text, return
    enriched dict.  Does NOT write to DB.  Mutation class: read_only.

    Returns the villager scaffold fields merged with the comms classification.
    Raises ValueError if vuuid is not found.
    """
    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT vuuid, name, persona, alias, ternary_state, "
                "       relevance_confidence_bps, seed, authority, slots "
                "FROM lucidota_go.percyphon_village "
                "WHERE vuuid = %s",
                (vuuid,),
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        raise ValueError(f"villager {vuuid!r} not found in percyphon_village")

    classification = classify_comms_pattern(text)

    return {
        "vuuid": row["vuuid"],
        "name": row["name"],
        "persona": row["persona"],
        "alias": row["alias"],
        "ternary_state": row["ternary_state"],
        "relevance_confidence_bps": row["relevance_confidence_bps"],
        "seed": row["seed"],
        "authority": row["authority"],
        "slot_count": len(row["slots"]) if isinstance(row["slots"], list) else 0,
        "comms_classification": classification,
        "enriched_text_snippet": text[:200],
        "mutation_class": "read_only",
        "schema": "lucidota.percyphon.comms_filter.v1",
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

_SMOKE_TEXTS = [
    # vpn
    (
        "Using NordVPN with WireGuard protocol. Kill switch enabled. "
        "Split tunneling for streaming services. Exit node in Netherlands.",
        "vpn",
    ),
    # proxy
    (
        "Set up a SOCKS5 proxy at localhost:1080 with HTTP proxy on port 8080. "
        "Rotating proxy pool for scraping. Proxy pool size 500.",
        "proxy",
    ),
    # burn_phone
    (
        "Bought a prepaid SIM, pay-as-you-go plan. Using TextNow as a burner app "
        "for a temporary number. No-contract phone registered to nothing.",
        "burn_phone",
    ),
    # identity_change
    (
        "Created a new alias with a fake name. Sock puppet account with alternate identity. "
        "Legend: cover story as a travel blogger. Fresh identity package ready.",
        "identity_change",
    ),
    # anonymous_relay
    (
        "Routing through Tor network, .onion address. Anonymous email via Mailinator. "
        "Signal app for messaging. Zero-knowledge proof for authentication.",
        "anonymous_relay",
    ),
    # none
    (
        "Ordered a pizza from the local shop. Weather is nice today. "
        "Will check the garden later. Nothing suspicious here.",
        "none",
    ),
]


def run_smoke_test() -> None:
    """Run 6 example texts; print classification results and pattern class counts."""
    print("[percyphon_comms_filter] smoke test")
    print("-" * 60)

    counts: dict[str, int] = {cls: 0 for cls in PATTERN_CLASSES}
    results = []

    for text, expected in _SMOKE_TEXTS:
        result = classify_comms_pattern(text)
        cls = result["pattern_class"]
        bps = result["confidence_bps"]
        sigs = result["signals"]
        match_marker = "OK" if cls == expected else f"MISMATCH (expected {expected})"
        print(f"  class={cls:<18} bps={bps:>4}  signals={sigs}  [{match_marker}]")
        print(f"    text: {text[:80]}...")
        counts[cls] += 1
        results.append(result)

    print("-" * 60)
    print("[pattern class counts]")
    for cls, n in counts.items():
        bar = "#" * n
        print(f"  {cls:<18}: {n} {bar}")

    mismatches = sum(
        1 for (_, exp), res in zip(_SMOKE_TEXTS, results)
        if res["pattern_class"] != exp
    )
    print(f"\n[smoke test] {len(_SMOKE_TEXTS)} texts, {mismatches} mismatches")
    if mismatches == 0:
        print("[smoke test] PASS")
    else:
        print("[smoke test] FAIL — check keyword coverage above")


if __name__ == "__main__":
    run_smoke_test()
