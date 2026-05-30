# DARWIN HAMMER — match 1312, survivor 1
# gen: 5
# parent_a: hybrid_hybrid_decision_hygi_decreasing_pruning_m17_s4.py (gen2)
# parent_b: hybrid_hybrid_hybrid_fisher_hybrid_rlct_grokking_m224_s0.py (gen4)
# born: 2026-05-29T23:35:17Z

import numpy as np
import math
import random
import sys
import pathlib
import re

def gaussian_beam(theta: float, center: float, width: float) -> float:
    if width <= 0.0:
        raise ValueError("width must be positive")
    z = (theta - center) / width
    return math.exp(-0.5 * z * z)

def fisher_score(theta: float, center: float, width: float, eps: float = 1e-12) -> float:
    intensity = max(gaussian_beam(theta, center, width), eps)
    derivative = intensity * (-(theta - center) / (width * width))
    return (derivative * derivative) / intensity

def calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn):
    dVdt = (I_syn - g_L * (V - E_L) - g_Na * m * m * m * h * (V - E_Na) - g_K * n * n * n * n * (V - E_K)) / C_m
    return dVdt

def _raw_counts(text: str) -> dict[str, int]:
    EVIDENCE_RE = re.compile(
        r"\b(?:evidence|verify|verified|confirm|confirmed|source|sourced|citation|receipt|hash|sha256|screenshot|record|log|document|proof|fact|facts|check|checked|audit)\b",
        re.I,
    )
    PLANNING_RE = re.compile(
        r"\b(?:plan|checklist|steps?|sequence|timeline|roadmap|phase|priority|prioritize|triage|criteria|protocol|procedure|schedule|budget|scope|test|smoke)\b",
        re.I,
    )
    DELAY_RE = re.compile(
        r"\b(?:pause|sleep|wait|tomorrow|later|hold|cool down|de[- ]?escalate|not now|before i|first|after|review)\b",
        re.I,
    )
    SUPPORT_RE = re.compile(
        r"\b(?:ask|call|text|friend|friends|rowyn|kai|chance|doctor|therapist|lawyer|advocate|support|help|community|team|handoff|delegate)\b",
        re.I,
    )
    BOUNDARY_RE = re.compile(
        r"\b(?:boundary|boundaries|walk away|no contact|do not|don't|stop|limit|refuse|protect|safe|safety|lock|redact|privacy|permission|consent)\b",
        re.I,
    )
    OUTCOME_RE = re.compile(
        r"\b(?:done|shipped|finished|complete|resolved|succeeded|passed|paid|safe|sent|filed|closed|fixed|working|green|verified)\b",
        re.I,
    )
    IMPULSIVE_RE = re.compile(
        r"\b(?:rage|impulsive|panic|panicking|spiral|doom|fuck it|right now|immediately|tonight|destroy|revenge|scorched|burn it|quit|give up)\b",
        re.I,
    )
    SCARCITY_RE = re.compile(
        r"\b(?:broke|homeless|last\s+\$?\d+|\$\s?\d+\s+to my name|can't afford|cannot afford|rent due|no money|starving|desperate|no sleep|exhausted)\b",
        re.I,
    )
    RISK_RE = re.compile(
        r"\b(?:kill|die|suicide|suicidal|self[- ]?harm|overdose|hurt myself|danger|unsafe|crisis|collapse)\b",
        re.I,
    )

    return {
        "evidence_count": len(EVIDENCE_RE.findall(text or "")),
        "planning_count": len(PLANNING_RE.findall(text or "")),
        "delay_count": len(DELAY_RE.findall(text or "")),
        "support_count": len(SUPPORT_RE.findall(text or "")),
        "boundary_count": len(BOUNDARY_RE.findall(text or "")),
        "outcome_count": len(OUTCOME_RE.findall(text or "")),
        "impulsive_count": len(IMPULSIVE_RE.findall(text or "")),
        "scarcity_count": len(SCARCITY_RE.findall(text or "")),
        "risk_count": len(RISK_RE.findall(text or "")),
    }

def hybrid_decision_hygi_decreasing_pruning_rlct_grokking(text: str) -> dict[str, float]:
    raw_counts = _raw_counts(text)
    evidence_count = raw_counts["evidence_count"]
    planning_count = raw_counts["planning_count"]
    delay_count = raw_counts["delay_count"]
    support_count = raw_counts["support_count"]
    boundary_count = raw_counts["boundary_count"]
    outcome_count = raw_counts["outcome_count"]
    impulsive_count = raw_counts["impulsive_count"]
    scarcity_count = raw_counts["scarcity_count"]
    risk_count = raw_counts["risk_count"]

    V = -70.0  # initial membrane potential
    C_m = 1.0  # membrane capacitance
    g_L = 0.1  # passive leak conductance
    E_L = -54.4  # leak reversal potential
    g_Na = 120.0  # maximum Na+ conductance
    E_Na = 50.0  # Na+ reversal potential
    m = 0.5  # Na+ activation gate variable
    h = 0.5  # Na+ inactivation gate variable
    g_K = 36.0  # maximum K+ conductance
    E_K = -77.0  # K+ reversal potential
    n = 0.5  # K+ activation gate variable
    I_syn = 10.0  # synaptic current

    dVdt = calculate_membrane_potential(V, C_m, g_L, E_L, g_Na, E_Na, m, h, g_K, E_K, n, I_syn)

    theta = 0.5  # angle
    center = 0.0  # center of gaussian beam
    width = 1.0  # width of gaussian beam
    fisher = fisher_score(theta, center, width)

    decision = (evidence_count * 0.1 + planning_count * 0.2 + delay_count * 0.1 + 
                 support_count * 0.1 + boundary_count * 0.1 + outcome_count * 0.2 + 
                 impulsive_count * 0.05 + scarcity_count * 0.05 + risk_count * 0.05) * np.tanh(dVdt)

    return {
        "membrane_potential": dVdt,
        "fisher_score": fisher,
        "decision": decision,
    }

if __name__ == "__main__":
    text = "I have evidence that this plan will work. I have a checklist and a timeline. I will delay the decision until tomorrow."
    result = hybrid_decision_hygi_decreasing_pruning_rlct_grokking(text)
    print(result)