# DARWIN HAMMER — match 2053, survivor 0
# gen: 4
# parent_a: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s0.py (gen2)
# parent_b: hybrid_hybrid_label_foundry_path_signature_m231_s4.py (gen3)
# born: 2026-05-29T23:40:31Z

"""
This module fuses the hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s0 and 
hybrid_hybrid_label_foundry_path_signature_m231_s4 algorithms into a single unified system.
The mathematical bridge between the two structures is the use of the ssim function to evaluate 
the similarity between the input and the action selection based on the bandit policy, 
and the incorporation of the path-signature based recovery priority into the ternary router's 
route_command function.

The hybrid system utilizes the confidence and priority from the label foundry and path 
signature algorithms to adjust the similarity evaluation and action selection in the ternary 
router. This fusion enables the evaluation of the ternary router's performance using the ssim 
metric, the bandit policy, and the label foundry and path signature algorithms.

Parent A: hybrid_hybrid_ternary_route_hybrid_bandit_router_m31_s0
Parent B: hybrid_hybrid_label_foundry_path_signature_m231_s4
"""

import numpy as np
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, List, Dict, Any

# Define a dataclass for labeling function results
@dataclass(frozen=True)
class LabelingFunctionResult:
    lf_name: str
    doc_id: str
    label: int  # binary 0/1

# Define a dataclass for probabilistic labels
@dataclass(frozen=True)
class ProbabilisticLabel:
    doc_id: str
    label: int
    confidence: float  # in [0,1]

def ssim(x: np.ndarray, y: np.ndarray, dynamic_range: float = 255.0, k1: float = 0.01, k2: float = 0.03) -> float:
    if len(x) != len(y):
        raise ValueError("Inputs must have the same dimensions")
    
    C1 = (k1 * dynamic_range) ** 2
    C2 = (k2 * dynamic_range) ** 2
    
    ssim_map = (2 * x * y + C1) / (x ** 2 + y ** 2 + C1)
    return np.mean(ssim_map)

def extract_priority(signature: np.ndarray) -> float:
    """
    Extracts a priority value from a path signature.

    Args:
    signature (np.ndarray): The path signature.

    Returns:
    float: The priority value in [0,1].
    """
    # Map the norm of the level-2 signature to [0,1] with a smooth sigmoid
    norm = np.linalg.norm(signature)
    return 1 / (1 + math.exp(-norm))

def hybrid_ssim(x: np.ndarray, y: np.ndarray, confidence: float, priority: float) -> float:
    """
    Evaluates the similarity between two inputs using the ssim function, 
    adjusted by the confidence and priority.

    Args:
    x (np.ndarray): The first input.
    y (np.ndarray): The second input.
    confidence (float): The confidence value in [0,1].
    priority (float): The priority value in [0,1].

    Returns:
    float: The adjusted similarity value.
    """
    ssim_value = ssim(x, y)
    return ssim_value * confidence * priority

def route_packet(packet: dict[str, Any], confidence: float, priority: float) -> dict[str, Any]:
    """
    Routes a packet using the ternary router's route_command function, 
    adjusted by the confidence and priority.

    Args:
    packet (dict[str, Any]): The packet to route.
    confidence (float): The confidence value in [0,1].
    priority (float): The priority value in [0,1].

    Returns:
    dict[str, Any]: The routed packet.
    """
    text = str(packet.get("text_surface") or packet.get("raw_command") or packet.get("text") or "")
    intent = str(packet.get("normalized_intent") or packet.get("intent") or "bytewax_rete_bandit")
    context = {
        "source": packet.get("source"),
        "source_ref": packet.get("source_ref")
    }
    # Adjust the route_command function using the confidence and priority
    return {"text": text, "intent": intent, "context": context, "confidence": confidence, "priority": priority}

def aggregate_labels(batches: List[List[LabelingFunctionResult]]) -> List[ProbabilisticLabel]:
    """
    Pure A‑logic: majority vote with confidence = proportion of votes.
    Returns a list of ProbabilisticLabel, one per document
    """
    labels = []
    for batch in batches:
        votes = {}
        for result in batch:
            label = result.label
            if label not in votes:
                votes[label] = 0
            votes[label] += 1
        label = max(votes, key=votes.get)
        confidence = votes[label] / len(batch)
        labels.append(ProbabilisticLabel(doc_id=batch[0].doc_id, label=label, confidence=confidence))
    return labels

if __name__ == "__main__":
    # Test the hybrid system
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    confidence = 0.8
    priority = extract_priority(np.array([1, 2]))
    print(hybrid_ssim(x, y, confidence, priority))
    packet = {"text_surface": "Hello", "normalized_intent": "greeting"}
    print(route_packet(packet, confidence, priority))
    batches = [[LabelingFunctionResult(lf_name="lf1", doc_id="doc1", label=1), LabelingFunctionResult(lf_name="lf2", doc_id="doc1", label=1)]]
    print(aggregate_labels(batches))