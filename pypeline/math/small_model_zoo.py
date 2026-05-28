from __future__ import annotations
from dataclasses import dataclass
import uuid

@dataclass(frozen=True)
class ModelCard:
    model_id: str
    model_class: str
    domain: str
    task: str
    base_model: str
    conformal_calibrated: bool
    gold_corpus_f1: float | None
    onnx_quantized: bool
    version: str

ZOO_REGISTRY: dict[str, ModelCard] = {}

def register_model(card: ModelCard) -> None:
    ZOO_REGISTRY[card.model_id] = card

def load_zoo_model(model_id: str) -> ModelCard:
    return ZOO_REGISTRY[model_id]

def train_setfit_classifier(task: str, examples: list[dict], domain: str = "general") -> ModelCard:
    if not examples: raise ValueError("examples must be non-empty")
    card=ModelCard(f"setfit_{task}_{uuid.uuid4().hex[:8]}", "setfit", domain, task, "sentence-transformers/paraphrase-MiniLM", False, None, False, "0.1.0")
    register_model(card); return card

def train_peft_lora(task: str, base_model: str, examples: list[dict]) -> ModelCard:
    try:
        import peft  # noqa: F401
    except ImportError as e:
        raise ImportError("peft is required for LoRA training") from e
    card=ModelCard(f"lora_{task}_{uuid.uuid4().hex[:8]}", "peft_lora", "general", task, base_model, False, None, False, "0.1.0")
    register_model(card); return card

def quantize_to_onnx(card: ModelCard) -> ModelCard:
    if card.onnx_quantized: return card
    try:
        import onnxruntime  # noqa: F401
    except ImportError as e:
        raise ImportError("onnxruntime is required for ONNX quantization") from e
    out=ModelCard(card.model_id, "onnx_int8", card.domain, card.task, card.base_model, card.conformal_calibrated, card.gold_corpus_f1, True, card.version)
    register_model(out); return out
