"""Tests for pypeline.math.small_model_zoo — RFC-0018 §3.14."""
import pytest
from pypeline.math.small_model_zoo import (
    ModelCard,
    ZOO_REGISTRY,
    register_model,
    train_setfit_classifier,
    train_peft_lora,
    quantize_to_onnx,
    load_zoo_model,
)


def _labeled_examples(n: int = 10) -> list[dict]:
    texts = [
        "the project is complete and ready",
        "this is a total failure and broken",
        "looks great, approved",
        "serious issues found in the report",
        "everything checks out fine",
        "major violation detected",
        "passed all quality gates",
        "rejected due to non-compliance",
        "satisfactory result across the board",
        "unacceptable performance metrics",
    ][:n]
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1][:n]
    return [{"text": t, "label": l} for t, l in zip(texts, labels)]


class TestRegisterAndLoadModel:
    def test_register_and_load(self):
        card = ModelCard(
            model_id="test_register_001",
            model_class="sklearn_tfidf",
            domain="general",
            task="smoke_test",
            base_model="tfidf+logreg",
            conformal_calibrated=False,
            gold_corpus_f1=None,
            onnx_quantized=False,
            version="0.1.0",
        )
        register_model(card)
        loaded = load_zoo_model("test_register_001")
        assert loaded == card

    def test_load_missing_raises_key_error(self):
        with pytest.raises(KeyError):
            load_zoo_model("nonexistent_model_xyz")

    def test_zoo_registry_updated(self):
        card = ModelCard(
            model_id="test_reg_002",
            model_class="setfit",
            domain="forum",
            task="test_task",
            base_model="paraphrase-MiniLM",
            conformal_calibrated=False,
            gold_corpus_f1=None,
            onnx_quantized=False,
            version="0.1.0",
        )
        register_model(card)
        assert "test_reg_002" in ZOO_REGISTRY


class TestTrainSetfitClassifier:
    def test_returns_model_card(self):
        examples = _labeled_examples(10)
        card = train_setfit_classifier("test_task", examples)
        assert isinstance(card, ModelCard)

    def test_card_registered_in_zoo(self):
        examples = _labeled_examples(10)
        card = train_setfit_classifier("test_task_reg", examples)
        assert card.model_id in ZOO_REGISTRY

    def test_card_has_correct_task(self):
        examples = _labeled_examples(10)
        card = train_setfit_classifier("my_specific_task", examples)
        assert card.task == "my_specific_task"

    def test_card_has_version(self):
        examples = _labeled_examples(10)
        card = train_setfit_classifier("versioned_task", examples)
        assert card.version

    def test_domain_propagated(self):
        examples = _labeled_examples(10)
        card = train_setfit_classifier("forum_task", examples, domain="forum")
        assert card.domain == "forum"

    def test_empty_examples_raises(self):
        with pytest.raises((ValueError, Exception)):
            train_setfit_classifier("empty_task", [])

    def test_unique_model_ids(self):
        examples = _labeled_examples(10)
        c1 = train_setfit_classifier("dedup_task", examples)
        c2 = train_setfit_classifier("dedup_task", examples)
        assert c1.model_id != c2.model_id


class TestTrainPeftLora:
    def test_raises_import_error_without_peft(self):
        """Without peft installed, must raise ImportError, NOT NotImplementedError."""
        try:
            import peft  # noqa: F401
            pytest.skip("peft is installed — skipping ImportError test")
        except ImportError:
            pass

        with pytest.raises(ImportError, match="peft"):
            train_peft_lora("test", "bert-base-uncased", _labeled_examples())

    def test_not_raises_not_implemented_error(self):
        """Must not raise NotImplementedError (stub is closed)."""
        try:
            import peft  # noqa: F401
            pytest.skip("peft is installed — stub behaviour changes")
        except ImportError:
            pass

        try:
            train_peft_lora("test", "bert-base-uncased", _labeled_examples())
        except ImportError:
            pass  # Expected
        except NotImplementedError:
            pytest.fail("train_peft_lora must not raise NotImplementedError")


class TestQuantizeToOnnx:
    def _make_card(self, model_id: str = "q_test") -> ModelCard:
        card = ModelCard(
            model_id=model_id,
            model_class="sklearn_tfidf",
            domain="general",
            task="quant_test",
            base_model="tfidf+logreg",
            conformal_calibrated=False,
            gold_corpus_f1=None,
            onnx_quantized=False,
            version="0.1.0",
        )
        register_model(card)
        return card

    def test_already_quantized_returns_same(self):
        card = ModelCard(
            model_id="already_onnx",
            model_class="onnx_int8",
            domain="general",
            task="t",
            base_model="x",
            conformal_calibrated=False,
            gold_corpus_f1=None,
            onnx_quantized=True,
            version="0.1.0",
        )
        result = quantize_to_onnx(card)
        assert result.onnx_quantized is True
        assert result == card

    def test_raises_import_error_without_onnxruntime(self):
        try:
            import onnxruntime  # noqa: F401
            pytest.skip("onnxruntime installed")
        except ImportError:
            pass

        card = self._make_card("ort_missing_test")
        with pytest.raises(ImportError, match="onnxruntime"):
            quantize_to_onnx(card)

    def test_not_raises_not_implemented_error(self):
        try:
            import onnxruntime  # noqa: F401
            pytest.skip("onnxruntime installed")
        except ImportError:
            pass

        card = self._make_card("ort_nie_test")
        try:
            quantize_to_onnx(card)
        except ImportError:
            pass
        except NotImplementedError:
            pytest.fail("quantize_to_onnx must not raise NotImplementedError")
