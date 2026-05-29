import json
import os
import urllib.request
import ast
import random
import math
import pathlib
from datetime import datetime, timezone
from typing import Tuple, List


class WorkflowMaker:
    ARMS = ["groq", "local", "self"]

    def __init__(self, gamma: float = 0.2) -> None:
        n = len(self.ARMS)
        self.w: List[float] = [1.0] * n
        self.gamma = gamma

    # -------------------- Bandit utilities --------------------
    def _probs(self) -> List[float]:
        """Return the probability distribution over the arms."""
        n = len(self.ARMS)
        s = sum(self.w)
        return [(1 - self.gamma) * wi / s + self.gamma / n for wi in self.w]

    def choose(self) -> int:
        """Select an arm according to the current probability distribution."""
        p = self._probs()
        r = random.random()
        c = 0.0
        for i, pi in enumerate(p):
            c += pi
            if r <= c:
                return i
        return 0

    def update(self, arm: int, reward: float) -> None:
        """Update the weight of the selected arm using the received reward."""
        p = self._probs()
        # Protect against division by zero
        if p[arm] == 0:
            return
        self.w[arm] *= math.exp(self.gamma * reward / p[arm] / len(self.ARMS))

    # -------------------- Model callers --------------------
    def _call_groq(self, prompt: str) -> str:
        """Call the Groq API (or a compatible OpenAI endpoint)."""
        key = os.environ.get("GROQ_API_KEY", "")
        base = os.environ.get(
            "GROQ_BASE_URL", "https://api.groq.com/openai/v1"
        )  # default Groq endpoint
        payload = {
            "model": "openai/gpt-oss-120b",
            "max_tokens": 800,
            "messages": [{"role": "user", "content": prompt}],
        }
        data = json.dumps(payload).encode()
        request = urllib.request.Request(
            f"{base}/chat/completions",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as resp:
            resp_data = json.loads(resp.read())
        return resp_data["choices"][0]["message"]["content"]

    def _call_local(self, prompt: str) -> str:
        """Fallback local model stub. In a real implementation this would invoke a
        locally hosted LLM. Here we simply echo the prompt with a note."""
        return f"[LOCAL MODEL] Generated based on prompt:\n{prompt}"

    def _call_self(self, prompt: str) -> str:
        """Self‑reflection stub. Returns a short deterministic string."""
        return f"[SELF] I attempted to follow the instruction: {prompt[:60]}..."

    # -------------------- Code audit --------------------
    def _audit(self, code: str) -> Tuple[bool, str]:
        """Very light static analysis: syntax check and line count limit."""
        try:
            ast.parse(code)
        except SyntaxError:
            return False, "syntax error"
        if code.count("\n") > 100:
            return False, "too many LOC"
        return True, "pass"

    # -------------------- Main workflow --------------------
    def run(
        self,
        instruction: str,
        quality_bar: str,
        max_iter: int = 5,
    ) -> str:
        """Iteratively generate code until it passes the audit or max_iter is hit."""
        best = ""
        feedback = ""
        ok = False

        for i in range(max_iter):
            arm = self.choose()
            prompt = f"{instruction}\nFeedback: {feedback}\nQuality bar: {quality_bar}"

            if arm == 0:
                result = self._call_groq(prompt)
            elif arm == 1:
                result = self._call_local(prompt)
            else:  # arm == 2
                result = self._call_self(prompt)

            ok, msg = self._audit(result)
            reward = 1.0 if ok else 0.0
            self.update(arm, reward)

            best = result
            if ok:
                break
            feedback = msg

        # Persist a concise version of the result
        out_dir = pathlib.Path("05_OUTPUTS/workflow_maker")
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output_path = out_dir / f"result_{ts}.json"
        output_path.write_text(
            json.dumps(
                {
                    "status": "ok" if ok else "partial",
                    "iterations": i + 1,
                    "result": best[:500],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return best


if __name__ == "__main__":
    wm = WorkflowMaker()
    print(
        wm.run(
            "Write a hello world function in Python.",
            "must be <10 LOC",
            max_iter=2,
        )
    )