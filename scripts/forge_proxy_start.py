import sys
import asyncio
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "01_REPOS" / "forge" / "src"))

from forge.clients.vllm import VLLMClient
from forge.proxy.server import HTTPServer
from forge.context.manager import ContextManager
from forge.context.strategies import TieredCompact

client = VLLMClient(model_path=str(ROOT/"03_VAULT/models/tensorblock/Falcon3-Mamba-7B-Instruct-GGUF/Falcon3-Mamba-7B-Instruct-Q2_K.gguf"), base_url="http://127.0.0.1:8083/v1")
ctx = ContextManager(strategy=TieredCompact(), budget_tokens=4096)
server = HTTPServer(client=client, context_manager=ctx, host="127.0.0.1", port=9000, serialize_requests=True, max_retries=3, rescue_enabled=True)

async def _run(s):
    await s.start()
    await asyncio.get_event_loop().create_future()
asyncio.run(_run(server))