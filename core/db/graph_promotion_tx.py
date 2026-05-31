import inspect
import json
import os
from pathlib import Path


class SecurityError(Exception):
    pass


# Load allowlist at module import time
_ALLOWLIST_PATH = Path("/home/mfspx/LUCIDOTA/00_PROJECT_BRAIN/canonical_graph_write_allowlist.json")
with open(_ALLOWLIST_PATH, "r") as f:
    _ALLOWLIST_DATA = json.load(f)

# Build the allowlist from all permitted module categories
_ALLOWLIST: set[str] = set()
for key in [
    "allowed_materialization_helpers",
    "staging_only_modules",
    "test_fixture_modules",
    "helper_owned_internal_modules",
]:
    for module in _ALLOWLIST_DATA.get(key, []):
        _ALLOWLIST.add(module)


def _get_caller_script_path() -> str:
    """Get the absolute path of the calling script using inspect.stack()."""
    # Walk up the stack to find the first frame that is not this module
    for frame_info in inspect.stack():
        frame_path = frame_info.filename
        # Skip frames from this module
        if frame_path == __file__:
            continue
        # Resolve to absolute path
        return os.path.abspath(frame_path)
    raise RuntimeError("Unable to determine caller script path")


def _check_allowlist() -> None:
    """Raise SecurityError if caller is not in the allowlist."""
    caller_path = _get_caller_script_path()
    # Also check against just the basename for relative path matches
    caller_basename = os.path.basename(caller_path)
    
    # Check if the full path or basename is in the allowlist
    # Normalize paths for comparison (forward slashes)
    normalized_caller = caller_path.replace("\\", "/")
    
    for allowed in _ALLOWLIST:
        normalized_allowed = allowed.replace("\\", "/")
        if normalized_caller == normalized_allowed or normalized_caller.endswith(f"/{normalized_allowed}"):
            return
        if caller_basename == os.path.basename(normalized_allowed):
            return
    
    raise SecurityError(
        f"Caller script '{caller_path}' is not in the graph write allowlist. "
        f"Allowed modules: {sorted(_ALLOWLIST)}"
    )


async def graph_promotion_tx(conn):
    """
    Async context manager for graph promotion transactions.
    
    Sets the graph_promotion_path to 'on' for the duration of the transaction.
    Validates that the calling script is in the allowlist before proceeding.
    
    Args:
        conn: An open psycopg (v3) async connection.
    
    Usage:
        async with graph_promotion_tx(conn) as tx_conn:
            # tx_conn is the same conn, now in a transaction with graph_promotion_path='on'
            await tx_conn.execute("INSERT INTO ...")
    """
    _check_allowlist()
    
    class _TxContext:
        def __init__(self, conn):
            self.conn = conn
        
        async def __aenter__(self):
            # Start transaction and set graph promotion path
            await self.conn.execute("BEGIN")
            await self.conn.execute("SET LOCAL lucidota.graph_promotion_path='on'")
            return self.conn
        
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            if exc_type is None:
                await self.conn.commit()
            else:
                await self.conn.rollback()
                # Re-raise the exception
                return False
    
    return _TxContext(conn)
