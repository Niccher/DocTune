"""
web/core/ids.py
---------------
Collision-resistant, time-ordered identifier generator using ULID specification.
Produces human-readable, sortable, URL-safe identifiers (e.g., run_01HP..., job_01HP...).
"""

import time
import secrets
from typing import Optional

# Crockford's Base32 character set (excluding I, L, O, U to avoid confusion)
CROCKFORD_32 = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


_last_ms = 0
_last_rand = 0


def generate_ulid() -> str:
    """Generate a standard 26-character Crockford Base32 monotonic ULID."""
    global _last_ms, _last_rand
    ms = int(time.time() * 1000)
    if ms > _last_ms:
        _last_ms = ms
        _last_rand = secrets.randbits(80)
    else:
        # Same millisecond: monotonically increment the random component
        _last_rand += 1
        ms = _last_ms

    time_chars = []
    t = ms
    for _ in range(10):
        time_chars.append(CROCKFORD_32[t & 0x1F])
        t >>= 5
    time_str = "".join(reversed(time_chars))

    rand_chars = []
    r = _last_rand
    for _ in range(16):
        rand_chars.append(CROCKFORD_32[r & 0x1F])
        r >>= 5
    rand_str = "".join(reversed(rand_chars))
    return time_str + rand_str



def generate_id(prefix: str = "") -> str:
    """
    Generate a typed, time-ordered, collision-resistant identifier.
    Example: generate_id("run") -> "run_01HPX7K..."
    """
    ulid = generate_ulid()
    if prefix:
        clean_prefix = prefix.strip("_")
        return f"{clean_prefix}_{ulid}"
    return ulid


def generate_run_id(method: str = "qlora") -> str:
    """Generate a collision-resistant run ID preserving the tuning method."""
    clean_method = "".join(c for c in method.lower() if c.isalnum())
    return f"run_{clean_method}_{generate_ulid()}"


def generate_job_id(job_type: str = "train") -> str:
    """Generate a collision-resistant job ID."""
    clean_type = "".join(c for c in job_type.lower() if c.isalnum())
    return f"job_{clean_type}_{generate_ulid()}"
