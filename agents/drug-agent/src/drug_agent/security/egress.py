"""Network egress policy.

Default posture is deny-all. Optional connectors must pass an explicit
allowlist of hostnames. Tests run with network denied so a stray live call
raises loudly instead of silently succeeding.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from urllib.parse import urlparse


class NetworkDeniedError(RuntimeError):
    """Raised when a network call is attempted under a deny policy."""


@dataclass
class EgressPolicy:
    allow_network: bool = False
    allowlist: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls) -> EgressPolicy:
        allow = os.environ.get("DRUGAGENT_ALLOW_NETWORK", "0") in {"1", "true", "True"}
        raw = os.environ.get("DRUGAGENT_EGRESS_ALLOWLIST", "")
        allowlist = [h.strip() for h in raw.split(",") if h.strip()]
        return cls(allow_network=allow, allowlist=allowlist)

    def check(self, url: str) -> None:
        if not self.allow_network:
            raise NetworkDeniedError(
                f"Network egress denied by policy (attempted {url!r}). "
                "Set DRUGAGENT_ALLOW_NETWORK=1 and an egress allowlist to enable."
            )
        host = urlparse(url).hostname or ""
        if self.allowlist and not any(host == a or host.endswith("." + a) for a in self.allowlist):
            raise NetworkDeniedError(f"Host {host!r} not in egress allowlist {self.allowlist!r}.")


def check_url(url: str, policy: EgressPolicy | None = None) -> None:
    (policy or EgressPolicy.from_env()).check(url)
