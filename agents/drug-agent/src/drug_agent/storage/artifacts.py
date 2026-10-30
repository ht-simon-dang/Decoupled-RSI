"""Content-addressed artifact store.

Artifacts are immutable and addressed by the SHA-256 of their bytes, so writing
the same content twice is idempotent and de-duplicated. The local backend lays
content out under ``<root>/cas/<aa>/<hash>``; the S3 stub demonstrates the
production seam without requiring a live bucket.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from ..contracts.evidence import ArtifactRef
from ..util import canonical_json, hash_bytes


class ArtifactStore(ABC):
    """Abstract immutable artifact store."""

    @abstractmethod
    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        media_type: str | None = None,
        produced_by: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef: ...

    @abstractmethod
    def get_bytes(self, ref: ArtifactRef | str) -> bytes: ...

    @abstractmethod
    def exists(self, content_hash: str) -> bool: ...

    # --- convenience serializers ------------------------------------------
    def put_json(self, obj: Any, **kw: Any) -> ArtifactRef:
        data = canonical_json(obj).encode("utf-8")
        kw.setdefault("kind", "json")
        kw.setdefault("media_type", "application/json")
        return self.put_bytes(data, **kw)

    def put_text(self, text: str, **kw: Any) -> ArtifactRef:
        kw.setdefault("kind", "text")
        kw.setdefault("media_type", "text/plain")
        return self.put_bytes(text.encode("utf-8"), **kw)

    def put_model(self, model: Any, **kw: Any) -> ArtifactRef:
        """Persist a pydantic model as canonical JSON."""
        payload = model.model_dump(mode="json") if hasattr(model, "model_dump") else model
        return self.put_json(payload, **kw)

    def get_json(self, ref: ArtifactRef | str) -> Any:
        return json.loads(self.get_bytes(ref).decode("utf-8"))

    def put_dataframe(self, df: Any, **kw: Any) -> ArtifactRef:
        """Persist a pandas DataFrame as Parquet."""
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        kw.setdefault("kind", "parquet")
        kw.setdefault("media_type", "application/vnd.apache.parquet")
        return self.put_bytes(buf.getvalue(), **kw)


class LocalArtifactStore(ArtifactStore):
    """Filesystem content-addressed store rooted at ``root``."""

    scheme = "cas"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.cas = self.root / "cas"
        self.cas.mkdir(parents=True, exist_ok=True)

    def _path_for(self, digest_hex: str) -> Path:
        shard = digest_hex[:2]
        d = self.cas / shard
        d.mkdir(parents=True, exist_ok=True)
        return d / digest_hex

    def put_bytes(
        self,
        data: bytes,
        *,
        kind: str,
        media_type: str | None = None,
        produced_by: str | None = None,
        run_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ArtifactRef:
        ch = hash_bytes(data)  # "sha256:<hex>"
        digest_hex = ch.split(":", 1)[1]
        path = self._path_for(digest_hex)
        if not path.exists():
            # The staging name has to be unique per writer, not per digest. Two
            # processes storing the *same* content — which is the normal case in
            # a content-addressed store, and exactly what two concurrent
            # campaign experiments do — would otherwise share one ``.tmp`` path:
            # the first ``replace`` consumes it and the second raises
            # FileNotFoundError on a file it believed it had just written.
            # ``replace`` is atomic within a directory, so the final step is
            # still safe; only the staging name needed to stop colliding.
            tmp = path.with_name(f"{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
            tmp.write_bytes(data)
            try:
                tmp.replace(path)
            except OSError:  # pragma: no cover - lost a benign race; the bytes are identical
                tmp.unlink(missing_ok=True)
                if not path.exists():
                    raise
        return ArtifactRef(
            kind=kind,
            uri=f"{self.scheme}://sha256/{digest_hex}",
            content_hash=ch,
            size_bytes=len(data),
            media_type=media_type,
            produced_by=produced_by,
            run_id=run_id,
            metadata=metadata or {},
        )

    def _resolve(self, ref: ArtifactRef | str) -> Path:
        uri = ref.uri if isinstance(ref, ArtifactRef) else ref
        if uri.startswith(f"{self.scheme}://sha256/"):
            digest_hex = uri.rsplit("/", 1)[1]
        elif uri.startswith("sha256:"):
            digest_hex = uri.split(":", 1)[1]
        else:
            digest_hex = uri
        return self._path_for(digest_hex)

    def get_bytes(self, ref: ArtifactRef | str) -> bytes:
        path = self._resolve(ref)
        if not path.exists():
            raise FileNotFoundError(f"artifact not found: {ref}")
        return path.read_bytes()

    def exists(self, content_hash: str) -> bool:
        digest_hex = content_hash.split(":", 1)[-1]
        return self._path_for(digest_hex).exists()

    def export_to(self, ref: ArtifactRef | str, dest: Path) -> Path:
        """Copy an artifact out to a human-facing path (for run bundles)."""
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._resolve(ref), dest)
        return dest


class S3StubArtifactStore(LocalArtifactStore):
    """Production seam: same interface, ``s3://`` URIs, local bytes for the demo.

    A real deployment swaps ``put_bytes``/``get_bytes`` for boto3 calls; the
    URI scheme and content-addressing contract are identical so callers are
    unchanged.
    """

    scheme = "s3"

    def __init__(self, root: Path | str, bucket: str = "drug-agent-artifacts") -> None:
        super().__init__(root)
        self.bucket = bucket

    def put_bytes(self, data: bytes, **kw: Any) -> ArtifactRef:  # type: ignore[override]
        ref = super().put_bytes(data, **kw)
        digest_hex = ref.content_hash.split(":", 1)[1]
        ref.uri = f"s3://{self.bucket}/cas/sha256/{digest_hex}"
        return ref

    def _resolve(self, ref: ArtifactRef | str) -> Path:
        uri = ref.uri if isinstance(ref, ArtifactRef) else ref
        digest_hex = uri.rsplit("/", 1)[1]
        return self._path_for(digest_hex)


def make_artifact_store(backend: str, root: Path | str) -> ArtifactStore:
    if backend == "s3_stub":
        return S3StubArtifactStore(root)
    return LocalArtifactStore(root)
