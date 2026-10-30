"""The experience store — normalized episodes over the raw journals.

Deliberately thin. The journals are the system of record and are append-only;
this keeps a cached normalized view beside them and offers the splitting that
every honest evaluation needs.

**Splitting is chronological.** The evolver studies older episodes and is
measured on newer ones, because that is the only split that answers the question
anyone actually has: *would this change have helped the requests that came next?*
A random split of five episodes answers nothing and hides leakage.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path

from ..config import EvolveConfig
from ..schemas import Episode
from .normalize import normalize_request


class ExperienceStore:
    """Reads request journals; caches the normalized form."""

    def __init__(self, cfg: EvolveConfig | None = None) -> None:
        self.cfg = cfg or EvolveConfig()
        self.journal_root = self.cfg.resolved_journal_dir()
        self.cache_dir = self.cfg.resolved_store_dir() / "episodes"

    # --- reading --------------------------------------------------------
    def journal_dirs(self) -> list[Path]:
        """Every request directory under the history root, in date order.

        Resolution is delegated to the journal module so that the dated
        ``running-history/requests/<day>/<request>`` layout and the older flat
        one are both readable — this store must never be the reason a migrated
        history stops being analyzable.
        """
        from ...observability.journal import iter_request_dirs

        return iter_request_dirs(self.journal_root)

    @staticmethod
    def _episode_id(journal_dir: Path) -> str:
        """The request id, read from the manifest — never from the directory name.

        The directory name carries a timestamp and a slug for humans; the id is
        the key. Deriving it from the manifest is what keeps cached episodes
        valid across a move or a rename.
        """
        try:
            manifest = json.loads((journal_dir / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return journal_dir.name
        return str(manifest.get("request_id") or journal_dir.name)

    def load(self, *, refresh: bool = False) -> list[Episode]:
        """Every complete episode, oldest first."""
        episodes: list[Episode] = []
        for d in self.journal_dirs():
            ep = self._cached(d, refresh=refresh)
            if ep is not None:
                episodes.append(ep)
        episodes.sort(key=lambda e: (e.started_at or 0.0, e.episode_id))
        return episodes

    def _cached(self, journal_dir: Path, *, refresh: bool) -> Episode | None:
        cache = self.cache_dir / f"{self._episode_id(journal_dir)}.json"
        if not refresh and cache.is_file():
            try:
                src_mtime = (journal_dir / "manifest.json").stat().st_mtime
                if cache.stat().st_mtime >= src_mtime:
                    return Episode.model_validate_json(cache.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
        episode = normalize_request(journal_dir)
        if episode is None:
            return None
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache.write_text(episode.model_dump_json(indent=2), encoding="utf-8")
        except OSError:
            pass  # a cache that cannot be written is not a reason to fail
        return episode

    def get(self, episode_id: str) -> Episode | None:
        for d in self.journal_dirs():
            if self._episode_id(d) == episode_id:
                return self._cached(d, refresh=False)
        return None

    # --- splitting ------------------------------------------------------
    def split(
        self, episodes: Iterable[Episode] | None = None, *, holdout_fraction: float = 0.4
    ) -> dict[str, list[Episode]]:
        """Chronological ``{"discovery": [...], "holdout": [...]}``.

        With very few episodes a holdout is a gesture rather than a measurement,
        and the caller is told so: the returned dict carries the counts and every
        report built on it prints them. We keep the split anyway, because the
        alternative — studying everything and then claiming generalization — is
        exactly the leak the brief warns about.
        """
        eps = sorted(episodes or self.load(), key=lambda e: (e.started_at or 0.0, e.episode_id))
        if not eps:
            return {"discovery": [], "holdout": []}
        n_holdout = max(1, round(len(eps) * holdout_fraction)) if len(eps) > 2 else 0
        cut = len(eps) - n_holdout
        return {"discovery": eps[:cut], "holdout": eps[cut:]}

    # --- export ---------------------------------------------------------
    def write_normalized(self, path: str | Path | None = None) -> Path:
        """Dump every episode as one JSONL file — the machine-readable view."""
        out = Path(path) if path else self.cfg.resolved_store_dir() / "episodes.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ep in self.load():
                fh.write(json.dumps(json.loads(ep.model_dump_json()), ensure_ascii=False) + "\n")
        return out

    def __iter__(self) -> Iterator[Episode]:
        return iter(self.load())


__all__ = ["ExperienceStore"]
