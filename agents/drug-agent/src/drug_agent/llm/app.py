#!/usr/bin/env python
"""Launcher for the master-agent web console.

Brings up the console and the JSON API for the LLM-driven master agent: it plans
a drug-discovery program, transcribes the user's intent into precise chemical
requirements, dispatches the CHEMICAL and EVALUATION specialists against the
deterministic pipeline, and adaptively re-prompts generation when results fall
short. Every request is journaled (see ``drug_agent.observability.journal``).

Usage::

    drugagent console                       # http://127.0.0.1:2003/
    drugagent console --port 2003 --open
    python -m drug_agent.llm.app --model qwen3.8:27b --max-cycles 3

Requires a local Ollama with the chosen model::

    ollama serve
    ollama pull qwen3.8:27b
"""

from __future__ import annotations

import argparse
import os

from .config import DEFAULT_MODEL, Config, load_config


def build_parser(cfg: Config) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Launch the LLM master-agent web console.")
    ap.add_argument("--host", default=cfg.host)
    ap.add_argument("--port", type=int, default=cfg.port)
    ap.add_argument("--model", default=cfg.model, help=f"Ollama model (default {DEFAULT_MODEL})")
    ap.add_argument("--ollama-host", default=cfg.ollama_host)
    ap.add_argument("--strategy", choices=["adaptive", "naive"], default=cfg.refine_strategy)
    ap.add_argument("--max-cycles", type=int, default=cfg.max_cycles)
    ap.add_argument(
        "--generator",
        choices=["auto", "library", "pfm", "library+pfm"],
        default=cfg.generator,
        help=(
            "pin the candidate source for every run; 'auto' (the default) leaves "
            "the choice to the master agent (see README)"
        ),
    )
    ap.add_argument("--n-generated", type=int, default=cfg.n_generated, help="de-novo molecules per cycle")
    ap.add_argument("--pfm-pocket", default=cfg.pfm_pocket_pdb, help="prepared pocket PDB handed to PFM")
    ap.add_argument(
        "--structured",
        choices=["auto", "on", "off"],
        default=cfg.structured_outputs,
        help="constrain decoding to the target JSON schema (default auto)",
    )
    ap.add_argument("--journal-dir", default=cfg.journal_dir, help="where per-request journals are written")
    ap.add_argument("--no-journal", dest="journal", action="store_false", help="disable request journaling")
    ap.add_argument("--open", dest="open_browser", action="store_true", help="open the console in a browser")
    ap.set_defaults(journal=cfg.journal_enabled)
    return ap


def main(argv: list[str] | None = None) -> None:
    cfg = load_config()
    args = build_parser(cfg).parse_args(argv)

    cfg = cfg.with_overrides(
        host=args.host,
        port=args.port,
        model=args.model,
        ollama_host=args.ollama_host,
        refine_strategy=args.strategy,
        max_cycles=args.max_cycles,
        generator=args.generator,
        n_generated=args.n_generated,
        pfm_pocket_pdb=args.pfm_pocket,
        structured_outputs=args.structured,
        journal_enabled=args.journal,
        journal_dir=args.journal_dir,
    )
    # Make settings visible to child imports / the pipeline bridge.
    os.environ.setdefault("DRUGAGENT_PROFILE", cfg.drug_agent_profile)

    from .server import create_app

    app = create_app(cfg)

    shown = "127.0.0.1" if args.host in ("0.0.0.0", "::") else args.host
    url = f"http://{shown}:{args.port}/"
    print(f"[console] master-agent console → {url}", flush=True)
    print(
        f"[console] model={cfg.model}  ollama={cfg.ollama_host}  structured={cfg.structured_outputs}",
        flush=True,
    )
    print(
        f"[console] strategy={cfg.refine_strategy}  max_cycles={cfg.max_cycles}  API docs → {url}docs",
        flush=True,
    )
    gen = cfg.generator_status()
    from ..agents.llm.master_agent import installed_generators

    engines = installed_generators()
    from ..llm.schemas import candidate_source_label

    source = (
        "chosen per run by the master agent"
        if cfg.generator == "auto"
        else candidate_source_label(cfg.generator)
    )
    print(
        f"[console] candidate source={source}  "
        f"de-novo generators installed={len(engines)} ({', '.join(engines) or 'none'})",
        flush=True,
    )
    if "pfm" in cfg.generator and not gen["available"]:
        print(f"[console] ⚠ {gen['reason']}", flush=True)
    print(
        f"[console] journal {'→ ' + str(cfg.resolved_journal_dir()) if cfg.journal_enabled else 'disabled'}",
        flush=True,
    )

    if args.open_browser:
        try:
            import threading
            import webbrowser

            threading.Timer(1.5, lambda: webbrowser.open(url)).start()
        except Exception:
            pass

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
