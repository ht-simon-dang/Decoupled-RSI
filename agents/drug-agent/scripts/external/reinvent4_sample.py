"""Sample de-novo SMILES from a REINVENT4 prior — the drug-agent ⇄ REINVENT4 bridge.

This file runs inside REINVENT4's own environment, so it imports nothing from
``drug_agent`` and nothing outside the standard library except REINVENT4 itself.
REINVENT4 requires Python >= 3.11 (``pyproject.toml``), so this driver is written
for 3.11 rather than pinned down to an older dialect.

Everything below was read out of the pinned checkout (ee0d56f4a074). Nothing here
is guessed.

``the entry point``
    ``reinvent/Reinvent.py:main(args)``. The console script ``reinvent`` is
    declared in ``pyproject.toml`` as ``reinvent = "reinvent.Reinvent:main_script"``,
    and ``main_script`` does exactly two things: ``args = parse_command_line()``
    then ``main(args)``. So the real entry point is ``main``, and this driver
    fills the same ``argparse.Namespace`` that ``reinvent/utils/cli.py`` would
    have produced — ``config_filename``, ``config_format``, ``device``,
    ``log_filename``, ``log_level``, ``seed``, ``dotenv_filename``,
    ``enable_rdkit_log_levels`` — and calls it directly. No second process, no
    reimplementation of the run mode.

``the config``
    ``reinvent/utils/config_parse.py:read_config`` accepts ``toml``, ``json`` and
    ``yaml``, and ``main`` picks the format from the file suffix. JSON is written
    here because the standard library can emit it and ``tomllib`` cannot. The
    top-level schema is ``reinvent/validation.py:ReinventConfig``, which is
    pydantic with ``extra="forbid"`` — only ``run_type``, ``device``, ``seed``,
    ``tb_logdir``, ``json_out_config``, ``parameters`` and the run-mode sections
    are accepted. The ``parameters`` block is validated against
    ``reinvent/runmodes/samplers/validation.py:SectionParameters``, also
    ``extra="forbid"``: ``model_file``, ``num_smiles``, ``smiles_file``,
    ``target_smiles_path``, ``sample_strategy``, ``output_file``,
    ``target_nll_file``, ``unique_molecules``, ``randomize_smiles``,
    ``isomeric_smiles``, ``temperature``.

``the checkpoint argument``
    ``parameters.model_file``, resolved by
    ``reinvent/runmodes/create_adapter.py:resolve_model_filename``. It is either a
    key of ``reinvent/utils/prior_registry.py`` (``.reinvent``, ``.libinvent``,
    ``.m2m_medium`` …) or a plain filename; the adapter always passes an absolute
    path, so the registry — and with it the ``REINVENT_PRIOR_BASE`` environment
    variable — is bypassed. An unreadable path raises ``RuntimeError: model file
    ... is not accessible`` there, before any sampling.

``the output format``
    ``reinvent/runmodes/samplers/run_sampling.py`` writes ``parameters.output_file``
    as CSV whose header is ``HEADERS[model_type]``. For the unconditional prior
    that is ``SMILES,SMILES_state,NLL``; the fragment generators emit five
    columns instead, which is why this driver reports the header it actually read
    rather than assuming one. ``SMILES_state`` is the integer value of
    ``reinvent/models/model_factory/sample_batch.py:SmilesState`` — 0 INVALID,
    1 VALID, 2 DUPLICATE — assigned by ``samplers/sampler.py:validate_smiles``.

``unique_molecules``
    Left to the caller and set to false by the adapter. When true,
    ``run_sampling`` calls ``filter_valid`` and only ``SmilesState.VALID`` rows
    survive, which would destroy the denominator the platform needs to compute
    validity honestly. With it false the CSV carries every row the model produced.

Only the unconditional ``Reinvent`` de-novo mode is driven here. LibInvent,
LinkInvent, Mol2Mol and PepInvent need ``parameters.smiles_file`` seed structures
and write a different set of columns; they are a different driver, not a flag.

Request JSON:
    model_file        absolute path to the prior (``.prior`` torch pickle)
    num_smiles        number of SMILES to sample
    device            torch device string, e.g. "cpu" or "cuda:0"
    seed              RNG seed, passed to ``reinvent.utils.set_seed``
    randomize_smiles  optional, default true (SectionParameters default)
    unique_molecules  optional, default false (see above)
    temperature       optional, default 1.0
    log_level         optional, one of REINVENT4's levels; default "info"
    out_dir           where to write result.json

Response JSON (``out_dir/result.json``):
    molecules[]       {smiles, state, nll}
    headers[]         the CSV header row REINVENT4 wrote
    model_type        the generator name matched from ``run_sampling.HEADERS``
    n_rows            data rows read
    n_malformed_rows  rows whose field count did not match the header
    reinvent_version  ``reinvent.version.__version__``
    device            the device string the config asked for
"""

from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
# The sys.path line above must run first; this directory is not a package.
import _driver


def build_config(request: dict, samples_csv: Path) -> dict:
    """The ReinventConfig body for one unconditional sampling run.

    Keys are exactly those ``reinvent/validation.py:ReinventConfig`` and
    ``reinvent/runmodes/samplers/validation.py:SectionParameters`` accept; both
    are ``extra="forbid"``, so an invented key is a startup error, not a silent
    no-op.
    """
    parameters = {
        "model_file": str(request["model_file"]),
        "num_smiles": int(request["num_smiles"]),
        "output_file": str(samples_csv),
        "unique_molecules": bool(request.get("unique_molecules", False)),
        "randomize_smiles": bool(request.get("randomize_smiles", True)),
        "temperature": float(request.get("temperature", 1.0)),
    }
    return {
        "run_type": "sampling",
        "device": str(request.get("device") or "cpu"),
        "parameters": parameters,
    }


def read_samples(path: Path) -> tuple[list[str], list[dict], int]:
    """Parse REINVENT4's sampling CSV into ``(headers, molecules, n_malformed)``.

    Rows whose field count disagrees with the header are counted, not guessed at:
    the adapter reports them rather than silently shortening the batch.
    """
    with path.open(newline="") as fh:
        rows = list(csv.reader(fh))
    if not rows:
        raise RuntimeError(f"REINVENT4 wrote an empty CSV at {path}")

    headers = [name.strip() for name in rows[0]]
    molecules: list[dict] = []
    malformed = 0

    for row in rows[1:]:
        if not any(field.strip() for field in row):
            continue
        if len(row) != len(headers):
            malformed += 1
            continue
        record = dict(zip(headers, row, strict=True))
        entry: dict = {"smiles": record.get("SMILES", "").strip()}
        entry["state"] = _as_int(record.get("SMILES_state"))
        entry["nll"] = _as_float(record.get("NLL"))
        molecules.append(entry)

    return headers, molecules, malformed


def _as_int(raw: str | None) -> int | None:
    try:
        return int(float(str(raw).strip()))
    except (TypeError, ValueError):
        return None


def _as_float(raw: str | None) -> float | None:
    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return None


def match_model_type(headers: list[str]) -> str:
    """Name the generator from its header row, using REINVENT4's own table.

    ``run_sampling.HEADERS`` maps model type to column names, so the header the
    run actually wrote identifies the generator without loading the prior twice.
    """
    from reinvent.runmodes.samplers.run_sampling import HEADERS

    for name, columns in HEADERS.items():
        if list(columns) == headers:
            return name
    return ""


def run(request: dict) -> dict:
    out_dir = Path(request["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_csv = out_dir / "sampling.csv"
    config_path = out_dir / "sampling.json"
    log_path = out_dir / "reinvent.log"

    config = build_config(request, samples_csv)
    config_path.write_text(json.dumps(config, indent=2))

    from reinvent import version
    from reinvent.Reinvent import main as reinvent_main

    # The Namespace reinvent/utils/cli.py:parse_command_line() would have built.
    # ``device`` stays None so the config file's value is used, which is the path
    # reinvent/runmodes/utils/helpers.py:set_torch_device takes for a config-only
    # device; passing both would only duplicate it.
    args = Namespace(
        config_filename=config_path,
        config_format="json",
        device=None,
        log_filename=str(log_path),
        log_level=str(request.get("log_level") or "info"),
        seed=request.get("seed"),
        dotenv_filename=None,
        enable_rdkit_log_levels=None,
    )
    reinvent_main(args)

    if not samples_csv.is_file():
        raise RuntimeError(
            f"REINVENT4 finished without writing {samples_csv}; see {log_path} for its own log"
        )

    headers, molecules, malformed = read_samples(samples_csv)
    return {
        "molecules": molecules,
        "headers": headers,
        "model_type": match_model_type(headers),
        "n_rows": len(molecules),
        "n_malformed_rows": malformed,
        "reinvent_version": version.__version__,
        "device": config["device"],
        "log_path": str(log_path),
    }


if __name__ == "__main__":
    _driver.main(run)
