# Data card — offline demo fixtures

[English](DATA_CARD.md) | [简体中文](DATA_CARD_zh.md)

All files in this directory are generated deterministically by
[`../generate_fixtures.py`](../generate_fixtures.py). They exist so the platform
can run end-to-end offline with no external data access.

> **Provenance and trust.** Chemical **structures** are well-known public
> drug/drug-like molecules (SMILES). All **bioactivity labels, target–disease
> scores, structure metadata, literature abstracts, and single-cell counts are
> SYNTHETIC** — produced by documented latent rules plus seeded noise. They are
> for demonstration only and must never be treated as measured/real data.

| File | Kind | Records | Trust | Notes |
| ---- | ---- | ------- | ----- | ----- |
| `seh_bioactivity.csv` | bioactivity (ChEMBL-like) | 50 | synthetic labels | `IC50` in nM with `pchembl_value`; latent SAR rule on descriptors |
| `compound_library.csv` | compounds (PubChem-like) | 50 | public structures | candidate library (public SMILES) |
| `targets.json` | target–disease (Open-Targets-like) | 1 target / 3 assoc | synthetic scores | sEH / EPHX2 (UniProt P34913) |
| `structures.json` | structure metadata (RCSB-PDB-like) | 2 | synthetic metadata | PDB ids referenced; no coordinates |
| `literature.jsonl` | literature | 3 | synthetic; 1 adversarial | includes one prompt-injection abstract for security tests |
| `sc_perturbation.npz` | single-cell | 120 cells × 24 genes | synthetic | disease/healthy over 4 donors; donor-aware splits |

- **Target of the demo:** soluble epoxide hydrolase (sEH / EPHX2), a
  well-characterized, non-pathogen, small-molecule-tractable enzyme. Chosen so
  no novel-discovery or clinical claim is implied.
- **Licenses:** structures are public-domain identifiers; synthetic labels are
  released under the repository's Apache-2.0 license.
- **Snapshot IDs / hashes:** each connector computes a content hash over its
  records (`DataSnapshot.snapshot_id`), recorded in every run's `manifest.json`.
- **Regeneration:** `make fixtures` (deterministic; byte-stable given the seed).
