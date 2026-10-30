# Scientific runtime adapters

Native interfaces were checked against upstream documentation on 2026-10-30:
[RFantibody](https://github.com/RosettaCommons/RFantibody),
[AntiFold](https://github.com/oxpig/AntiFold),
[ANARCII](https://github.com/oxpig/ANARCII/wiki),
[ImmuneBuilder](https://github.com/oxpig/ImmuneBuilder),
[Boltz-2 prediction](https://github.com/jwohlwend/boltz/blob/main/docs/prediction.md).
Install backend environments and weights separately. Record the actual immutable
commit/release and weight hashes from that installation, rather than using the
documentation date as a model version.

## Operator configuration

Copy only entries you have actually installed from
`configs/runtimes.example.json` into `.antibody-agent/runtimes.json`. Use absolute
paths. `ANTIBODY_RUNTIMES` can point to another trusted local configuration. No
HTTP request or LLM output may set executable commands or runtime versions.

Every entry requires `version` and `weights_version`. A native entry specifies a
`python` interpreter plus the adapter-specific asset paths. A bridge entry
specifies a `command` argv list. Commands never go through a shell. The gateway
appends the input and output JSON file paths as the final two arguments, sets the
package import path, uses a private working directory, and starts a process group
that it terminates on cancellation or timeout.

The backend Python must import the agent's lightweight base dependencies as well
as its own scientific dependencies. It need not install sibling agents. Inference
sets Hugging Face offline flags and package-local cache directories. It does not
upload antigen sequences to an MSA service. A custom bridge is trusted operator
code; its own network and dependency behavior must be understood before deployment.

## Native interfaces

**RFantibody.** The adapter invokes `rfdiffusion`, `proteinmpnn`, and `rf2` from
the configured `bin_dir`. It currently accepts one positive target state with an
explicit antigen sequence/PDB and a complete HLT-labelled reference framework.
Antibody chains are `H`/`L`, HLT CDR remarks must agree with the supplied numbering,
and editable regions must be entire fixed-length CDR loops. Epitope sequence
indices are mapped to actual target PDB residue ids. Antigen sequence/PDB mismatch,
incomplete HLT annotations, indels and unsupported mappings stop the run. The
adapter uses deterministic sampling, one MPNN sequence per generated backbone,
and retains designed coordinates. Internal RF2 is provenance/self-consistency,
not the independent acceptance evaluator. Pinned installations must provide the
documented CLI flags; incompatible older layouts require their own JSON bridge.

**AntiFold.** Runs `python -m antifold.main` using explicit heavy/light or nanobody
chain labels, installed checkpoint, sampling count/seed, numeric IMGT editable
positions and `--skip_anarcii_numbering`. The PDB sequence and residue labels must
exactly match the supplied IMGT map. Paired FASTA outputs remain paired; the
parent FASTA record is excluded. Insertion-label edits require a mapping bridge.
Every sampled change is subsequently checked against protected labels and the
declared mutation budget. AntiFold likelihoods do not stand in for KD.

**ANARCII.** Uses `Anarcii(seq_type="antibody").number(...)` and accepts only
complete, ungapped, chain-type-consistent IMGT numbering. Any trimming is an
error, requiring the user to specify the actual variable domain. It does not
label an arbitrary ordinary VH as a functional VHH.

**ImmuneBuilder.** Uses ABodyBuilder2 for VH/VL and NanoBodyBuilder2 for VHH.
`weights_dir` must contain the four upstream `antibody_model_1`…`4` or
`nanobody_model_1`…`4` assets as applicable. Their availability is checked before
constructing the predictor. Actual predicted PDB sequences are checked exactly;
CA geometry is computed locally. OpenMM, pdbfixer and numbering dependencies are
part of the separate scientific environment.

**Boltz-2.** Requires `cache` and `checkpoint`. Input contains the paired protein
chains plus full antigen sequence, each with `msa: empty`; no template, pocket,
contact constraint, remote MSA or affinity property is added. Output is PDB plus
the upstream confidence JSON. The adapter reports the maximum antibody-chain to
antigen pair-chain ipTM, not global ipTM dominated by the VH/VL interface. This is
a structural confidence convention, not an antibody binding probability. Both
single-sequence mode and missing membrane/glycan geometry are recorded. The
gateway independently checks the coordinates and antigen sequence.

## Versioned JSON bridge

The private request envelope is:

```json
{
  "protocol": "antibody-tools-v1",
  "tool": "generation.iggm",
  "arguments": {"brief": {}, "count": 4},
  "runtime": {"version": "actual-code-version", "weights_version": "actual-weight-version"}
}
```

Here `{}` is a schema illustration, not a valid design brief. The complete Brief
schema is available through `antibody-agent schema` and `GET /api/schema`.

The bridge writes one JSON object to the provided output path:

```json
{
  "tool": "generation.iggm",
  "status": "unavailable",
  "data": {},
  "error": "Explain the actual missing dependency, input or model asset"
}
```

Valid status values are `ok`, `unavailable`, `error`, `timeout`. A nonzero process
exit, absent/oversized JSON, mismatched tool name or invalid structure cannot be
accepted even if an output file says `ok`. The gateway owns provenance: model,
code version, weight version, seed, request hash and output hash. Backend text
cannot replace these fields. Errors remain private; public summaries expose the
reason a required check could not be satisfied, without dumping raw candidates.

The `ok` data contracts are:

| Tool | Arguments | Required successful data |
| --- | --- | --- |
| `generation.<registered generator>` | `brief`, integer `count` | `binders`: nonempty list, at most `count`; each a complete paired/VHH Binder with product, source and pairing provenance |
| `numbering.anarcii` | `binder`, `scheme` | `binder`: unchanged sequences/product, complete reversible numbering and provenance |
| `fold.immunebuilder` | `binder`, `seed` | actual `pdb`; gateway recalculates `sequence_verified`, `ca_clashes` |
| `complex.boltz2` / `complex.chai1` | `binder`, `state`, `context_id`, `seed` | actual `pdb`, echoed `context_id`, `antigen_iptm` and `complex_plddt` in [0,1], explicit `templates_used: false`, `forced_contacts: false`; native-context checks if applicable |
| `profile.tap` / `profile.tnp` | `binder`, independently predicted `pdb`, `seed` | matching `profile` name, `applicable` boolean, `severe_flags` list, nonempty measured-from-model `metrics`, documented reference-panel domain |

Complex PDB chain names are `H` (VH/VHH), `L` (VL if present), `A` (antigen), with
exact complete input sequences. The gateway recalculates contacts and clashes;
bridge-supplied `sequence_verified` or geometric counts cannot bypass it. Do not
include a small-molecule affinity head or describe ipTM/pLDDT as experimental KD.
`native_context_checked` may only be true after actual contextual geometry checks;
the built-in protein-only Boltz adapter leaves it false.

For TAP/TNP, compute the actual format-appropriate metrics and apply a fixed,
versioned reference panel to derive severe flags. A list of regex sequence motifs
is **not** a TAP/TNP implementation. This repository intentionally provides no
mock profiling bridge. Consult [TAP](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/tap)
and [TNP](https://github.com/oxpig/TNP), their dependencies and deployed terms.

Use the same protocol for BoltzGen, IgGM, IgDesign, p-IgGen, LICHEN, Sapiens and
Humatch. Preserve each tool's real task boundary: paired-sequence priors are not
antigen-conditioned binders, inverse folding does not establish a novel pose,
and human similarity does not demonstrate clinical immunogenicity. Code/weight
licenses and unknown training overlap belong in the deployed registry manifest.

## Experimental property model files

`train-property` consumes a `TrainingRequest`:

- `model_id`: immutable local version identifier.
- `domain`: endpoint, unit, exact conditions, full AntigenState and Product.
- `rows`: `id`, complete `binder`, `lineage`, `split` (`train`, `calibration`, `test`),
  `status`, numeric `value` when measured, `source`, and original `measurement_id`.

Declare the same assay/product/antigen domain for every record; do not merge
incompatible measurements under a shared endpoint name. Preserve censored and
technical-failure rows with their status; they are counted and retained, not
assigned synthetic numeric labels. All training and split data is saved privately.

`next-batch` consumes, for example:

```json
{"objectives":[{"model_id":"project-v1","direction":"min"}],"count":4}
```

Each objective must refer to an existing, integrity-checked local model. Predictions
remain condition-specific model evidence and cannot overwrite observations or
relax the independent candidate-delivery gates.
