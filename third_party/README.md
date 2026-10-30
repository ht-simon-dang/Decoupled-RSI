# Third-party dependencies

[English](README.md) | [简体中文](README_zh.md)

Start with the [repository installation and file-layout index](../THIRD_PARTY.md).
Each family documents its dependency manifest, expected files and available
installation entrypoints:

- [Drug](drug/README.md)
- [Peptide](peptide/README.md)
- [mRNA](mrna/README.md)
- [Enzyme](enzyme/README.md)
- [Antibody](antibody/README.md)

The Antibody inventory describes the existing agent-local installer and manual
scientific backend setup; it does not add a separate third-party installer.

Source checkouts, environments, models, data, and generated local readiness
records are acquired separately and stay outside the published source snapshot.
