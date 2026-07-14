# Adding A Dataset

Dataset support is split between declarative configuration and a small Python
adapter. Configuration owns upstream identity; the adapter owns source-specific
inventory parsing.

1. Add a dataset YAML file with its adapter name, upstream release, URLs,
   archive checksums, license, expected extracted paths, and canonical
   `expected_inventory` counts by media kind. Templated sources must provide
   an exact `artifact_checksums` mapping for every expanded artifact.
2. Implement the adapter's `fetch` and `inventory` hooks. Use
   `download_remote_sources("source-name", ...)` for ordinary fixed-name
   downloads and `build_file_inventory` for stem-based file trees; keep source-specific
   archive transformations and metadata joins explicit. Inventory records must
   expose stable source IDs and must not apply curation or split policy.
3. Add the dataset to the edition YAML and provide edition-owned curation and
   exclusion files where applicable.
4. Add checked-in metadata fixtures that cover normal rows, malformed rows,
   duplicate IDs, and source-specific path handling.
5. Run `lrac-data plan` for both selection modes and the network-free test
   suite before attempting a full corpus build.

Shared code performs selection, mandatory exclusion checks, audio conversion,
manifest serialization, and state fingerprinting. An adapter must not create
its own `.done` files, choose validation members, or write a final manifest.
Transformations such as multipart streaming, split-archive joining, nested
archives, or parquet decoding belong in the dataset adapter.

URLs and archive contents can change independently. Pin a release identifier
and archive checksum, fail on a mismatch, and document any credentials or
manual license acceptance required to fetch the source.
