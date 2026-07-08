from __future__ import annotations

from pathlib import Path

import pytest

from lrac_data.config import ConfigError, load_edition_config
from lrac_data.models import MediaKind, Split


def _write_config_fixture(root: Path) -> None:
    datasets = root / "configs" / "datasets"
    editions = root / "configs" / "editions"
    metadata = root / "metadata"
    datasets.mkdir(parents=True)
    editions.mkdir(parents=True)
    metadata.mkdir()
    (datasets / "fixture.yaml").write_text(
        """\
id: fixture
adapter: libritts
release: fixture-v1
license: fixture-only
media_kinds: [speech]
sources: []
expected_files: []
""",
        encoding="utf-8",
    )
    (metadata / "quality-ids.csv").write_text("uid\nexact-id\n", encoding="utf-8")
    (metadata / "quality-template.csv").write_text(
        "speaker,filename\nspk_a,path/to/clip.wav\n", encoding="utf-8"
    )
    (metadata / "exclusions.csv").write_text(
        "name,partition,dataset,source_id,speaker_id\n"
        "validation-items,validation,fixture,validation-a,\n"
        "validation-items,validation,fixture,validation-b,\n"
        "evaluation-items,evaluation,fixture,evaluation-x,\n",
        encoding="utf-8",
    )
    (editions / "fixture.yaml").write_text(
        """\
schema_version: 1
edition: fixture
datasets: [fixture]
curation_files:
  - name: exact-quality
    dataset: fixture
    media_kind: speech
    action: include
    path: metadata/quality-ids.csv
    source_id_column: uid
  - name: template-quality
    dataset: fixture
    media_kind: speech
    action: include
    path: metadata/quality-template.csv
    source_id_template: "{speaker}_{filename_stem}"
exclusion_files:
  - metadata/exclusions.csv
""",
        encoding="utf-8",
    )


def test_load_edition_expands_curation_csvs_and_groups_exclusions(
    tmp_path: Path,
) -> None:
    _write_config_fixture(tmp_path)

    loaded = load_edition_config("fixture", repo_root=tmp_path)

    assert loaded.config.edition == "fixture"
    assert len(loaded.config.datasets) == 1
    assert [rule.source_ids for rule in loaded.config.curations] == [
        ("exact-id",),
        ("spk_a_clip",),
    ]
    assert all(rule.media_kind is MediaKind.SPEECH for rule in loaded.config.curations)
    assert len(loaded.config.exclusions) == 2
    assert loaded.config.exclusions[0].partition is Split.VALIDATION
    assert loaded.config.exclusions[0].source_ids == (
        "validation-a",
        "validation-b",
    )
    assert loaded.config.exclusions[1].partition is Split.EVALUATION
    assert loaded.config.exclusions[1].source_ids == ("evaluation-x",)


def test_exclusion_csv_requires_the_exact_normalized_header(tmp_path: Path) -> None:
    _write_config_fixture(tmp_path)
    (tmp_path / "metadata" / "exclusions.csv").write_text(
        "name,partition,dataset,source_id\nvalidation-items,validation,fixture,validation-a\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="exclusion header must be exactly"):
        load_edition_config("fixture", repo_root=tmp_path)


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        ("uid\n   \n", "blank CSV row"),
        ("uid\nexact-id\nexact-id\n", "duplicate curation source ID"),
    ],
)
def test_curation_csv_rejects_blank_and_duplicate_rows(
    tmp_path: Path, contents: str, message: str
) -> None:
    _write_config_fixture(tmp_path)
    (tmp_path / "metadata" / "quality-ids.csv").write_text(contents, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_edition_config("fixture", repo_root=tmp_path)


def test_uncurated_config_does_not_require_curation_files(tmp_path: Path) -> None:
    _write_config_fixture(tmp_path)
    (tmp_path / "metadata" / "quality-ids.csv").unlink()
    (tmp_path / "metadata" / "quality-template.csv").unlink()

    loaded = load_edition_config("fixture", repo_root=tmp_path, selection="uncurated")

    assert loaded.config.curations == ()


def test_evaluation_exclusions_must_use_exact_source_ids(tmp_path: Path) -> None:
    _write_config_fixture(tmp_path)
    (tmp_path / "metadata" / "exclusions.csv").write_text(
        "name,partition,dataset,source_id,speaker_id\n"
        "evaluation-speaker,evaluation,fixture,,speaker-x\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="evaluation exclusions must use exact source IDs"):
        load_edition_config("fixture", repo_root=tmp_path)
