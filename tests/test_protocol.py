from pathlib import Path

import pytest

from audio_deepfake.data.protocol import LABEL_TO_TARGET, load_protocol


def _write_protocol(root: Path, content: str) -> None:
    protocol_dir = root / "ASVspoof2019_LA_cm_protocols"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "ASVspoof2019.LA.cm.train.trn.txt").write_text(
        content, encoding="utf-8"
    )


def test_protocol_fields_and_paths(tmp_path: Path) -> None:
    _write_protocol(
        tmp_path,
        "speaker1 LA_T_0000001 - - bonafide\n"
        "speaker2 LA_T_0000002 - A01 spoof\n",
    )
    entries = load_protocol(tmp_path, "train")
    assert [entry.target for entry in entries] == [1, 0]
    assert entries[1].attack_id == "A01"
    assert entries[0].audio_path == (
        tmp_path / "ASVspoof2019_LA_train/flac/LA_T_0000001.flac"
    )
    assert LABEL_TO_TARGET == {"bonafide": 1, "spoof": 0}


def test_bad_protocol_row_is_rejected(tmp_path: Path) -> None:
    _write_protocol(tmp_path, "not enough fields\n")
    with pytest.raises(ValueError, match="expected 5 fields"):
        load_protocol(tmp_path, "train")


def test_unknown_split_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        load_protocol(tmp_path, "test")
