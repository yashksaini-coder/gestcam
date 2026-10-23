import pathlib

from gestcam.cli import _out_devices


def test_out_devices():
    """The only branchy part of the multi-device fan-out."""
    two = ["/dev/video10", "/dev/video11"]
    assert _out_devices("none", two) == []                       # preview-only
    assert _out_devices("auto", two) == two                      # the whole feature
    assert _out_devices("auto", []) == [None]                    # no loopback: pre-fanout behaviour
    assert _out_devices("/dev/video10, /dev/video11", []) == two  # copy-paste keeps a space


def test_setup_runs_packaged_script_then_models_and_doctor(monkeypatch):
    from gestcam import cli, doctor, recognizer, segment

    calls = []
    monkeypatch.setattr(cli.subprocess, "run", lambda cmd: calls.append(cmd) or type("R", (), {"returncode": 0}))
    monkeypatch.setattr(recognizer, "ensure_model", lambda *a, **k: calls.append("gesture"))
    monkeypatch.setattr(segment, "ensure_segmenter", lambda *a, **k: calls.append("seg"))
    monkeypatch.setattr(doctor, "run_doctor", lambda **k: 0)
    monkeypatch.setattr(cli.sys, "platform", "linux")
    assert cli.main(["setup"]) == 0
    assert calls[0][0] == "bash" and pathlib.Path(calls[0][1]).is_file()  # shipped in the package
    assert calls[1:] == ["gesture", "seg"]
