from gestcam.cli import _out_devices


def test_out_devices():
    """The only branchy part of the multi-device fan-out."""
    two = ["/dev/video10", "/dev/video11"]
    assert _out_devices("none", two) == []                       # preview-only
    assert _out_devices("auto", two) == two                      # the whole feature
    assert _out_devices("auto", []) == [None]                    # no loopback: pre-fanout behaviour
    assert _out_devices("/dev/video10, /dev/video11", []) == two  # copy-paste keeps a space
