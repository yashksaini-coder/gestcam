"""gestcam command line.

  gestcam doctor [--json] [--no-probe]        preflight: what will/won't work here, with fixes
  gestcam run [options]                       webcam → gestures → reactions → virtual camera
  gestcam demo [--out demo.mp4 | --preview]   render every reaction on a synthetic scene
  gestcam eval <clip.mp4 | dir-of-images>     offline: what would have fired, frame by frame
  gestcam cameras                             list video devices
"""

from __future__ import annotations

import argparse
import signal
import sys


def _add_model_opts(p: argparse.ArgumentParser):
    """Flags `run` and `eval` share, so eval measures the settings you run with."""
    p.add_argument("--infer-width", type=int, default=640, help="downscale width fed to the model")
    p.add_argument("--num-hands", type=int, default=2, help="max hands the model tracks")
    p.add_argument("--detect-conf", type=float, default=0.5, help="palm detection confidence floor")
    p.add_argument("--min-score", type=float, default=0.5, help="canned gesture confidence threshold")
    p.add_argument("--hold", type=float, default=0.7, help="seconds a gesture must be held")
    p.add_argument("--cooldown", type=float, default=2.5, help="seconds between reactions")
    p.add_argument("--defer", type=float, default=0.15,
                   help="extra hold for a one-hand gesture while a second hand is in frame")
    p.add_argument("--model", default=None, help="path to gesture_recognizer.task (default: auto-download to cache)")


def _model_cfg(a) -> dict:
    return dict(infer_width=a.infer_width, num_hands=a.num_hands, detect_conf=a.detect_conf,
                min_score=a.min_score, hold_s=a.hold, cooldown_s=a.cooldown, defer_s=a.defer,
                model=a.model)


def _add_run_opts(p: argparse.ArgumentParser):
    p.add_argument("--device", default="0", help="camera index or /dev/videoN (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-mjpg", action="store_true", help="don't request MJPG from the webcam")
    p.add_argument("--mirror", action="store_true", help="flip output horizontally")
    p.add_argument("--infer-every", type=int, default=1, help="run the model on every Nth frame")
    p.add_argument("--no-hold-ring", action="store_true", help="hide the hold-progress ring")
    p.add_argument("--out", default="auto",
                   help="auto (every GestCam loopback) | /dev/video10,/dev/video11 | none")
    p.add_argument("--preview", action="store_true", help="also show a preview window")
    p.add_argument("--record", default=None, help="also write output to this .mp4")
    p.add_argument("--record-raw", default=None,
                   help="also write the frames the model sees (mirrored if --mirror, no effects) to this .mp4, for `gestcam eval`")
    p.add_argument("--bg", default=None, metavar="remove|blur|IMAGE",
                   help="replace the background behind you: flat grey, blurred, or an image")
    p.add_argument("--no-segment", action="store_true",
                   help="no person mask: every effect is drawn over you (saves ~5 ms of model CPU)")
    p.add_argument("--debug", action="store_true",
                   help="draw landmarks, poses, the decision and hold timers on the output")
    _add_model_opts(p)


def _dev(s: str):
    return int(s) if s.isdigit() else s


def _out_devices(out: str, found: list[str]) -> list[str | None]:
    """--out → the device list for VirtualCamOutput. [None] = let pyvirtualcam pick.

    `found or [None]` is deliberate: with no GestCam-labelled device (module not
    loaded, or not Linux) we hand pyvirtualcam None, which is byte-for-byte the
    behaviour before multi-device output existed.
    """
    if out == "none":
        return []
    if out != "auto":
        return [d.strip() for d in out.split(",") if d.strip()]
    return found or [None]


def cmd_run(a) -> int:
    from .pipeline import RunConfig, run
    from .streams import FileOutput, MultiOutput, PreviewOutput, VirtualCamOutput, gestcam_devices

    cfg = RunConfig(device=_dev(a.device), width=a.width, height=a.height, fps=a.fps, mjpg=not a.no_mjpg,
                    mirror=a.mirror, infer_every=a.infer_every, hold_ring=not a.no_hold_ring,
                    segment=not a.no_segment, background=a.bg, debug=a.debug, record_raw=a.record_raw, **_model_cfg(a))
    def make_sink(w, h, fps):
        sinks = []
        for d in _out_devices(a.out, gestcam_devices()):
            try:
                sinks.append(VirtualCamOutput(w, h, fps, device=d))
            except Exception as e:
                # A rival producer fails in the CONSTRUCTOR, not in send(): S_FMT
                # returns EBUSY while another process holds the OUTPUT token. One
                # busy device must not take the others down with it.
                print(f"[gestcam] skipping {d}: {e}")
        if a.out != "none" and not sinks:
            if a.out != "auto":
                raise RuntimeError("no virtual camera could be opened")
            # No loopback loaded: still useful as a local mirror, so fall back to a window.
            print("[gestcam] no virtual camera (run scripts/loopback.sh to feed Discord/Zoom); "
                  "showing a preview window instead")
        if a.preview or not sinks:
            sinks.append(PreviewOutput())
        if a.record:
            sinks.append(FileOutput(a.record, w, h, fps))
        return MultiOutput(sinks)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("flag", True))
    run(cfg, make_sink, should_stop=lambda: stop["flag"])
    return 0


def cmd_demo(a) -> int:
    from .pipeline import demo
    from .streams import FileOutput, MultiOutput, PreviewOutput, VirtualCamOutput, gestcam_devices

    source = None
    if a.camera is not None:  # open first: the camera decides the real frame size
        from .streams import Capture
        source = Capture(_dev(a.camera), a.width, a.height, a.fps)
        a.width, a.height = source.width, source.height
    sinks = []
    if a.out:
        sinks.append(FileOutput(a.out, a.width, a.height, a.fps))
    if a.vcam:
        # Same fan-out as `run`, which makes `gestcam demo --vcam` the check for
        # multi-device output: no webcam and no model needed to see both light up.
        sinks += [VirtualCamOutput(a.width, a.height, a.fps, device=d)
                  for d in _out_devices("auto", gestcam_devices())]
    if a.preview or not (sinks or a.sheet):
        sinks.append(PreviewOutput("gestcam demo (q to quit)"))
    demo(MultiOutput(sinks), a.width, a.height, a.fps, a.seconds, source=source, sheet=a.sheet)
    if source:
        source.close()
    return 0


def cmd_eval(a) -> int:
    from .pipeline import RunConfig, evaluate
    try:
        return evaluate(a.path, RunConfig(**_model_cfg(a)), expect=a.expect, as_json=a.json)
    except ValueError as e:  # nothing to evaluate: never a vacuous exit 0
        print(f"[eval] {e}", file=sys.stderr)
        return 2


def cmd_cameras(_a) -> int:
    from .doctor import Report, check_video_devices
    r = Report()
    check_video_devices(r)
    print(r.render())
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gestcam", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("doctor", help="preflight diagnostics")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-probe", action="store_true", help="don't open the webcam")
    p.add_argument("--device", default=None)
    p.set_defaults(fn=lambda a: __import__("gestcam.doctor", fromlist=["run_doctor"]).run_doctor(a.json, not a.no_probe, a.device))

    p = sub.add_parser("run", help="run the pipeline")
    _add_run_opts(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("demo", help="render all reactions without a model or gestures")
    p.add_argument("--out", default=None, help="write .mp4")
    p.add_argument("--vcam", action="store_true", help="also send to the virtual camera")
    p.add_argument("--preview", action="store_true")
    p.add_argument("--camera", default=None, help="use a real camera as background (index or /dev/videoN)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seconds", type=float, default=24.0)
    p.add_argument("--sheet", default=None, help="write a contact sheet of all 8 reactions (.png)")
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("eval", help="replay a clip or a folder of photos through recognition + hold logic")
    p.add_argument("path", help="video file (its own timestamps) or a directory of images "
                   "(each held 1.2 s, or hold + defer + 0.35 s if longer)")
    p.add_argument("--expect", default=None, help="reaction every clip should fire; exit 1 otherwise")
    p.add_argument("--json", action="store_true", help="per-frame records + summary as JSON")
    _add_model_opts(p)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("cameras", help="list video devices")
    p.set_defaults(fn=cmd_cameras)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
