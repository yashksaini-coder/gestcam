"""gestcam command line.

  gestcam doctor [--json] [--no-probe]        preflight: what will/won't work here, with fixes
  gestcam run [options]                       webcam → gestures → reactions → virtual camera
  gestcam demo [--out demo.mp4 | --preview]   render every reaction on a synthetic scene
  gestcam cameras                             list video devices
"""

from __future__ import annotations

import argparse
import signal
import sys


def _add_run_opts(p: argparse.ArgumentParser):
    p.add_argument("--device", default="0", help="camera index or /dev/videoN (default 0)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--no-mjpg", action="store_true", help="don't request MJPG from the webcam")
    p.add_argument("--mirror", action="store_true", help="flip output horizontally")
    p.add_argument("--infer-width", type=int, default=640, help="downscale width fed to the model")
    p.add_argument("--infer-every", type=int, default=1, help="run the model on every Nth frame")
    p.add_argument("--min-score", type=float, default=0.5, help="canned gesture confidence threshold")
    p.add_argument("--hold", type=float, default=0.7, help="seconds a gesture must be held")
    p.add_argument("--cooldown", type=float, default=2.5, help="seconds between reactions")
    p.add_argument("--no-hold-ring", action="store_true", help="hide the hold-progress ring")
    p.add_argument("--model", default=None, help="path to gesture_recognizer.task (default: auto-download to cache)")
    p.add_argument("--out", default="auto", help="auto | /dev/videoN | none")
    p.add_argument("--preview", action="store_true", help="also show a preview window")
    p.add_argument("--record", default=None, help="also write output to this .mp4")


def _dev(s: str):
    return int(s) if s.isdigit() else s


def cmd_run(a) -> int:
    from .pipeline import RunConfig, run
    from .streams import FileOutput, MultiOutput, PreviewOutput, VirtualCamOutput

    cfg = RunConfig(device=_dev(a.device), width=a.width, height=a.height, fps=a.fps, mjpg=not a.no_mjpg,
                    mirror=a.mirror, infer_width=a.infer_width, infer_every=a.infer_every,
                    min_score=a.min_score, hold_s=a.hold, cooldown_s=a.cooldown,
                    hold_ring=not a.no_hold_ring, model=a.model)
    def make_sink(w, h, fps):
        sinks = []
        if a.out != "none":
            sinks.append(VirtualCamOutput(w, h, fps, device=None if a.out == "auto" else a.out))
        if a.preview or a.out == "none":
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
    from .streams import FileOutput, MultiOutput, PreviewOutput, VirtualCamOutput

    source = None
    if a.camera is not None:  # open first: the camera decides the real frame size
        from .streams import Capture
        source = Capture(_dev(a.camera), a.width, a.height, a.fps)
        a.width, a.height = source.width, source.height
    sinks = []
    if a.out:
        sinks.append(FileOutput(a.out, a.width, a.height, a.fps))
    if a.vcam:
        sinks.append(VirtualCamOutput(a.width, a.height, a.fps))
    if a.preview or not sinks:
        sinks.append(PreviewOutput("gestcam demo (q to quit)"))
    demo(MultiOutput(sinks), a.width, a.height, a.fps, a.seconds, source=source)
    if source:
        source.close()
    return 0


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
    p.set_defaults(fn=cmd_demo)

    p = sub.add_parser("cameras", help="list video devices")
    p.set_defaults(fn=cmd_cameras)

    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
