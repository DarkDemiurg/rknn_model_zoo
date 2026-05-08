#!/usr/bin/env python3
"""
Camera capture benchmark for Radxa Dragon Q6A.
Discovers cameras, enumerates resolutions/formats, tests capture FPS
via multiple backends (OpenCV, GStreamer, FFmpeg, VidGear).
Outputs a summary table: rows = backends, columns = resolutions.
Results saved to benchmark_results.txt.

Usage:
  python3 benchmark_camera.py
  python3 benchmark_camera.py --frames 300 --device /dev/video0
"""

import argparse
import subprocess
import re
import time
import sys
import os
import struct
import fcntl
import array
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np

# Suppress warnings globally
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
warnings.filterwarnings("ignore")

# ─── V4L2 ioctl constants ──────────────────────────────────────────────────

_VIDIOC_ENUM_FMT = 0xC0405602
_VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
_VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B
_VIDIOC_QUERYCAP = 0x80685600

_V4L2_BUF_TYPE_VIDEO_CAPTURE = 1
_V4L2_FRMSIZE_TYPE_DISCRETE = 1
_V4L2_FRMIVAL_TYPE_DISCRETE = 1

_V4L2_CAP_VIDEO_CAPTURE = 0x00000001


def _fourcc_to_str(fourcc):
    return "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))


# ─── Output helper ─────────────────────────────────────────────────────────

class Tee:
    """Write to both stdout and a file."""
    def __init__(self, filepath):
        self.file = open(filepath, "w")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


# ─── Camera discovery via ioctl ─────────────────────────────────────────────

def find_cameras():
    """Return list of /dev/videoN paths that support VIDEO_CAPTURE."""
    cams = []
    for dev in sorted(Path("/dev").glob("video*")):
        try:
            fd = os.open(str(dev), os.O_RDWR | os.O_NONBLOCK)
            try:
                buf = bytearray(104)
                fcntl.ioctl(fd, _VIDIOC_QUERYCAP, buf)
                caps = struct.unpack_from("<I", buf, 84)[0]
                if caps == 0:
                    caps = struct.unpack_from("<I", buf, 80)[0]
                if caps & _V4L2_CAP_VIDEO_CAPTURE:
                    cams.append(str(dev))
            finally:
                os.close(fd)
        except Exception:
            pass
    return cams


def get_resolutions(device):
    """Return sorted list of (width, height, fps, fmt_str)."""
    modes = _get_resolutions_v4l2ctl(device)
    if modes:
        return modes
    modes = _get_resolutions_ioctl(device)
    if modes:
        return modes
    return _get_resolutions_opencv_probe(device)


def _get_resolutions_v4l2ctl(device):
    try:
        out = subprocess.check_output(
            ["v4l2-ctl", "-d", device, "--list-formats-ext"],
            stderr=subprocess.DEVNULL, text=True, timeout=5
        )
    except Exception:
        return []

    results = set()
    current_fmt = None
    current_res = None
    for line in out.splitlines():
        m = re.match(r"\s+\[\d+\].*'(\w+)'", line)
        if m:
            current_fmt = m.group(1)
        m = re.search(r"Size:.*?(\d+)x(\d+)", line)
        if m:
            current_res = (int(m.group(1)), int(m.group(2)))
        m = re.search(r"\((\d+(?:\.\d+)?)\s*fps\)", line)
        if m and current_fmt and current_res:
            results.add((current_res[0], current_res[1], float(m.group(1)), current_fmt))
    return sorted(results, key=lambda x: (x[0] * x[1], -x[2]))


def _get_resolutions_ioctl(device):
    try:
        fd = os.open(device, os.O_RDWR | os.O_NONBLOCK)
    except Exception:
        return []

    results = set()
    try:
        fmt_idx = 0
        while True:
            buf = bytearray(64)
            struct.pack_into("<II", buf, 0, fmt_idx, _V4L2_BUF_TYPE_VIDEO_CAPTURE)
            try:
                fcntl.ioctl(fd, _VIDIOC_ENUM_FMT, buf)
            except OSError:
                break
            pixfmt = struct.unpack_from("<I", buf, 12)[0]
            fmt_str = _fourcc_to_str(pixfmt)
            fmt_idx += 1

            size_idx = 0
            while True:
                buf2 = bytearray(44)
                struct.pack_into("<II", buf2, 0, size_idx, pixfmt)
                try:
                    fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMESIZES, buf2)
                except OSError:
                    break
                ftype = struct.unpack_from("<I", buf2, 8)[0]
                if ftype != _V4L2_FRMSIZE_TYPE_DISCRETE:
                    for w, h in [(640, 480), (960, 720), (1280, 720), (1920, 1080)]:
                        results.add((w, h, 30.0, fmt_str))
                    break
                w = struct.unpack_from("<I", buf2, 12)[0]
                h = struct.unpack_from("<I", buf2, 16)[0]
                size_idx += 1

                ival_idx = 0
                best_fps = 30.0
                while True:
                    buf3 = bytearray(52)
                    struct.pack_into("<IIII", buf3, 0, ival_idx, pixfmt, w, h)
                    try:
                        fcntl.ioctl(fd, _VIDIOC_ENUM_FRAMEINTERVALS, buf3)
                    except OSError:
                        break
                    itype = struct.unpack_from("<I", buf3, 16)[0]
                    if itype != _V4L2_FRMIVAL_TYPE_DISCRETE:
                        num = struct.unpack_from("<I", buf3, 20)[0]
                        den = struct.unpack_from("<I", buf3, 24)[0]
                        if num > 0:
                            best_fps = max(best_fps, den / num)
                        break
                    num = struct.unpack_from("<I", buf3, 20)[0]
                    den = struct.unpack_from("<I", buf3, 24)[0]
                    if num > 0:
                        best_fps = max(best_fps, den / num)
                    ival_idx += 1

                results.add((w, h, best_fps, fmt_str))
    finally:
        os.close(fd)

    return sorted(results, key=lambda x: (x[0] * x[1], -x[2]))


def _get_resolutions_opencv_probe(device):
    import cv2
    common = [
        (640, 480), (800, 600), (960, 720), (1280, 720), (1280, 960),
        (1920, 1080), (2560, 1440), (3840, 2160),
    ]
    results = []
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        try:
            idx = int(device.replace("/dev/video", ""))
            cap = cv2.VideoCapture(idx)
        except Exception:
            return []
    if not cap.isOpened():
        return []

    for fourcc_str in ["MJPG", None]:
        if fourcc_str:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc_str))
        for w, h in common:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if aw == w and ah == h:
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                fmt = fourcc_str or "YUYV"
                results.append((w, h, fps, fmt))
    cap.release()
    return sorted(set(results), key=lambda x: (x[0] * x[1], -x[2]))


# ─── Capture backends ───────────────────────────────────────────────────────

def bench_opencv(device, w, h, fps, fmt, num_frames):
    import cv2
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if fmt == "MJPG":
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        return None
    for _ in range(10):
        cap.read()
    t0 = time.monotonic()
    count = 0
    for _ in range(num_frames):
        ret, _ = cap.read()
        if ret:
            count += 1
    elapsed = time.monotonic() - t0
    cap.release()
    return count / elapsed if elapsed > 0 and count > 0 else None


def bench_gstreamer(device, w, h, fps, fmt, num_frames):
    import cv2
    if "GStreamer" not in cv2.getBuildInformation():
        return None
    if fmt == "MJPG":
        src = (f"v4l2src device={device}"
               f" ! image/jpeg,width={w},height={h},framerate={int(fps)}/1"
               f" ! jpegdec ! videoconvert ! video/x-raw,format=BGR"
               f" ! appsink drop=true sync=false")
    else:
        src = (f"v4l2src device={device}"
               f" ! video/x-raw,width={w},height={h},framerate={int(fps)}/1"
               f" ! videoconvert ! video/x-raw,format=BGR"
               f" ! appsink drop=true sync=false")
    cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        return None
    for _ in range(10):
        cap.read()
    t0 = time.monotonic()
    count = 0
    for _ in range(num_frames):
        ret, _ = cap.read()
        if ret:
            count += 1
    elapsed = time.monotonic() - t0
    cap.release()
    return count / elapsed if elapsed > 0 and count > 0 else None


def bench_ffmpeg(device, w, h, fps, fmt, num_frames):
    input_fmt = "mjpeg" if fmt == "MJPG" else "rawvideo"
    cmd = [
        "ffmpeg", "-y",
        "-f", "v4l2", "-input_format", input_fmt,
        "-video_size", f"{w}x{h}",
        "-framerate", str(int(fps)),
        "-i", device,
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-an", "-sn", "pipe:1"
    ]
    frame_size = w * h * 3
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL, bufsize=frame_size * 4)
    except FileNotFoundError:
        return None
    for _ in range(10):
        d = proc.stdout.read(frame_size)
        if len(d) != frame_size:
            proc.kill(); proc.wait()
            return None
    t0 = time.monotonic()
    count = 0
    for _ in range(num_frames):
        d = proc.stdout.read(frame_size)
        if len(d) != frame_size:
            break
        count += 1
    elapsed = time.monotonic() - t0
    proc.kill(); proc.wait()
    return count / elapsed if elapsed > 0 and count > 0 else None


def bench_vidgear(device, w, h, fps, fmt, num_frames):
    import cv2
    try:
        from vidgear.gears import CamGear
    except ImportError:
        return None
    src = device
    try:
        src = int(device.replace("/dev/video", ""))
    except ValueError:
        pass
    try:
        stream = CamGear(source=src, backend=cv2.CAP_V4L2).start()
    except Exception:
        return None
    if hasattr(stream, 'stream') and stream.stream is not None:
        if fmt == "MJPG":
            stream.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        stream.stream.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        stream.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        stream.stream.set(cv2.CAP_PROP_FPS, fps)
    for _ in range(10):
        stream.read()
    t0 = time.monotonic()
    count = 0
    for _ in range(num_frames):
        f = stream.read()
        if f is None:
            break
        count += 1
    elapsed = time.monotonic() - t0
    stream.stop()
    return count / elapsed if elapsed > 0 and count > 0 else None


BACKENDS = [
    ("OpenCV", bench_opencv),
    ("GStreamer", bench_gstreamer),
    ("FFmpeg", bench_ffmpeg),
]


# ─── Inference benchmark ────────────────────────────────────────────────────

IMAGE_SIZE = 640

def bench_inference(device, modes, num_frames, model_key, qai_libs):
    """Benchmark inference FPS: capture one frame per resolution, run model N times."""
    import cv2
    from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig

    class YoloDetector(QNNContext):
        def Inference(self, input_data):
            return super().Inference([input_data])

    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir.parent / "model"

    MODEL_DIRS = {"yolov5": "yolov5", "yolov8": "yolov8", "yolo11": "yolo11", "yolo26": "yolo26"}
    d = model_dir / MODEL_DIRS[model_key]
    # Look in variant subdirectories first (w8a8, w8a16), then root
    bins = list(d.glob("*/*.bin"))
    if not bins:
        bins = list(d.glob("*.bin"))
    if not bins:
        print(f"  ERROR: No .bin model in {d}")
        return {}
    model_path = bins[0]

    # Resolve qai_libs
    if qai_libs is None:
        for c in [script_dir / "qai_libs", script_dir.parent / "qai_libs", Path("qai_libs"),
                  Path.home() / "venv/lib/python3.12/site-packages/qai_appbuilder/libs"]:
            if c.exists():
                qai_libs = str(c)
                break
    if qai_libs is None:
        print("  ERROR: Cannot find qai_libs. Use --qai-libs")
        return {}

    QNNConfig.Config(qai_libs, Runtime.HTP, LogLevel.WARN, ProfilingLevel.BASIC)
    try:
        detector = YoloDetector(model_key, str(model_path))
    except Exception as e:
        print(f"  ERROR loading model: {e}")
        return {}

    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    results = {}
    for w, h, fps, fmt in modes:
        label = f"{w}x{h} [{fmt}]"
        # Capture one frame
        cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        if fmt == "MJPG":
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            results[label] = None
            continue

        # Preprocess
        inp = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(inp, axis=0).astype(np.float32) / 255.0

        # Warmup
        for _ in range(10):
            detector.Inference(inp)

        # Measure
        t0 = time.monotonic()
        for _ in range(num_frames):
            detector.Inference(inp)
        elapsed = time.monotonic() - t0
        results[label] = num_frames / elapsed

        sys.stdout.write(f"  {label:20s} ... {results[label]:.1f} fps\n")
        sys.stdout.flush()

    PerfProfile.RelPerfProfileGlobal()
    del detector
    return results

# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Camera capture FPS benchmark")
    parser.add_argument("--frames", type=int, default=900, help="Frames per test")
    parser.add_argument("--device", default=None, help="Specific /dev/videoN")
    parser.add_argument("--max-resolutions", type=int, default=None,
                        help="Limit number of resolutions to test")
    parser.add_argument("--yuyv", action="store_true",
                        help="Also test YUYV format (default: MJPG only)")
    parser.add_argument("--inference", action="store_true",
                        help="Also benchmark inference FPS per resolution")
    parser.add_argument("--model", default="yolov8",
                        choices=["yolov5", "yolov8", "yolo11", "yolo26"],
                        help="Model for inference benchmark")
    parser.add_argument("--qai-libs", default=None,
                        help="Path to QNN libs (for inference benchmark)")
    parser.add_argument("-o", "--output", default="benchmark_results.txt",
                        help="Output file for results")
    args = parser.parse_args()

    # Redirect output to both console and file
    tee = Tee(args.output)
    sys.stdout = tee

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Camera Benchmark — {timestamp}")
    print(f"Frames per test: {args.frames}")

    if args.device:
        cameras = [args.device]
    else:
        cameras = find_cameras()
        if not cameras:
            print("No cameras found.")
            sys.exit(1)

    for cam in cameras:
        print(f"\n{'='*60}")
        print(f"Camera: {cam}")
        print(f"{'='*60}")

        modes = get_resolutions(cam)
        if not modes:
            print("  No supported modes found.")
            continue

        # Filter by format
        allowed_fmts = {"MJPG"}
        if args.yuyv:
            allowed_fmts.add("YUYV")
        modes = [m for m in modes if m[3] in allowed_fmts]

        if not modes:
            print("  No supported modes found (for selected formats).")
            continue

        # Deduplicate by (w, h, fmt), keep highest fps
        best = {}
        for w, h, fps, fmt in modes:
            key = (w, h, fmt)
            if key not in best or fps > best[key]:
                best[key] = fps
        modes = [(w, h, best[(w, h, fmt)], fmt) for (w, h, fmt) in sorted(best.keys(),
                 key=lambda k: k[0]*k[1])]

        if args.max_resolutions:
            modes = modes[:args.max_resolutions]

        print(f"\nAvailable modes ({len(modes)}):")
        for w, h, fps, fmt in modes:
            print(f"  {w}x{h} @ {fps:.0f}fps [{fmt}]")

        # Results: {backend_name: {res_label: fps}}
        results = {name: {} for name, _ in BACKENDS}
        res_labels = []

        for w, h, fps, fmt in modes:
            label = f"{w}x{h} [{fmt}]"
            res_labels.append(label)
            print(f"\n--- Testing {w}x{h} [{fmt}] @ {fps:.0f}fps, {args.frames} frames ---")
            for bname, bfunc in BACKENDS:
                sys.stdout.write(f"  {bname:12s} ... ")
                sys.stdout.flush()
                try:
                    measured = bfunc(cam, w, h, fps, fmt, args.frames)
                except Exception:
                    measured = None
                if measured is not None:
                    print(f"{measured:.1f} fps")
                else:
                    print("N/A")
                results[bname][label] = measured

        # ─── Summary table ──────────────────────────────────────────────
        print(f"\n{'='*60}")
        print("SUMMARY TABLE (FPS)")
        print(f"{'='*60}\n")

        col_w = max(len(l) for l in res_labels) + 2
        header = f"{'Backend':<12s}" + "".join(f"{r:>{col_w}s}" for r in res_labels)
        print(header)
        print("-" * len(header))
        for bname, _ in BACKENDS:
            row = f"{bname:<12s}"
            for label in res_labels:
                v = results[bname].get(label)
                cell = f"{v:.1f}" if v else "N/A"
                row += f"{cell:>{col_w}s}"
            print(row)

        # ─── Inference benchmark ────────────────────────────────────────
        if args.inference:
            print(f"\n{'='*60}")
            print(f"INFERENCE BENCHMARK — {args.model} (input {IMAGE_SIZE}x{IMAGE_SIZE})")
            print(f"{'='*60}")
            print(f"Frames: {args.frames}\n")

            inf_results = bench_inference(cam, modes, args.frames,
                                          args.model, args.qai_libs)

            print(f"\n{'─'*40}")
            print(f"{'Resolution':<20s} {'Inference FPS':>14s}")
            print(f"{'─'*40}")
            for label in res_labels:
                v = inf_results.get(label)
                cell = f"{v:.1f}" if v else "N/A"
                print(f"{label:<20s} {cell:>14s}")

            # Add inference row to capture table
            results["Inference"] = inf_results
            print(f"\n{'='*60}")
            print("COMBINED TABLE (FPS)")
            print(f"{'='*60}\n")
            header = f"{'Backend':<12s}" + "".join(f"{r:>{col_w}s}" for r in res_labels)
            print(header)
            print("-" * len(header))
            for bname, _ in BACKENDS:
                row = f"{bname:<12s}"
                for label in res_labels:
                    v = results[bname].get(label)
                    cell = f"{v:.1f}" if v else "N/A"
                    row += f"{cell:>{col_w}s}"
                print(row)
            row = f"{'Inference':<12s}"
            for label in res_labels:
                v = inf_results.get(label)
                cell = f"{v:.1f}" if v else "N/A"
                row += f"{cell:>{col_w}s}"
            print(row)

        print()

    sys.stdout = tee.stdout
    tee.close()
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
