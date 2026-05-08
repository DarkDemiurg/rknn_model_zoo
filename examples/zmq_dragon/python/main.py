#!/usr/bin/env python3
"""
ZMQ detection sender for Radxa Dragon Q6A (Qualcomm QCS6490).
Supports YOLOv5, YOLOv8, YOLO11, YOLO26 via QAI AppBuilder (Hexagon NPU).

Architecture:
  - Capture process: ffmpeg (MJPEG decode + normalize) → shared memory (float32)
  - Main process: inference on NPU from shared memory
  - ZMQ: sends detections text + BGR image

ZMQ protocol:
  Part 1: "name@x1,y1,x2,y2@confidence;..." (text detections)
  Part 2: raw BGR image bytes (640x640)

Usage:
  python3 main.py --model yolov8 --source 0
  python3 main.py --model yolov8 --variant w8a16 --source 0 --crop 0.5
"""

import sys
import argparse
import time
import os
import cv2
import zmq
import numpy as np
from qai_appbuilder import QNNContext, Runtime, LogLevel, ProfilingLevel, PerfProfile, QNNConfig
from pathlib import Path

# ---------- constants ----------
IMAGE_SIZE = 640
FPS = 90
ZMQ_ADDR = "tcp://127.0.0.1:5757"

CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

MODEL_DIRS = {
    "yolov5": "yolov5",
    "yolov8": "yolov8",
    "yolo11": "yolo11",
    "yolo26": "yolo26",
}


# ---------- model wrapper ----------

class YoloDetector(QNNContext):
    def Inference(self, input_data):
        return super().Inference([input_data])


def find_model(model_key, base_dir, variant=None, runtime_ver=None):
    d = base_dir / MODEL_DIRS[model_key]
    if runtime_ver:
        d = d / runtime_ver
    if variant:
        d = d / variant
    if not d.exists():
        print(f"ERROR: Model directory not found: {d}")
        sys.exit(1)
    bins = list(d.glob("*.bin"))
    if not bins:
        print(f"ERROR: No .bin model found in {d}")
        sys.exit(1)
    return bins[0]


# ---------- postprocess ----------

def nms_numpy(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.array([], dtype=int)
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[np.where(ovr <= iou_threshold)[0] + 1]
    return np.array(keep, dtype=int)


def postprocess(outputs, scale_x, scale_y, score_thresh, iou_thresh):
    """Parse model outputs → list of (name, x1, y1, x2, y2, confidence)."""
    out_arrays = [np.asarray(o) for o in outputs]

    # Identify outputs by shape
    boxes = None
    flat_outputs = []
    for a in out_arrays:
        if a.ndim >= 2 and a.shape[-1] == 4:
            boxes = a.reshape(-1, 4)
        else:
            flat_outputs.append(a.ravel())

    if boxes is None or len(flat_outputs) < 2:
        return []

    n = boxes.shape[0]
    arr0 = flat_outputs[0][:n]
    arr1 = flat_outputs[1][:n]

    # Determine scores vs class_ids by value range
    if arr0.max() > 1.0:
        class_ids = arr0.astype(np.int32)
        scores = arr1
    elif arr1.max() > 1.0:
        scores = arr0
        class_ids = arr1.astype(np.int32)
    else:
        if len(np.unique(arr0[:1000])) < len(np.unique(arr1[:1000])):
            class_ids = np.round(arr0 * 79).astype(np.int32)
            scores = arr1
        else:
            scores = arr0
            class_ids = np.round(arr1 * 79).astype(np.int32)

    # Score filter
    mask = scores >= score_thresh
    if not mask.any():
        return []
    boxes, scores, class_ids = boxes[mask], scores[mask], class_ids[mask]

    # NMS
    keep = nms_numpy(boxes, scores, iou_thresh)
    if len(keep) == 0:
        return []
    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]

    # Scale boxes from 640x640 to original resolution
    boxes[:, 0] *= scale_x
    boxes[:, 2] *= scale_x
    boxes[:, 1] *= scale_y
    boxes[:, 3] *= scale_y

    return [(CLASSES[c] if 0 <= c < len(CLASSES) else str(c),
             int(b[0]), int(b[1]), int(b[2]), int(b[3]), float(s))
            for b, s, c in zip(boxes, scores, class_ids)]


def format_zmq_msg(detections):
    if not detections:
        return ""
    return "".join(f"{n}@{x1},{y1},{x2},{y2}@{c:.2f};" for n, x1, y1, x2, y2, c in detections)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="ZMQ YOLO detector for Dragon Q6A")
    parser.add_argument("--model", default="yolov8", choices=["yolov5", "yolov8", "yolo11", "yolo26"])
    parser.add_argument("--variant", default="w8a8", choices=["w8a8", "w8a16"],
                        help="Model quantization variant")
    parser.add_argument("--runtime-ver", default="2.46",
                        help="QAIRT runtime version (model subdirectory)")
    parser.add_argument("--source", default="0", help="Camera index (/dev/videoN)")
    parser.add_argument("-w", "--width", type=int, default=960, help="Capture width")
    parser.add_argument("-ht", "--height", type=int, default=720, help="Capture height")
    parser.add_argument("--zmq-addr", default=ZMQ_ADDR)
    parser.add_argument("--score-thresh", type=float, default=0.45)
    parser.add_argument("--iou-thresh", type=float, default=0.7)
    parser.add_argument("--qai-libs", default=None, help="Path to QNN libs")
    parser.add_argument("--fisheye", nargs='?', const="-0.3:-0.1",
                        default=None, metavar="k1:k2",
                        help="Fisheye lens correction (default k1=-0.3:k2=-0.1)")
    parser.add_argument("--crop", type=float, default=None, metavar="RATIO",
                        help="Crop center of frame before scaling (e.g. 0.5 = keep 50%%)")
    parser.add_argument("--save-frames", type=int, default=0, metavar="N",
                        help="Save every Nth frame with detections to debug_frames/")
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()
    model_dir = script_dir.parent / "model"

    # Resolve QNN libraries path
    qai_libs = args.qai_libs
    if qai_libs is None:
        for c in [script_dir / "qai_libs", script_dir.parent / "qai_libs", Path("qai_libs"),
                  Path.home() / "venv/lib/python3.12/site-packages/qai_appbuilder/libs"]:
            if c.exists():
                qai_libs = str(c)
                break
    if qai_libs is None:
        print("ERROR: Cannot find qai_libs. Use --qai-libs")
        sys.exit(1)

    model_path = find_model(args.model, model_dir, args.variant, args.runtime_ver)
    print(f"Model: {args.model}/{args.runtime_ver}/{args.variant} ({model_path})")
    print(f"Resolution: {args.width}x{args.height}")

    # Init QNN runtime + load model onto Hexagon NPU
    QNNConfig.Config(qai_libs, Runtime.HTP, LogLevel.ERROR, ProfilingLevel.OFF)
    try:
        detector = YoloDetector(args.model, str(model_path))
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        sys.exit(1)

    # Init ZMQ publisher
    ctx = zmq.Context()
    sock = ctx.socket(zmq.PUB)
    sock.setsockopt(zmq.SNDBUF, IMAGE_SIZE * IMAGE_SIZE * 8 * 3 * 2)
    sock.setsockopt(zmq.SNDHWM, 2)
    sock.bind(args.zmq_addr)
    print(f"ZMQ PUB: {args.zmq_addr}")

    # --- Capture in separate process ---
    import multiprocessing, multiprocessing.shared_memory, ctypes

    input_size_f32 = IMAGE_SIZE * IMAGE_SIZE * 3 * 4
    input_size_u8 = IMAGE_SIZE * IMAGE_SIZE * 3
    scale_x = args.width / IMAGE_SIZE
    scale_y = args.height / IMAGE_SIZE

    N_BUFS = 3
    shm_input = [multiprocessing.shared_memory.SharedMemory(
        create=True, size=input_size_f32) for _ in range(N_BUFS)]
    shm_rgb = multiprocessing.shared_memory.SharedMemory(
        create=True, size=input_size_u8)

    sig_r, sig_w = os.pipe()
    latest_idx = multiprocessing.RawValue(ctypes.c_int, -1)
    cam_running = multiprocessing.RawValue(ctypes.c_int, 1)

    def capture_process(source, width, height, shm_in_names, shm_rgb_name,
                        latest, running, pipe_w, fisheye, crop):
        import numpy as np, subprocess
        shm_in = [multiprocessing.shared_memory.SharedMemory(name=n) for n in shm_in_names]
        inp_bufs = [np.ndarray((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32,
                               buffer=s.buf) for s in shm_in]
        shm_r = multiprocessing.shared_memory.SharedMemory(name=shm_rgb_name)
        rgb_buf = np.ndarray((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8, buffer=shm_r.buf)

        cam = int(source)
        filters = []
        if fisheye:
            k1, k2 = [float(x) for x in fisheye.split(":")]
            filters.append(f"lenscorrection=cx=0.5:cy=0.5:k1={k1}:k2={k2}")
        if crop:
            filters.append(f"crop=iw*{crop}:ih*{crop}")
        filters.append(f"scale={IMAGE_SIZE}:{IMAGE_SIZE}")
        vf = ",".join(filters)

        proc = subprocess.Popen([
            "ffmpeg", "-y",
            "-f", "v4l2", "-input_format", "mjpeg",
            "-video_size", f"{width}x{height}",
            "-framerate", str(FPS),
            "-i", f"/dev/video{cam}",
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-an", "-sn", "pipe:1"
        ], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
           bufsize=IMAGE_SIZE * IMAGE_SIZE * 3 * 4)

        fsz = IMAGE_SIZE * IMAGE_SIZE * 3
        widx = 0
        while running.value:
            raw = proc.stdout.read(fsz)
            if len(raw) != fsz:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
            np.multiply(frame, 1.0 / 255.0, out=inp_bufs[widx], casting='unsafe')
            rgb_buf[:] = frame[0]
            latest.value = widx
            os.write(pipe_w, b'\x01')
            widx = (widx + 1) % N_BUFS

        proc.kill()
        proc.wait()
        os.close(pipe_w)
        for s in shm_in:
            s.close()
        shm_r.close()

    cap_proc = multiprocessing.Process(target=capture_process, args=(
        args.source, args.width, args.height,
        [s.name for s in shm_input], shm_rgb.name,
        latest_idx, cam_running, sig_w, args.fisheye, args.crop
    ), daemon=True)
    cap_proc.start()
    os.close(sig_w)

    PerfProfile.SetPerfProfileGlobal(PerfProfile.BURST)

    inp_views = [np.ndarray((1, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32,
                            buffer=s.buf) for s in shm_input]
    rgb_view = np.ndarray((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8, buffer=shm_rgb.buf)
    bgr_buf = np.empty((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)

    save_interval = args.save_frames
    if save_interval:
        debug_dir = script_dir.parent / "debug_frames"
        debug_dir.mkdir(exist_ok=True)

    try:
        os.read(sig_r, 1)

        _read = os.read
        _sig_r = sig_r
        _Inference = detector.Inference
        _latest = latest_idx

        loop_start = time.monotonic()
        frame_counter = 0
        cam_frame_counter = 0

        while True:
            _read(_sig_r, 64)
            cur_idx = _latest.value

            outputs = _Inference(inp_views[cur_idx])

            cam_frame_counter += 1
            frame_counter += 1
            detections = postprocess(outputs, scale_x, scale_y,
                                     args.score_thresh, args.iou_thresh)

            if frame_counter == 30:
                elapsed = time.monotonic() - loop_start
                print(f"\t FPS: {frame_counter / elapsed:.1f}")
                frame_counter = 0
                loop_start = time.monotonic()

            msg_str = format_zmq_msg(detections)
            cv2.cvtColor(rgb_view, cv2.COLOR_RGB2BGR, dst=bgr_buf)
            sock.send_string(msg_str, zmq.SNDMORE | zmq.NOBLOCK)
            sock.send(bgr_buf, zmq.NOBLOCK)

            if save_interval and cam_frame_counter % save_interval == 0:
                debug_img = bgr_buf.copy()
                for name, x1, y1, x2, y2, conf in detections:
                    dx1, dy1 = int(x1 / scale_x), int(y1 / scale_y)
                    dx2, dy2 = int(x2 / scale_x), int(y2 / scale_y)
                    cv2.rectangle(debug_img, (dx1, dy1), (dx2, dy2), (0, 255, 0), 2)
                    cv2.putText(debug_img, f"{name} {conf:.2f}",
                                (dx1, dy1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1)
                cv2.imwrite(str(debug_dir / f"frame_{cam_frame_counter}.jpg"), debug_img)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        cam_running.value = 0
        cap_proc.join(timeout=3)
        os.close(sig_r)
        PerfProfile.RelPerfProfileGlobal()
        del detector
        for s in shm_input:
            s.close()
            s.unlink()
        shm_rgb.close()
        shm_rgb.unlink()
        sock.close()
        ctx.term()


if __name__ == "__main__":
    main()
