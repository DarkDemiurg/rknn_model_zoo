PYTHON ?= python3
DEVICE ?= /dev/video0
SOURCE ?= 0
MODEL ?= yolov8
VARIANT ?= w8a8
RUNTIME_VER ?= 2.46
WIDTH ?= 960
HEIGHT ?= 720
FRAMES ?= 900
FISHEYE ?= -0.2:0.0
CROP ?= 0.5
SCORE_THRESH ?= 0.45
SAVE_FRAMES ?= 0

COMMON_ARGS = --model $(MODEL) --variant $(VARIANT) --runtime-ver $(RUNTIME_VER) \
              --source $(SOURCE) -w $(WIDTH) -ht $(HEIGHT) --score-thresh $(SCORE_THRESH) \
              --save-frames $(SAVE_FRAMES)

.PHONY: bench bench-inference run run-fisheye run-crop help

help:
	@echo "Usage: make <target> [VAR=value ...]"
	@echo ""
	@echo "Targets:"
	@echo "  run             Run detector (normal camera)"
	@echo "  run-fisheye     Run detector with fisheye lens correction"
	@echo "  run-crop        Run detector with center crop (for fisheye cameras)"
	@echo "  bench           Benchmark camera capture"
	@echo "  bench-inference Benchmark camera + inference"
	@echo "  help            Show this help"
	@echo ""
	@echo "Variables:"
	@echo "  MODEL=$(MODEL)      yolov5|yolov8|yolo11|yolo26"
	@echo "  VARIANT=$(VARIANT)    w8a8|w8a16"
	@echo "  SOURCE=$(SOURCE)        Camera index"
	@echo "  WIDTH=$(WIDTH)      Capture width"
	@echo "  HEIGHT=$(HEIGHT)      Capture height"
	@echo "  SCORE_THRESH=$(SCORE_THRESH)  Detection confidence threshold"
	@echo "  CROP=$(CROP)      Center crop ratio (0.0-1.0)"
	@echo "  FISHEYE=$(FISHEYE)  k1:k2 coefficients for lens correction"
	@echo "  SAVE_FRAMES=$(SAVE_FRAMES)   Save every Nth frame (0=disabled)"
	@echo "  FRAMES=$(FRAMES)      Frames for benchmark"

bench:
	$(PYTHON) python/benchmark_camera.py --device $(DEVICE) --frames $(FRAMES)

bench-inference:
	$(PYTHON) python/benchmark_camera.py --device $(DEVICE) --frames $(FRAMES) --inference --model $(MODEL)

run:
	$(PYTHON) python/main.py $(COMMON_ARGS)

run-fisheye:
	$(PYTHON) python/main.py $(COMMON_ARGS) --fisheye="$(FISHEYE)"

run-crop:
	$(PYTHON) python/main.py $(COMMON_ARGS) --crop $(CROP)
