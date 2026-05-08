#!/usr/bin/env python3
"""ZMQ detection client — prints received detections."""
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.RCVHWM, 2)
socket.connect("tcp://127.0.0.1:5757")
socket.subscribe(b'')

while True:
    msg = socket.recv_multipart()
    text = msg[0].decode()
    if text:
        for det in text.rstrip(";").split(";"):
            name, coords, conf = det.split("@")
            print(f"  {name:12s} [{coords}] {float(conf)*100:.0f}%")
        print()
