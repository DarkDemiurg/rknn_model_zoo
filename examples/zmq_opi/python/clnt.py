import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
#socket.setsockopt(zmq.CONFLATE, 1)
socket.setsockopt(zmq.SNDHWM, 2)
socket.setsockopt(zmq.RCVBUF, 640 * 640 * 8 * 3 * 2)
socket.connect("tcp://127.0.0.1:5555")
socket.subscribe(b'')


while True:
    message = socket.recv_multipart()
    print(f"{len(message)=} {len(message[0])=} {len(message[1])=}")
