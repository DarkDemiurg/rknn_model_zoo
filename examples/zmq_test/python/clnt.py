import time
import zmq

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.CONFLATE, 1)
socket.connect("tcp://127.0.0.1:5555")
socket.subscribe(b'')


while True:
    message = socket.recv()
    print("Received detect: %s" % message)
