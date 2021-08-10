import os
import socket
import time

import cv2
import numpy as np
import torch.cuda
from PIL import Image

from michigan_driver import Driver
# https://webnautes.tistory.com/1382
def recvall(sock: socket, count: int) -> bytes:
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return buf
        buf += newbuf
        count -= len(newbuf)
    return buf


class SocketDriver(Driver):
    def __init__(self, gpu_id: int = -1):
        super(SocketDriver, self).__init__(gpu_id)
        self.server_socket: socket = socket.socket(
            family=socket.AF_INET,
            type=socket.SOCK_STREAM
        )
        self.server_socket.setsockopt(
            socket.SOL_SOCKET,
            socket.SO_REUSEADDR,
            1
        )
        self.server_socket.bind(
            (
                '127.0.0.1',
                8080
            )
        )

    def start(self):
        self.server_socket.listen()
        print('server just got started at 127.0.0.1:8080')

        # datas 를 받으면 michiGAN 으로 변환해서 respond
        while True:
            print('waiting')
            client_socket, addr = self.server_socket.accept()
            print('receiving')
            # try_except 필수!
            data_keys = (
                'label_ref', 'label_tag', 'orient_mask', 'orient_tag', 'orient_ref', 'image_ref', 'image_tag')
            datas = {}

            for key in data_keys:
                length = recvall(client_socket, 16).decode()
                payload = recvall(client_socket, int(length))
                print(f"receiving key:{key}, length:{length}")
                buffered_data = np.frombuffer(payload, dtype='uint8')
                if key in ("image_ref", "image_tag"):
                    np_data = np.resize(buffered_data, (512, 512, 3))
                else:
                    np_data = np.resize(buffered_data, (512, 512))

                data = Image.fromarray(np_data)
                # save it to dictionary
                datas[key] = data

            gen_payload: bytes = generated.tobytes()
            gen_length: bytes = str(len(gen_payload)).ljust(16).encode()

            print(f"sending generated image {gen_length}")
            client_socket.send(gen_length)
            client_socket.send(gen_payload)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if torch.cuda.is_available():
    SD = SocketDriver(0)
else:
    SD = SocketDriver(-1)

SD.start()
