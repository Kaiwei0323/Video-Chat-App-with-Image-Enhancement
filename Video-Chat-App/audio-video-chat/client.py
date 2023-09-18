import socket
import struct
import threading
import cv2
import random
# from server import Client
from tkinter import *
import pickle
import numpy as np
import argparse
import pyshine as ps
import tensorflow as tf
import os
import sys
import torch

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


rtsr = os.path.abspath('audio-video-chat/PyTorch-SRResNet/model/model_srresnet.pth')
print(rtsr)
sys.path.append(rtsr)

# from model import model_srresnet
#from utils.common import rgb2ycbcr, ycbcr2rgb
from skimage.metrics import structural_similarity as compute_ssim
from time import perf_counter

ClientSocket1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ClientSocket2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ClientSocket3 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
ClientSocket4 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

parser = argparse.ArgumentParser(description='Multi-threaded Video Chat App')
parser.add_argument("--host", type=str, help="Server IP Address")
parser.add_argument("--vid_port", type=str, help="Server Video Port")
parser.add_argument("--aud_port", type=str, help="Server Audio Port")
args = parser.parse_args()

sys.path.append('audio-video-chat/PyTorch-SRResNet')
import numpy as np
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as compute_ssim
from time import perf_counter
host = args.host
port1 = int(args.vid_port)
port2 = int(args.aud_port)
# host = input("Enter server IP: ")
# port1 = int(input("Enter video port: "))
# port2 = int(input("Enter audio port: "))
# host = '127.0.0.1'
# port = 1236

print('Waiting for connection')
try:
    ClientSocket1.connect((host, port1))
    ClientSocket2.connect((host, port2))
except socket.error as e:
    print(str(e))

encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


def accum_mean(curr, new, count):
    return (curr * count) / (count + 1.0) + new / (count + 1.0)


def compress(img):
    lr = cv2.resize(img, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
    return encode_jpeg(lr)


def decompress(e):
    lr = decode_jpeg(e)
    sr = super_resolution(lr)
    return np.array(sr)



def inputs():
    while True:
        vid = cv2.VideoCapture(0)

        psnr = 0
        ssim = 0
        bench = 0
        count = 0

        while vid.isOpened():
            ret, frame = vid.read()

            res, imgenc = cv2.imencode(".jpg", frame, encode_params)
            data = np.array(imgenc)
            stringData = data.tostring()

            ClientSocket1.sendall(str.encode(str(len(stringData)).ljust(16)))
            ClientSocket1.sendall(stringData)


def audio_input():
    mode = 'send'
    name = 'SERVER TRANSMITTING AUDIO'
    audio, context = ps.audioCapture(mode=mode)
    while True:
        frame = audio.get()  # audio
        aud = pickle.dumps(frame)  # audio

        length = str.encode(str(len(aud)).ljust(16))
        if ClientSocket2:
            ClientSocket2.sendall(length)
            ClientSocket2.sendall(aud)


def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf:
            return None
        buf += newbuf
        count -= len(newbuf)
    return buf

if torch.cuda.is_available():
    device = torch.device('cuda')
    print(f'Using GPU ({torch.cuda.get_device_name()})')
else:
    device = torch.device('cpu')
    print('PyTorch CUDA device not available. Using CPU')
def output():
    model = torch.load('audio-video-chat/PyTorch-SRResNet/model/model_srresnet.pth', map_location=device)
    model = model['model']

    while True:
        threadNo = recvall(ClientSocket1, 16).decode("utf-8")
        length = recvall(ClientSocket1, 16).decode("utf-8")
        # print("length:", length)
        stringData = recvall(ClientSocket1, int(length))
        data = np.fromstring(stringData, dtype="uint8")
        # print("data:", data)
        imgdec = cv2.imdecode(data, cv2.IMREAD_COLOR)
        lr = cv2.resize(imgdec, dsize=(int(imgdec.shape[1] / 4), int(imgdec.shape[0] / 4)),
                        interpolation=cv2.INTER_AREA)
        if choice == '1':
            # Compress

            #lr = rgb2ycbcr(lr)
            #lr = lr / 255
            #lr = tf.expand_dims(lr, axis=0)
            lr = (lr / 255.).astype(float).transpose(2, 0, 1)
            lr = np.expand_dims(lr, 0)
            lr = torch.from_numpy(lr).to(device).type(torch.float32)
            # Decompress
            sr = model(lr)
            sr = sr.cpu().data[0].numpy()
            sr = np.clip(sr * 255, 0, 255).astype(np.uint8)
            sr = sr.transpose(1, 2, 0)
            cv2.imshow("Clinet " + threadNo, sr)
        else:
            sr = cv2.resize(lr, dsize=(imgdec.shape[1], imgdec.shape[0]), interpolation=cv2.INTER_CUBIC)
            cv2.imshow("Thread " + threadNo, sr)
        cv2.waitKey(1)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    ClientSocket1.close()


def audio_output():
    mode = 'get'
    name = 'CLIENT RECEIVING AUDIO'
    audio, context = ps.audioCapture(mode=mode)
    aud_data = b""
    while True:
        threadNo = recvall(ClientSocket2, 16).decode("utf-8")
        length = recvall(ClientSocket2, 16).decode("utf-8")
        frame_data = recvall(ClientSocket2, int(length))
        frame = pickle.loads(frame_data)
        audio.put(frame)
    ClientSocket2.close()


Response = ClientSocket1.recv(1024)
print(Response.decode('utf-8'))

choice = input('0 for original resolution | 1 for super resolution:')
inp = threading.Thread(target=inputs)
out = threading.Thread(target=output)
audio_inp = threading.Thread(target=audio_input)
audio_out = threading.Thread(target=audio_output)

audio_inp.start()
inp.start()
audio_out.start()
out.start()