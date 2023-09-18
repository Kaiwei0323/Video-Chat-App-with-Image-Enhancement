import cv2
import torch
from RealESRGAN import RealESRGAN
import numpy as np

def super_resolution(model, img):
    sr = np.array(model.predict(np.asarray(img)))
    sr = cv2.resize(sr, dsize=(512, 512), interpolation=cv2.INTER_CUBIC)
    return sr

def test():
    device = torch.device('cuda')
    model = RealESRGAN(device, scale=2)
    model.load_weights('weights/RealESRGAN_x2.pth', download=True)
    vid = cv2.VideoCapture(0)
    while True:
        _, frame = vid.read()
        # Downscale frame
        frame = frame[:,80:560,:]
        lr = cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        sr = super_resolution(model, lr)
        #comp = np.concatenate([frame, sr], axis=1)
        cv2.imshow('frame', sr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    test()
