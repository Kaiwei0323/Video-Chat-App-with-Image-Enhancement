import cv2
import tensorflow as tf
import os
import sys
import numpy as np
rtsr = os.path.abspath('ESPCN-TF')
print(rtsr)
sys.path.append(rtsr)

from model import ESPCN
from utils.common import rgb2ycbcr, ycbcr2rgb
from skimage.metrics import structural_similarity as compute_ssim
from time import perf_counter

def accum_mean(curr, new, count):
    return (curr * count) / (count + 1.0) + new / (count + 1.0)

def compress(img):
    lr = cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
    return encode_jpeg(lr)

def decompress(e):
    lr = decode_jpeg(e)
    sr = super_resolution(lr)
    return np.array(sr)

def test():
    vid = cv2.VideoCapture(0)

    model = ESPCN(4)
    ckpt_path = 'ESPCN-TF/checkpoint/x4/ESPCN-x4.h5'
    model.load_weights(ckpt_path)

    psnr = 0
    ssim = 0
    bench = 0
    count = 0

    while True:
        _, frame = vid.read()
        frame = frame[:,80:560,:]

        # Compress
        lr = cv2.resize(frame, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        lr = rgb2ycbcr(lr)
        lr = lr / 255
        lr = tf.expand_dims(lr, axis=0)

        # Decompress
        start = perf_counter()
        sr = model.predict(lr)[0]
        end = perf_counter()
        sr = tf.cast(sr * 255, tf.uint8)
        sr = ycbcr2rgb(sr).numpy()

        # Score
        frame = cv2.resize(frame, dsize=(512,512), interpolation=cv2.INTER_CUBIC)

        bench = accum_mean(bench, end-start, count)

        score_psnr = cv2.PSNR(sr, frame)
        psnr = accum_mean(psnr, score_psnr, count)

        score_ssim = compute_ssim(sr, frame, channel_axis=2)
        ssim = accum_mean(ssim, score_ssim, count)

        count += 1

        # comparison
        comp = np.concatenate([frame, sr], axis=1)

        cv2.imshow(f'Frame', comp) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    print(f'Time: {bench}')
    print(f'PSNR: {psnr}')
    print(f'SSIM: {ssim}')

if __name__=='__main__':
    test()
