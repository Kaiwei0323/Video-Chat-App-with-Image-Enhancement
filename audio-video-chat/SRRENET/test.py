import cv2
import torch
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    print('PyTorch CUDA device not available')
    assert False
import os
import sys
sys.path.append('pytorch-SRResNet')
import numpy as np
from torchvision.transforms.functional import to_tensor
from skimage.metrics import structural_similarity as compute_ssim
from time import perf_counter

def mean(curr, new, count):
    return curr * count / (count + 1) + new / (count + 1)

def test():
    vid = cv2.VideoCapture(0)
    size_mean = 0

    model = torch.load('pytorch-SRResNet/model/model_srresnet.pth', map_location=device)
    model = model['model']

    psnr = 0
    ssim = 0
    bench = 0
    count = 0

    while True:
        _, frame = vid.read()
        frame = frame[:,80:560,:]

        # Compress
        lr = cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        lr = (lr / 255.).astype(float).transpose(2,0,1)
        lr = np.expand_dims(lr, 0)
        lr = torch.from_numpy(lr).to(device).type(torch.float32)
        
        # Decompress
        start = perf_counter()
        sr = model(lr)
        end = perf_counter()
        sr = sr.cpu().data[0].numpy()
        sr = np.clip(sr * 255, 0, 255).astype(np.uint8)
        sr = sr.transpose(1,2,0)

        # Score
        frame = cv2.resize(frame, dsize=(sr.shape[1], sr.shape[0]), interpolation=cv2.INTER_CUBIC)

        bench = mean(bench, end-start, count)

        score_psnr = cv2.PSNR(sr, frame)
        psnr = mean(psnr, score_psnr, count)

        score_ssim = compute_ssim(sr, frame, channel_axis=2)
        ssim = mean(ssim, score_ssim, count)

        count += 1

        # Display
        comp = np.concatenate([frame, sr], axis=1)

        cv2.imshow(f'SR: {sr.shape[0]}x{sr.shape[1]}', comp) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    print(f'Time: {bench}')
    print(f'PSNR: {psnr}')
    print(f'SSIM: {ssim}')

if __name__=='__main__':
    test()
