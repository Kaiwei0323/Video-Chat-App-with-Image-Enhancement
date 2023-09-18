import cv2
import torch
from torchsr.models import ninasr_b2
from torchvision.transforms.functional import to_tensor
import numpy as np
from skimage.metrics import structural_similarity as compute_ssim
from time import perf_counter

def mean(current, new, count):
    return current * count / (count + 1) + new / (count + 1)

def test():
    device = torch.device('cuda')
    model = ninasr_b2(scale=4, pretrained=True)
    model.to(device).eval()
    vid = cv2.VideoCapture(0)

    psnr = 0
    ssim = 0
    bench = 0
    count = 0

    while True:
        _, frame = vid.read()
        frame = frame[:,80:560,:]

        # Compress
        lr = cv2.resize(frame, dsize=(128, 128), interpolation=cv2.INTER_CUBIC)
        lr = to_tensor(lr).unsqueeze(0).to(device)

        # Decompress
        start = perf_counter()
        sr = model(lr).squeeze(0)
        end = perf_counter()
        sr = torch.permute(sr, (1, 2, 0))
        sr = sr.detach().cpu().numpy()
        sr = np.clip(sr * 255, 0, 255).astype(np.uint8)
        # print(type(sr))
        # print(sr.shape)
        # print(sr.dtype)

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
        cv2.imshow('frame', comp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    vid.release()
    cv2.destroyAllWindows()
    print(f'Time: {bench}')
    print(f'PSNR: {psnr}')
    print(f'SSIM: {ssim}')

if __name__=='__main__':
    test()
