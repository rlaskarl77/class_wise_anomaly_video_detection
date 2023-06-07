import os
import argparse
import cv2
import numpy as np

from torch.utils.data import DataLoader

from general import smart_inference_mode
from learner import Learner
from dataset import *
from utils import *

@smart_inference_mode()
def run(opt, model, anomaly_test_loader):
    model.eval()

    for i, data in enumerate(anomaly_test_loader):
        inputs, gts, frames = data
        inputs = inputs.view(-1, inputs.size(-1)).to(torch.device('cuda'))

        score, fea = model(inputs)

        if opt.mode == 'amc':
            score = get_amc_score(score.unsqueeze(0), fea.unsqueeze(0), mean=None)
            score = score.squeeze().unsqueeze(1)

        score = score.cpu().detach().numpy()
        score_list = np.zeros(frames[0])
        step = np.round(np.linspace(0, frames[0] // 16, 33))

        for j in range(32):
            score_list[int(step[j]) * 16:(int(step[j + 1])) * 16] = score[j]

        gt_list = np.zeros(frames[0])
        for k in range(len(gts) // 2):
            s = gts[k * 2]
            e = min(gts[k * 2 + 1], frames)
            gt_list[s - 1:e] = 1

        vid = cv2.VideoCapture('./Stealing058_x264.mp4')
        frame_width = int(vid.get(3))
        frame_height = int(vid.get(4))
        out = cv2.VideoWriter('result.avi', cv2.VideoWriter_fourcc(*'DIVX'), 120, (frame_width, frame_height))
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.4
        color = (255, 255, 255)
        thickness = 1
        frame_number = 0
        if (vid.isOpened() == False):
            print("Error opening video stream or file")
        while (vid.isOpened()):
            # Capture frame-by-frame
            ret, frame = vid.read()
            if ret == True:
                frame_number += 1
                if score_list[frame_number-1] > 0.99:
                    r = frame.copy()
                    r[:, :, 0] = r[:, :, 0] * 0.5
                    r[:, :, 1] = r[:, :, 1] * 0.5
                    r = cv2.putText(r, 'Score:' + str(round(score_list[frame_number - 1], 3)), (10, 50), font, fontScale,
                                        color, thickness, cv2.LINE_AA)
                    out.write(r)
                else:
                    frame = cv2.putText(frame, 'Score:' + str(round(score_list[frame_number - 1], 3)), (10, 50), font, fontScale,
                                        color, thickness, cv2.LINE_AA)
                    out.write(frame)
            else:
                break
        vid.release()
        out.release()

def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # model configurations
    parser.add_argument('--mode', type=str, help='amc or noamc', default='noamc')
    parser.add_argument('--ckpt', type=str, default='./best_135_0.844.pt')

    return parser.parse_known_args()[0] if known else parser.parse_args()

def main(opt):
    opt = parse_opt()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Learner(input_dim=2048, drop_p=0.0, mode=opt.mode).to(device)
    assert opt.ckpt is not None, 'need checkpoint to validate'
    ckpt = torch.load(opt.ckpt)
    model.load_state_dict(ckpt['model_state_dict'])

    anomaly_test_dataset = Anomaly_Loader(is_train=0, version='test_anomaly.txt')
    anomaly_test_loader = DataLoader(anomaly_test_dataset, batch_size=1)

    run(opt, model, anomaly_test_loader)

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
