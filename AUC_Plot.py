import os
import argparse
import matplotlib.pyplot as plt

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

        x = range(0, len(gt_list))
        plt.plot(x, gt_list, label="gt")
        plt.plot(x, score_list, label="detector")

        plt.xlabel('frame number')
        plt.ylabel('score')
        plt.title('Stealing058_x264.mp4')

        plt.legend()
        plt.show()
        print()

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
