from torch.utils.data import Dataset
from src.cityscapes_tools import evaluate_cityscapes, semseg_area_features, img2results
from PIL import Image
import numpy as np
import torch
from sklearn.model_selection import train_test_split


class CityscapesPredictions:
    def __init__(self, pred_dir, ann_dir, ann_suffix="gtFine_labelIds.png", grid=1):
        self.eval_results = sorted(evaluate_cityscapes(pred_dir, ann_dir, ann_suffix).items(), key=lambda item: item[1])
        # print(eval_results[:10], eval_results[-10:])
        eval_results = np.array([
            k
            for k, _ in self.eval_results
        ])
        self.paths = np.concatenate([eval_results[:150], eval_results[-150:]])
        target = torch.cat([torch.ones(150), torch.zeros(150)])

        self.train_paths, self.test_paths, self.y_train, self.y_test = train_test_split(
            self.paths,
            target,
            test_size=0.2,
            random_state=1,
        )
        self.X_train = torch.stack(
            [
                semseg_area_features(np.array(Image.open(k)), grid)
                for k in self.train_paths
            ],
            dim=0,
        )
        self.X_test = torch.stack(
            [
                semseg_area_features(np.array(Image.open(k)), grid)
                for k in self.test_paths
            ],
            dim=0,
        )

    def __len__(self):
        return len(self.imgs)
