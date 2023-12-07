# Code for evaluating a semantic segmentation model on a dataset and getting the
# output data. Also include code for evaluating the inpainting model and getting
# the resulting samples.
import torch
import sys
import os
import numpy as np
from PIL import Image
from src.source_target_dataset import SourceTargetDataset, ImageDataset
from src.cityscapes_tools import img2results, calculateIOU
from mmdet.apis import init_detector, inference_detector
from omegaconf import OmegaConf
import yaml
from mmdet.datasets.cityscapes import PALETTE
from torch.utils.data._utils.collate import default_collate
sys.path.append(os.path.join(os.path.dirname(__file__), "../lama/"))
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.utils import register_debug_signal_handlers
from saicinpainting.evaluation.utils import move_to_device
from saicinpainting.evaluation.data import pad_img_to_modulo
import cv2

class SemanticSegModel:
    def __init__(self, cfg_path: str, ckpt_path: str) -> None:
        PALETTE.append([0,0,0])
        self.colors = np.array(PALETTE, dtype=np.uint8)
        self.model = init_detector(cfg_path, ckpt_path, device="cuda:0")

    def run(self, dataset: ImageDataset) -> ImageDataset:
        """Run model on the given dataset.

        Args:
            dataset: Image data.

        Returns:
            Dataset of output images from model and GT image as label. The order
            is the same as given in the input.
        """
        semsegs = []
        targets = []
        img_ind = 0
        for img, gt_semseg in dataset:
            result = inference_detector(self.model, img, filename=f"img{img_ind}.jpg", eval='panoptic')
            pan_pred, cat_pred, _ = result[0]
            sem = cat_pred[pan_pred].numpy()
            sem[sem == 255] = self.colors.shape[0] - 1
            sem_img = Image.fromarray(self.colors[sem])
            semsegs.append(sem_img)
            targets.append(gt_semseg)
            img_ind += 1

        semsegs = np.stack(semsegs)
        targets = np.stack(targets)
        return ImageDataset(semsegs, targets)


def eval_predicted_semseg(preds: ImageDataset) -> np.ndarray:
    results = []
    preds.set_batch(1)
    for pred_semseg, gt_semseg in preds:
        size = gt_semseg.shape[:2]
        pred_semseg = cv2.resize(pred_semseg, size[::-1])
        gt_semseg = gt_semseg[:,:,2]
        results.append(calculateIOU(img2results(pred_semseg), gt_semseg))
    return np.stack(results)


class InpaintingModel:
    def __init__(self) -> None:
        with open("../lama/big-lama/config.yaml", "r") as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
        train_config.training_model.predict_only = True
        train_config.visualizer.kind = 'noop'
        self.model = load_checkpoint(train_config, "../lama/big-lama/models/best.ckpt", strict=False, map_location='cpu')
        self.model.freeze()
        self.device = torch.device("cuda")
        self.model.to(self.device)

    def run(self, dataset: ImageDataset) -> ImageDataset:
        res = []
        for img, mask in dataset:
            img = np.transpose(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (2, 0, 1))
            img = img.astype('float32') / 255
            mask = (cv2.GaussianBlur(mask.astype(float), (15, 15), cv2.BORDER_DEFAULT) > 0).astype("float32")
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # mask = mask.astype('float32') / 255

            batch = dict(
                upad_to_size=img.shape[1:],
                image=pad_img_to_modulo(img, 8),
                mask=pad_img_to_modulo(mask[None, ...], 8)
            )
            batch = default_collate([batch])
            with torch.no_grad():
                batch = move_to_device(batch, self.device)
                batch = self.model(batch)
                cur_res = batch["inpainted"][0].permute(1, 2, 0).detach().cpu().numpy()
                unpad_to_size = batch.get('unpad_to_size', None)
                if unpad_to_size is not None:
                    orig_height, orig_width = unpad_to_size
                    cur_res = cur_res[:orig_height, :orig_width]

            cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
            cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
            res.append(cur_res)

        res = np.stack(res)
        return ImageDataset(res, dataset.targets())