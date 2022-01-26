import os
import sys
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data._utils.collate import default_collate

from generator import make_generator
from data import make_default_val_dataset

# from generator_jit import make_generator
# from apex import amp


def make_multiscale_noise(base_tensor, scales=6, scale_mode="bilinear"):
    batch_size, _, height, width = base_tensor.shape
    cur_height, cur_width = height, width
    result = []
    align_corners = False if scale_mode in ("bilinear", "bicubic") else None
    for _ in range(scales):
        cur_sample = torch.randn(
            batch_size, 1, cur_height, cur_width, device=base_tensor.device
        )
        cur_sample_scaled = F.interpolate(
            cur_sample,
            size=(height, width),
            mode=scale_mode,
            align_corners=align_corners,
        )
        result.append(cur_sample_scaled)
        cur_height //= 2
        cur_width //= 2
    return torch.cat(result, dim=1)


class BaseInpaintingModule(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.generator = make_generator(config.generator)

    def forward(self, batch):
        """Pass data through generator and obtain at leas 'predicted_image' and 'inpainted' keys"""
        raise NotImplementedError()


class DefaultInpaintingModule(BaseInpaintingModule):
    def __init__(
        self,
        config,
        concat_mask=True,
        add_noise_kwargs=None,
        noise_fill_hole=False,
    ):
        super().__init__(config)
        self.concat_mask = concat_mask
        self.add_noise_kwargs = add_noise_kwargs
        self.noise_fill_hole = noise_fill_hole

    def forward(self, batch):
        img = batch["image"]
        mask = batch["mask"]
        masked_img = img * (1 - mask)
        # if self.add_noise_kwargs is not None:
        #     noise = make_multiscale_noise(masked_img, **self.add_noise_kwargs)
        #     if self.noise_fill_hole:
        #         masked_img = masked_img + mask * noise[:, :masked_img.shape[1]]
        #     masked_img = torch.cat([masked_img, noise], dim=1)

        if self.concat_mask:
            masked_img = torch.cat([masked_img, mask], dim=1)

        batch["predicted_image"] = self.generator(masked_img)
        batch["inpainted"] = (
            mask * batch["predicted_image"] + (1 - mask) * batch["image"]
        )

        return batch


def load_checkpoint(train_config, ckpt_path, map_location="cpu", strict=False):
    model = DefaultInpaintingModule(train_config)
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["state_dict"], strict=strict)
    return model


def move_to_device(obj, device):
    if isinstance(obj, nn.Module):
        return obj.to(device)
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (tuple, list)):
        return [move_to_device(el, device) for el in obj]
    if isinstance(obj, dict):
        return {name: move_to_device(val, device) for name, val in obj.items()}
    raise ValueError(f'Unexpected type {type(obj)}')


def predict(indir, outdir, img_suffix='.png', gpu_id='0'):
    import yaml
    from omegaconf import OmegaConf
    import time
    if gpu_id != 'cpu':
        torch.cuda.set_device(int(gpu_id))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_config_path = "/home/ubuntu/yha/lama_lightning/big-lama/config.yaml"
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    model = load_checkpoint(
        train_config,
        "/home/ubuntu/yha/lama_lightning/big-lama/models/best.ckpt",
        map_location="cpu",
    )
    model.eval()
    model.to(device)

    if not indir.endswith("/"):
        indir += "/"

    dataset = make_default_val_dataset(
        indir, img_suffix=img_suffix, pad_out_to_modulo=8
    )
    with torch.no_grad():
        mask_fname = dataset.mask_filenames[0]
        cur_out_fname = os.path.join(
            outdir, os.path.splitext(mask_fname[len(indir) :])[0] + ".png"
        )
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = move_to_device(default_collate([dataset[0]]), device)
        batch["mask"] = (batch["mask"] > 0) * 1
        
        start = time.time()
        batch = model(batch)
        end = time.time() - start
        print(end)
        cur_res = (
            batch['inpainted'][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)
        
        
def batch_predict(indir, outdir, img_suffix='.png', gpu_id='0'):
    import yaml
    from omegaconf import OmegaConf
    import time
    if gpu_id != 'cpu':
        torch.cuda.set_device(int(gpu_id))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    train_config_path = "/home/ubuntu/yha/lama_lightning/big-lama/config.yaml"
    with open(train_config_path, "r") as f:
        train_config = OmegaConf.create(yaml.safe_load(f))
    model = load_checkpoint(
        train_config,
        "/home/ubuntu/yha/lama_lightning/big-lama/models/best.ckpt",
        map_location="cpu",
    )
    model.eval()
    model.to(device)

    if not indir.endswith("/"):
        indir += "/"

    dataset = make_default_val_dataset(
        indir, img_suffix=img_suffix, pad_out_to_modulo=8
    )
    with torch.no_grad():
        mask_fname = dataset.mask_filenames[0]
        cur_out_fname = os.path.join(
            outdir, os.path.splitext(mask_fname[len(indir) :])[0] + ".png"
        )
        os.makedirs(os.path.dirname(cur_out_fname), exist_ok=True)
        batch = move_to_device(default_collate([dataset[0]]), device)
        batch["mask"] = (batch["mask"] > 0) * 1
        
        start = time.time()
        batch = model(batch)
        end = time.time() - start
        print(end)
        cur_res = (
            batch['inpainted'][0]
            .permute(1, 2, 0)
            .detach()
            .cpu()
            .numpy()
        )

        cur_res = np.clip(cur_res * 255, 0, 255).astype("uint8")
        cur_res = cv2.cvtColor(cur_res, cv2.COLOR_RGB2BGR)
        cv2.imwrite(cur_out_fname, cur_res)
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='input image and mask directory')
    parser.add_argument('--outdir', type=str, required=True, help='output image directory')
    parser.add_argument('--img_suffix', type=str, default='.png', help='image extension')
    parser.add_argument('--device', type=str, default='0', help='gpu id')
    
    args = parser.parse_args()
    predict(args.indir, args.outdir, args.img_suffix, args.device)
    
    # import yaml
    # from omegaconf import OmegaConf
    # import time

    # train_config_path = "/home/ubuntu/yha/lama_lightning/lama/big-lama/config.yaml"
    # with open(train_config_path, "r") as f:
    #     train_config = OmegaConf.create(yaml.safe_load(f))
    # device = torch.device("cuda:1")
    # batch = {}
    # batch["image"] = torch.rand(1, 3, 1280, 720).to(device)
    # batch["mask"] = torch.zeros(1, 1, 1280, 720).to(device)
    # # model = DefaultInpaintingModule(train_config)
    # model = load_checkpoint(
    #     train_config,
    #     "/home/ubuntu/yha/lama_lightning/lama/big-lama/models/best.ckpt",
    #     map_location="cpu",
    # )
    # model.eval()
    # model.to(device)
    # # opt_level = 'O1'
    # # optimizer = torch.optim.Adam(model.parameters())

    # # model, optimizer = amp.initialize(model, optimizer, opt_level=opt_level)

    # # model = torch.jit.script(model, (batch))
    # start = time.time()
    # batch = model(batch)
    # end = time.time() - start
    # print(end)
