# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import random

from pathlib import Path

import numpy as np
import torch

from PIL import Image
from torch.utils.data import Dataset

from compressai.registry import register_dataset


@register_dataset("VideoFolder")
class VideoFolder(Dataset):
    """Load a video folder database. Training and testing video clips
    are stored in a directorie containing mnay sub-directorie like Vimeo90K Dataset:

    .. code-block::

        - rootdir/
            train.list
            test.list
            - sequences/
                - 00010/
                    ...
                    -0932/
                    -0933/
                    ...
                - 00011/
                    ...
                - 00012/
                    ...

    training and testing (valid) clips are withdrew from sub-directory navigated by
    corresponding input files listing relevant folders.

    This class returns a set of three video frames in a tuple.
    Random interval can be applied to if subfolders includes more than 6 frames.

    Args:
        root (string): root directory of the dataset
        rnd_interval (bool): enable random interval [1,2,3] when drawing sample frames
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'test')
    """

    def __init__(
        self,
        root,
        rnd_interval=False,
        rnd_temp_order=False,
        transform=None,
        split="train",
    ):
        if transform is None:
            raise RuntimeError("Transform must be applied")

        splitfile = Path(f"{root}/{split}.txt")
        splitdir = Path(f"{root}/sequences")

        if not splitfile.is_file():
            raise RuntimeError(f'Invalid file "{root}"')

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        with open(splitfile, "r") as f_in:
            self.sample_folders = [Path(f"{splitdir}/{f.strip()}") for f in f_in]   # sample_floders的长度等于txt中的行数

        self.max_frames = 3  # hard coding for now
        self.rnd_interval = rnd_interval
        self.rnd_temp_order = rnd_temp_order
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """

        sample_folder = self.sample_folders[index]  #sample_folder即txt中第index行代表的子文件夹的路径
        samples = sorted(f for f in sample_folder.iterdir() if f.is_file()) # 对每个子文件夹中的7帧图片的路径按顺序排序，

        max_interval = (len(samples) + 2) // self.max_frames    # (7+2) // 3 = 3
        interval = random.randint(1, max_interval) if self.rnd_interval else 1  # train时 interval = rand（1，3）  test时 interval = 1
        frame_paths = (samples[::interval])[: self.max_frames]

        # axis = 0，则表示合并后第一个维度数据要变（axis是从0开始计算的，即第一维表示0）
        # axis = 1，则表示合并后第二个维度的数据要变
        # axis = 2，则表示合并后第三个维度数据要变
        # axis = -1，则表示最后一个维度
        frames = np.concatenate(                                                        # convert("RGB") 将RGBA 格式 转变成 RGB 格式
            [np.asarray(Image.open(p).convert("RGB")) for p in frame_paths], axis=-1    # np.array.shape 由一帧(480,720,3) → 三帧(480,720,9)
        )
        frames = torch.chunk(self.transform(frames), self.max_frames)   # transform中的toTensor返回shape = torch.Size([9, 480, 720])
                                                                        # transform中的CenterCrop返回shape = torch.Size([9, 256, 256])
                                                                # torch.chunk 返回由3个tensor组成的元组，每个的shape = torch.Size([3, 256, 256])

        if self.rnd_temp_order:
            if random.random() < 0.5:
                return frames[::-1]     # 数组倒序

        return frames      # 返回的是max_frames大小的元组

    def __len__(self):
        return len(self.sample_folders)
