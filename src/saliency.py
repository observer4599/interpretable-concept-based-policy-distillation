# Code inspired by https://github.com/greydanus/visualize_atari
# The repository did not come with a license file, but it stated that
# it was licensed under the MIT License. Thus, I added the information below
# given the provided information in the repository.

# The MIT License

# Copyright (c) 2017 Sam Greydanus

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import cv2
from tqdm import tqdm


def get_mask(center, r, input_size):
    y, x = np.ogrid[-center[0]:input_size[0] -
                    center[0], -center[1]:input_size[1]-center[1]]
    keep = x*x + y*y <= 1
    mask = np.zeros(input_size)
    mask[keep] = 1  # select a circle of pixels
    # blur the circle of pixels. this is a 2D Gaussian for r=r^2=1
    mask = gaussian_filter(mask, sigma=r)
    return mask/mask.max()


def pertubation_saliency(obs: torch.Tensor, model, device, stride: int = 5,
                         radius: int = 5, sigma: int = 3):
    model.eval()
    input_size = obs.shape[-2:]
    size = obs.shape[-1]

    obs_np = obs.numpy(force=True)
    action_probs = torch.softmax(model(obs), dim=1).numpy(force=True)

    scores = np.zeros((len(obs), int(size/stride)+1, int(size/stride)+1))
    cols = list(range(0, size, stride))
    rows = list(range(0, size))

    pbar = tqdm(total=len(cols) * len(rows))
    for col in cols:
        for row in rows:
            pbar.update(1)

            # Masking
            mask = get_mask((row, col), radius, input_size)
            masked_obs = (1 - mask) * obs_np + mask * \
                gaussian_filter(obs_np, sigma=sigma)

            masked_action_probs = torch.softmax(
                model(torch.tensor(masked_obs, device=device, dtype=torch.float32)), dim=1).numpy(force=True)

            scores[:, int(row/stride), int(col/stride)] = .5 * \
                np.power(action_probs - masked_action_probs, 2).sum(1)
    pbar.close()

    batch_size = 256
    # Move batch to last dim
    scores = np.swapaxes(scores, 0, 1)
    scores = np.swapaxes(scores, 1, 2)
    rezied_scores = []
    for i in range(-(len(obs) // -batch_size)):
        rezied_scores.append(cv2.resize(
            scores[:, :, i * batch_size: (i + 1) * batch_size], dsize=input_size))
    rezied_scores = np.concatenate(rezied_scores, axis=2)
    rezied_scores = np.swapaxes(rezied_scores, 2, 1)
    rezied_scores = np.swapaxes(rezied_scores, 1, 0)

    return rezied_scores / rezied_scores.max(axis=(1, 2), keepdims=True)
