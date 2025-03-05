import os
import time
import warnings
from typing import List

import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.fake_recollection_dataset import (
    FakeTeacherRecollectionDataset,
)
from vlnce_baselines.dagger_trainer import ObservationsDict
from collections import defaultdict

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

from PIL import Image
import numpy as np

def collate_fn(batch):
    """Each sample in batch: (
        episode_id,
        obs,
        prev_actions,
        oracle_actions,
        inflec_weight,
    )
    """

    def _pad_helper(t, max_len, fill_val=0):
        pad_amount = max_len - t.size(0)
        if pad_amount == 0:
            return t

        pad = torch.full_like(t[0:1], fill_val).expand(
            pad_amount, *t.size()[1:]
        )
        return torch.cat([t, pad], dim=0)

    transposed = list(zip(*batch))
    episode_ids_batch = list(transposed[0])
    trajectory_ids_batch = list(transposed[1])
    observations_batch = list(transposed[2])
    prev_actions_batch = list(transposed[3])
    corrected_actions_batch = list(transposed[4])
    weights_batch = list(transposed[5])
    B = len(prev_actions_batch)

    new_observations_batch = defaultdict(list)
    for sensor in observations_batch[0]:
        for bid in range(B):
            new_observations_batch[sensor].append(
                observations_batch[bid][sensor]
            )

    observations_batch = new_observations_batch

    max_traj_len = max(ele.size(0) for ele in prev_actions_batch)
    for bid in range(B):
        for sensor in observations_batch:
            observations_batch[sensor][bid] = _pad_helper(
                observations_batch[sensor][bid], max_traj_len, fill_val=1.0
            )

        prev_actions_batch[bid] = _pad_helper(
            prev_actions_batch[bid], max_traj_len
        )
        corrected_actions_batch[bid] = _pad_helper(
            corrected_actions_batch[bid], max_traj_len
        )
        weights_batch[bid] = _pad_helper(weights_batch[bid], max_traj_len)

    for sensor in observations_batch:
        observations_batch[sensor] = torch.stack(
            observations_batch[sensor], dim=1
        )
        observations_batch[sensor] = observations_batch[sensor].view(
            -1, *observations_batch[sensor].size()[2:]
        )

    prev_actions_batch = torch.stack(prev_actions_batch, dim=1)
    corrected_actions_batch = torch.stack(corrected_actions_batch, dim=1)
    weights_batch = torch.stack(weights_batch, dim=1)
    not_done_masks = torch.ones_like(
        corrected_actions_batch, dtype=torch.uint8
    )
    not_done_masks[0] = 0

    observations_batch = ObservationsDict(observations_batch)

    return (
        episode_ids_batch,
        trajectory_ids_batch,
        observations_batch,
        prev_actions_batch.view(-1, 1),
        not_done_masks.view(-1, 1),
        corrected_actions_batch,
        weights_batch,
    )

@baseline_registry.register_trainer(name="fake_recollect_trainer")
class RecollectTrainer(BaseVLNCETrainer):
    """A Teacher Forcing trainer that re-collects episodes from simulation
    rather than saving them all to disk. Not a real trainer, just a data collection script.
    """

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(
            os.path.dirname(
                self.config.IL.RECOLLECT_TRAINER.trajectories_file
            ),
            exist_ok=True,
        )
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def train(self) -> None:
        split = self.config.TASK_CONFIG.DATASET.SPLIT
        self.config.defrost()
        self.config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        self.config.TASK_CONFIG.TASK.SDTW.SPLIT = split
        self.config.IL.RECOLLECT_TRAINER.gt_path = (
            self.config.TASK_CONFIG.TASK.NDTW.GT_PATH
        )
        self.config.use_pbar = not is_slurm_batch_job()
        self.config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        self.config.freeze()

        dataset = FakeTeacherRecollectionDataset(self.config)

        diter = iter(
            torch.utils.data.DataLoader(
                dataset,
                batch_size=dataset.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                pin_memory=False,
                drop_last=True,
                num_workers=1,
            )
        )

        if self.config.IL.RECOLLECT_TRAINER.effective_batch_size > 0:
            assert (
                self.config.IL.RECOLLECT_TRAINER.effective_batch_size
                % self.config.IL.batch_size
                == 0
            ), (
                "Gradient accumulation: effective_batch_size"
                " should be a multiple of batch_size."
            )


        batches_per_epoch = dataset.length // dataset.batch_size

        t = (
            tqdm.trange(
                batches_per_epoch, leave=False, dynamic_ncols=True
            )
            if self.config.use_pbar
            else range(batches_per_epoch)
        )

        for batch_idx in t:
            batch_time = time.time()
            batch_str = f"{batch_idx + 1}/{batches_per_epoch}"

            (
                episode_ids_batch,
                trajectory_ids_batch,
                observations_batch,
                prev_actions_batch,
                not_done_masks,
                corrected_actions_batch,
                weights_batch,
            ) = next(diter)

            observations_batch = apply_obs_transforms_batch(
                {
                    k: v.to(device=self.device, non_blocking=True)
                    for k, v in observations_batch.items()
                },
                dataset.obs_transforms,
            )

            save_dir = ("./rgb_images")
            os.makedirs(save_dir, exist_ok=True)

            rgb_tensor = observations_batch["rgb"].cpu().numpy()
            batch_size = rgb_tensor.shape[0]

            for i in range(batch_size):
                rgb_img = rgb_tensor[i]
                filename = os.path.join(save_dir, "episode_"+episode_ids_batch[0]+"trajectory_"+trajectory_ids_batch[0]+f"_sample{i}.png")
                Image.fromarray(rgb_img).save(filename)

            save_dir = ("./depth_images")
            os.makedirs(save_dir, exist_ok=True)

            depth_tensor = observations_batch["depth"].cpu().numpy()
            batch_size = depth_tensor.shape[0]

            for i in range(batch_size):
                depth_image = depth_tensor[i]
                filename = os.path.join(save_dir, "episode_"+episode_ids_batch[0]+"trajectory_"+trajectory_ids_batch[0]+f"_sample{i}.npz")
                np.savez_compressed(filename, depth_image)
        dataset.close_sims()
