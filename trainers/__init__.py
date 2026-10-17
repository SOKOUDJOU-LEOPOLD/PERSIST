# fmt:off
import faulthandler
import gc
import json
import os
import shutil
import signal
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import time
import uuid
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import *

import torch
from accelerate.utils import set_seed as accelerate_set_seed, send_to_device as accelerate_send_to_device, gather_object
from loguru import logger
from tqdm import tqdm
from tyro.conf import Fixed

from utils.accelerate_util import accelerator_load_random_state

# `kill -USR1 <pid>` prints every thread's Python stack to stderr (inherited by dataloader workers),
# to diagnose hung multi-node jobs without stopping them
faulthandler.register(signal.SIGUSR1, all_threads=True)
from utils.train_util import (
    infinite_loader,
    compute_grad_norm,
    compute_step_and_weight_norm,
)


@dataclass
class BaseTrainingArgs:
    run_name: str

    output_dir: str

    remote_output_dir: str | None

    resume: str = "auto"
    """Whether to automatically resume runs from the most recent run in remote_output_dir or output_dir"""

    ckpt_to_load_for_azure_runs: str | None = None
    """Path to checkpoint to load for Azure runs, if any."""

    dataloader_workers: int = 4
    """Number of workers for train data loader per device."""

    val_dataloader_workers: int = 4
    """Number of workers for val data loader per device."""

    dataloader_prefetch_factor: int = 2
    """Prefetch factor used for the train dataloader"""

    val_dataloader_prefetch_factor: int = 2
    """Prefetch factor used for the validation dataloader"""

    use_ema: bool = True
    """Whether to maintain an exponential moving average (EMA) of model weights."""

    ema_decay: float = 0.9999
    """EMA decay factor (closer to 1.0 = slower update). Typical values: 0.999, 0.9999."""

    ema_update_every: int = 1
    """Update EMA every N optimizer steps (after gradient sync)."""

    pretrained_checkpoint: str | None = None
    """Path to a raw model.safetensors to initialize weights from before training starts
    (weight-only load, step counter stays at 0 -- distinct from --resume, which restores
    step/optimizer/RNG state too, for resuming THIS run after an interruption)."""

    pretrained_checkpoint_model_key: str = "denoiser"
    """Key into self.models to load pretrained_checkpoint into."""

    pretrained_checkpoint_extra: str | None = None
    """Optional second raw model.safetensors to load into a second model key (see
    pretrained_checkpoint_extra_model_key) -- e.g. the pixel denoiser's voxel_class_embedder,
    which has no other loading path and is otherwise always randomly initialized."""

    pretrained_checkpoint_extra_model_key: str | None = None
    """Key into self.models for pretrained_checkpoint_extra. Required if pretrained_checkpoint_extra is set."""

    use_wandb: bool = False
    """Whether to log metrics to Weights & Biases. Default off so existing non-finetune configs
    are unaffected."""

    wandb_project: str = "persist-finetune"

    wandb_entity: str | None = "leopoldgatsing-texas-a-m-university"

    wandb_run_name: str | None = None
    """Defaults to run_id if not set."""

    full_pass_validation: bool = False
    """If True, each validation check iterates the ENTIRE val_dataloader (averaging loss/metrics
    across all val episodes) instead of drawing one random batch via next() on an infinite
    iterator. Default off so existing non-finetune configs/runs are unaffected."""

    dataloader_start_method: Literal["fork", "forkserver", "spawn"] | None = None
    """How dataloader worker processes are started (None = platform default, fork on Linux). 'forkserver' never forks
    the training process itself, which avoids rare deadlocks when workers are (re)created while CUDA/NCCL threads
    are running -- seen as multi-node hangs at epoch boundaries."""

    no_epoch_boundaries: bool = False
    """Train loader draws from one long chain of shuffled passes (utils.train_util.ChainedShuffleSampler) instead of
    restarting each epoch, so multi-GPU runs never hit accelerate's per-epoch RNG sync at mismatched steps (a hang)."""

    checkpoint_retention: Literal["all", "latest_and_best"] = "all"
    """'all' keeps every checkpoint_<step> (stock behavior). 'latest_and_best' keeps only the newest
    checkpoint_<step> (full state, for resuming) plus checkpoints/best/ (model + EMA weights of the
    lowest full-pass val loss so far; needs full_pass_validation)."""

    # ---- Arguments to be filled in at runtime ----
    run_id: Fixed[str | None] = None
    """Unique identifier for this training run."""
    loaded_run: Fixed[str | None] = None
    """Run dir to resume training"""
    loaded_ckpt: Fixed[str | None] = None
    """Checkpoint step to resume training."""

    def __post_init__(self):
        if self.checkpoint_retention == "latest_and_best":
            assert self.full_pass_validation, "checkpoint_retention='latest_and_best' picks best/ by full-pass val loss: set --full-pass-validation."
        if "AMLT_OUTPUT_DIR" in os.environ:
            unique_run_id = f"{uuid.uuid4()}"[:8]
            self.run_id = f"{self.run_name}-uid-{unique_run_id}"
        else:
            self.run_id, self.loaded_run, self.loaded_ckpt = self.check_existing_ckpt()
            if self.loaded_run:
                logger.info(f"Overriding output_dir from <<{self.output_dir}>> to...")
            self.output_dir = self.loaded_run or os.path.join(self.output_dir, self.run_id)
            os.makedirs(self.output_dir, exist_ok=True)
            logger.info(f"Output directory: {self.output_dir}")
    def check_existing_ckpt(self):
        loaded_run_path, loaded_ckpt_path = None, None

        if self.resume == "auto":
            resume_dir = self.remote_output_dir or self.output_dir
            if os.path.exists(resume_dir):
                runlist = [
                    run
                    for run in os.listdir(resume_dir)
                    if os.path.isdir(os.path.join(resume_dir, run))
                    and run.split("-uid-")[0] == self.run_name
                ]
                if runlist:
                    recent_run_id = max(
                        runlist, key=lambda d: os.path.getmtime(os.path.join(resume_dir, d))
                    )
                    recent_run_path = os.path.join(resume_dir, recent_run_id)
                    if os.path.exists(os.path.join(recent_run_path, "checkpoints")):
                        checkpoint_dirs_and_iter = [
                            (ch_name, ch_name.split("checkpoint_")[-1])
                            for ch_name in os.listdir(os.path.join(recent_run_path, "checkpoints"))
                            if ch_name.startswith("checkpoint_")
                            and ch_name.split("checkpoint_")[-1].isdigit()
                        ]
                        if checkpoint_dirs_and_iter:
                            last_checkpoint_dir = max(
                                checkpoint_dirs_and_iter, key=lambda x: int(x[1])
                            )[0]
                            checkpoint_path = os.path.join(
                                recent_run_path, "checkpoints", last_checkpoint_dir
                            )
                            if os.path.isfile(os.path.join(checkpoint_path, "model.safetensors")):
                                run_id = recent_run_id
                                loaded_run_path = recent_run_path
                                loaded_ckpt_path = checkpoint_path
                                logger.info(
                                    f"Most recent checkpoint dir is {checkpoint_path}, resume training from it."
                                )
        elif self.resume != "disable":
            checkpoint_path = Path(self.resume)
            if os.path.isfile(os.path.join(checkpoint_path, "model.safetensors")):
                loaded_ckpt_path = checkpoint_path
                loaded_run_path = os.path.dirname(os.path.dirname(checkpoint_path))
                logger.info(f"Using provided checkpoint path: {checkpoint_path}")
            else:
                logger.error(f"Provided checkpoint path does not exist or lacks model.safetensors: {checkpoint_path}")
        else:
            logger.info(f"--resume is set to 'disable', training from scratch.")

        if loaded_run_path is None:
            logger.info("No checkpoints to resume. Train from scratch instead.")
            unique_run_id = f"{uuid.uuid4()}"[:8]
            run_id = f"{self.run_name}-uid-{unique_run_id}"
        else:
            run_id = os.path.basename(loaded_run_path)

        return run_id, loaded_run_path, loaded_ckpt_path

class EMA:
    """
    Simple EMA tracker for one nn.Module.

    - Keeps a shadow copy of trainable parameters.
    - Checkpointable via state_dict/load_state_dict (compatible with Accelerate's register_for_checkpointing).
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {}
        self._init_from(model)

    def _named_trainable_params(self, model: torch.nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad:
                yield name, p

    @torch.no_grad()
    def _init_from(self, model: torch.nn.Module):
        # Initialize shadow weights to current params
        self.shadow = {name: p.detach().clone() for name, p in self._named_trainable_params(model)}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        d = self.decay
        for name, p in self._named_trainable_params(model):
            if name not in self.shadow:
                # In case the model structure changed mid-run; be robust.
                self.shadow[name] = p.detach().clone()
            else:
                self.shadow[name].mul_(d).add_(p.detach(), alpha=(1.0 - d))

    @torch.no_grad()
    def copy_to(self, model: torch.nn.Module):
        for name, p in self._named_trainable_params(model):
            if name in self.shadow:
                p.data.copy_(self.shadow[name].data)

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "shadow": self.shadow,
        }

    def load_state_dict(self, state_dict: dict):
        self.decay = float(state_dict["decay"])
        self.shadow = state_dict["shadow"]
        return self

class BaseTrainer:

    def __init__(self, cfg, accelerator):
        self.cfg = cfg
        self.accelerator = accelerator
        if cfg.dataloader_start_method is not None:
            # DataLoaders built without an explicit multiprocessing_context use this default
            torch.multiprocessing.set_start_method(cfg.dataloader_start_method, force=True)
        self.init_distributed()
        self.init_seeding()
        self.init_logging()
        self.init_directories()
        self._save_checkpoint_next_sync = False
        self._step = 0
        # restored on resume so a resumed run only replaces best/ when it actually beats it
        best_info = self.ckpt_dir / "best" / "best.json"
        self.best_val_loss = json.load(open(best_info))["val_loss"] if best_info.exists() else float("inf")

        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    def prepare_dataset(self):
        raise NotImplementedError("Needs to be implemented in subclass.")

    def prepare_models(self):
        raise NotImplementedError("Needs to be implemented in subclass.")

    def load_pretrained_checkpoint(self):
        """Weight-only load from self.cfg.pretrained_checkpoint (+ optional
        pretrained_checkpoint_extra) into self.models, for FINE-TUNING from an existing
        checkpoint. Must be called after prepare_models() (so self.models exists) and before
        prepare_accelerate() (so weights land in the raw, unwrapped model and the step counter
        is untouched) -- distinct from the resume path in load_state(), which always restores the
        step counter too, for resuming THIS run after an interruption rather than starting a new
        run from existing weights.
        """
        if self.cfg.pretrained_checkpoint is None:
            return
        from safetensors.torch import load_file

        def _load_into(path, key):
            checkpoint = load_file(path)
            if checkpoint and next(iter(checkpoint)).startswith("_orig_mod."):
                checkpoint = {k[len("_orig_mod."):]: v for k, v in checkpoint.items()}
            missing, unexpected = self.models[key].load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded pretrained weights for '{key}' from {path} "
                        f"(missing={len(missing)}, unexpected={len(unexpected)})")

        _load_into(self.cfg.pretrained_checkpoint, self.cfg.pretrained_checkpoint_model_key)
        if self.cfg.pretrained_checkpoint_extra is not None:
            assert self.cfg.pretrained_checkpoint_extra_model_key is not None, (
                "pretrained_checkpoint_extra_model_key must be set when pretrained_checkpoint_extra is set"
            )
            _load_into(self.cfg.pretrained_checkpoint_extra, self.cfg.pretrained_checkpoint_extra_model_key)

    def training_losses(self, batch, verbose=True) -> Tuple[Dict, Dict]:
        raise NotImplementedError("Needs to be implemented in subclass.")

    def train(self):
        time_last_print = 0.0
        time_elapsed = 0.0
        loss_dict = defaultdict(float)  # Store loss per rank and gather every i_log steps
        metrics = defaultdict(float)  # Store main-rank metrics other than loss
        data_iterator = infinite_loader(self.dataloader)
        val_data_iterator = infinite_loader(self.val_dataloader)

        if self.cfg.use_wandb and self.accelerator.is_main_process:
            import wandb
            wandb.init(
                project=self.cfg.wandb_project, entity=self.cfg.wandb_entity,
                name=self.cfg.wandb_run_name or self.cfg.run_id, config=vars(self.cfg),
                # run_id is reused when --resume picks this run back up, so a resumed job continues the same curves
                id=self.cfg.run_id, resume="allow",
            )

        logger.info("Training starts ...")
        self.models.train()
        progress_bar = tqdm(
            initial=self.step,
            total=self.cfg.total_steps,
            desc="Training",
            unit="step",
            dynamic_ncols=True,
            disable=not self.accelerator.is_main_process,
        )
        while self.step < self.cfg.total_steps:
            with self.accelerator.accumulate(self.models):
                # Fetching data
                time_step_start = time.time()
                batch = next(data_iterator)
                metrics["time/data"] += (time.time() - time_step_start) / self.i_log

                # Training step
                time_train_start = time.time()
                with self.accelerator.autocast():
                    losses, status = self.training_losses(batch, verbose=self.cfg.log_train_metrics)
                    self.accelerator.backward(losses["loss"])
                for k, v in losses.items():
                    avg_v_detached = v.detach().mean() / self.i_log
                    loss_dict[f"loss/{k}"] += avg_v_detached  # Store loss for later logging

                if self.accelerator.sync_gradients:
                    if self.accelerator.is_main_process:
                        grad_stats_noclip = self.compute_grad_stats()
                        for k, v in grad_stats_noclip.items():
                            metrics[f"{k}_noclip"] += v / (self.i_log / self.cfg.grad_accumulation_steps)
                    self.accelerator.clip_grad_norm_(self.models.parameters(), self.cfg.max_grad_norm)
                    if self.accelerator.is_main_process:
                        grad_stats = self.compute_grad_stats()
                        for k, v in grad_stats.items():
                            metrics[k] += v / (self.i_log / self.cfg.grad_accumulation_steps)

                self.optimizer.step()
                self.optimizer.zero_grad()

                # ---- EMA update (only on real optimizer steps) ----
                if self.accelerator.sync_gradients and getattr(self.cfg, "use_ema", False) and self.ema is not None:
                    # Update every N grad steps
                    if ((self.grad_step + 1) % self.cfg.ema_update_every) == 0:
                        for name, model in self._iter_models():
                            unwrapped = self.accelerator.unwrap_model(model)
                            self.ema[name].update(unwrapped)

                # manually warm up lr without a scheduler
                if self.accelerator.sync_gradients and self.step < self.cfg.lr_warmup_steps:
                    lr_scale = min(1.0, float(self.step + 1) / (self.cfg.lr_warmup_steps * self.cfg.grad_accumulation_steps))
                    for pg in self.optimizer.param_groups:
                        pg["lr"] = lr_scale * self.cfg.learning_rate
                metrics["time/train"] += (time.time() - time_train_start) / self.i_log
                self.step += 1
                progress_bar.update(1)

            # Validation step
            if self.i_val > 0 and self.step % self.i_val == 0:
                del batch
                with torch.no_grad(), self.accelerator.autocast():
                    if self.cfg.full_pass_validation:
                        # Full pass over the ENTIRE val_dataloader (e.g. all 15 fixed val
                        # episodes), averaged -- not just one random batch via next() on an
                        # infinite iterator. Opt-in (see full_pass_validation docstring) so
                        # existing non-finetune runs keep today's lightweight behavior.
                        val_sums, val_counts = defaultdict(float), defaultdict(int)
                        for val_batch in self.val_dataloader:
                            val_batch = accelerate_send_to_device(val_batch, self.accelerator.device, non_blocking=True)
                            val_losses, status = self.training_losses(val_batch, verbose=True)
                            for k, v in val_losses.items():
                                val_sums[k] += v.mean().float().cpu().item()
                                val_counts[k] += 1
                        # each rank only sees its shard of the val set: combine all ranks before averaging
                        if self.world_size > 1:
                            per_rank = gather_object([(dict(val_sums), dict(val_counts))])
                            val_sums, val_counts = defaultdict(float), defaultdict(int)
                            for rank_sums, rank_counts in per_rank:
                                for k in rank_sums:
                                    val_sums[k] += rank_sums[k]
                                    val_counts[k] += rank_counts[k]
                        for k, v in val_sums.items():
                            metrics[f"val_loss/{k}"] += (v / val_counts[k]) / max(self.i_log // self.i_val, 1)
                        if self.cfg.checkpoint_retention == "latest_and_best" and self.accelerator.is_main_process:
                            val_loss = val_sums["loss"] / val_counts["loss"]
                            if val_loss < self.best_val_loss:
                                self.best_val_loss = val_loss
                                self.save_best(val_loss)
                    else:
                        val_batch = next(val_data_iterator)
                        val_batch = accelerate_send_to_device(val_batch, self.accelerator.device, non_blocking=True)
                        val_losses, status = self.training_losses(val_batch, verbose=True)
                        for k, v in val_losses.items():
                            metrics[f"val_loss/{k}"] += v.mean().float().cpu().item() / max(self.i_log // self.i_val, 1)
                        del val_batch

            # # Gather losses across ranks only when logging
            self.accelerator.wait_for_everyone()
            if self.step % self.i_log == 0:
                time_log_gather_start = time.time()
                loss2log = {
                    k: self.accelerator.gather(v).cpu().mean().item() for k, v in loss_dict.items()
                }
                loss_dict.clear()
                metrics["time/log_gather"] += time.time() - time_log_gather_start
                # Logging (console only)
                if self.accelerator.is_main_process:
                    log_values = {**loss2log, **metrics}
                    msg = " | ".join(f"{k}: {v:.4g}" for k, v in log_values.items())
                    logger.info(f"[step {self.step} | grad_step {self.grad_step}] {msg}")
                    if "loss/loss" in loss2log:
                        progress_bar.set_postfix(loss=loss2log["loss/loss"])
                    if self.cfg.use_wandb:
                        wandb.log(log_values, step=self.step)
                metrics.clear()
                gc.collect()

            time_elapsed += time.time() - time_step_start
            if self.accelerator.is_main_process and self.step % self.i_print == 0:
                speed = self.i_print / (time_elapsed - time_last_print) * 3600
                columns = [
                    f"Step: {self.step}/{self.cfg.total_steps} ({self.step / self.cfg.total_steps * 100:.2f}%)",
                    f"Grad step: {self.grad_step}/{self.cfg.total_grad_steps}",
                    f"Elapsed: {time_elapsed / 3600:.2f} h",
                    f"Speed: {speed:.2f} steps/h",
                    f"ETA: {(self.cfg.total_steps - self.step) / speed:.2f} h",
                ]
                logger.info(" | ".join([c.ljust(25) for c in columns]), flush=True)
                time_last_print = time_elapsed

            # Save checkpoint
            if self.i_save > 0 and self.step % self.i_save == 0:
                self._save_checkpoint_next_sync = True
            if self._save_checkpoint_next_sync and self.accelerator.sync_gradients:
                if self.accelerator.is_main_process:
                    self.save_state()
                self._save_checkpoint_next_sync = False

        progress_bar.close()
        if self.accelerator.is_main_process:
            logger.info("Training finished.")
            if self.cfg.use_wandb:
                wandb.finish()

    def shutdown(self):
        """Exit the process once every rank is done. Everything is saved by now, but persistent dataloader
        workers / the NCCL process group can keep a finished multi-node job alive until its time limit."""
        self.accelerator.wait_for_everyone()
        self.accelerator.end_training()
        logger.info("Exiting.")
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)

    def init_distributed(self):
        self.device = self.accelerator.device
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes
        if self.accelerator.mixed_precision == "fp16":
            self.weight_dtype = torch.float16
        elif self.accelerator.mixed_precision == "bf16":
            self.weight_dtype = torch.bfloat16
        else:
            warnings.warn("Half-precision is required. Consider using either fp16 or bf16.")

    def init_seeding(self):
        accelerate_set_seed(self.cfg.random_seed)

    def init_logging(self):
        logger.remove()
        log_format = f"[RANK {self.rank}/{self.world_size - 1}] - {{time:YYYY-MM-DD HH:mm:ss.SSS}} | {{level: <8}} | {{message}}"
        logger.add(
            sys.stderr,
            format=log_format,
            level="INFO",
            filter=lambda record: self.accelerator.is_main_process,
        )
    def init_directories(self):
        if "AMLT_OUTPUT_DIR" in os.environ and "AMLT_DIRSYNC_DIR" in os.environ:
            self.remote_output_dir = Path(os.environ.get("AMLT_OUTPUT_DIR", "./"))
            self.local_output_dir = Path(os.environ.get("AMLT_DIRSYNC_DIR", "./"))
        else:
            self.remote_output_dir = Path(self.cfg.output_dir)
            self.local_output_dir = Path(self.cfg.output_dir)
        self.ckpt_dir = self.remote_output_dir / "checkpoints"
        self.local_ckpt_dir = self.local_output_dir / "checkpoints"
        logger.info(f"Checkpoint dir: {self.ckpt_dir}")
        logger.info(f"Local checkpoint dir: {self.local_ckpt_dir}")

    def _iter_models(self):
        # self.models is treated as a mapping elsewhere; keep it robust
        if isinstance(self.models, dict):
            return self.models.items()
        if hasattr(self.models, "items"):
            return self.models.items()
        return [("model", self.models)]

    def prepare_trainable_parameters(self):
        pass

    def prepare_optimizer(self):
        logger.info("Initialize optimizer and lr scheduler ...")
        self.optimizer = torch.optim.AdamW(
            self.models.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay, betas=self.cfg.opt_beta
        )

    def prepare_accelerate(self):
        logger.info("Prepare accelerate for distributed training ...")
        for key in self.models:
            self.models[key] = self.accelerator.prepare(self.models[key])
        self.optimizer, self.dataloader, self.val_dataloader = \
            (self.accelerator.prepare(self.optimizer, self.dataloader, self.val_dataloader,
                                      device_placement=[True, True, False]))

        if getattr(self.cfg, "use_ema", False):
            self.ema = {}
            for name, model in self._iter_models():
                unwrapped = self.accelerator.unwrap_model(model)
                self.ema[name] = EMA(unwrapped, decay=self.cfg.ema_decay)
            logger.info(f"EMA enabled for models: {list(self.ema.keys())} (decay={self.cfg.ema_decay})")
        else:
            self.ema = None

        ckpt_is_loaded = False
        if "AMLT_OUTPUT_DIR" in os.environ:
            if self.cfg.resume == 'auto':
                ckpts = sorted(
                    [p for p in self.ckpt_dir.glob("checkpoint_*") if p.is_dir()],
                    key=lambda p: int(p.name.split("_")[-1]),
                )
                if len(ckpts) == 0:
                    logger.info(f"No checkpoints found in {self.ckpt_dir} to auto resume.")
                while len(ckpts) > 0 and not ckpt_is_loaded:
                    try:
                        ckpt_is_loaded = self.load_state(ckpts[-1])
                    except Exception as e:
                        logger.info(f"Error loading checkpoint from {ckpts[-1]}: {e}")
                        ckpts.pop()
            if self.cfg.ckpt_to_load_for_azure_runs is not None and not ckpt_is_loaded:
                ckpt_is_loaded = self.load_state(self.cfg.ckpt_to_load_for_azure_runs)
        else:
            if self.cfg.loaded_ckpt is not None:
                ckpt_is_loaded = self.load_state(self.cfg.loaded_ckpt)

        if not ckpt_is_loaded:
            logger.info("No checkpoint to resume, training from scratch instead.")

    @torch.no_grad()
    def compute_grad_stats(self, model=None, optimizer=None):
        """
        Compute gradient statistics for the model.
        """
        if model is None:
            model = self.models
        if optimizer is None:
            optimizer = self.optimizer
        grad_norm = compute_grad_norm(model)
        step_norm, weight_norm = compute_step_and_weight_norm(optimizer)
        grad_stats = {
            "train_stats/grad_norm": grad_norm,
            "train_stats/update_to_weight_ratio": step_norm / (weight_norm + 1e-10),
        }

        return grad_stats

    def save_state(self):
        logger.info(f"Saving checkpoint at step {self.step} to {self.ckpt_dir}")
        current_ckpt_dir = self.local_ckpt_dir / f"checkpoint_{self.step}"
        current_ckpt_dir.mkdir(exist_ok=True, parents=True)
        self.accelerator.save_state(current_ckpt_dir)

        if getattr(self.cfg, "use_ema", False) and getattr(self, "ema", None) is not None:
            from safetensors.torch import save_file as safetensors_save_file

            ema_tensors = {}
            for model_name, ema_obj in self.ema.items():
                for k, v in ema_obj.shadow.items():
                    ema_tensors[f"{model_name}.{k}"] = v.detach().cpu()

            safetensors_save_file(ema_tensors, str(current_ckpt_dir / "ema.safetensors"))

        if self.cfg.checkpoint_retention == "latest_and_best":
            # the new checkpoint is fully written above, so older ones are no longer needed to resume
            for old_ckpt_dir in self.local_ckpt_dir.glob("checkpoint_*"):
                if old_ckpt_dir.is_dir() and old_ckpt_dir != current_ckpt_dir:
                    logger.info(f"Removing older checkpoint {old_ckpt_dir}")
                    shutil.rmtree(old_ckpt_dir)

    def save_best(self, val_loss):
        """Write model + EMA weights (no optimizer/RNG state) to checkpoints/best/, replacing the previous best.
        File names match accelerate's save_state (model.safetensors, model_1.safetensors, ...)."""
        from safetensors.torch import save_file as safetensors_save_file, save_model as safetensors_save_model

        logger.info(f"New best val loss {val_loss:.6f} at step {self.step}, saving to {self.local_ckpt_dir / 'best'}")
        best_dir, tmp_dir = self.local_ckpt_dir / "best", self.local_ckpt_dir / "best_tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        tmp_dir.mkdir(parents=True)
        for i, (name, model) in enumerate(self._iter_models()):
            filename = "model.safetensors" if i == 0 else f"model_{i}.safetensors"
            safetensors_save_model(self.accelerator.unwrap_model(model), str(tmp_dir / filename))
        if getattr(self, "ema", None) is not None:
            ema_tensors = {f"{model_name}.{k}": v.detach().cpu()
                           for model_name, ema_obj in self.ema.items() for k, v in ema_obj.shadow.items()}
            safetensors_save_file(ema_tensors, str(tmp_dir / "ema.safetensors"))
        json.dump({"step": self.step, "val_loss": val_loss}, open(tmp_dir / "best.json", "w"))
        # swap in only once fully written, so a crash mid-save never leaves best/ broken
        shutil.rmtree(best_dir, ignore_errors=True)
        tmp_dir.rename(best_dir)

    def load_state(self, checkpoint_path):
        checkpoint_path = Path(checkpoint_path)
        self.accelerator.load_state(checkpoint_path, strict=False)
        accelerator_load_random_state(str(checkpoint_path), self.accelerator, self.rank)

        if getattr(self.cfg, "use_ema", False):
            ema_path = checkpoint_path / "ema.safetensors"
            if ema_path.exists():
                from safetensors.torch import load_file
                ema_sd = load_file(ema_path)
                # ema.safetensors is stored as flat tensors keyed by "<model_name>.<param_name>"
                for model_name, ema_obj in self.ema.items():
                    prefix = f"{model_name}."
                    shadow = {}
                    for k, t in ema_sd.items():
                        if k.startswith(prefix):
                            shadow[k[len(prefix):]] = t
                    if shadow:
                        ema_obj.load_state_dict({"decay": ema_obj.decay, "shadow": shadow})
                        for k in ema_obj.shadow.keys():
                            ema_obj.shadow[k] = ema_obj.shadow[k].to(self.accelerator.device)
                        logger.info(f"EMA weights loaded successfully for {model_name} from ema.safetensors")
            else:
                logger.info("No ema.safetensors found; EMA will start from loaded model weights.")
                for name, model in self._iter_models():
                    unwrapped = self.accelerator.unwrap_model(model)
                    self.ema[name] = EMA(unwrapped, decay=self.cfg.ema_decay)

        self.step = int(checkpoint_path.name.replace("checkpoint_", ""))
        self.accelerator.project_configuration.iteration = self.step
        logger.info(f"Loaded checkpoint from {checkpoint_path}.")
        return True

    def get_model(self, model):
        return model.module if hasattr(model, 'module') else model

    @property
    def step(self):
        return self._step

    @step.setter
    def step(self, value):
        self._step = value

    @property
    def grad_step(self):
        return self.step // self.cfg.grad_accumulation_steps

    @property
    def i_log(self):
        return self.cfg.i_log * self.cfg.grad_accumulation_steps

    @property
    def i_print(self):
        return self.cfg.i_print * self.cfg.grad_accumulation_steps

    @property
    def i_val(self):
        return self.cfg.i_val * self.cfg.grad_accumulation_steps

    @property
    def i_save(self):
        return self.cfg.i_save * self.cfg.grad_accumulation_steps
