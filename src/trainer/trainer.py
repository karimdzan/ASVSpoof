import random

import PIL
import torch
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from src.trainer.base_trainer import BaseTrainer
from src.logger.utils import plot_spectrogram_to_buf
from src.metric.utils import compute_eer
from src.utils import inf_loop, MetricTracker
from typing import Optional
import gc


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            lr_scheduler,
            config,
            device,
            dataloaders,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.accum_grad = self.config["trainer"].get("accum_grad", 1)

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["audio", "target"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                    index=batch_idx
                    )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                # self._log_predictions(**batch)
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker,
                      index: Optional[int] = None, total: Optional[int] = None):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(batch["audio"])
        if type(outputs) is dict:
            batch.update(outputs)
        else:
            batch["logits"] = outputs

        # batch["log_probs"] = F.log_softmax(batch["logits"], dim=-1)
        # batch["log_probs_length"] = self.model.transform_input_lengths(
        #     batch["spectrogram_length"]
        # )
        batch["loss"] = self.criterion(batch["logits"], batch["target"])
        if is_train:
            if (index + 1) % self.accum_grad == 0 or index + 1 == self.len_epoch:
                self.optimizer.zero_grad()
            batch["loss"].backward()
            if (index + 1) % self.accum_grad == 0 or index + 1 == self.len_epoch:
                self._clip_grad_norm()
                self.optimizer.step()
                self.lr_scheduler.step()
            torch.cuda.empty_cache()
            gc.collect()
            for met in self.metrics:
                metrics.update(met.name, met(**batch))

        metrics.update("loss", batch["loss"].item())
        # for met in self.metrics:
        #     metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()

        logits = []
        target = []

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
                logits.append(batch["logits"])
                target.append(batch["target"])
            self.writer.set_step(epoch * self.len_epoch, part)
            # self._log_scalars(self.evaluation_metrics)
            # self._log_predictions(**batch)
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        logits = torch.cat(logits)
        target = torch.cat(target)
        for metric in self.metrics:
            self.evaluation_metrics.update(metric.name, metric(logits=logits, target=target))
        self._log_scalars(self.evaluation_metrics)
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
