"""
# Portions of this file are based on code from the “jbloomAus/SAELens” and "HugoFry/mats_sae_training_for_ViTs" repositories (MIT-licensed):
    https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/sae_trainer.py
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/config.py
"""

from typing import Any

import torch
import wandb
from tqdm import tqdm

from src.sae_training.config import Config
from src.sae_training.hooked_vit import Hook, HookedVisionTransformer
from src.sae_training.sparse_autoencoder import SparseAutoencoder
from src.sae_training.vit_activations_store import ViTActivationsStore


class SAETrainer:
    def __init__(
        self,
        sae: SparseAutoencoder,
        model: HookedVisionTransformer,
        activation_store: ViTActivationsStore,
        cfg: Config,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler,
        device: torch.device,
    ):
        self.sae = sae
        self.model = model
        self.activation_store = activation_store
        self.cfg = cfg
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.act_freq_scores = torch.zeros(sae.cfg.d_sae, device=device)
        self.n_forward_passes_since_fired = torch.zeros(sae.cfg.d_sae, device=device)
        self.n_frac_active_tokens = 0
        self.n_training_tokens = 0
        self.ghost_grad_neuron_mask = None
        self.n_training_steps = 0

        self.checkpoint_thresholds = list(
            range(
                0,
                cfg.total_training_tokens,
                cfg.total_training_tokens // self.cfg.n_checkpoints,
            )
        )[1:]

    def _build_sparsity_log_dict(self) -> dict[str, Any]:
        feature_freq = self.act_freq_scores / self.n_frac_active_tokens
        log_feature_freq = torch.log10(feature_freq + 1e-10).detach().cpu()

        return {
            "plots/feature_density_line_chart": wandb.Histogram(
                log_feature_freq.numpy()
            ),
            "metrics/mean_log10_feature_sparsity": log_feature_freq.mean().item(),
        }

    @torch.no_grad()
    def _reset_running_sparsity_stats(self) -> None:
        self.act_freq_scores = torch.zeros(self.cfg.d_sae, device=self.device)
        self.n_frac_active_tokens = 0

    def _train_step(
        self,
        sae_in: torch.Tensor,
    ):
        self.optimizer.zero_grad()

        self.sae.train()
        self.sae.set_decoder_norm_to_unit_norm()

        # log and then reset the feature sparsity every feature_sampling_window steps
        if (self.n_training_steps + 1) % self.cfg.feature_sampling_window == 0:
            if self.cfg.log_to_wandb:
                sparsity_log_dict = self._build_sparsity_log_dict()
                wandb.log(sparsity_log_dict, step=self.n_training_steps)
            self._reset_running_sparsity_stats()

        ghost_grad_neuron_mask = (
            self.n_forward_passes_since_fired > self.cfg.dead_feature_window
        ).bool()
        sae_out, feature_acts, loss_dict = self.sae(sae_in, ghost_grad_neuron_mask)

        with torch.no_grad():
            if self.cfg.class_token:
                did_fire = (feature_acts > 0).float().sum(-2) > 0
                self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0)

            else:
                # default for PatchSAE
                did_fire = (((feature_acts > 0).float().sum(-2) > 0).sum(-2)) > 0
                self.act_freq_scores += (feature_acts.abs() > 0).float().sum(0).sum(0)

            self.n_forward_passes_since_fired += 1
            self.n_forward_passes_since_fired[did_fire] = 0
            self.n_frac_active_tokens += sae_out.size(0)

        self.ghost_grad_neuron_mask = ghost_grad_neuron_mask

        loss_dict["loss"].backward()
        self.sae.remove_gradient_parallel_to_decoder_directions()

        self.optimizer.step()
        self.scheduler.step()

        return sae_out, feature_acts, loss_dict

    def _calculate_sparsity_metrics(self) -> dict:
        """Calculate sparsity-related metrics."""
        feature_freq = self.act_freq_scores / self.n_frac_active_tokens

        return {
            "sparsity/mean_passes_since_fired": self.n_forward_passes_since_fired.mean().item(),
            "sparsity/n_passes_since_fired_over_threshold": self.ghost_grad_neuron_mask.sum().item(),
            "sparsity/below_1e-5": (feature_freq < 1e-5).float().mean().item(),
            "sparsity/below_1e-6": (feature_freq < 1e-6).float().mean().item(),
            "sparsity/dead_features": (feature_freq < self.cfg.dead_feature_threshold)
            .float()
            .mean()
            .item(),
        }

    @torch.no_grad()
    def _log_train_step(
        self,
        feature_acts: torch.Tensor,
        loss_dict: dict[str, torch.Tensor],
        sae_out: torch.Tensor,
        sae_in: torch.Tensor,
    ):
        """Log training metrics to wandb."""
        metrics = self._calculate_metrics(feature_acts, sae_out, sae_in)
        sparsity_metrics = self._calculate_sparsity_metrics()

        log_dict = {
            "losses/overall_loss": loss_dict["loss"].item(),
            "losses/mse_loss": loss_dict["mse_loss"].item(),
            "losses/l1_loss": loss_dict["l1_loss"].item(),
            "losses/ghost_grad_loss": loss_dict["mse_loss_ghost_resid"].item(),
            **metrics,
            **sparsity_metrics,
            "details/n_training_tokens": self.n_training_tokens,
            "details/current_learning_rate": self.optimizer.param_groups[0]["lr"],
        }

        wandb.log(log_dict, step=self.n_training_steps)

    @torch.no_grad()
    def _calculate_metrics(
        self, feature_acts: torch.Tensor, sae_out: torch.Tensor, sae_in: torch.Tensor
    ) -> dict:
        """Calculate model performance metrics."""
        if self.cfg.class_token:
            l0 = (feature_acts > 0).float().sum(-1).mean()
        else:
            l0 = (feature_acts > 0).float().sum(-1).mean(-1).mean()
        per_token_l2_loss = (sae_out - sae_in).pow(2).sum(dim=-1).mean().squeeze()
        total_variance = sae_in.pow(2).sum(-1).mean()
        explained_variance = 1 - per_token_l2_loss / total_variance

        return {
            "metrics/explained_variance": explained_variance.mean().item(),
            "metrics/explained_variance_std": explained_variance.std().item(),
            "metrics/l0": l0.item(),
        }

    @torch.no_grad()
    def _update_pbar(self, loss_dict, pbar, batch_size):
        pbar.set_description(
            f"{self.n_training_steps}| MSE Loss {loss_dict['mse_loss'].item():.3f} | L1 {loss_dict['l1_loss'].item():.3f}"
        )
        pbar.update(batch_size)

    @torch.no_grad()
    def _checkpoint_if_needed(self):
        if (
            self.checkpoint_thresholds
            and self.n_training_tokens > self.checkpoint_thresholds[0]
        ):
            self.save_checkpoint()
            self.run_evals()  # TODO: Implement this
            self.checkpoint_thresholds.pop(0)

    def save_checkpoint(self, is_final=False):
        if is_final:
            path = f"{self.cfg.checkpoint_path}/final_{self.sae.get_name()}.pt"
        else:
            path = f"{self.cfg.checkpoint_path}/{self.n_training_tokens}_{self.sae.get_name()}.pt"
        self.sae.save_model(path)

    def fit(self) -> SparseAutoencoder:
        pbar = tqdm(total=self.cfg.total_training_tokens, desc="Training SAE")

        try:
            # Train loop
            while self.n_training_tokens < self.cfg.total_training_tokens:
                # Do a training step.
                sae_acts = self.activation_store.get_batch_activations()
                self.n_training_tokens += sae_acts.size(0)

                sae_out, feature_acts, loss_dict = self._train_step(sae_in=sae_acts)

                if (
                    self.cfg.log_to_wandb
                    and (self.n_training_steps + 1) % self.cfg.wandb_log_frequency == 0
                ):
                    self._log_train_step(
                        feature_acts=feature_acts,
                        loss_dict=loss_dict,
                        sae_out=sae_out,
                        sae_in=sae_acts,
                    )

                self._checkpoint_if_needed()
                self.n_training_steps += 1
                self._update_pbar(loss_dict, pbar, sae_out.size(0))
        finally:
            print("Saving final checkpoint")
            self.save_checkpoint(is_final=True)
            self.run_evals()

        pbar.close()
        return self.sae

    @torch.no_grad()
    def run_evals(self):
        self.sae.eval()

        def _create_hook(hook_fn):
            return Hook(
                self.sae.cfg.block_layer,
                self.sae.cfg.module_name,
                hook_fn,
                return_module_output=False,
            )

        def _zero_ablation_hook(activations):
            activations[:, 0, :] = torch.zeros_like(activations[:, 0, :]).to(
                activations.device
            )
            return (activations,)

        def _sae_reconstruction_hook(activations):
            activations[:, 0, :] = self.sae(activations[:, 0, :])[0]
            return (activations,)

        # Get model inputs and compute baseline loss
        # model_inputs = self.activation_store.get_batch_of_images_and_labels()
        model_inputs = self.activation_store.get_batch_model_inputs(process_labels=True)
        original_loss = self.model(return_type="loss", **model_inputs).item()

        # Compute loss with SAE reconstruction
        sae_hooks = [_create_hook(_sae_reconstruction_hook)]
        reconstruction_loss = self.model.run_with_hooks(
            sae_hooks, return_type="loss", **model_inputs
        ).item()

        # Compute loss with zeroed activations
        zero_hooks = [_create_hook(_zero_ablation_hook)]
        zero_ablation_loss = self.model.run_with_hooks(
            zero_hooks, return_type="loss", **model_inputs
        ).item()

        # Calculate reconstruction score
        reconstruction_score = (reconstruction_loss - original_loss) / (
            zero_ablation_loss - original_loss
        )

        # Log metrics if configured
        if self.cfg.log_to_wandb:
            wandb.log(
                {
                    "metrics/contrastive_loss_score": reconstruction_score,
                    "metrics/original_contrastive_loss": original_loss,
                    "metrics/contrastive_loss_with_sae": reconstruction_loss,
                    "metrics/contrastive_loss_with_ablation": zero_ablation_loss,
                },
                step=self.n_training_steps,
            )

        del model_inputs
        torch.cuda.empty_cache()

        self.sae.train()
