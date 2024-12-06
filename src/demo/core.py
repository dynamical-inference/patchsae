import os
from copy import deepcopy
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image


class UtilMixin:
    def _get_max_activating_images_and_labels(
        self, neuron_idx, dataset, max_activating_image_indices
    ):
        img_list = max_activating_image_indices[neuron_idx]
        images = []
        labels = []
        for i in img_list:
            try:
                images.append(dataset[i.item()]["image"])
                labels.append(dataset[i.item()]["label"])
            except Exception:
                images.append(dataset[i.item()]["jpg"])
                labels.append(dataset[i.item()]["cls"])
        return images, labels

    def _create_patches(self, patch=256):
        temp = self.processed_image["pixel_values"].clone()
        patches = temp[0].data.unfold(0, 3, 3)
        patches = patches.unfold(1, patch, patch)
        patches = patches.unfold(2, patch, patch)
        return patches


class VisualizeMixin:
    def _plot_input_image(self):
        plt.imshow(self.input_image)

    def _plot_feature_mask(self, patches, feat_idx, mask=None, plot=True):
        if mask is None:
            mask = self.sae_act[0, :, feat_idx].cpu()

        fig, axs = plt.subplots(patches.size(1), patches.size(2), figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)

        for i in range(patches.size(1)):
            for j in range(patches.size(2)):
                patch = patches[0, i, j].permute(1, 2, 0)
                patch *= torch.tensor(self.vit.processor.image_processor.image_std)
                patch += torch.tensor(self.vit.processor.image_processor.image_mean)
                masked_patch = patch * mask[i * patches.size(2) + j + 1]
                masked_patch = (masked_patch - masked_patch.min()) / (
                    masked_patch.max() - masked_patch.min() + 1e-8
                )
                axs[i, j].imshow(masked_patch)
                axs[i, j].axis("off")

        fig.suptitle(feat_idx)
        plt.close()
        return fig

    def _plot_patches(self, patches, highlight_patch_idx=None):
        fig, axs = plt.subplots(patches.size(1), patches.size(2), figsize=(6, 6))
        plt.subplots_adjust(wspace=0.01, hspace=0.01)
        for i in range(patches.size(1)):
            for j in range(patches.size(2)):
                patch = patches[0, i, j].permute(1, 2, 0)
                patch *= torch.tensor(self.vit.processor.image_processor.image_std)
                patch += torch.tensor(self.vit.processor.image_processor.image_mean)
                axs[i, j].imshow(patch)
                if i * patches.size(2) + j == highlight_patch_idx:
                    for spine in axs[i, j].spines.values():
                        spine.set_edgecolor("red")
                        spine.set_linewidth(3)
                    axs[i, j].set_xticks([])
                    axs[i, j].set_yticks([])
                else:
                    axs[i, j].axis("off")
        plt.show()

    def _plot_union_top_neruons(
        self, top_k, union_top_neurons, token_idx, token_act, save=False
    ):
        print(f"Union of top {top_k} neurons: {union_top_neurons}")

        plt.figure(figsize=(10, 5))
        plt.plot(token_act)
        plt.plot(
            union_top_neurons,
            token_act[union_top_neurons],
            "ro",
            label="Top neurons",
            markersize=5,
        )

        # Annotate feature indices
        for idx in union_top_neurons:
            plt.text(
                idx, token_act[idx] + 0.05, str(idx), fontsize=9, ha="center"
            )  # Adjust the 0.05 value as needed for spacing

        plt.legend()
        plt.title(f"token {token_idx} activation")

        if save:
            img_name = os.path.basename(self.img_url).replace(".jpg", "")
            save_name = f"{self.save_dir}/{img_name}/activation/{token_idx}.jpg"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)

        plt.show()
        plt.close()

    def _plot_images(
        self,
        dataset_name,
        images,
        neuron_idx,
        labels=None,
        suptitle=None,
        top_k=5,
        save=False,
    ):
        images = [img.resize((224, 224)) for img in images]
        num_cols = min(top_k, 5)
        num_rows = (top_k + num_cols - 1) // num_cols
        fig, axes = plt.subplots(
            num_rows, num_cols, figsize=(4.5 * num_cols, 5 * num_rows)
        )
        axes = axes.flatten()  # Flatten the 2D array of axes

        for i in range(top_k):
            axes[i].imshow(images[i])  # Display the image
            axes[i].axis("off")  # Hide axes
            if labels is not None:
                class_name = self.class_names[dataset_name][int(labels[i])]
                axes[i].set_title(f"{labels[i]} {class_name}", fontsize=25)
        # plt.suptitle(suptitle)
        plt.tight_layout()

        if save:
            img_name = os.path.basename(self.img_url).replace(".jpg", "")
            save_name = (
                f"{self.save_dir}/{img_name}/top_images/{dataset_name}/{neuron_idx}.jpg"
            )
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)

        plt.close()

        return fig

    def _fig_to_img(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        img = np.frombuffer(canvas.tostring_rgb(), dtype="uint8")
        img = img.reshape(canvas.get_width_height()[::-1] + (3,))
        return img

    def _plot_multiple_images(self, figs, neuron_idx, top_k=5, save=False):
        # Create a new figure to hold all subplots
        num_plots = len(figs)
        cols = 1  # Number of columns in the subplot grid
        rows = (num_plots + cols - 1) // cols  # Calculate rows required

        combined_fig = plt.figure(figsize=(20, 12))  # Adjust figsize as needed

        for i, fig in enumerate(figs):
            ax = combined_fig.add_subplot(rows, cols, i + 1)
            img = self._fig_to_img(fig)
            ax.imshow(img)
            ax.axis("off")

        if save:
            img_name = os.path.basename(self.img_url).replace(".jpg", "")
            save_name = f"{self.save_dir}/{img_name}/top_images/{neuron_idx}.jpg"
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name)

        combined_fig.show()
        # plt.close(combined_fig)


class SAETester(VisualizeMixin, UtilMixin):
    def __init__(
        self,
        vit,
        cfg,
        sae,
        mean_acts,
        max_act_images,
        datasets,
        class_names,
        noisy_threshold=0.1,
        device="cpu",
    ):
        self.vit = vit
        self.cfg = cfg
        self.sae = sae
        self.mean_acts = mean_acts
        self.max_act_images = max_act_images
        self.datasets = datasets
        self.class_names = class_names
        self.noisy_threshold = noisy_threshold
        self.device = device

    def register_image(self, img_url: str) -> None:
        """Load and process an image from a URL or local path."""
        if isinstance(img_url, str):
            image = self._load_image(img_url)
        else:
            image = img_url
        self.input_image = image
        self.processed_image = self.vit.processor(
            images=image, text="", return_tensors="pt", padding=True
        )

    def _load_image(self, img_url: str) -> Image.Image:
        """Helper method to load image from URL or local path."""
        if "http" in img_url:
            response = requests.get(img_url)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        return Image.open(img_url)

    @property
    def processed_image(self):
        return self._processed_image

    @processed_image.setter
    def processed_image(self, value):
        self._processed_image = value

    @property
    def input_image(self):
        return self._input_image

    @input_image.setter
    def input_image(self, value):
        self._input_image = value

    def show_input_image(self):
        self._plot_input_image()

    def run(
        self, highlight_patch_idx, patch_size=16, top_k=5, num_images=5, seg_mask=True
    ):
        # idx = 0 is cls token
        self.show_patches(
            highlight_patch_idx=highlight_patch_idx - 1, patch_size=patch_size
        )
        top_neurons = self.get_top_neurons(highlight_patch_idx, top_k=top_k)
        self.show_ref_images_of_neuron_indices(
            top_neurons, top_k=num_images, seg_mask=True
        )

    def show_patches(self, highlight_patch_idx=None, patch_size=16):
        if not hasattr(self, "input_image"):
            assert not hasattr(self, "input_image"), "register image first"

        patches = self._create_patches(patch=patch_size)
        self._plot_patches(patches.cpu().data, highlight_patch_idx=highlight_patch_idx)

    def show_segmentation_mask(self, feat_idx, patch_size=16, mask=None, plot=True):
        patches = self._create_patches(patch=patch_size)
        fig = self._plot_feature_mask(
            patches.cpu().data, feat_idx, mask=None, plot=plot
        )
        return fig

    def get_segmentation_mask(self, image, feat_idx: int):
        if image.mode == "L":
            image = image.convert("RGB")

        vit_act = self._run_vit_hook(image)
        sae_act = self._run_sae_hook(vit_act)
        token_act = sae_act[0].detach().cpu().numpy()
        filtered_mean_act = self._filter_out_nosiy_activation(token_act)

        temp = filtered_mean_act[:, feat_idx]
        mask = torch.Tensor(temp[1:,].reshape(14, 14)).view(1, 1, 14, 14)
        mask = torch.nn.functional.interpolate(mask, (image.height, image.width))[0][
            0
        ].numpy()
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-10)

        base_opacity = 30
        image_array = np.array(image)[..., :3]
        rgba_overlay = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        rgba_overlay[..., :3] = image_array[..., :3]

        darkened_image = (image_array[..., :3] * (base_opacity / 255)).astype(np.uint8)
        rgba_overlay[mask == 0, :3] = darkened_image[mask == 0]
        rgba_overlay[..., 3] = 255  # Fully opaque

        return Image.fromarray(rgba_overlay)

    def get_top_neurons(self, token_idx=None, top_k=5, plot=True):
        if token_idx is None:
            token_acts, top_neurons, self.sae_act = self._get_img_acts_and_top_neurons(
                top_k=top_k
            )
        else:
            token_acts, top_neurons, self.sae_act = (
                self._get_token_acts_and_top_neurons(token_idx=token_idx, top_k=top_k)
            )
        if plot:
            self._plot_union_top_neruons(top_k, top_neurons, token_idx, token_acts)
        return top_neurons

    def get_top_images(self, neuron_idx: int, top_k=5, show_seg_mask=False):
        out_top_images = []
        for dataset_name in self.datasets.keys():
            if self.datasets[dataset_name] is None:
                continue
            images, labels = self._get_max_activating_images_and_labels(
                neuron_idx,
                self.datasets[dataset_name],
                self.max_act_images[dataset_name],
            )

            if show_seg_mask:
                images = [
                    self.get_segmentation_mask(img, neuron_idx)
                    for img in images[:top_k]
                ]
            suptitle = f"{dataset_name} - {neuron_idx}"
            fig = self._plot_images(
                dataset_name,
                images,
                neuron_idx,
                labels,
                suptitle=suptitle,
                top_k=top_k,
                save=False,
            )
            out_top_images.append(fig)

        return out_top_images

    def show_ref_images_of_neuron_indices(
        self, neuron_indices: list[int], top_k=5, save=False, seg_mask=False
    ):
        for neuron_idx in neuron_indices:
            figs = self.get_top_images(neuron_idx, top_k=top_k, show_seg_mask=False)
            self._plot_multiple_images(figs, neuron_indices, top_k=top_k, save=False)

            if seg_mask:
                figs = self.get_top_images(neuron_idx, top_k=top_k, show_seg_mask=True)
                self._plot_multiple_images(
                    figs, neuron_indices, top_k=top_k, save=False
                )

    def get_activation_distribution(self):
        vit_act = self._run_vit_hook()
        sae_act = self._run_sae_hook(vit_act)
        token_act = sae_act[0].detach().cpu().numpy()
        filtered_mean_act = self._filter_out_nosiy_activation(token_act)
        self.sae_act = sae_act
        return filtered_mean_act

    def _get_img_acts_and_top_neurons(self, top_k=5, threshold=0.2):
        vit_act = self._run_vit_hook()
        sae_act = self._run_sae_hook(vit_act)

        token_act = sae_act[0].detach().cpu().numpy()
        filtered_mean_act = self._filter_out_nosiy_activation(token_act)
        token_act = (filtered_mean_act > threshold).sum(0)
        filtered_mean_act = filtered_mean_act.sum(0)
        top_neurons = np.argsort(filtered_mean_act)[::-1][:top_k]

        return token_act, top_neurons, sae_act

    def _get_token_acts_and_top_neurons(self, token_idx, top_k=5):
        vit_act = self._run_vit_hook()
        sae_act = self._run_sae_hook(vit_act)

        token_act = sae_act[0, token_idx, :].detach().cpu().numpy()
        filtered_mean_act = self._filter_out_nosiy_activation(token_act)
        top_neurons = np.argsort(filtered_mean_act)[::-1][:top_k]

        return token_act, top_neurons, sae_act

    def _run_vit_hook(self, image=None):
        if image is None:
            inputs = self.processed_image.to(self.device)
        else:
            inputs = self.vit.processor(
                images=image, text="", return_tensors="pt", padding=True
            )
        list_of_hook_locations = [(self.cfg.block_layer, self.cfg.module_name)]
        vit_out, vit_cache_dict = self.vit.run_with_cache(
            list_of_hook_locations, **inputs
        )
        vit_act = vit_cache_dict[(self.cfg.block_layer, self.cfg.module_name)]
        return vit_act

    def _run_sae_hook(self, vit_act):
        sae_out, sae_cache_dict = self.sae.run_with_cache(vit_act)
        sae_act = sae_cache_dict["hook_hidden_post"]
        if sae_act.shape[0] != 1:
            sae_act = sae_act.permute(1, 0, 2)
        return sae_act[:, :197, :]

    def _filter_out_nosiy_activation(self, features):
        noisy_features_indices = (
            (self.mean_acts["imagenet"] > self.noisy_threshold).nonzero()[0].tolist()
        )
        features_copy = deepcopy(features)
        if len(features_copy.shape) == 1:
            features_copy[noisy_features_indices] = 0
        elif len(features_copy.shape) == 2:
            features_copy[:, noisy_features_indices] = 0
        return features_copy
