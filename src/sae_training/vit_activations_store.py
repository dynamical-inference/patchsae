"""
# Portions of this file are based on code from the “jbloomAus/SAELens” and "HugoFry/mats_sae_training_for_ViTs" repositories (MIT-licensed):
    https://github.com/jbloomAus/SAELens/blob/main/sae_lens/training/activations_store.py
    https://github.com/HugoFry/mats_sae_training_for_ViTs/blob/main/sae_training/vit_activations_store.py
"""

from torch.utils.data import DataLoader, TensorDataset

from src.sae_training.hooked_vit import HookedVisionTransformer
from src.sae_training.utils import get_model_activations, process_model_inputs


class ViTActivationsStore:
    """
    Class for streaming tokens and generating and storing activations
    while training SAEs.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        device: str,
        seed: int,
        model: HookedVisionTransformer,
        block_layer: int,
        module_name: str,
        class_token: bool,
    ):
        self.device = device
        self.model = model
        self.batch_size = batch_size
        self.block_layer = block_layer
        self.module_name = module_name
        self.class_token = class_token

        self.dataset = dataset.shuffle(seed=seed)
        self.dataset_iter = iter(self.dataset)

    def get_batch_model_inputs(self, process_labels=False):
        """Get model activations for a batch of data"""
        batch_dict = {"image": [], "label": []}

        def _add_data(current_item, batch_dict):
            for key, value in current_item.items():
                batch_dict[key].append(value)

        for _ in range(self.batch_size):
            try:
                current_item = next(self.dataset_iter)
            except StopIteration:
                self.dataset_iter = iter(self.dataset)
                current_item = next(self.dataset_iter)
            _add_data(current_item, batch_dict)

        inputs = process_model_inputs(
            batch_dict, self.model, self.device, process_labels=process_labels
        )
        return inputs

    def get_batch_activations(self):
        # """Get model activations for a batch of data"""
        inputs = self.get_batch_model_inputs()
        return get_model_activations(
            self.model, inputs, self.block_layer, self.module_name, self.class_token
        )

    def _create_new_dataloader(self) -> DataLoader:
        """Create a new dataloader with fresh activations"""
        activations = self._get_batch_activations()
        dataset = TensorDataset(activations)
        return iter(DataLoader(dataset, batch_size=self.batch_size, shuffle=True))

    def get_next_batch(self):
        """Get next batch, creating new dataloader if current one is exhausted"""
        try:
            return self._get_batch_activations()
        except StopIteration:
            self.dataloader = self._create_new_dataloader()
            return next(self.dataloader)
