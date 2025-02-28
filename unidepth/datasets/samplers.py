import torch


class DistributedSamplerNoDuplicate(torch.utils.data.DistributedSampler):
    """A distributed sampler that doesn't add duplicates. Arguments are the same as DistributedSampler"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self.drop_last and len(self.dataset) % self.num_replicas != 0:
            # some ranks may have less samples, that's fine
            if self.rank >= len(self.dataset) % self.num_replicas:
                self.num_samples -= 1
            self.total_size = len(self.dataset)
