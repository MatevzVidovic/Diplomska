

from torch.utils.data import Dataset, DataLoader, Sampler
import random



class ResamplingSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # Generate random indices with replacement
        return iter(random.choices(range(len(self.data_source)), k=self.num_samples))

    def __len__(self):
        return self.num_samples


class BalancedRandomSampler(Sampler):
    def __init__(self, data_source, num_samples):
        self.data_source = data_source
        self.num_samples = num_samples

    def __iter__(self):
        # Generate random indices with replacement
        items = []
        diff = self.num_samples - len(items)
        while diff > 0:
            random_permutation = random.sample(range(len(self.data_source)), len(self.data_source))
            items += random_permutation[:diff] # will take all elements, unless diff is smaller than len(random_permutation)
            diff = self.num_samples - len(items)

        return iter(items)

    def __len__(self):
        return self.num_samples


class LimitedSampler(Sampler):
    def __init__(self, data_source, num_samples, shuffle=False):
        self.data_source = data_source
        self.num_samples = min(num_samples, len(data_source))
        self.shuffle = shuffle

    def __iter__(self):
        
        if self.shuffle:
            items = random.sample(range(len(self.num_samples)), self.num_samples)
        else:
            items = list(range(self.num_samples))

        return iter(items)

    def __len__(self):
        return self.num_samples