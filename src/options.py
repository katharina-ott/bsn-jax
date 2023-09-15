from typing import List, Type

from data import IntegrationDataset


class Options:
    step_size: float
    num_epochs: int
    layer_sizes: List[List[int]]
    n: int
    data_class: Type[IntegrationDataset]

    def __init__(self, step_size, num_epochs, layers_sizes, n, data_class):
        self.step_size = step_size
        self.num_epochs = num_epochs
        self.layer_sizes = layers_sizes
        self.n = n
        self.data_class = data_class
