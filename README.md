# grafog \[WIP\]
Graph Data Augmentation Library for PyTorch Geometric

## What is it?
Data augmentations are heavily used in Computer Vision and Natural Language Processing to improve variety of training instances. They have also proven to yield good results in both supervised and self-supervised settings. 

`grafog` (portmanteau of "graph" and "augmentation") provides a set of methods to perform data augmentation on graph-structured data, especially meant for self-supervised node classification. It is built on top of `torch_geometric` and is easily integrable with its homogenous [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) API.

This library is heavily inspired by `GAug` from the paper "Data Augmentation for Graph Neural Networks" [[abs](https://arxiv.org/abs/2006.06830), [pdf](https://arxiv.org/pdf/2006.06830.pdf)].

## Installation
You can install the library via `pip`:

```
$ pip install grafog
```

You can also install the library from source:

```
$ git clone https://github.com/rish-16/grafog
$ cd grafog
$ pip install -e .
```

## Usage
The library comes with the following data augmentations:

1. Random Node Removal
2. Random Edge Removal
3. Noisy Edge Removal
4. Ideal Edge Addition

> There are many more features to be added over time, so stay tuned!

```python
from torch_geometric.datasets import CoraFull
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import grafog.transforms as T

transforms = T.Compose([
    T.RandomNodeDrop(),
    T.RandomEdgeDrop(),
    T.NoisyEdgeRemoval(),
    T.IdealEdgeAddition(),
])

data = CoraFull()
data = T(data) # apply the augmentation(s)
train_loader = DataLoader(data, ...)
```

## Contributions
If you spot any issues, feel free to raise a PR or Issue. All meaningful contributions welcome!

## Remarks
This library was built as a project for a class ([UIT2201]()) at NUS. I planned and built it over the span of 12 weeks. I thank Prof. Mikhail Filippov for his guidance, feedback, and support!

## License
[MIT](https://github.com/rish-16/grafog/blob/main/LICENSE)