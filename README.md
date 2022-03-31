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

1. Random Node Drop
2. Random Edge Drop
3. Normalize Features
3. MixUp Strategy

> There are many more features to be added over time, so stay tuned!

```python
from torch_geometric.datasets import CoraFull
from torch_geometric.loader import DataLoader
import grafog.transforms as T

# compose graph augmentations
transforms = T.Compose([
  T.DropNode(p=0.2),
  T.DropEdge(p=0.25),
  T.MixUp(),
  ...
])

data = CoraFull()
new_data = transforms(data) # apply the augmentation(s)
```

## Contributions
If you spot any issues, feel free to raise a PR or Issue. All meaningful contributions welcome!

## Remarks
This library was built as a project for a class ([UIT2201]()) at NUS. I planned and built it over the span of 12 weeks. I thank Prof. Mikhail Filippov for his guidance, feedback, and support!

## License
[MIT](https://github.com/rish-16/grafog/blob/main/LICENSE)
