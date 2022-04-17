# grafog
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
import grafog.transforms as T

node_aug = T.Compose([
    T.NodeDrop(p=0.45),
    T.NodeMixUp(lamb=0.5, classes=7),
    ...
])

edge_aug = T.Compose([
    T.EdgeDrop(0=0.15),
    T.EdgeFeatureMasking()
])

data = CoraFull()
model = ...

for epoch in range(10): # begin training loop
    new_data = node_aug(data) # apply the node augmentation(s)
    new_data = edge_aug(new_data) # apply the edge augmentation(s)
    
    x, y = new_data.x, new_data.y
    ...
```

## Contributions
If you spot any issues, feel free to raise a PR or Issue. All meaningful contributions welcome!

## Remarks
This library was built as a project for a class ([UIT2201](https://nusmods.com/modules/UIT2201/computer-science-the-i-t-revolution)) at NUS. I planned and built it over the span of 10 weeks. I thank Prof. Mikhail Filippov for his guidance, feedback, and support!

## License
[MIT](https://github.com/rish-16/grafog/blob/main/LICENSE)
