<img src="banner.png">

# grafog
Graph Data Augmentation Library for PyTorch Geometric.

---

## What is it?
Data augmentations are heavily used in Computer Vision and Natural Language Processing to address data imbalance, data scarcity, and prevent models from overfitting. They have also proven to yield good results in both supervised and self-supervised (contrastive) settings. 

`grafog` (portmanteau of "graph" and "augmentation") provides a set of methods to perform data augmentation on graph-structured data, especially meant for self-supervised node classification. It is built on top of `torch_geometric` and is easily integrable with its [`Data`](https://pytorch-geometric.readthedocs.io/en/latest/modules/data.html#torch_geometric.data.Data) API.

---

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

#### Dependencies
```
torch==1.10.2
torch_geometric==2.0.3
```
---

## Usage
The library comes with the following data augmentations:

1. Random Node Drop
2. Random Edge Drop
3. Normalize Features
4. MixUp Strategy
5. Node Feature Masking
6. Edge Feature Masking

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

---

## Remarks
This library was built as a project for a class ([UIT2201](https://nusmods.com/modules/UIT2201/computer-science-the-i-t-revolution)) at NUS. I planned and built it over the span of 10 weeks. I thank _Prof. Mikhail Filippov_ for his guidance, feedback, and support!

If you spot any issues, feel free to raise a PR or Issue. All meaningful contributions welcome!

---

## License
[MIT](https://github.com/rish-16/grafog/blob/main/LICENSE)
