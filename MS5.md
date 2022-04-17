# `grafog`: Graph Data Augmentations Made Easy

_Rishabh Anand_ <br>
Milestone 5, UIT2201 (online)

## Progress Report
The past 10 weeks of development have been great – I found an interesting underexplored niche and was able to capitalise on it, and even found an interesting group of people who were willing to talk to me about it. As Tony Stark famously said at the beginning of Iron Man 3: 

> “let’s track this from the beginning”

## Problem Statement and Solution
Graphs have special properties (non-euclidean, unstructured, translation/rotation-invariant) not learnable by traditional neural networks. Capturing the structure of graphs is very important, requiring us to invoke so-called Graph Neural Networks [scarcelli, gori]. On another note, augmentations have remained the primary data preprocessing technique to help with data imbalance, data scarcity, and preventing models from overfitting, among other benefits. However, unlike regular augmentations for images or text, graph augmentations are not so straightforward – it makes no sense to flip or invert graphs, or to apply transformations on features, which may lead to corruption.

As a result, I built `grafog`, a graph augmentation library built on top of `torch_geometric` (popular Graph Deep Learning library). The library is finally complete with 6 different graph augmentations listed below. The library allows users to chain together graph augmentations that introduce subgraphs, higher-order structures, and carefully transformed features, into the original graphs, allowing for more robust learning of graph representations. Users even have the ability to string individual augmentations together as done in other augmentation libraries like `albumentations` for Computer Vision or `torchtext` for Natural Language Processing.

### List of Augmentations:
| Augmentation                 | Remarks                                                            | When to use                      |
|------------------------------|--------------------------------------------------------------------|----------------------------------|
| `NodeDrop(p=0.05)`           | Randomly drops nodes from the graph with the given probability     | before, during training          |
| `EdgeDrop(p=0.05)`           | Randomly drops edges from the graph with the given probability     | before, during training          |
| `Normalize()`                | Normalizes the node or edge features                               | before training                  |
| `NodeMixUp(lamb, classes)`   | Performs MixUp on node features with the given lambda `lamb`       | during training                  |
| `NodeFeatureMasking(p=0.15)` | Randomly masks node features                                       | during training                  |
| `EdgeFeatureMasking(p=0.15)` | Randomly masks edge features                                       | during training                  |

> As per usual practice, augmentations are not applied during validation or testing.

### Usage
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

Before this semester, for my own research at NUS Computing on Graph Deep Learning, I was reading _Neil Shah_’s paper (from Snap Research) introducing the `GAug` graph augmentation technique. The paper went on to state graph augmentations should be carefully crafted to ensure maximal effectiveness given the non-euclidean nature of graphs. I searched up for graph augmentation libraries but could only find isolated projects from former research papers – there was no single package incorporating everything. This motivated my reaching out to _Neil Shah_, _Chaitanya Joshi_ (Cambridge), and _Derek Lim_ (MIT) to discuss possible approaches and solutions. They all believed graph augmentation is an area worth looking into as it has major implications on Graph Contrastive Learning, Graph Representation Learning, and Graph Adversarial Attacks research. These conversations spawned the idea for `grafog` and prompted me to build it for UIT22021!

## Outreach
Comically, `grafog` received [some interest](https://github.com/rish-16/grafog/issues/1) from the Graph DL community even before it was built! Now that the library has been built, I have released the product into the wild on Twitter, GitHub, LinkedIn, and Reddit. The GitHub repo itself already has 3 stars and I'm sure it'll climb once it's public. Twitter and LinkedIn also seem to like it, with my post garnering __XYZ__ and __ABC__ likes (respectively) in a day. I've posted on `r/MachineLearning` and `r/learnmachinelearning` and am yet to see any results. More updates in due time!

## Reflections
I was flipping through MS1 and realised I was very fortunate throughout the project, in that there was minimal pivoting or changing of ideas. I was able to zero-in on a niche area early on and build something that addressed a specific problem a specific group of people faced. In my endeavours building people-centric innovations in the past and present, I’ve found inexperienced people make mistakes that fall into four overlapping categories:

- Inventing problems that don’t exist and building a product no one wants
- Building a product first and force-fitting a problem to it
- Finding a problem and force-fitting the wrong solution to it
- Building a product without talking to a single potential user or gaining feedback

I'm happy the lessons from past projects have helped me focus on `grafog` and bring it to reality faster. My prototyping and iterative development skills have immensely improved this semester having taken this module and I'm sure it'll be of good use in future projects.

## Future Work
The summer of 2022 is going to be spent triaging [issues](https://github.com/rish-16/grafog/issues) and merging potential PRs from the community on GitHub. Additionally, I'll be working with Neil Shah much more closely (it might even develop into an internship at Snap Research!) to study graph augmentations using `grafog` and its impact on graph neural network performance and optimisation. I'll also be introducing new augmentations from the literature such as learnable node/edge addition, node shuffling, and random walk subgraph generation.

## Acknowledgements
I’d like to thank _Chaitanya Joshi_, first year PhD student at Cambridge University studing GDL in Science, for the feedback and insights into the project and sharing resources to examine; _Derek Lim_, first year PhD student at MIT studying GNN expressiveness, for our chat on expressive graph neural networks and how certain graph augmentations imbue graphs with “nice” properties that enable better generalisation; _Dr. Neil Shah_, Lead Research Scientist at Snap Research and author of the `GAug` paper I referenced in MS1/MS2, for letting me pick his brain on graph augmenations.

I also wish to express my gratitude to _Prof. Mikhail Filippov_ for the chance to work on this exciting project! I spoke to a lot of interesting people and learned a lot from [UIT2201](https://nusmods.com/modules/UIT2201/computer-science-the-i-t-revolution) and look forward to studying some of these topics with more depth in the near future. It was nice learning stuff from a high-level perspective and having a birds-eye view of the entire technology landscape.
