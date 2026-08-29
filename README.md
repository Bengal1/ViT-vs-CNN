# Vision Transformer vs. CNN: Comparative Study

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)


<img align ="center" width="100%" alt="cnn_vs_vit_wide2" src="https://github.com/user-attachments/assets/513a5e7d-4be8-4be8-995e-caf68a53e9e9" />


A comparative study of Vision Transformers (ViT) and Convolutional Neural Networks (CNNs) for image classification. This project explores how model performance evolves across datasets with different levels of visual complexity, class diversity, and scale, highlighting the transition from CNN dominance on simple tasks to ViT advantages on more challenging visual data.

This repository also showcases the [*Vision Transformer (ViT)*](https://en.wikipedia.org/wiki/Vision_transformer), providing insight into its core architectural ideas, training dynamics, and its ability to capture global relationships through self-attention. It offers a practical view of the strengths and limitations of ViTs, and how they differ from traditional convolutional approaches.


## 🚀 Quick Start

Run a training experiment in one command:
```bash
python main.py --model vit --dataset tiny_imagenet
```

## 🧪 Experimental Setup

All experiments use the same training pipeline and evaluation procedure for the
CNN and ViT within each dataset. The official test sets are kept separate from
training and validation and are used only for final evaluation.

| Dataset | Input Size | Train / Val Split | Batch Size | Optimizer | Learning Rate | Max Epochs | Early Stopping |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| MNIST | 28×28 | 85% / 15% | 64 | AdamW | 2e-4 | 200 | Patience 15 |
| CIFAR-10 | 32×32 | 85% / 15% | 64 | AdamW | 2e-4 | 200 | Patience 15 |
| Food-101 | 128×128 | 85% / 15% | 64 | AdamW | 2e-4 | 200* | Patience 15 |
| Tiny ImageNet | 64×64 | 85% / 15% | 64 | AdamW | 2e-4 | 200 | Patience 15 |

\* Some Food-101 ViT experiments were continued beyond the initial training
window from saved checkpoints.

### Training Configuration

- **Optimizer:** AdamW
- **Weight decay:** `8e-2`
- **Label smoothing:** `0.1`
- **Gradient clipping:** maximum norm of `1.0`
- **Learning-rate schedule:** 10-epoch linear warmup followed by cosine annealing
- **Minimum learning rate:** `1e-6`
- **Random seed:** `1755900008`
- **Checkpoint selection:** best validation-performing checkpoint

### Data Processing

MNIST and CIFAR-10 use dataset-specific normalization. Food-101 and Tiny
ImageNet use stronger training augmentation, including random cropping,
horizontal flipping, color jitter, RandAugment, and random erasing.

Validation and test preprocessing is deterministic, and the same reproducible
train/validation split is used for both architectures within each dataset.

> **Note:** CNN and ViT configurations are not parameter-matched. The study
> compares the implemented versions of both architectures rather than a
> strictly parameter-equivalent benchmark.


## 📊 Results

<table>
  <tr>
    <th>Dataset</th>
    <th>Model</th>
    <th>Parameters</th>
    <th>Test Accuracy (%)</th>
    <th>Accuracy Gap</th>
    <th>Best Model 🏆</th>
  </tr>

  <tr>
    <td rowspan="2" ><b>MNIST</b></td>
    <td><code>CNN</code></td>
    <td>843,850</td>
    <td><b>99.44</b></td>
    <td rowspan="2" align="center"><b>+0.36 pp</b></td>
    <td rowspan="2" align="center"><b>CNN</b></td>
  </tr>
  <tr>
    <td><code>ViT</code></td>
    <td>14,226,442</td>
    <td>99.08</td>
  </tr>

  <tr>
    <td rowspan="2"><b>CIFAR-10</b></td>
    <td><code>CNN</code></td>
    <td>1,204,874</td>
    <td><b>76.10</b></td>
    <td rowspan="2" align="center"><b>+10.05 pp</b></td>
    <td rowspan="2" align="center"><b>CNN</b></td>
  </tr>
  <tr>
    <td><code>ViT</code></td>
    <td>14,281,354</td>
    <td>66.05</td>
  </tr>

  <tr>
    <td rowspan="2"><b>Food-101</b></td>
    <td><code>CNN</code></td>
    <td>29,563,109</td>
    <td>27.96</td>
    <td rowspan="2" align="center"><b>+39.38 pp</b></td>
    <td rowspan="2" align="center"><b>ViT</b></td>
  </tr>
  <tr>
    <td><code>ViT</code></td>
    <td>14,408,549</td>
    <td><b>67.34</b></td>
  </tr>

  <tr>
    <td rowspan="2"><b>Tiny ImageNet</b></td>
    <td><code>CNN</code></td>
    <td>6,545,224</td>
    <td>27.45</td>
    <td rowspan="2" align="center"><b>+15.33 pp</b></td>
    <td rowspan="2" align="center"><b>ViT</b></td>
  </tr>
  <tr>
    <td><code>ViT</code></td>
    <td>14,372,936</td>
    <td><b>42.78</b></td>
  </tr>
</table>

### Results Analysis

The experiments reveal a clear shift in relative performance between the two
architectures across the four datasets. The CNN performs better on MNIST and
CIFAR-10, while the ViT achieves substantially better results on Food-101 and
Tiny ImageNet.

#### CNN advantage on MNIST and CIFAR-10

On **MNIST**, both models achieve very high accuracy, with the CNN reaching
**99.44%** compared with **99.08%** for the ViT. The difference is only
**0.36 percentage points**, indicating that both architectures are highly
effective on this relatively simple classification task.

The CNN, however, reaches this result with approximately **0.84M parameters**,
compared with **14.23M parameters** for the ViT. This makes the CNN considerably
more parameter-efficient on MNIST.

<img align="right" width="40%" alt="mnist_validation_accuracy_cnn_vs_vit" src="https://github.com/user-attachments/assets/8365ea22-e397-4cae-b6ee-4d9c39b83816" />

The training dynamics also reveal an important difference between the two
architectures. The CNN reaches approximately **99% validation accuracy by
epoch 7** and then quickly stabilizes near its final performance. The ViT
converges considerably more slowly and requires substantially more epochs to
approach its best validation accuracy.

This difference is more informative than the final test-accuracy gap alone.
Although both models eventually achieve similar performance, the CNN learns
the MNIST task much faster and with far fewer parameters. This behavior is
consistent with the strong spatial inductive bias of convolutional layers,
which is particularly well suited to a dataset such as MNIST, where the
classification signal is dominated by simple local structures and shapes.

The difference becomes more pronounced on **CIFAR-10**. The CNN achieves
**76.10%** test accuracy, while the ViT reaches **66.05%**, giving the CNN a
**10.05 percentage-point advantage**. The CNN also uses only **1.20M
parameters**, compared with **14.28M** for the ViT.

These results are consistent with the strong inductive biases built into
convolutional architectures. Local connectivity and translation equivariance
make CNNs particularly effective at learning local visual features from
relatively small image-classification datasets.

#### ViT advantage on Food-101 and Tiny ImageNet

The relative performance changes substantially on the more challenging
datasets.

On **Food-101**, the ViT achieves **67.34%** accuracy compared with only
**27.96%** for the CNN, an improvement of **39.38 percentage points**.

This result is particularly notable because the ViT uses approximately
**14.41M parameters**, while the CNN contains approximately **29.56M
parameters**. The large performance difference therefore cannot be explained
simply by a larger ViT model.

<img align="right" width="40%" alt="food101_validation_accuracy_cnn_vs_vit" src="https://github.com/user-attachments/assets/268e84eb-98f9-4fbf-97fa-f79dca40894d" />

The training curves reveal an additional difference beyond the final accuracy.
While the CNN improves initially, its validation performance gradually
plateaus in the mid-20% range, reaching a best validation accuracy of
**25.91%**. In contrast, the ViT continues to improve over a much longer
training period, eventually reaching **62.66%** validation accuracy.

This suggests that, under the configurations used in this project, additional
optimization continues to benefit the ViT on Food-101, while the CNN reaches
its effective performance ceiling much earlier.

This behavior is the opposite of what is observed on MNIST, where the CNN
converges to near-maximum accuracy within only a few epochs while the ViT
requires substantially longer training.

On **Tiny ImageNet**, the same overall trend appears. The ViT reaches
**42.78%** accuracy compared with **27.45%** for the CNN, giving the ViT a
**15.33 percentage-point advantage**. In this experiment the ViT is the larger
model, with approximately **14.37M parameters** compared with **6.55M** for the
CNN.

Food-101 and Tiny ImageNet contain considerably more classes and greater
visual variability than MNIST and CIFAR-10. The results suggest that the
ViT becomes increasingly competitive when classification requires learning
richer relationships across different regions of an image.


#### Loss and accuracy

The test-loss measurements are consistent with the accuracy results. The
winning model on every dataset also achieves the lower test loss:

- **MNIST:** CNN — 0.519 vs. ViT — 0.532
- **CIFAR-10:** CNN — 1.087 vs. ViT — 1.574
- **Food-101:** ViT — 2.001 vs. CNN — 3.326
- **Tiny ImageNet:** ViT — 3.173 vs. CNN — 3.668

This provides additional evidence that the observed differences are not
limited to the top-1 accuracy metric.

#### Overall observation

<img align="right" width="40%"  alt="vit_vs_cnn_test_accuracy" src="https://github.com/user-attachments/assets/471d3cf5-713e-4164-8676-9cfd7b82b9fb" />

Across the experiments, the comparison follows a clear pattern:

**MNIST → CNN**  
**CIFAR-10 → CNN**  
**Food-101 → ViT**  
**Tiny ImageNet → ViT**

More importantly, the experiments reveal contrasting training behavior across
the datasets. On MNIST, the CNN reaches near-ceiling performance within only a
few epochs, while the ViT converges much more gradually. On Food-101, the pattern
reverses: the CNN plateaus relatively early while the ViT continues improving
through substantially longer training.

The strongest CNN advantage appears on **CIFAR-10**, where it is both more
accurate and significantly smaller. The strongest ViT advantage appears on
**Food-101**, where it exceeds the CNN by **39.38 percentage points** despite
using fewer parameters.

These experiments illustrate the different strengths and inductive biases of
the two architectures rather than establishing that one architecture is
universally superior.

> **Note:** The models in this project are not parameter-matched, and their
> parameter counts vary between datasets. The results should therefore be
> interpreted as a comparison of the implemented CNN and ViT configurations
> across different image-classification tasks, rather than as a strictly
> controlled architecture benchmark.


## ⚙️ Training & Execution

Follow the steps below to set up and run the project.

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run training
```bash
python main.py --model <model> --dataset <dataset>
```

### Configuration
Choose a model and dataset:

**Models:**
- `vit` — Vision Transformer
- `cnn` — Convolutional Neural Network

**Datasets:**
- `mnist` — MNIST dataset
- `cifar10` — CIFAR-10 dataset
- `food101` — Food-101 dataset
- `tiny_imagenet` — Tiny ImageNet dataset

### Output
After training:
- Metrics are saved to `results/`
- Training and validation plots are generated automatically
- GPU is used if available

### Notes
- Datasets are downloaded automatically if not present
- Hyperparameters can be modified in `config.py`


## 🧠 Vision Transformer
<img align="right" width="300" alt="ViT Architecture" src="https://github.com/user-attachments/assets/41934e26-ecd0-4aec-89a3-b78fee241ebb" />

The Vision Transformer (ViT) is a deep learning architecture that adapts the Transformer, originally developed for natural language processing, to image recognition tasks. Introduced by Dosovitskiy et al. in “An Image is Worth 16x16 Words” (2020), ViT replaces traditional convolutional feature extractors with a sequence of image patches processed by self-attention. This approach demonstrated that, with sufficient data and compute, Transformers can outperform convolutional neural networks (CNNs) in computer vision benchmarks, paving the way for a broad family of vision transformer models.<br/>

In practice, ViT transforms an image into a sequence of smaller patches, which are then processed using the same self-attention mechanism that made Transformers successful in language tasks. By modeling relationships between patches directly, ViT captures both local details and long-range dependencies within an image, offering a flexible alternative to the strictly hierarchical representations of CNNs. Positional information is incorporated to maintain awareness of spatial structure, and a dedicated representation is used for classification. This design shifts the focus from handcrafted inductive biases toward a more data-driven approach, where the model learns to interpret visual structure primarily from large-scale training data.

### Patch Embedding
<img align="right" width="400" alt="patch_embedding_data" src="https://github.com/user-attachments/assets/49aa0282-b19e-4fab-a494-de5e708b8478" />

A key step in the Vision Transformer (ViT) is the patch embedding stage, which transforms an image into a sequence suitable for a Transformer. Instead of processing pixels individually or relying on convolutional filters, the input image is divided into fixed-size patches (for example, 16×16 pixels). Each patch is then flattened into a vector and projected through a linear layer to a chosen embedding space. The result is a sequence of patch embeddings that can be treated similarly to word tokens in natural language processing, allowing the Transformer to apply self-attention mechanisms across the entire image.<br/>

### CLS Token
The [CLS] token is a learnable embedding prepended to the sequence of patch embeddings in a Vision Transformer. Its primary purpose is to serve as a global representation of the entire image. During the forward pass, the Transformer encoder processes the sequence of patch embeddings along with the [CLS] token, allowing the self-attention mechanism to integrate information from all patches into this special token. After the final encoder block, the [CLS] token contains a summary of the image’s content and is typically fed into the classification head to produce the output logits. By using the [CLS] token in this way, ViT can perform classification based on a single learned representation rather than aggregating information from all patch embeddings.

### Positional Encoding
Since Transformers process input sequences without any inherent notion of order, it is necessary to provide information about the position of each patch in the image. In the Vision Transformer, this is achieved through positional encoding, which adds a vector to each patch embedding to indicate its location within the image. Unlike the fixed sinusoidal encodings used in the original Transformer for NLP, ViT often uses learnable positional embeddings, which are initialized randomly and updated during training. These learnable embeddings allow the model to adaptively encode spatial relationships between patches, helping the self-attention mechanism capture both local and global structure in the image.

### Transformer Encoder
<img align="right" width="250" alt="Encoder" src="https://github.com/user-attachments/assets/ce78de70-696e-4968-bf7d-345d23c2bbc1" />

The Vision Transformer (ViT) is built on the Transformer encoder architecture, which processes images as a sequence of patch embeddings. Each encoder block combines multi-head self-attention and a feed-forward network, with normalization and residual connections to stabilize training.

The key component is self-attention, which allows each image patch to interact with every other patch. Unlike convolutional layers that focus on local neighborhoods, self-attention captures global relationships across the entire image. This enables the model to learn long-range dependencies and complex visual patterns more effectively.

By stacking multiple encoder blocks, ViT builds increasingly rich representations of the input, integrating both local details and global context for tasks such as image classification.

For a deeper explanation of the Transformer architecture, see:
[Transformer-From-Scratch-NLP](https://github.com/Bengal1/Transformer-From-Scratch-NLP)


## ViT vs CNN
Convolutional Neural Networks (CNNs), introduced with LeNet-5 and popularized by AlexNet, have long been the dominant approach in computer vision. They use convolutional filters over local receptive fields, along with pooling and fully connected layers, to build hierarchical representations. This design encodes strong inductive biases such as locality and translation equivariance, making CNNs highly effective and data-efficient for many visual tasks.  
To learn more about CNNs, see [CNN-Architecture-Guide](https://github.com/Bengal1/CNN-Architecture-Guide).

The Vision Transformer (ViT), introduced by Dosovitskiy et al. (2020), replaces convolutions with a Transformer encoder. It represents an image as a sequence of fixed-size patches, which are embedded and processed using multi-head self-attention. This allows the model to capture global relationships between all patches directly, with a [CLS] token used to aggregate information for classification.

<img align="right" width="400" alt="CNN vs ViT - Receptive Field" src="https://github.com/user-attachments/assets/a81742f9-b714-4a54-b3d9-dc0130135be3" />
In terms of architecture, CNNs build hierarchical representations through stacked convolutions, gradually expanding their receptive fields and excelling at capturing local patterns such as edges and textures. In contrast, Vision Transformers operate in patch-embedding space, where self-attention provides a global receptive field from the first layer.

This difference leads to distinct trade-offs: CNNs benefit from strong inductive biases, making them data-efficient and effective on smaller datasets, while ViTs rely more on large-scale data to learn spatial relationships. As a result, CNNs tend to perform well in data-limited settings, whereas ViTs scale more effectively with increased data and model size, often achieving superior performance on more complex tasks.

## 🗂️ Data

The experiments are conducted on four standard image classification datasets of increasing complexity: MNIST, CIFAR-10, Food-101, and Tiny ImageNet. Together they span a progression from simple, low-resolution, single-object images to large-scale, high-variability, real-world photos, allowing model performance to be evaluated as visual and semantic complexity increases.

### MNIST
<img align="right" width="300" alt="dataset_samples" src="https://github.com/user-attachments/assets/e591cfab-09f5-452b-a9f0-8671baa55670" />

MNIST is a grayscale dataset of handwritten digits (0–9) at 28×28 resolution, with 60,000 training and 10,000 test samples. Images are centered, single-channel, and contain simple, consistent strokes with minimal background or variation between samples of the same digit. Of the four datasets, it requires the least ability to relate distant parts of an image to each other: most digits can be classified from small, localized patterns alone, without connecting information from opposite corners of the image.

### CIFAR-10

CIFAR-10 consists of 32×32 RGB images across 10 object classes (animals, vehicles, etc.), with 50,000 training and 10,000 test samples. It adds color, cluttered backgrounds, and greater variation within each class, while still using very low resolution. This makes it a useful midpoint: harder than MNIST, since classification can no longer rely on a single consistent local pattern, but still small enough in scale that CNNs, which are built to prioritize nearby pixels over distant ones, perform well.

### Food-101
<img align="right" width="300" alt="dataset_dif" src="https://github.com/user-attachments/assets/968c0a03-c022-4570-a29d-364a477139e9" />
Food-101 contains 101,000 real-world photos across 101 food categories, split into 750 training and 250 test images per class. Its training images are intentionally left uncleaned, some mislabeled or visually noisy, and resolution varies up to 512×512. Many dishes look similar to one another, while photos of the same dish can look very different depending on angle, plating, and lighting. This combination pushes classification to depend increasingly on relating different regions of an image to each other, rather than on any single local pattern.

### Tiny ImageNet


Tiny ImageNet is a scaled-down version of ImageNet with 64×64 RGB images across 200 classes, comprising 100,000 training and 10,000 test samples. It has the highest number of classes of the four datasets, but, unlike Food-101, only a modest number of training images per class relative to that count. This combination of high complexity and limited data makes it one of the most demanding dataset in the study, testing not just which architecture handles complexity better, but which one still performs well when there is little data to learn from.

Across all four datasets, difficulty is driven by the same four factors in different proportions: image resolution, number of classes, variation within each class, and training data available per class. MNIST and CIFAR-10 keep all four modest, favoring models built to prioritize nearby pixels. Food-101 raises visual and semantic complexity while keeping data per class high, favoring models able to relate distant regions of an image to one another. Tiny ImageNet raises complexity further without a matching increase in data, making it the clearest test of how each architecture holds up when that data runs short.</br>


## 📚 Reference

[An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

[Attention Is All You Need](https://arxiv.org/abs/1706.03762)

[Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)


## 📄 License

This project is licensed under the MIT License.

See the [LICENSE](LICENSE) file for more details.
