# Deep Learning Image Classifier: CNN-from-Scratch vs. Transfer Learning

A direct, measured comparison between training a CNN from scratch and
fine-tuning a pretrained ResNet18 on CIFAR-10 — including a real,
counter-intuitive finding about what "transfer learning is faster"
actually means in practice.

## Results

| Model | Epochs | Final Accuracy | Total Training Time |
|---|---|---|---|
| SimpleCNN (from scratch) | 10 | **75.0%** | ~116s |
| ResNet18 (fine-tuned) | 5 | **92.0%** | ~413s |

ResNet18 reached **88.7% accuracy after just its first epoch** — already
beating the from-scratch CNN's entire 10-epoch result.

## The nuance most people get wrong

It's tempting to say "ResNet18 needed fewer epochs, so transfer learning is
faster." **That's not quite right.** Checking actual wall-clock time:

- SimpleCNN: ~11.4–12.6s per epoch → ~116s total
- ResNet18: ~82.3–82.8s per epoch → ~413s total

**ResNet18 actually took ~3.6x longer in raw training time**, despite
needing fewer epochs — because it's a much deeper network, and each epoch
is far more computationally expensive.

**The real, honest takeaway:** transfer learning reduced the amount of
*data and epochs* needed to reach high accuracy — not the total compute.
That's a genuinely different (and more accurate) claim than "transfer
learning is faster," and it's the kind of distinction worth catching before
you present a result, not after.

## Why the two models differ

| | SimpleCNN | ResNet18 |
|---|---|---|
| Starting point | Random weights | Pretrained on ImageNet (1.4M images, 1000 classes) |
| What it has to learn | Everything — edges, textures, shapes, objects | Only how to adapt existing features to CIFAR-10's 10 classes |
| Layers trained | All | Only `layer4` + final classifier (early layers frozen) |

Freezing the early layers preserves ResNet18's already-learned general
visual features (edges, textures, basic shapes) — retraining everything
risks losing that knowledge, especially on a dataset as small as CIFAR-10
compared to ImageNet.

## Fair comparison, by design

Both models were trained with **identical data augmentation** (random crop
+ horizontal flip) and the same overall training setup — so the accuracy
gap can be attributed to architecture and pretraining, not an inconsistent
experimental setup.

## Running it

Built and tested on [Kaggle](https://kaggle.com):

1. Create a new Kaggle notebook.
2. **Settings → Accelerator → GPU T4 x2**
3. **Settings → Internet → ON** (needed for CIFAR-10 and pretrained weights)
4. Upload `train.py`'s cells into notebook cells and run top to bottom.

To run locally instead:
```bash
pip install -r requirements.txt
python train.py
```
(Training will be significantly slower without a GPU — expect well over an
hour instead of ~9 minutes combined.)

## Dataset

[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): 60,000 32×32 color
images across 10 classes (plane, car, bird, cat, deer, dog, frog, horse,
ship, truck) — 50,000 for training, 10,000 held out for testing.

## License

MIT — see [LICENSE](LICENSE).
