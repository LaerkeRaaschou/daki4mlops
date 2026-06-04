---
datasets:
- tiny-imagenet-200
license: mit
tags:
- image-classification
- resnet
- tiny-imagenet
---

# Model Card for ResNet-18

<!-- Provide a quick summary of what the model is/does. -->

The goal of this model is classifying images. It is trained on tiny_imagenet, which has 200 classes with 100000 images in the training set and 10000 images in the test set.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->

This model is a ResNet-18 convolutional neural network trained from scratch for image classification on the tiny_imagenet dataset. It takes a 64x64 RGB image as input and predicts one of 200 object classes. The model was developed as part of an MLOps course project at Aalborg University, with the primary goal of demonstrating an end-to-end machine-learning pipeline (training, evaluation, versioning, CI/CD, compression, and monitoring) rather than achieving state-of-the-art accuracy.

- **Developed by:** Anne, Lærke og Stoyan
- **Model type:** Image classifier
- **Language(s) (NLP):** Python
- **Trained from architecture:** ResNet-18 (trained from scratch)

### Model Sources

<!-- Provide the basic links for the model. -->
- **Repository:** https://github.com/LaerkeRaaschou/daki4mlops/README.md

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

The model can be used directly for image classification: given a 64x64 RGB image, it predicts one of the 200 tiny_imagenet classes. It is intended for educational and demonstration purposes within this MLOps project.

### Downstream Use

<!-- This section is for the model use when fine-tuned for a task, or when plugged into a larger ecosystem/app -->

The trained network can serve as a basis for compression experiments (quantization and pruning).

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

Not intended for production use or for images outside the Tiny ImageNet distribution.

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

Trained only on the Tiny ImageNet classes at 64x64 resolution; performance is not guaranteed outside this distribution.

### Recommendations

<!-- This section is meant to convey recommendations with respect to the bias, risk, and technical limitations. -->

Users (both direct and downstream) should be made aware of the risks, biases and limitations of the model. More information needed for further recommendations.

## How to Get Started with the Model

Download the code in the repository and follow the README file.
https://github.com/LaerkeRaaschou/daki4mlops/README.md

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

Trained on the tiny_imagenet dataset, using 90000 training images and 10000 images for validation.Download data here: https://www.kaggle.com/competitions/tiny-imagenet/data.

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

The model have been trained on data using following data preprocessing and augmentation:transforms.RandomResizedCrop(64, scale=(0.7, 1.0)), transforms.RandomHorizontalFlip(), transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), transforms.ToTensor(), transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))


#### Training Hyperparameters

- **Training regime:** fp16 mixed precision <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

[More Information Needed]

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

The model is testet on the validation set from tiny_imagenet as the original testset does not include labels.This test set contains 10000 images.

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

Evaluation is done per class.Results are reported overall and per individual class across all 200 Tiny ImageNet classes, including identification of the best- and worst-performing classes.

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

Accuracy as the primary metric. Prediction confidence is also reported overall and per class.

### Results

[More Information Needed]

#### Summary



## Environmental Impact

<!-- Total emissions (in grams of CO2eq) and additional considerations, such as electricity usage, go here. Edit the suggested text below accordingly -->

Carbon emissions can be estimated using the [Machine Learning Impact calculator](https://mlco2.github.io/impact#compute) presented in [Lacoste et al. (2019)](https://arxiv.org/abs/1910.09700).

- **Hardware Type:** NVIDIA L4 GPU(s), single- or multi-GPU via PyTorch DDP / DeepSpeed.
- **Hours used:** [More Information Needed]
- **Cloud Provider:** AI-Lab
- **Carbon Emitted:** [More Information Needed]

## Technical Specifications

### Model Architecture and Objective

The model is a ResNet-18 convolutional neural network, consisting of an initial 3x3 convolution and max-pooling stem followed by four stages of residual blocks (2 blocks each, with channel widths 64, 128, 256, 512), global average pooling, and a fully connected classification head. The architecture is adapted for Tiny ImageNet with 200 output classes and 64x64 RGB input. Residual (skip) connections allow gradients to flow through the network, enabling stable training of the deeper stack. The training objective is multi-class image classification, optimized with torch.nn.CrossEntropyLoss.

### Compute Infrastructure

AI-Lab

#### Hardware

NVIDIA L4 GPU(s), single- or multi-GPU via PyTorch DDP / DeepSpeed.

#### Software

PyTorch, Hydra, OmegaConf, scikit-learn, Weights & Biases, MLflow.

## Citation

<!-- If there is a paper or blog post introducing the model, the APA and Bibtex information for that should go in this section. -->

**BibTeX:**

@inproceedings{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016},
  doi       = {https://doi.org/10.1109/CVPR.2016.90}
}

**APA:**

He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 770-778.
