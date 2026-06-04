---
{{ card_data }}
model_id: RESNET-18
model_summary: An image classifier trained on Tiny ImageNet-200 (200 classes, 100,000 training images).
model_description: A ResNet-18 image classifier trained from scratch on Tiny ImageNet-200 (200 classes, 64x64 RGB images). Trained as part of an MLOps course project.
developers: Anne, Lærke og Stoyan
model_type: Image classifier
base_arch: RESNET-18
repo: https://github.com/LaerkeRaaschou/daki4mlops/blob/main/model/README.md
bias_risks_limitations: The model is trained only on the 200 Tiny ImageNet classes at 64x64 resolution and should not be expected to perform well outside this distribution.
training_data: For training, the training set from Tiny ImageNet-200 is used (100,000 images across 200 classes, 500 per class, 64x64 RGB).
testing_data: For testing, the validation set from Tiny ImageNet-200 is used (10,000 images, 50 per class).
model_specs: The base architecture is ResNet-18, adapted for 200 output classes and 64x64 input images.
---


# Model Card for RESNET-18 <!-- model_id -->

An image classifier trained on Tiny ImageNet-200 (200 classes, 100,000 training images). <!-- model_summary -->

## Model Details

### Model Description

A ResNet-18 image classifier trained from scratch on Tiny ImageNet-200 (200 classes, 64x64 RGB images). Trained as part of an MLOps course project. <!-- model_description -->

- **Developed by: Anne, Lærke og Stoyan** <!-- developers -->
- **Model type: Image Classifier** <!-- model_type -->
- **Trained from architecture: RESNET-18** <!-- base_arch -->

### Model Sources

- **Repository: https://github.com/LaerkeRaaschou/daki4mlops/blob/main/model/README.md** <!-- repo -->

## Uses

### Direct Use

{{ direct_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

The model is trained only on the 200 Tiny ImageNet classes at 64x64 resolution and should not be expected to perform well outside this distribution. <!-- bias_risks_limitations -->

## Training Details

### Training Data

For training, the training set from Tiny ImageNet-200 is used (100,000 images across 200 classes, 500 per class, 64x64 RGB). <!-- training_data -->

### Training Procedure

#### Preprocessing

{{ preprocessing | default("[More Information Needed]", true)}}

#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!-- fp32, fp16 mixed precision, bf16 mixed precision, etc. -->

{{ training_hyperparameters | default("[More Information Needed]", true)}}

#### Speeds, Sizes, Times

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

For testing, the validation set from Tiny ImageNet-200 is used (10,000 images, 50 per class). <!-- testing_data -->

#### Factors

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination

{{ model_examination | default("[More Information Needed]", true)}}

## Technical Specifications

### Model Architecture and Objective

The base architecture is ResNet-18, adapted for 200 output classes and 64x64 input images. <!-- model_specs -->

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}
