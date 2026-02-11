---
{{ card_data }}
model_id: RESNET-18
model_summary: The goal of this model is classifing images. It is trained on Tiny-imagenet which has xx classes with xxx images.
model_description: The goal of this model is classifing images. It is trained on Tiny-imagenet which has xx classes with xxx images. With some more information.
developers: Anne, Lærke og Stoyan
model_type: Image classifier
base_arch: RESNET-18
repo: https://github.com/LaerkeRaaschou/daki4mlops/blob/main/model/README.md


bias_risks_limitations: The model has following bias and risk when used xxxx

training_data: For training the training set from the Tiny-imagenet-200 dataset is used. It contains XXXX.
testing_data: For testing the validation set from Tiny-imagenet-200 dataset is used, containing XXXX.

model_specs: The base architecture is the RESNET-18 xxxxx
---


# Model Card for RESNET-18 <!-- model_id -->
Model summary: The goal of this model is classifing images. It is trained on Tiny-imagenet which has xx classes with xxx images. <!-- model_summary -->

## Model Details

### Model Description

Model description: The goal of this model is classifing images. It is trained on Tiny-imagenet which has xx classes with xxx images. With some more information. <!-- model_description -->

- **Developed by: Anne, Lærke og Stoyan** <!-- developers -->
- **Model type: Image Classifier** <!-- model_type -->
- **Trained from architecture: RESNET-18** <!-- base_arch -->

### Model Sources
<!-- Provide the basic links for the model. -->

- **Repository: https://github.com/LaerkeRaaschou/daki4mlops/blob/main/model/README.md** <!-- repo -->

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->

### Direct Use

<!-- This section is for the model use without fine-tuning or plugging into a larger ecosystem/app. -->

{{ direct_use | default("[More Information Needed]", true)}}

### Out-of-Scope Use

<!-- This section addresses misuse, malicious use, and uses that the model will not work well for. -->

{{ out_of_scope_use | default("[More Information Needed]", true)}}

## Bias, Risks, and Limitations

<!-- This section is meant to convey both technical and sociotechnical limitations. -->

The model has following bias and risk when used xxxx <!-- bias_risks_limitations -->

## Training Details

### Training Data

<!-- This should link to a Dataset Card, perhaps with a short stub of information on what the training data is all about as well as documentation related to data pre-processing or additional filtering. -->

For training the training set from the Tiny-imagenet-200 dataset is used. It contains XXXX. <!-- training_data -->

### Training Procedure

<!-- This relates heavily to the Technical Specifications. Content here should link to that section when it is relevant to the training procedure. -->

#### Preprocessing

{{ preprocessing | default("[More Information Needed]", true)}}


#### Training Hyperparameters

- **Training regime:** {{ training_regime | default("[More Information Needed]", true)}} <!--fp32, fp16 mixed precision, bf16 mixed precision, bf16 non-mixed precision, fp16 non-mixed precision, fp8 mixed precision -->

#### Speeds, Sizes, Times

<!-- This section provides information about throughput, start/end time, checkpoint size if relevant, etc. -->

{{ speeds_sizes_times | default("[More Information Needed]", true)}}

## Evaluation

<!-- This section describes the evaluation protocols and provides the results. -->

### Testing Data, Factors & Metrics

#### Testing Data

<!-- This should link to a Dataset Card if possible. -->

For testing the validation set from Tiny-imagenet-200 dataset is used, containing XXXX. <!-- testing_data -->

#### Factors

<!-- These are the things the evaluation is disaggregating by, e.g., subpopulations or domains. -->

{{ testing_factors | default("[More Information Needed]", true)}}

#### Metrics

<!-- These are the evaluation metrics being used, ideally with a description of why. -->

{{ testing_metrics | default("[More Information Needed]", true)}}

### Results

{{ results | default("[More Information Needed]", true)}}

#### Summary

{{ results_summary | default("", true) }}

## Model Examination

<!-- Relevant interpretability work for the model goes here -->

{{ model_examination | default("[More Information Needed]", true)}}

## Technical Specifications

### Model Architecture and Objective

The base architecture is the RESNET-18 xxxxx <!-- model_specs -->

### Compute Infrastructure

{{ compute_infrastructure | default("[More Information Needed]", true)}}

#### Hardware

{{ hardware_requirements | default("[More Information Needed]", true)}}

#### Software

{{ software | default("[More Information Needed]", true)}}