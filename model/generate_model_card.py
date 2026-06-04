from huggingface_hub import ModelCard, ModelCardData
from omegaconf import OmegaConf


def main():
    config_path = "outputs/2026-06-04/09-20-38/.hydra/config.yaml"
    cfg = OmegaConf.load(config_path)

    # Frontmatter metadata (the {{ card_data }} block at the top)
    card_data = ModelCardData(
        license="mit",
        tags=["image-classification", "resnet", "tiny-imagenet"],
        datasets=["tiny-imagenet-200"],
    )

    card = ModelCard.from_template(
        card_data,
        template_path="model/modelcard_template.md",
        # Model Details
        model_id="ResNet-18",
        model_summary=(
            f"The goal of this model is classifying images. It is trained on "
            f"{cfg.data.name}, which has {cfg.data.classes} classes with "
            f"{cfg.data.train_img} images in the training set and "
            f"{cfg.data.test_img} images in the test set."
        ),
        model_description=(
            f"This model is a ResNet-18 convolutional neural network trained from "
            f"scratch for image classification on the {cfg.data.name} dataset. It "
            f"takes a 64x64 RGB image as input and predicts one of {cfg.data.classes} "
            f"object classes. The model was developed as part of an MLOps course "
            f"project at Aalborg University, with the primary goal of demonstrating "
            f"an end-to-end machine-learning pipeline (training, evaluation, "
            f"versioning, CI/CD, compression, and monitoring) rather than achieving "
            f"state-of-the-art accuracy."
        ),
        developers="Anne, Lærke og Stoyan",
        model_type="Image classifier",
        language="Python",
        base_arc="ResNet-18 (trained from scratch)",
        # Model Sources
        repo="https://github.com/LaerkeRaaschou/daki4mlops/README.md",
        # Uses
        direct_use=(
            f"The model can be used directly for image classification: given a "
            f"64x64 RGB image, it predicts one of the {cfg.data.classes} "
            f"{cfg.data.name} classes. It is intended for educational and "
            f"demonstration purposes within this MLOps project."
        ),
        downstream_use=(
            "The trained network can serve as a basis for compression "
            "experiments (quantization and pruning)."
        ),
        out_of_scope_use=(
            "Not intended for production use or for images outside the "
            "Tiny ImageNet distribution."
        ),
        # Bias, Risks, and Limitations
        bias_risks_limitations=(
            "Trained only on the Tiny ImageNet classes at 64x64 resolution; "
            "performance is not guaranteed outside this distribution."
        ),
        bias_recommendations="",
        # How to Get Started
        get_started_code="Download the code in the repository and follow the README file.",
        # Training Details
        training_data=(
            f"Trained on the {cfg.data.name} dataset, using "
            f"{int(cfg.data.train_img * (1 - cfg.data.val_split))} training images "
            f"and {int(cfg.data.train_img * cfg.data.val_split)} images for validation."
            "Download data here: https://www.kaggle.com/competitions/tiny-imagenet/data."
        ),
        preprocessing=(
            "The model have been trained on data using following data preprocessing and augmentation:"
            "transforms.RandomResizedCrop(64, scale=(0.7, 1.0)), "
            "transforms.RandomHorizontalFlip(), "
            "transforms.ColorJitter(0.2, 0.2, 0.2, 0.1), "
            "transforms.ToTensor(), "
            "transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))"
        ),
        training_regime="fp16 mixed precision" if cfg.amp.use else "fp32",
        speeds_sizes_times="",
        # Evaluation
        testing_data=(
            f"The model is testet on the validation set from {cfg.data.name} as the original testset does not include labels."
            f"This test set contains {cfg.data.test_img} images."
        ),
        testing_factors=(
            "Evaluation is done per class."
            "Results are reported overall and per individual class across all 200 Tiny ImageNet classes, "
            "including identification of the best- and worst-performing classes."
        ),
        testing_metrics="Accuracy as the primary metric. Prediction confidence is also reported overall and per class.",
        results="",
        results_summary="",
        # Environmental Impact
        hardware_type="NVIDIA L4 GPU(s), single- or multi-GPU via PyTorch DDP / DeepSpeed.",
        hours_used="",
        cloud_provider="AI-Lab",
        co2_emitted="",
        # Technical Specifications
        model_specs=(
            f"The model is a ResNet-18 convolutional neural network, consisting of "
            f"an initial 3x3 convolution and max-pooling stem followed by four stages "
            f"of residual blocks (2 blocks each, with channel widths 64, 128, 256, 512), "
            f"global average pooling, and a fully connected classification head. The "
            f"architecture is adapted for Tiny ImageNet with {cfg.data.classes} output "
            f"classes and 64x64 RGB input. Residual (skip) connections allow gradients "
            f"to flow through the network, enabling stable training of the deeper stack. "
            f"The training objective is multi-class image classification, optimized with "
            f"{cfg.loss._target_}."
        ),
        compute_infrastructure="AI-Lab",
        hardware_requirements="NVIDIA L4 GPU(s), single- or multi-GPU via PyTorch DDP / DeepSpeed.",
        software="PyTorch, Hydra, OmegaConf, scikit-learn, Weights & Biases, MLflow.",
        # Citation
        citation_bibtex="""@inproceedings{he2016deep,
  title     = {Deep Residual Learning for Image Recognition},
  author    = {He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {770--778},
  year      = {2016},
  doi       = {https://doi.org/10.1109/CVPR.2016.90}
}""",
        citation_apa=(
            "He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning "
            "for Image Recognition. Proceedings of the IEEE Conference on Computer "
            "Vision and Pattern Recognition (CVPR), 770-778."
        ),
    )

    card.save("model_card.md")
    print("Model card written to model_card.md")


if __name__ == "__main__":
    main()
