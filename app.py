import os
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import gradio as gr

from inference import (
    SiameseUNet,
    preprocess,
    make_rgb,
    calculate_metrics,
    make_overlay,
    THRESHOLD,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "best_siamese_unet_v22.pt"

DATA_DIRS = [
    "siamese_dataset_v2",
    "siamese_dataset",
]

EXCLUDED_REGIONS = {
    "brazil_para",
    "malaysia_sarawak",
}

VALIDATION_REGIONS = {
    "sweden_central",
    "spain_galicia",
}

TEST_REGIONS = {
    "australia_newsouthwales",
    "chile_biobio",
    "greece",
}


# --------------------------------------------------
# Device + model
# --------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", DEVICE)

MODEL = SiameseUNet().to(DEVICE)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if (
    isinstance(checkpoint, dict)
    and "model_state_dict" in checkpoint
):
    checkpoint = checkpoint["model_state_dict"]

MODEL.load_state_dict(checkpoint)
MODEL.eval()

print("Model loaded successfully.")


# --------------------------------------------------
# Discover available regions
# --------------------------------------------------

def discover_regions():

    regions = {}

    for data_dir in DATA_DIRS:

        if not os.path.isdir(data_dir):
            continue

        pattern = os.path.join(
            data_dir,
            "*_2018.npy"
        )

        for path2018 in glob.glob(pattern):

            filename = os.path.basename(path2018)

            region = filename.replace(
                "_2018.npy",
                ""
            )

            if region in EXCLUDED_REGIONS:
                continue

            path2025 = os.path.join(
                data_dir,
                f"{region}_2025.npy"
            )

            pathmask = os.path.join(
                data_dir,
                f"{region}_masks.npy"
            )

            if (
                os.path.exists(path2025)
                and os.path.exists(pathmask)
            ):

                regions[region] = {
                    "2018": path2018,
                    "2025": path2025,
                    "mask": pathmask,
                }

    return dict(sorted(regions.items()))


REGIONS = discover_regions()

if not REGIONS:
    raise RuntimeError(
        "No complete region datasets were found."
    )


# --------------------------------------------------
# Split information
# --------------------------------------------------

def get_split(region):

    if region in TEST_REGIONS:
        return "Held-out Test"

    if region in VALIDATION_REGIONS:
        return "Validation"

    return "Training"


def pretty_region(region):

    return (
        region
        .replace("_", " ")
        .title()
    )


# --------------------------------------------------
# Region information
# --------------------------------------------------

def region_changed(region):

    paths = REGIONS[region]

    data = np.load(
        paths["2018"],
        mmap_mode="r"
    )

    number_of_patches = len(data)

    split = get_split(region)

    info = (
        f"**{pretty_region(region)}**  \n"
        f"Dataset split: **{split}**  \n"
        f"Available patches: **{number_of_patches}**"
    )

    return (
        gr.update(
            minimum=0,
            maximum=number_of_patches - 1,
            value=0,
            step=1,
        ),
        info,
    )


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def run_prediction(region, patch_index):

    patch_index = int(patch_index)

    paths = REGIONS[region]

    region2018 = np.load(
        paths["2018"],
        mmap_mode="r"
    )

    region2025 = np.load(
        paths["2025"],
        mmap_mode="r"
    )

    masks = np.load(
        paths["mask"],
        mmap_mode="r"
    )

    if patch_index >= len(region2018):
        raise gr.Error(
            f"Patch must be between 0 and "
            f"{len(region2018) - 1}."
        )

    raw2018 = np.array(
        region2018[patch_index],
        dtype=np.float32
    )

    raw2025 = np.array(
        region2025[patch_index],
        dtype=np.float32
    )

    truth = np.array(
        masks[patch_index],
        dtype=np.float32
    )

    x2018 = preprocess(
        raw2018
    ).to(DEVICE)

    x2025 = preprocess(
        raw2025
    ).to(DEVICE)

    with torch.no_grad():

        logits = MODEL(
            x2018,
            x2025
        )

        probabilities = torch.sigmoid(
            logits
        )

        prediction = (
            probabilities >= THRESHOLD
        ).float()

    prediction_mask = (
        prediction[0, 0]
        .cpu()
        .numpy()
    )

    probability_map = (
        probabilities[0, 0]
        .cpu()
        .numpy()
    )

    actual_loss = (
        truth.mean() * 100
    )

    predicted_loss = (
        prediction_mask.mean() * 100
    )

    precision, recall, f1, iou = (
        calculate_metrics(
            prediction_mask,
            truth
        )
    )

    rgb2018 = make_rgb(
        raw2018
    )

    rgb2025 = make_rgb(
        raw2025
    )

    overlay = make_overlay(
        prediction_mask,
        truth
    )


    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

    axes[0].imshow(
        rgb2018
    )
    axes[0].set_title(
        "2018 Sentinel-2"
    )

    axes[1].imshow(
        rgb2025
    )
    axes[1].set_title(
        "2025 Sentinel-2"
    )

    axes[2].imshow(
        truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[2].set_title(
        f"Hansen Ground Truth\n"
        f"{actual_loss:.1f}% loss"
    )

    axes[3].imshow(
        prediction_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[3].set_title(
        f"Model Prediction\n"
        f"{predicted_loss:.1f}% loss"
    )

    axes[4].imshow(
        overlay
    )
    axes[4].set_title(
        f"Diagnostic Overlay\n"
        f"F1 = {f1:.3f}"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()


    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------

    split = get_split(region)

    metrics = f"""
### Results

| Metric | Value |
|---|---:|
| Dataset split | **{split}** |
| Patch | **{patch_index}** |
| Actual forest loss | **{actual_loss:.2f}%** |
| Predicted forest loss | **{predicted_loss:.2f}%** |
| Precision | **{precision:.3f}** |
| Recall | **{recall:.3f}** |
| F1 | **{f1:.3f}** |
| IoU | **{iou:.3f}** |

**Diagnostic overlay:** green = true positive, orange = false positive, magenta = false negative.

Decision threshold: **{THRESHOLD}**
"""

    if split == "Training":

        metrics += """

> This region was included in training. Its results demonstrate learned performance but should not be interpreted as unseen-region generalization.
"""

    elif split == "Validation":

        metrics += """

> This region was used for model validation and threshold/model selection.
"""

    else:

        metrics += """

> This is a geographically held-out test region and was not used for model training.
"""

    return fig, metrics


# --------------------------------------------------
# Initial values
# --------------------------------------------------

initial_region = list(
    REGIONS.keys()
)[0]

initial_count = len(
    np.load(
        REGIONS[initial_region]["2018"],
        mmap_mode="r"
    )
)

initial_info = (
    f"**{pretty_region(initial_region)}**  \n"
    f"Dataset split: **{get_split(initial_region)}**  \n"
    f"Available patches: **{initial_count}**"
)


# --------------------------------------------------
# Gradio interface
# --------------------------------------------------

with gr.Blocks(
    title="Forest Loss Detection"
) as demo:

    gr.Markdown(
        """
# Forest Loss Detection from Satellite Imagery

Multispectral Siamese U-Net change detection using paired Sentinel-2 imagery from **2018 and 2025**.

Select a geographic region and a 128 × 128 satellite patch to run the trained model.
"""
    )

    with gr.Row():

        with gr.Column(scale=1):

            region_dropdown = gr.Dropdown(
                choices=list(REGIONS.keys()),
                value=initial_region,
                label="Region",
            )

            patch_slider = gr.Slider(
                minimum=0,
                maximum=initial_count - 1,
                value=0,
                step=1,
                label="Patch number",
            )

            region_info = gr.Markdown(
                initial_info
            )

            predict_button = gr.Button(
                "Run Forest-Loss Detection",
                variant="primary",
            )

        with gr.Column(scale=3):

            output_image = gr.Plot(
                label="Prediction"
            )

            output_metrics = gr.Markdown()


    region_dropdown.change(
        fn=region_changed,
        inputs=region_dropdown,
        outputs=[
            patch_slider,
            region_info,
        ],
    )

    predict_button.click(
        fn=run_prediction,
        inputs=[
            region_dropdown,
            patch_slider,
        ],
        outputs=[
            output_image,
            output_metrics,
        ],
    )


# --------------------------------------------------
# Launch
# --------------------------------------------------

if __name__ == "__main__":

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
