import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


THRESHOLD = 0.15

BAND_MEAN = np.array([
    0.043106921371243716,
    0.06289281560990558,
    0.0576524661529695,
    0.2567089260417136,
    0.17755819701799685,
    0.10268084334833795
], dtype=np.float32)

BAND_STD = np.array([
    0.04395384521952954,
    0.043629188586751334,
    0.051786338695320885,
    0.08738248514022778,
    0.08483382836073564,
    0.06507501083818537
], dtype=np.float32)


# --------------------------------------------------
# Model
# --------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class SiameseEncoder(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()

        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)
        self.enc4 = DoubleConv(128, 256)

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        x4 = self.enc4(self.pool(x3))

        return x1, x2, x3, x4


class SiameseUNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = SiameseEncoder(6)

        self.up3 = nn.ConvTranspose2d(
            256, 128, 2, stride=2
        )
        self.dec3 = DoubleConv(256, 128)

        self.up2 = nn.ConvTranspose2d(
            128, 64, 2, stride=2
        )
        self.dec2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(
            64, 32, 2, stride=2
        )
        self.dec1 = DoubleConv(64, 32)

        self.out = nn.Conv2d(
            32, 1, kernel_size=1
        )

    def forward(self, image_2018, image_2025):

        a1, a2, a3, a4 = self.encoder(image_2018)
        b1, b2, b3, b4 = self.encoder(image_2025)

        d1 = torch.abs(a1 - b1)
        d2 = torch.abs(a2 - b2)
        d3 = torch.abs(a3 - b3)
        d4 = torch.abs(a4 - b4)

        x = self.up3(d4)
        x = torch.cat([x, d3], dim=1)
        x = self.dec3(x)

        x = self.up2(x)
        x = torch.cat([x, d2], dim=1)
        x = self.dec2(x)

        x = self.up1(x)
        x = torch.cat([x, d1], dim=1)
        x = self.dec1(x)

        return self.out(x)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

def preprocess(image):

    image = image.astype(np.float32)

    image = np.clip(
        image,
        0,
        10000
    )

    image = image / 10000.0

    image = (
        image - BAND_MEAN[:, None, None]
    ) / BAND_STD[:, None, None]

    return torch.from_numpy(
        image
    ).unsqueeze(0)


# --------------------------------------------------
# RGB display
# --------------------------------------------------

def make_rgb(image):

    # Band order:
    # B2, B3, B4, B8, B11, B12
    #
    # RGB uses:
    # B4, B3, B2

    rgb = image[
        [2, 1, 0]
    ].astype(np.float32)

    for i in range(3):

        low, high = np.percentile(
            rgb[i],
            [2, 98]
        )

        rgb[i] = np.clip(
            (rgb[i] - low)
            /
            (high - low + 1e-8),
            0,
            1
        )

    return np.transpose(
        rgb,
        (1, 2, 0)
    )


# --------------------------------------------------
# Metrics
# --------------------------------------------------

def calculate_metrics(prediction, truth):

    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    tp = np.logical_and(
        prediction,
        truth
    ).sum()

    fp = np.logical_and(
        prediction,
        np.logical_not(truth)
    ).sum()

    fn = np.logical_and(
        np.logical_not(prediction),
        truth
    ).sum()

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    iou = (
        tp / (tp + fp + fn)
        if (tp + fp + fn) > 0
        else 0.0
    )

    return precision, recall, f1, iou


# --------------------------------------------------
# Diagnostic overlay
# --------------------------------------------------

def make_overlay(prediction, truth):

    prediction = prediction.astype(bool)
    truth = truth.astype(bool)

    height, width = truth.shape

    overlay = np.zeros(
        (height, width, 3),
        dtype=np.float32
    )

    true_positive = np.logical_and(
        prediction,
        truth
    )

    false_positive = np.logical_and(
        prediction,
        np.logical_not(truth)
    )

    false_negative = np.logical_and(
        np.logical_not(prediction),
        truth
    )

    # True positives = green
    overlay[true_positive] = [
        0,
        1,
        0
    ]

    # False positives = orange
    overlay[false_positive] = [
        1,
        0.5,
        0
    ]

    # False negatives = magenta
    overlay[false_negative] = [
        1,
        0,
        1
    ]

    return overlay


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run forest-loss inference "
            "with the Siamese U-Net."
        )
    )

    parser.add_argument(
        "--image2018",
        required=True,
        help=(
            "Path to 2018 region "
            ".npy file"
        )
    )

    parser.add_argument(
        "--image2025",
        required=True,
        help=(
            "Path to 2025 region "
            ".npy file"
        )
    )

    parser.add_argument(
        "--mask",
        required=True,
        help=(
            "Path to Hansen mask "
            ".npy file"
        )
    )

    parser.add_argument(
        "--patch",
        type=int,
        default=0,
        help=(
            "Patch index inside "
            "the region files"
        )
    )

    parser.add_argument(
        "--model",
        default=(
            "best_siamese_unet_v22.pt"
        ),
        help=(
            "Path to trained "
            "model checkpoint"
        )
    )

    parser.add_argument(
        "--output",
        default="prediction.png",
        help=(
            "Output visualization "
            "filename"
        )
    )

    args = parser.parse_args()


    # ----------------------------------------------
    # Device
    # ----------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Device:",
        device
    )


    # ----------------------------------------------
    # Load region files
    # ----------------------------------------------

    region_2018 = np.load(
        args.image2018,
        mmap_mode="r"
    )

    region_2025 = np.load(
        args.image2025,
        mmap_mode="r"
    )

    region_masks = np.load(
        args.mask,
        mmap_mode="r"
    )

    print(
        "2018 region shape:",
        region_2018.shape
    )

    print(
        "2025 region shape:",
        region_2025.shape
    )

    print(
        "Mask region shape:",
        region_masks.shape
    )


    # ----------------------------------------------
    # Check shapes
    # ----------------------------------------------

    if (
        region_2018.ndim != 4
        or region_2018.shape[1:]
        != (6, 128, 128)
    ):

        raise ValueError(
            "2018 file should have "
            "shape (N,6,128,128)"
        )


    if (
        region_2025.ndim != 4
        or region_2025.shape[1:]
        != (6, 128, 128)
    ):

        raise ValueError(
            "2025 file should have "
            "shape (N,6,128,128)"
        )


    if (
        region_masks.ndim != 3
        or region_masks.shape[1:]
        != (128, 128)
    ):

        raise ValueError(
            "Mask file should have "
            "shape (N,128,128)"
        )


    if not (
        len(region_2018)
        ==
        len(region_2025)
        ==
        len(region_masks)
    ):

        raise ValueError(
            "2018, 2025 and mask "
            "files must contain the "
            "same number of patches."
        )


    if (
        args.patch < 0
        or args.patch
        >= len(region_2018)
    ):

        raise ValueError(
            f"Patch must be between "
            f"0 and "
            f"{len(region_2018)-1}"
        )


    print(
        "Using patch:",
        args.patch
    )


    # ----------------------------------------------
    # Extract patch
    # ----------------------------------------------

    raw_2018 = np.array(
        region_2018[
            args.patch
        ],
        dtype=np.float32
    )

    raw_2025 = np.array(
        region_2025[
            args.patch
        ],
        dtype=np.float32
    )

    ground_truth = np.array(
        region_masks[
            args.patch
        ],
        dtype=np.float32
    )


    # ----------------------------------------------
    # Preprocess
    # ----------------------------------------------

    x2018 = preprocess(
        raw_2018
    ).to(device)

    x2025 = preprocess(
        raw_2025
    ).to(device)


    # ----------------------------------------------
    # Load model
    # ----------------------------------------------

    model = SiameseUNet().to(
        device
    )

    checkpoint = torch.load(
        args.model,
        map_location=device
    )

    if (
        isinstance(
            checkpoint,
            dict
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        checkpoint = (
            checkpoint[
                "model_state_dict"
            ]
        )

    model.load_state_dict(
        checkpoint
    )

    model.eval()


    # ----------------------------------------------
    # Prediction
    # ----------------------------------------------

    with torch.no_grad():

        logits = model(
            x2018,
            x2025
        )

        probabilities = torch.sigmoid(
            logits
        )

        prediction = (
            probabilities
            >= THRESHOLD
        ).float()


    probability_map = (
        probabilities[
            0,
            0
        ]
        .cpu()
        .numpy()
    )

    prediction_mask = (
        prediction[
            0,
            0
        ]
        .cpu()
        .numpy()
    )


    # ----------------------------------------------
    # Metrics
    # ----------------------------------------------

    predicted_loss = (
        prediction_mask.mean()
        * 100
    )

    actual_loss = (
        ground_truth.mean()
        * 100
    )

    precision, recall, f1, iou = (
        calculate_metrics(
            prediction_mask,
            ground_truth
        )
    )

    print()
    print(
        f"Actual forest loss: "
        f"{actual_loss:.2f}%"
    )

    print(
        f"Predicted forest loss: "
        f"{predicted_loss:.2f}%"
    )

    print(
        f"Precision: "
        f"{precision:.3f}"
    )

    print(
        f"Recall: "
        f"{recall:.3f}"
    )

    print(
        f"F1: "
        f"{f1:.3f}"
    )

    print(
        f"IoU: "
        f"{iou:.3f}"
    )


    # ----------------------------------------------
    # Visualization
    # ----------------------------------------------

    rgb2018 = make_rgb(
        raw_2018
    )

    rgb2025 = make_rgb(
        raw_2025
    )

    overlay = make_overlay(
        prediction_mask,
        ground_truth
    )


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
        ground_truth,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[2].set_title(
        f"Hansen Ground Truth\n"
        f"{actual_loss:.2f}% loss"
    )


    axes[3].imshow(
        prediction_mask,
        cmap="gray",
        vmin=0,
        vmax=1
    )

    axes[3].set_title(
        f"Model Prediction\n"
        f"{predicted_loss:.2f}% loss"
    )


    axes[4].imshow(
        overlay
    )

    axes[4].set_title(
        f"Diagnostic Overlay\n"
        f"F1 = {f1:.3f}"
    )


    for ax in axes:
        ax.axis(
            "off"
        )


    plt.tight_layout()


    plt.savefig(
        args.output,
        dpi=200,
        bbox_inches="tight"
    )


    plt.close()


    print()
    print(
        "Saved:",
        args.output
    )


if __name__ == "__main__":
    main()
