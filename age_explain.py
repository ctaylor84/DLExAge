import os
import pickle
import lzma
import argparse
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import torch
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from captum.attr import IntegratedGradients, NoiseTunnel

from vit_explain.baselines.ViT.ViT_explanation_generator import LRP
from models import resnet50, efficientnetv2_m, prepare_model

SEGMENTATION_COLOURS = [
    "blue",
    "cyan",
    "orange",
    "green",
    "red",
    "purple",
    "brown",
    "pink",
    "gray",
    "olive",
    "tan",
]


def parse_args():
    parser = argparse.ArgumentParser(description="OCT attribution map generator")
    parser.add_argument(
        "-i",
        "--img_path",
        default="../RETFound_MAE_RG/datasets/images",
        dest="img_path",
        help="Image path",
    )
    parser.add_argument(
        "-e",
        "--seg_path",
        default="../octage/segmentation/data/pkl",
        dest="seg_path",
        help="Segmentation path",
    )
    parser.add_argument(
        "-m",
        "--model_path",
        default="../RETFound_MAE_RG/finetune/bb_age_+0_+0/checkpoint-best.pth",
        dest="model_path",
        help="Model parameter path",
    )
    parser.add_argument(
        "--model",
        default="vit_large_patch16",
        choices=[
            "vit_large_patch16",
            "vit_base_patch16",
            "vit_small_patch16",
            "efficientnetv2_m",
            "resnet50",
        ],
        dest="model",
        help="Model type",
    )
    parser.add_argument(
        "--baseline",
        default=[0.0, 1.0],
        nargs="+",
        dest="baseline",
        type=float,
        help="Integrated gradients baselines (when relevant)",
    )
    parser.add_argument(
        "-n",
        "--norm_path",
        default="../RETFound_MAE_RG/datasets/bb_age_+0_+0/train.csv",
        dest="norm_path",
        help="Training set CSV file path (to obtain norm parameters)",
    )
    parser.add_argument(
        "-c",
        "--csv_path",
        default="../RETFound_MAE_RG/datasets/bb_age_+0_+0/test.csv",
        dest="csv_path",
        help="CSV file containing central fovea slice values",
    )
    parser.add_argument(
        "-o",
        "--out_path",
        default="attr/attr.xz",
        dest="out_path",
        help="Output file path",
    )
    parser.add_argument(
        "-s",
        "--slice_count",
        default=1,
        dest="slice_count",
        type=int,
        help="Number of slices to use (including the central slice)",
    )
    parser.add_argument(
        "-g",
        "--gap_size",
        default=5,
        dest="gap_size",
        type=int,
        help="Pixel threshold for layer filtering (-1 for no filter)",
    )
    parser.add_argument(
        "--cls_token",
        default=False,
        action="store_true",
        dest="cls_token",
        help="Use the class token to make predictions (ViT models only)",
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        dest="plot",
        help="Plot attributions",
    )
    parser.add_argument(
        "--max",
        default=False,
        action="store_true",
        dest="attr_max",
        help="Calculate the maximum attribution over white and black baselines (IG only)",
    )
    parser.add_argument(
        "--nt",
        default=False,
        action="store_true",
        dest="noise_tunnel",
        help="Use noise tunnel on input images (IG only)",
    )
    parser.add_argument(
        "--test",
        default=False,
        action="store_true",
        dest="mem_test",
        help="VRAM test mode",
    )
    parser.add_argument(
        "--pause",
        default=None,
        type=int,
        help="Save a checkpoint after this many scans, then exit",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="Resume from previous checkpoint",
    )
    args = vars(parser.parse_args())
    assert args["slice_count"] > 0
    args["img_path"] = os.path.expanduser(args["img_path"])
    args["seg_path"] = os.path.expanduser(args["seg_path"])
    args["norm_path"] = os.path.expanduser(args["norm_path"])
    args["csv_path"] = os.path.expanduser(args["csv_path"])
    return args


def get_norm_params(csv_path):
    targets_df = pd.read_csv(
        os.path.expanduser(csv_path), header=0, dtype={"file": str, "target": float}
    )

    target_values = torch.tensor(list(targets_df["target"]))
    dataset_mean = torch.mean(target_values).item()
    dataset_std = torch.std(target_values).item()
    norm_params = {"mean": dataset_mean, "std": dataset_std}
    return norm_params


def build_transform(input_size, norm=True):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    t = []
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
    )
    t.append(transforms.CenterCrop(input_size))

    if norm:
        t.append(transforms.ToTensor())
        t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


def generate_lrp_attr(attr_gen, input_image, class_index=None):
    transformer_attribution = attr_gen.generate_LRP(
        input_image,
        method="transformer_attribution",
        index=class_index,
    ).detach()

    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(
        transformer_attribution, scale_factor=16, mode="bilinear"
    )
    transformer_attribution = (
        transformer_attribution.reshape(224, 224).data.cpu().numpy()
    )
    return transformer_attribution


def get_image_attr_lrp(img_t, model, attr_gen):
    with torch.cuda.amp.autocast():  # type: ignore
        output = model(img_t)
    attr = generate_lrp_attr(attr_gen, img_t)
    return attr, output.item()


def generate_ig_attr(attr_gen, input_image, baseline):
    if attr_gen.__class__ == IntegratedGradients:
        attributions_ig = attr_gen.attribute(
            input_image,
            target=None,
            baselines=baseline,
            n_steps=100,
        ).reshape(3, 224, 224)
    else:  # Using noise tunnel
        attributions_ig = attr_gen.attribute(
            input_image,
            nt_samples=5,
            nt_samples_batch_size=5,
            nt_type="smoothgrad",
            target=None,
            baselines=baseline,
            n_steps=100,
        ).reshape(3, 224, 224)

    attributions_ig = torch.sum(torch.abs(attributions_ig), dim=0).cpu().numpy()
    return attributions_ig


def get_image_attr_ig(img_t, model, attr_gen, baseline):
    with torch.cuda.amp.autocast():  # type: ignore
        output = model(img_t)
        output_b = model(baseline)
        attr = generate_ig_attr(attr_gen, img_t, baseline)
    fit_score = np.abs(np.sum(attr) - (output.item() - output_b.item()))
    return attr, output.item(), fit_score


def get_image_attr_ig_loop(img_t, model, attr_gen, baseline, attr_max=False):
    if not attr_max:
        attr_list = list()
    else:
        img_size = img_t.shape[-1]
        attr = np.zeros((img_size, img_size))
    fit_score_list = list()

    for i in range(len(baseline)):
        attr_out = get_image_attr_ig(img_t, model, attr_gen, baseline[i])
        attr_c, age_pred, fit_score_c = attr_out

        if len(baseline) == 1:
            return attr_c, age_pred, fit_score_c

        if not attr_max:
            attr_list.append(attr_c)
        else:
            greater_idx = attr_c > attr
            attr[greater_idx] = attr_c[greater_idx]
        fit_score_list.append(fit_score_c)

    if not attr_max:
        attr = np.stack(attr_list)
        attr = np.mean(attr, axis=0)

    fit_score = np.mean(fit_score_list)
    return attr, age_pred, fit_score


def draw_area(first_layer, second_layer, size=224):
    area = Image.new("1", (size, size))
    draw = ImageDraw.Draw(area)
    draw.line(list(zip(range(size), first_layer)), fill=1, width=1)

    layer_stack = np.stack((first_layer, second_layer), axis=1)
    area_start = np.amin(np.around(layer_stack), axis=1).astype(int)
    area_end = np.amax(np.around(layer_stack), axis=1).astype(int)

    # Draw lines between each segment boundary along the x-axis
    for j in range(size):
        draw.line([(j, area_start[j]), (j, area_end[j])], fill=1, width=1)
    return np.asarray(area, dtype=bool)


def get_layers(img_in_shape, seg, ds_indices):
    if img_in_shape == "256 x 256":
        seg_ds = (seg[ds_indices, :] - 69) / (512 / 224)
    elif img_in_shape == "256 x 171":
        seg_ds = (seg[ds_indices, :]) / (480 / 224)
    else:
        raise RuntimeError("Non-matching image size")

    # Apply RETFound augmentation transform
    seg_ds = (seg_ds / (224 / 256)) - 16
    for i in range(seg_ds.shape[1]):
        seg_ds[:, i] = np.interp(
            np.linspace(14, 210, 224), np.arange(seg_ds.shape[0]), seg_ds[:, i]
        )

    layers = np.zeros((224, 224, (seg.shape[1] - 1) + 2), dtype=bool)
    layers[:, :, 0] = draw_area(np.zeros((224)), seg_ds[:, 0])

    for i in range(seg.shape[1] - 1):
        layers[:, :, i + 1] = draw_area(seg_ds[:, i], seg_ds[:, i + 1])

    layers[:, :, -1] = draw_area(seg_ds[:, -1], np.full((224), 224))
    gap_dist = seg_ds[:, 5] - seg_ds[:, 1]
    return layers, gap_dist, seg_ds


def overlap_filter(layers):
    # Prevent layers from overlapping by giving priority to lower layers
    layers_filtered = np.copy(layers)
    for i in range(layers.shape[2] - 1):
        filter_idx = np.any(layers[:, :, i + 1 :], axis=2)
        layers_filtered[filter_idx, i] = False
    return layers_filtered


def gap_filter(layers, gap_dist, is_left_eye, gap_size=5, col_range=(90, 140)):
    # Filter upper retinal layer pixels that are close to the outer nuclear layer
    layers_out = np.flip(layers, axis=1) if is_left_eye else np.copy(layers)
    gap_dist_in = np.flip(gap_dist) if is_left_eye else gap_dist

    match_idx = np.zeros(gap_dist.shape, dtype=bool)
    match_idx[col_range[0] : col_range[1]] = (
        gap_dist_in[col_range[0] : col_range[1]] <= gap_size
    )
    rep_idx = np.zeros(layers.shape, dtype=bool)
    rep_idx[:, match_idx, 2:6] = layers_out[:, match_idx, 2:6] == 1
    layers_out[rep_idx] = 0
    layers_out[np.any(rep_idx, axis=2), 6] = 1

    if is_left_eye:
        layers_out = np.flip(layers_out, axis=1)
    return layers_out


def get_layer_attribution(attr, layers):
    col_sums = np.zeros((attr.shape[1], layers.shape[2]))
    col_sizes = np.zeros((attr.shape[1], layers.shape[2]), dtype=int)

    for i in range(layers.shape[2]):
        idx = layers[:, :, i] == 1
        layer_attr = attr[idx]

        if layer_attr.shape[0] > 0:
            for j in range(attr.shape[1]):
                col_attr = attr[layers[:, j, i] == 1, j]

                if col_attr.shape[0] > 0:
                    col_sums[j, i] = np.sum(col_attr)
                    col_sizes[j, i] = col_attr.shape[0]
    return col_sums, col_sizes


def get_csv_data(csv_path):
    in_data = pd.read_csv(csv_path, header=0)
    out_data = dict()

    for _, row in in_data.iterrows():
        ext_idx = row["file"].find(".")
        file_id = row["file"][:ext_idx]
        fn_split = file_id.split("_")
        scan_id = f"{fn_split[0]}_{fn_split[1]}_{fn_split[2]}_{fn_split[3]}"
        out_data[scan_id] = {
            "fovea_slice": int(fn_split[4]),
            "target": row["target"],
            "eye": fn_split[1],
            "file_ext": row["file"][ext_idx:],
        }
    return out_data


def get_img_sizes(data_path):
    img_sizes = dict()
    with open(data_path) as fh:
        file_data = fh.readlines()

    for line in file_data:
        size = line.split(",")[1][1:]
        splitn = line.split("_")
        scan_id = f"{splitn[0]}_{splitn[1]}_{splitn[2]}_{splitn[3]}"
        img_sizes[scan_id] = size
    return img_sizes


def plot_attr(img, attr, segmentation, layers, idx, directional=False, seg_lines=False):
    transform_no_mod = build_transform(224, norm=False)
    img = transform_no_mod(img)
    in_img = np.array(img)

    if directional:
        max_attr = max(np.amax(attr), -np.amin(attr))
        min_attr = -max_attr
        norm = mpl.colors.Normalize(vmin=min_attr, vmax=max_attr)
        attr_colors = plt.cm.seismic(norm(attr))[:, :, :3]
    else:
        attr_colors = plt.cm.jet(attr / np.amax(attr))[:, :, :3]

    colors = (in_img / (255 * 1)) + attr_colors
    colors /= np.amax(colors, axis=(0, 1))

    _, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25, 8.33))

    seg_img = np.zeros((layers.shape[0], layers.shape[1], 3))
    for i in range(layers.shape[2]):
        seg_color = mpl.colors.to_rgb(SEGMENTATION_COLOURS[i])
        seg_img[np.nonzero(layers[:, :, i])] = seg_color

    seg_img = seg_img + attr_colors
    seg_img /= np.amax(seg_img, axis=(0, 1))
    ax1.imshow(seg_img)
    ax1.axis("off")

    ax2.imshow(colors)
    ax2.axis("off")
    if seg_lines:
        for i in range(segmentation.shape[1]):  # -> 5, 7,8
            seg_clean = segmentation[:, i]
            seg_clean[seg_clean > 224] = 223
            seg_clean[seg_clean < 0] = 0
            ax2.plot(
                list(range(224)), seg_clean, linewidth=1, c=SEGMENTATION_COLOURS[i]
            )

    ax3.imshow(img)
    ax3.axis("off")
    plt.tight_layout()
    plt.savefig(f"plots/idv/{idx}.png", dpi=350)
    # plt.show()
    plt.close()


def plot_seg(img, segmentation, layers):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8.33))

    seg_img = np.zeros((layers.shape[0], layers.shape[1], 3))
    for i in range(layers.shape[2]):
        seg_color = mpl.colors.to_rgb(SEGMENTATION_COLOURS[i])
        seg_img[np.nonzero(layers[:, :, i])] = seg_color
    ax1.imshow(seg_img)
    ax1.axis("off")

    ax2.imshow(img, cmap="gray")
    ax2.axis("off")
    for i in range(segmentation.shape[1]):
        seg_clean = segmentation[:, i]
        seg_clean[seg_clean > 224] = 223
        seg_clean[seg_clean < 0] = 0
        ax2.plot(list(range(224)), seg_clean, linewidth=1, c=SEGMENTATION_COLOURS[i])
    plt.tight_layout()
    plt.show()


def generate_baselines(colors, transform):
    attr_baseline = list()
    for baseline_color in colors:
        baseline_color = int(255 * baseline_color)
        baseline_tuple = (baseline_color, baseline_color, baseline_color)
        attr_baseline_c = Image.new("RGB", (224, 224), color=baseline_tuple)  # type: ignore
        attr_baseline_c = transform(attr_baseline_c).unsqueeze(0).cuda()  # type: ignore
        attr_baseline.append(attr_baseline_c)
    return attr_baseline


def main():
    args = parse_args()
    print(args)
    transform = build_transform(224)
    norm_params = get_norm_params(args["norm_path"])
    print("Norm parameters:", norm_params)

    if args["model"][:3] == "vit":
        model = prepare_model(
            args["model"], args["model_path"], global_pool=not args["cls_token"]
        )
        attr_gen = LRP(model)
        use_lrp = True
    else:
        if args["model"] == "resnet50":
            model = resnet50(args["model_path"])
        elif args["model"] == "efficientnetv2_m":
            model = efficientnetv2_m(args["model_path"])

        print("Baseline values:", args["baseline"])
        baseline = generate_baselines(args["baseline"], transform)

        attr_gen = IntegratedGradients(model)
        if args["noise_tunnel"]:
            attr_gen = NoiseTunnel(attr_gen)
        use_lrp = False
        ig_score_sum = 0.0

    seg_files = set(os.listdir(args["seg_path"]))
    csv_data = get_csv_data(args["csv_path"])
    img_sizes = get_img_sizes("image_info.txt")

    ds_indices = np.around(np.arange(0, 512, 512 / 224)).astype(int)
    image_count = 0
    layer_count = 11
    out_meta = list()
    out_attr = list()
    out_sizes = list()

    if not args["mem_test"] and args["resume"]:
        with lzma.open(os.path.expanduser(args["out_path"]), "rb") as fh:
            checkpoint = pickle.load(fh)
        assert checkpoint["checkpoint"] is not None
        resume_count = checkpoint["checkpoint"]
    else:
        resume_count = None

    for scan_id, scan_data in csv_data.items():
        seg_fname = f"{scan_id}.pkl"

        if seg_fname not in seg_files:
            continue

        if resume_count is not None and image_count + 1 <= resume_count:
            image_count += 1
            continue

        scan_meta = dict(scan_data)
        scan_meta["scan_id"] = scan_id
        scan_meta["pred"] = list()
        scan_meta["slices"] = list()
        scan_sizes = np.zeros((args["slice_count"], 224, layer_count), dtype=int)
        scan_attr = np.zeros((args["slice_count"], 224, layer_count))

        is_left_eye = scan_data["eye"] == "21017"

        with open(os.path.join(args["seg_path"], seg_fname), "rb") as fh:
            seg = pickle.load(fh)

        # Add 6 pixel outer vitreous layer
        upper_seg = np.expand_dims(seg[:, :, 0] - 12, axis=2)
        seg = np.concatenate((upper_seg, seg), axis=2)

        for slice_j in range(args["slice_count"]):
            slice_r = scan_data["fovea_slice"] - args["slice_count"] // 2 + slice_j
            scan_meta["slices"].append(slice_r)

            img_ext = scan_data["file_ext"]
            img_fname = f"{scan_id}_{slice_r}{img_ext}"
            try:
                img_raw = Image.open(os.path.join(args["img_path"], img_fname))
            except FileNotFoundError:
                img_raw = Image.open(
                    os.path.join(args["img_path"], scan_id[0], img_fname)
                )
            img = img_raw.convert("RGB")
            img_t = transform(img).unsqueeze(0).cuda()

            if use_lrp:
                attr, age_pred = get_image_attr_lrp(img_t, model, attr_gen)
            else:
                attr, age_pred, ig_score = get_image_attr_ig_loop(
                    img_t, model, attr_gen, baseline, attr_max=args["attr_max"]
                )
                ig_score_sum += ig_score
            age_pred = (age_pred * norm_params["std"]) + norm_params["mean"]
            scan_meta["pred"].append(age_pred)

            # Remove unused last segmentation layer
            seg_slice = seg[slice_r - 1, :, :-1]
            layers, gap_dist, seg_ds = get_layers(
                img_sizes[scan_id], seg_slice, ds_indices
            )
            layers = overlap_filter(layers)

            if args["gap_size"] > 0:
                layers = gap_filter(
                    layers, gap_dist, is_left_eye, gap_size=args["gap_size"]
                )
            l_attr_sums, l_attr_sizes = get_layer_attribution(attr, layers)

            if is_left_eye:
                l_attr_sums = np.flip(l_attr_sums, axis=0)
                l_attr_sizes = np.flip(l_attr_sizes, axis=0)

            scan_attr[slice_j] = l_attr_sums
            scan_sizes[slice_j] = l_attr_sizes

            if args["plot"]:
                # plot_attr(img, attr, seg_ds, layers, directional=not use_lrp)
                # plot_attr(img, attr, seg_ds, layers, image_count, directional=False)
                plot_seg(img, seg_ds, layers)
        out_meta.append(scan_meta)
        out_attr.append(scan_attr)
        out_sizes.append(scan_sizes)

        image_count += 1
        print(scan_id, "|", image_count, "/", len(seg_files), flush=True)
        if args["mem_test"]:
            break
        if args["pause"] is not None and image_count >= args["pause"]:
            break

    if not use_lrp:
        print("Integrated gradients fit score:", ig_score_sum / image_count)

    out_data = {
        "args": args,
        "meta": out_meta,
        "attr": np.stack(out_attr, axis=0),
        "sizes": np.stack(out_sizes, axis=0),
        "score": ig_score_sum if not use_lrp else None,
        "checkpoint": args["pause"],
    }

    if args["resume"]:
        out_data["meta"] = checkpoint["meta"] + out_data["meta"]
        out_data["attr"] = np.concatenate(
            (checkpoint["attr"], out_data["attr"]), axis=0
        )
        out_data["sizes"] = np.concatenate(
            (checkpoint["sizes"], out_data["sizes"]), axis=0
        )
        out_data["score"] = ig_score_sum + checkpoint["score"] if not use_lrp else None
        out_data["checkpoint"] = None

    if not args["mem_test"]:
        with lzma.open(os.path.expanduser(args["out_path"]), "wb") as fh:
            pickle.dump(out_data, fh)


if __name__ == "__main__":
    main()
