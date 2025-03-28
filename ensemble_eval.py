import os
import pickle
import argparse
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from scipy.stats import pearsonr

HC_ID_LIST = "../biobank/UKBB-IDs/IDs-OCT-HC.txt"
DISEASE_ID_LIST = "../biobank/UKBB-IDs/IDs-OCT-eyeDiseases.txt"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate and evaluate ensemble predictions"
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        help="Ensemble name",
    )
    parser.add_argument(
        "-m",
        "--model",
        help="Model name",
    )
    parser.add_argument(
        "--slice",
        default=None,
        type=int,
        help="Number of slices to use per eye (None for maximum slices)",
    )
    parser.add_argument(
        "-u",
        "--include_unpaired",
        default=False,
        action="store_true",
        help="Include unpaired eyes",
    )
    parser.add_argument(
        "-a",
        "--avg",
        default=False,
        action="store_true",
        dest="eye_avg",
        help="Use eye average",
    )
    parser.add_argument(
        "--exclude",
        default=False,
        action="store_true",
        dest="exclude_nhc",
        help="Exclude non-healthy control patients",
    )
    parser.add_argument(
        "-t",
        "--traj",
        default=False,
        action="store_true",
        dest="plot_trajectories",
        help="Plot ageing trajectories",
    )
    parser.add_argument(
        "-d",
        "--traj_disease",
        default=False,
        action="store_true",
        help="Plot ageing trajectories labelleed by disease",
    )
    parser.add_argument(
        "-c",
        "--traj_count",
        default=20,
        type=int,
        help="Number of ageing trajectories to plot",
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="Plot results",
    )
    parser.add_argument(
        "-s",
        "--save",
        default=False,
        action="store_true",
        help="Save plot files",
    )
    parser.add_argument(
        "-o",
        "--out_pred",
        default=False,
        action="store_true",
        help="Save predictions",
    )
    args = vars(parser.parse_args())
    return args


def load_results(ensemble_name, model_name):
    path = os.path.join("data", ensemble_name, "bin", model_name + ".dat")
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    return data


def exclude_unpaired_eyes(slice_results):
    scan_id = slice_results["scanID"]
    id_unique, id_count = np.unique(scan_id, return_counts=True)
    multi_match = scan_id[..., None] == id_unique[id_count != 2][None, ...]
    multi_idx = np.logical_not(multi_match.any(axis=1))
    paired_true = slice_results["true"][multi_idx]
    paired_pred = slice_results["pred"][multi_idx]
    paired_ids = slice_results["scanID"][multi_idx]
    paired_eyes = slice_results["eye"][multi_idx]
    return paired_true, paired_pred, paired_ids, paired_eyes


def eye_avg_pred(slice_results):
    scan_id = slice_results["scanID"]
    id_unique, id_count = np.unique(scan_id, return_counts=True)
    multi_match = scan_id[..., None] == id_unique[id_count != 2][None, ...]
    multi_idx = np.logical_not(multi_match.any(axis=1))

    right_idx = np.logical_and(slice_results["eye"], multi_idx)
    left_idx = np.logical_and(np.logical_not(slice_results["eye"]), multi_idx)

    eye_diff = np.abs(
        slice_results["true"][left_idx] - slice_results["true"][right_idx]
    )
    assert (eye_diff <= 1).all()

    eye_true = slice_results["true"][left_idx]
    eye_pred = np.mean(
        (slice_results["pred"][left_idx], slice_results["pred"][right_idx]),
        axis=0,
    )
    eye_id_left = slice_results["scanID"][left_idx]
    eye_id_right = slice_results["scanID"][right_idx]
    assert (eye_id_left == eye_id_right).all()
    return eye_true, eye_pred, eye_id_left


def true_vs_pred_boxplot(
    target_list, prediction_list, plot_name, figsize=(5, 4), save=False
):
    value_max = int(np.floor(np.amax(target_list)))
    value_min = int(np.ceil(np.amin(target_list)))
    total_bins = value_max - value_min + 2
    bins = np.arange(value_min, value_max + 2, step=1.0) - 0.5
    digitized = np.digitize(target_list, bins)
    residuals_list = target_list - prediction_list

    mean_x = np.arange(value_min, value_max + 1)
    bin_loss_means = [
        np.mean(residuals_list[digitized == i]) for i in range(1, total_bins)
    ]
    bin_pred_means = [
        np.mean(prediction_list[digitized == i]) for i in range(1, total_bins)
    ]
    box_bins_loss = [residuals_list[digitized == i] for i in range(1, total_bins)]
    box_bins_pred = [prediction_list[digitized == i] for i in range(1, total_bins)]
    box_bins_counts = [
        target_list[digitized == i].shape[0] for i in range(1, total_bins)
    ]
    box_bin_x = list(range(1, total_bins))

    label_interval = 5 if np.amax(target_list) < 100 else 10
    box_labels = ["" if int(x) % label_interval != 0 else str(x) for x in mean_x]
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.boxplot(
        box_bins_loss,
        positions=box_bin_x,
        widths=0.5,
        flierprops=dict({"markersize": 3}),
    )
    ax.set_xticks(ticks=box_bin_x)
    ax.set_xticklabels(labels=box_labels)
    ax.plot(box_bin_x, bin_loss_means, c="blue")
    ax.plot(box_bin_x, np.zeros(len(box_bin_x)), c="red")
    ax.set_xlabel("True value (years)")
    ax.set_ylabel("Residual (years)")
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax2 = ax.twinx()
    ax2.plot(box_bin_x, box_bins_counts, c="green")
    ax2.set_ylabel("Number of samples")
    ax2.yaxis.get_ticklocs(minor=True)
    ax2.minorticks_on()
    ax2.tick_params(axis="x", which="minor", bottom=False)
    plt.tight_layout()

    if save:
        plt.savefig("plots/ensemble/" + plot_name + "_residuals.png", dpi=350)
        plt.savefig("plots/ensemble/" + plot_name + "_residuals.eps")
    else:
        plt.show()
    plt.close(fig)

    plt.figure(figsize=figsize)
    plt.boxplot(
        box_bins_pred,
        positions=box_bin_x,
        widths=0.5,
        flierprops=dict({"markersize": 3}),
    )
    plt.xticks(ticks=box_bin_x, labels=box_labels)
    plt.plot(box_bin_x, mean_x, c="red")
    plt.plot(box_bin_x, bin_pred_means, c="blue")
    plt.xlabel("True value (years)")
    plt.ylabel("Predicted value (years)")
    plt.tight_layout()
    ax = plt.gca()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if save:
        plt.savefig("plots/ensemble/" + plot_name + "_pred_vs_true.png", dpi=350)
        plt.savefig("plots/ensemble/" + plot_name + "_pred_vs_true.eps", dpi=350)
    else:
        plt.show()
    plt.close()


def load_id_list(file_path):
    with open(file_path) as fh:
        file_data = fh.readlines()

    file_ids = list()
    for line in file_data:
        if line != "":
            file_ids.append(line.strip())
    return file_ids


def generate_trajectories(ens_true, ens_pred, ens_scan_id, traj_count):
    age_trajectories = dict()

    hc_ids = set(load_id_list(HC_ID_LIST))

    for age_true, age_pred, scan_id in zip(ens_true, ens_pred, ens_scan_id):
        patient_id = scan_id.split("_")[0]

        patient_hc = patient_id in hc_ids

        if patient_id not in age_trajectories:
            age_trajectories[patient_id] = {"true": list(), "pred": list()}
        age_trajectories[patient_id]["true"].append(age_true)
        age_trajectories[patient_id]["pred"].append(age_pred)
        age_trajectories[patient_id]["hc"] = patient_hc

    patient_age_spans = list()
    patient_id_list = list()
    for patient_id, patient_data in age_trajectories.items():
        if len(patient_data["true"]) == 1:
            continue
        age_span = max(patient_data["true"]) - min(patient_data["true"])
        patient_age_spans.append(age_span)
        patient_id_list.append(patient_id)

    trajectories_filtered = OrderedDict()
    sorted_idx = np.flip(np.argsort(patient_age_spans))

    disease_counts = {False: 0, True: 0}

    for idx in sorted_idx:
        patient_id = patient_id_list[idx]
        is_hc = patient_id in hc_ids
        if disease_counts[is_hc] >= traj_count:
            continue
        trajectories_filtered[patient_id] = age_trajectories[patient_id]
        disease_counts[is_hc] += 1
    return trajectories_filtered


def plot_traj_disease(age_trajectories, plot_name, residual=False, save=False):
    plt.figure(figsize=(8, 6))
    disease_labels = dict()

    true_all = list()
    pred_all = list()
    for patient_id, patient_data in age_trajectories.items():
        hc_bool = patient_data["hc"]

        hc_str = "True" if hc_bool else "False"

        traj_data = {"x": list(), "y": list(), "id": list(), "Healthy control": list()}
        for age_true, age_pred in zip(patient_data["true"], patient_data["pred"]):
            if residual:
                traj_data["y"].append(age_true - age_pred)
            else:
                traj_data["y"].append(age_pred)
            traj_data["x"].append(age_true)
            traj_data["id"].append(patient_id)
            traj_data["Healthy control"].append(hc_str)

        if hc_str not in disease_labels:
            disease_labels[hc_str] = len(disease_labels)
            plot_label = hc_str
        else:
            plot_label = None
        hc_idx = disease_labels[hc_str]
        palette = sns.color_palette([sns.color_palette()[hc_idx]], n_colors=1)

        true_all += traj_data["x"]
        pred_all += traj_data["y"]

        sns.lineplot(
            traj_data,
            x="x",
            y="y",
            hue="Healthy control",
            palette=palette,
            legend=False,
            label=plot_label,
        )

        sns.scatterplot(
            traj_data,
            x="x",
            y="y",
            hue="Healthy control",
            palette=palette,
            legend=False,
            marker="x",
        )
        # sns.scatterplot(traj_data, x="true", y="pred", hue="id", marker="x", legend=False)

    if residual:
        diag_line = np.linspace(min(true_all), max(true_all), 10)
        plt.plot(diag_line, np.zeros((10)), "--", c="black")
        plt.ylabel("Residual (years)")
        residual_str = "_residual"
    else:
        min_value = min(min(true_all), min(pred_all))
        max_value = max(max(true_all), max(pred_all))
        diag_line = np.linspace(min_value, max_value, 10)
        plt.plot(diag_line, diag_line, "--", c="black")
        plt.ylabel("Predicted value (years)")
        residual_str = ""

    ax = plt.gca()
    ax.legend(title="Healthy control")

    plt.xlabel("True value (years)")
    plt.tight_layout()

    if save:
        out_dir = "plots/ensemble/"
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(out_dir + plot_name + residual_str + "_disease.png", dpi=350)
    else:
        plt.show()


def plot_trajectories(age_trajectories, residual=False):
    plot_data = {"true": list(), "pred": list(), "id": list()}
    for patient_id, patient_data in age_trajectories.items():
        for age_true, age_pred in zip(patient_data["true"], patient_data["pred"]):
            plot_data["true"].append(age_true)
            plot_data["pred"].append(age_pred)
            plot_data["id"].append(patient_id)

    plt.figure(figsize=(8, 6))
    min_value = min(min(plot_data["true"]), min(plot_data["pred"]))
    max_value = max(max(plot_data["true"]), max(plot_data["pred"]))
    diag_line = np.linspace(min_value, max_value, 10)
    plt.plot(diag_line, diag_line, "--", c="black")

    sns.lineplot(plot_data, x="true", y="pred", hue="id")
    # sns.scatterplot(traj, x="true", y="pred", hue="id", marker="x", legend=False)

    plt.xlabel("True value (years)")
    plt.ylabel("Predicted value (years)")
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    model_results = load_results(args["ensemble"], args["model"])

    if args["slice"] is None:
        args["slice"] = max(model_results.keys())

    if args["eye_avg"]:
        ens_true, ens_pred, ens_scanID = eye_avg_pred(model_results[args["slice"]])
        plot_avg_ext = "_avg"
        ens_eye = None
    elif args["include_unpaired"]:
        ens_true = model_results[args["slice"]]["true"]
        ens_pred = model_results[args["slice"]]["pred"]
        ens_scanID = model_results[args["slice"]]["scanID"]
        ens_eye = model_results[args["slice"]]["eye"]
        plot_avg_ext = ""
    else:
        ens_true, ens_pred, ens_scanID, ens_eye = exclude_unpaired_eyes(
            model_results[args["slice"]]
        )
        plot_avg_ext = ""

    if args["exclude_nhc"]:
        filtered_true = list()
        filtered_pred = list()
        filtered_scanID = list()
        filtered_eye = list()
        hc_ids = set(load_id_list(HC_ID_LIST))
        for i, scanID in enumerate(ens_scanID):
            if scanID.split("_")[0] in hc_ids:
                filtered_true.append(ens_true[i])
                filtered_pred.append(ens_pred[i])
                filtered_scanID.append(scanID)

                if ens_eye is not None:
                    filtered_eye.append(ens_eye[i])
        ens_true = np.asarray(filtered_true)
        ens_pred = np.asarray(filtered_pred)
        ens_scanID = np.asarray(filtered_scanID)

        if ens_eye is not None:
            ens_eye = filtered_eye

    mae = float(mean_absolute_error(ens_true, ens_pred))
    rmse = float(root_mean_squared_error(ens_true, ens_pred))
    r2 = float(r2_score(ens_true, ens_pred))

    print("Ensemble metrics:")
    print("Prediction count:", ens_true.shape[0])
    print("Slice count:", args["slice"])
    print("MAE:", round(mae, 4))
    print("RMSE:", round(rmse, 4))
    print("R2:", round(r2, 4))
    print("pearsonr:", pearsonr(ens_true, ens_pred - ens_true))

    out_name = args["ensemble"] + "_" + args["model"] + plot_avg_ext
    if args["out_pred"]:
        os.makedirs("pred", exist_ok=True)
        pred_dict = {
            "id": list(ens_scanID),
            "eye": ens_eye,
            "true": list(ens_true),
            "pred": list(ens_pred),
        }
        pred_df = pd.DataFrame.from_dict(pred_dict)
        pred_df.to_csv(os.path.join("pred", out_name + ".csv"))

    if args["plot"]:
        plot_name = out_name
        if args["plot_trajectories"]:
            age_traj = generate_trajectories(
                ens_true, ens_pred, ens_scanID, args["traj_count"]
            )
            if args["traj_disease"]:
                plot_traj_disease(
                    age_traj, plot_name, residual=False, save=args["save"]
                )
            else:
                plot_trajectories(age_traj, residual=False)
        else:
            true_vs_pred_boxplot(ens_true, ens_pred, plot_name, save=args["save"])


if __name__ == "__main__":
    main()
