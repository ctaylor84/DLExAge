import os
import time
import pickle
import argparse
from collections import defaultdict
from functools import partial
import multiprocessing as mp
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

NUM_THREADS = 24
NUTH_SLICE_MAX = 55
NUTH_SLICE_STEP = 5
NUTH_SLICE_LIMIT = 108
EVAL_TARGET = "all"
FOVEA_REL = True
FOVEA_SLICE_RANGE = {"min": 54, "max": 74}  # For UK Biobank


def parse_args():
    parser = argparse.ArgumentParser(description="OCT slice ensemble analysis")
    parser.add_argument(
        "-m",
        "--model",
        default="retfound",
        help="Ensemble model name",
    )
    parser.add_argument(
        "-t",
        "--type",
        default="ridge",
        choices=["lasso", "ridge", "rf", "gb"],
        help="Ensemble model type",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        default="bb_age",
        help="Model dataset name",
    )
    parser.add_argument(
        "-e",
        "--eval_dataset",
        default=None,
        help="Eval dataset name",
    )
    parser.add_argument(
        "-x",
        "--extra_train",
        default=None,
        help="Additional train dataset name",
    )
    parser.add_argument(
        "--slice_analysis",
        default=False,
        action="store_true",
        help="Plot per slice results",
    )
    parser.add_argument(
        "-p",
        "--plot",
        default=False,
        action="store_true",
        help="Plot model comparison",
    )
    parser.add_argument(
        "-b",
        "--basic",
        default=False,
        action="store_true",
        help="Add basic models to comparison plots",
    )
    parser.add_argument(
        "-s",
        "--save",
        default=False,
        action="store_true",
        help="Save plots to file",
    )
    return parser.parse_args()


def parse_models(dir, dataset, eval_dataset=None):
    dir_list = os.listdir(dir)
    model_paths = dict()
    for fname in dir_list:
        fname_split = fname.split("_")
        if fname[: len(dataset)] == dataset and fname_split[3] == EVAL_TARGET:
            if eval_dataset is None or fname_split[-1] == eval_dataset:
                model_paths[fname_split[2]] = fname
    return model_paths


def load_fovea_slices(data_path):
    with open(os.path.join(data_path, "fovea_slices.txt")) as fh:
        file_data = fh.readlines()

    fovea_slices = dict()
    for line in file_data:
        if line != "":
            line_split = line.split(",")
            fovea_slices[line_split[0]] = int(line_split[1])
    return fovea_slices


def slice_process(fname):
    fname_end = fname.split("_")[-1]
    slice = int(fname_end[: fname_end.find(".")])
    return slice


def fovea_slice_process(fname, fovea_slices):
    fname_end = fname.split("_")[-1]
    slice = int(fname_end[: fname_end.find(".")])
    image_id = fname[:17]
    try:
        fovea_slice = fovea_slices[image_id]
    except KeyError:
        fovea_slice = 64
    if fovea_slice < FOVEA_SLICE_RANGE["min"]:
        fovea_slice = 64
    elif fovea_slice > FOVEA_SLICE_RANGE["max"]:
        fovea_slice = 64
    slice -= fovea_slice
    return slice


def get_eye_process(fname):
    eye_str = fname.split("_")[1]
    return eye_str == "21018"


def get_scanID_process(fname):
    return fname[:7] + fname[13:17]


def load_targets(data_path, slice_value=None, fovea_rel=True, set_name="test"):
    csv_path = os.path.join(data_path, set_name + ".csv")
    targets_df = pd.read_csv(csv_path, header=0, dtype={"file": str, "target": float})
    target_files = list(targets_df["file"])
    pool = mp.Pool(NUM_THREADS)
    chunk_size = int(np.ceil(len(targets_df) / NUM_THREADS))

    if slice_value is None:
        if fovea_rel:
            fovea_slices = load_fovea_slices(data_path)
            process_f = partial(fovea_slice_process, fovea_slices=fovea_slices)
            slice_array = pool.map(process_f, target_files, chunk_size)
        else:
            slice_array = pool.map(slice_process, target_files, chunk_size)
        slice_array = np.asarray(slice_array)
    else:
        slice_array = np.repeat(slice_value, len(target_files))

    eye_array = pool.map(get_eye_process, target_files, chunk_size)
    id_array = pool.map(get_scanID_process, target_files, chunk_size)
    pool.close()

    eye_array = np.asarray(eye_array)
    id_array = np.asarray(id_array)
    return {
        "slices": slice_array,
        "true": targets_df["target"].values,
        "eye": eye_array,
        "scanID": id_array,
    }


def pair_eyes_nuth(anno_df, pair_max_gap=7):
    pair_gap_seconds = pair_max_gap * 86400
    image_count = 0
    data = dict()
    for _, row in anno_df.iterrows():
        if row["ScanPattern"] != "Volume scan in ART mode":
            continue
        if row["AnonID"] not in data:
            data[row["AnonID"]] = list()

        strc = time.strptime(row["VisitDate"], "%Y-%m-%d %H:%M:%S")  # type: ignore
        visit_time = time.mktime(strc)

        data[row["AnonID"]].append(
            {
                "file_id": row["FileID"],
                "visit_time": visit_time,
                "eye": row["Eye"],
                "slices": row["NumBScans"],
            }
        )
        image_count += 1

    pairs = dict()
    for patient_data in data.values():
        sorted_idx = np.argsort([x["slices"] for x in patient_data])
        sorted_idx = np.flip(sorted_idx)
        patient_data_sorted = [patient_data[x] for x in sorted_idx]
        image_used = [False] * len(patient_data)
        for i, image_i in enumerate(patient_data_sorted):
            if image_used[i]:
                continue
            for j, image_j in enumerate(patient_data_sorted):
                if image_used[j]:
                    continue
                if i != j and image_i["eye"] != image_j["eye"]:
                    if (
                        abs(image_i["visit_time"] - image_j["visit_time"])
                        <= pair_gap_seconds
                    ):
                        image_id_j = image_j["file_id"][
                            image_j["file_id"].find("_") + 1 :
                        ]
                        pair_id = image_i["file_id"] + "-" + image_id_j
                        pairs[image_i["file_id"]] = pair_id
                        pairs[image_j["file_id"]] = pair_id
                        image_used[i] = True
                        image_used[j] = True
                        break
    return pairs


def load_anno_nuth(csv_path):
    anno_df = pd.read_csv(csv_path, header=0, dtype={"FileID": str, "Eye": str})
    pair_ids = pair_eyes_nuth(anno_df)
    anno_data = dict()
    anno_data["eye"] = dict(zip(anno_df["FileID"], anno_df["Eye"]))
    anno_data["id"] = pair_ids
    return anno_data


def get_nuth_id_process(fname):
    id_split = fname.split("_")
    out_id = ""
    for i in range(len(id_split) - 1):
        out_id += id_split[i] + "_"
    out_id = out_id[:-1]
    return out_id


def load_targets_nuth(data_path, anno_data, set_name="test"):
    csv_path_targets = os.path.join(data_path, set_name + ".csv")
    targets_df = pd.read_csv(
        csv_path_targets, header=0, dtype={"file": str, "target": float, "rslice": int}
    )
    slice_array = list(targets_df["rslice"])
    id_array = list(map(get_nuth_id_process, targets_df["file"]))
    eye_array = [anno_data["eye"][x] == "right" for x in id_array]

    unpaired_count = 0
    pair_id_array = list()
    for scan_id in id_array:
        if scan_id in anno_data["id"]:
            pair_id_array.append(anno_data["id"][scan_id])
        else:
            pair_id_array.append(scan_id)
            unpaired_count += 1
    print("NUTH unpaired count:", unpaired_count, "/", len(pair_id_array))

    return {
        "slices": np.asarray(slice_array),
        "true": targets_df["target"].values,
        "eye": np.asarray(eye_array),
        "scanID": np.asarray(pair_id_array),
    }


def load_predictions(data_path, model_paths, targets, set_name="test"):
    pred = dict()
    slice_range = np.unique(targets["slices"])

    for model_name, model_path in model_paths.items():
        model_name_alpha = list(filter(lambda char: char.isalpha(), model_name))
        if model_name_alpha:
            model_name = model_name[: model_name.find(model_name_alpha[0])]

        pred_data = np.load(os.path.join(data_path, model_path, set_name + "_pred.npy"))
        pred[model_name] = dict()
        for slice in slice_range:
            slice_indices = targets["slices"] == slice
            pred[model_name][slice] = {
                "true": targets["true"][slice_indices],
                "pred": pred_data[slice_indices],
                "eye": targets["eye"][slice_indices],
                "scanID": targets["scanID"][slice_indices],
            }
    return pred


def load_predictions_nuth(
    data_path, model_paths, targets, set_name="test", all_slices=False
):
    pred = dict()
    scan_ids = np.unique(targets["scanID"])

    if not all_slices:
        max_bin = NUTH_SLICE_MAX + NUTH_SLICE_STEP // 2 + 1
        slice_bins = np.arange(-max_bin, max_bin + 1, NUTH_SLICE_STEP)
        out_slices = np.arange(-NUTH_SLICE_MAX, NUTH_SLICE_MAX + 1, NUTH_SLICE_STEP)

    missing_eye_count = 0
    imputed_ratios = list()
    for model_name, model_path in model_paths.items():
        pred_data = np.load(os.path.join(data_path, model_path, set_name + "_pred.npy"))
        pred_imputed = defaultdict(list)

        for eye in [True, False]:
            eye_idx = targets["eye"] == eye
            for scan_id in scan_ids:
                scan_id_idx = targets["scanID"] == scan_id
                scan_idx = np.logical_and(eye_idx, scan_id_idx)
                slices_c = targets["slices"][scan_idx]
                pred_c = pred_data[scan_idx]

                if slices_c.shape[0] == 0:
                    missing_eye_count += 1
                    continue

                sorted_idx = np.argsort(slices_c)
                slices_c = slices_c[sorted_idx]
                pred_c = pred_c[sorted_idx]

                if not all_slices:
                    out_pred = np.full(out_slices.shape, np.mean(pred_c))
                    is_missing = np.zeros(out_slices.shape, dtype=bool)
                    digitized = np.digitize(slices_c, slice_bins, right=True)
                    for i in range(1, slice_bins.shape[0]):
                        slice_idx = digitized == i
                        if slice_idx.any():
                            out_pred[i - 1] = np.mean(pred_c[slice_idx])
                        else:
                            is_missing[i - 1] = True
                    imputed_ratios.append(np.sum(is_missing) / out_slices.shape[0])
                else:
                    unique_slices = np.unique(slices_c)
                    if unique_slices.shape[0] < slices_c.shape[0]:
                        # Average multiple predictions for the same slices
                        unique_pred = np.zeros(unique_slices.shape[0])
                        for i, slice_value in enumerate(unique_slices):
                            unique_pred[i] = np.mean(pred_c[slices_c == slice_value])
                        slices_c = unique_slices
                        pred_c = unique_pred

                    if slices_c.shape[0] < slices_c[-1] - slices_c[0] + 1:
                        # Impute missing slice predictions using the mean of all slice predictions
                        out_slices = np.arange(slices_c[0], slices_c[-1] + 1)
                        mean_pred = np.mean(pred_c)
                        out_pred = np.repeat(mean_pred, out_slices.shape[0])
                        exist_idx = np.isin(out_slices, slices_c)
                        out_pred[exist_idx] = pred_c
                        imputed_ratios.append(
                            (out_slices.shape[0] - slices_c.shape[0])
                            / out_slices.shape[0]
                        )
                    else:
                        out_pred = pred_c
                        out_slices = slices_c
                        imputed_ratios.append(0.0)

                    if out_slices.shape[0] < NUTH_SLICE_LIMIT:
                        continue

                out_true = np.repeat(targets["true"][scan_idx][0], out_slices.shape[0])
                pred_imputed["pred"].append(out_pred)
                pred_imputed["slices"].append(out_slices)
                pred_imputed["true"].append(out_true)
                pred_imputed["eye"].append(np.repeat(eye, out_slices.shape[0]))
                pred_imputed["scanID"].append(np.repeat(scan_id, out_slices.shape[0]))

        for key in pred_imputed.keys():
            pred_imputed[key] = np.concatenate(pred_imputed[key])  # type: ignore

        slice_range: np.ndarray = (
            np.unique(pred_imputed["slices"]) if all_slices else out_slices
        )
        pred[model_name] = dict()
        for slice in slice_range:
            slice_indices = pred_imputed["slices"] == slice
            pred[model_name][slice] = {
                "true": pred_imputed["true"][slice_indices],
                "pred": pred_imputed["pred"][slice_indices],
                "eye": pred_imputed["eye"][slice_indices],
                "scanID": pred_imputed["scanID"][slice_indices],
            }
    print("Missing eye count:", missing_eye_count)
    print("Average imputed ratio:", np.mean(imputed_ratios))
    print("Max imputed ratio:", np.amax(imputed_ratios))
    print("95% percentile imputed ratio:", np.percentile(imputed_ratios, 95))
    print()
    return pred


def filter_missing_slices(pred, match_pred, central_model="+0"):
    target_slices = list(match_pred[central_model].keys())
    scan_ids = list()
    for slice in target_slices:
        scan_ids.append(pred[central_model][slice]["scanID"])
    scan_ids = np.concatenate(scan_ids)

    unique_ids, unique_counts = np.unique(scan_ids, return_counts=True)
    max_count = np.amax(unique_counts)
    count_matches = np.logical_or(
        unique_counts == max_count, unique_counts == max_count // 2
    )
    match_ids = unique_ids[count_matches]
    match_idx = dict()
    for slice in target_slices:
        match_idx[slice] = np.isin(pred[central_model][slice]["scanID"], match_ids)

    out_pred = dict()
    for model_name in pred.keys():
        out_pred[model_name] = dict()
        for slice in target_slices:
            out_pred[model_name][slice] = {
                "true": pred[model_name][slice]["true"][match_idx[slice]],
                "pred": pred[model_name][slice]["pred"][match_idx[slice]],
                "eye": pred[model_name][slice]["eye"][match_idx[slice]],
                "scanID": pred[model_name][slice]["scanID"][match_idx[slice]],
            }
    return out_pred


def get_age_weights(patient_ages):
    bins = np.arange(
        np.floor(np.amin(patient_ages)), np.ceil(np.amax(patient_ages) + 1), 1.0
    )
    inds = np.digitize(patient_ages, bins)

    weights = np.zeros(patient_ages.shape[0])
    for i in np.unique(inds):
        bin_matches = inds == i
        bin_indices = np.flatnonzero(bin_matches)
        bin_weight = 1 / len(bin_indices)
        weights[bin_indices] = bin_weight
    weights /= np.sum(weights)
    return weights


def combine_predictions(pred_1, pred_2, ratio: float | None = 1.0):
    pred_out = dict()
    ratio_ids = None
    for model_name in pred_1.keys():
        pred_out[model_name] = dict()
        for slice in pred_1[model_name].keys():
            if slice in pred_2[model_name]:
                if ratio is not None:
                    current_ids = pred_2[model_name][slice]["scanID"]
                    if ratio_ids is None:
                        rng = np.random.default_rng(555)
                        out_size = int(
                            pred_1[model_name][slice]["true"].shape[0] * ratio
                        )
                        age_weights = get_age_weights(pred_2[model_name][slice]["true"])
                        ratio_ids = rng.choice(
                            current_ids, out_size, replace=False, p=age_weights
                        )
                    ratio_idx = np.isin(current_ids, ratio_ids)
                    pred_2_true = pred_2[model_name][slice]["true"][ratio_idx]
                    pred_2_pred = pred_2[model_name][slice]["pred"][ratio_idx]
                    pred_2_eye = pred_2[model_name][slice]["eye"][ratio_idx]
                    pred_2_scanID = pred_2[model_name][slice]["scanID"][ratio_idx]
                else:
                    pred_2_true = pred_2[model_name][slice]["true"]
                    pred_2_pred = pred_2[model_name][slice]["pred"]
                    pred_2_eye = pred_2[model_name][slice]["eye"]
                    pred_2_scanID = pred_2[model_name][slice]["scanID"]

                pred_out[model_name][slice] = {
                    "true": np.concatenate(
                        (pred_1[model_name][slice]["true"], pred_2_true)
                    ),
                    "pred": np.concatenate(
                        (pred_1[model_name][slice]["pred"], pred_2_pred)
                    ),
                    "eye": np.concatenate(
                        (pred_1[model_name][slice]["eye"], pred_2_eye)
                    ),
                    "scanID": np.concatenate(
                        (pred_1[model_name][slice]["scanID"], pred_2_scanID)
                    ),
                }
    return pred_out


def slice_score(pred):
    scores = defaultdict(list)

    for model_name, model_results in pred.items():
        if model_results is None:
            continue

        for slice, slice_results in model_results.items():
            slice_mae = mean_absolute_error(
                slice_results["true"], slice_results["pred"]
            )
            slice_rmse = root_mean_squared_error(
                slice_results["true"], slice_results["pred"]
            )
            slice_r2 = r2_score(slice_results["true"], slice_results["pred"])

            scores["Model"].append(model_name)
            scores["Slice"].append(slice)
            scores["MAE (years)"].append(slice_mae)
            scores["RMSE (years)"].append(slice_rmse)
            scores["R2"].append(slice_r2)
    scores = pd.DataFrame.from_dict(scores)
    return scores


def plot_slice_scores(scores, ylim=None, fovea_rel=True, sort=True, palette="viridis"):
    slice_str = "Fovea slice offset" if fovea_rel else "Slice"
    plot_df = scores.rename(columns={"Slice": slice_str})
    if sort:
        plot_df = plot_df.sort_values(by="Model", key=lambda col: col.astype("int32"))
    plt.figure(figsize=(8, 5))
    ax = sns.lineplot(
        plot_df, x=slice_str, y="MAE (years)", hue="Model", palette=palette
    )
    sns.move_legend(ax, "upper right")
    plt.tight_layout()
    if ylim is not None:
        plt.ylim(ylim)


def get_slice_weights(pred, metric="MAE", inverted=True):
    weights = defaultdict(dict)

    for model_name, model_results in pred.items():
        for slice, slice_results in model_results.items():
            if metric == "MAE":
                slice_score = mean_absolute_error(
                    slice_results["true"], slice_results["pred"]
                )
            elif metric == "RMSE":
                slice_score = root_mean_squared_error(
                    slice_results["true"], slice_results["pred"]
                )
            elif metric == "R2":
                slice_score = r2_score(slice_results["true"], slice_results["pred"])
            else:
                raise ValueError("Unknown metric")

            if inverted:
                slice_score = 1 / slice_score

            weights[model_name][slice] = slice_score
    return weights


def get_slice_str(slice):
    return str(slice) if slice < 0 else "+" + str(slice)


def slice_combine_split(pred_in, weights_in=None, central_model="+0"):
    pred_out = dict()
    weights_out = dict()
    model_slices = np.asarray(list(map(int, pred_in.keys())))
    slice_range = np.asarray(list(pred_in[central_model].keys()))
    for slice in slice_range:
        match_index = np.argmin(np.abs(slice - model_slices))
        match_slice = model_slices[match_index]
        match_str = get_slice_str(match_slice)
        pred_out[slice] = pred_in[match_str][slice]
        if weights_in is not None:
            weights_out[slice] = weights_in[match_str][slice]

    if weights_in is None:
        return pred_out
    else:
        return pred_out, weights_out


def slice_combine_all(pred_in, weights_in=None, central_model="+0"):
    pred_out = dict()
    weights_out = dict()
    models_sorted = list(map(get_slice_str, sorted(map(int, pred_in.keys()))))
    for slice in pred_in[central_model].keys():
        slice_pred = list()
        slice_weights = list()
        for model in models_sorted:
            slice_pred.append(pred_in[model][slice]["pred"])
            if weights_in is not None:
                slice_weights.append(weights_in[model][slice])
        slice_pred = np.stack(slice_pred, axis=0)
        pred_out[slice] = {
            "pred": slice_pred,
            "true": pred_in[central_model][slice]["true"],
            "eye": pred_in[central_model][slice]["eye"],
            "scanID": pred_in[central_model][slice]["scanID"],
        }
        if weights_in is not None:
            weights_out[slice] = np.stack(slice_weights, axis=0)

    if weights_in is None:
        return pred_out
    else:
        return pred_out, weights_out


def slice_ensemble(
    pred_all,
    weights,
    type="mean",
    stride=1,
    step_size=1,
    single_model=None,
    split_models=True,
):
    central_model = "+0" if FOVEA_REL else "64"
    use_all_pred = single_model is None and not split_models
    if single_model is not None:  # Only use predictions from the foveal slice model
        pred = pred_all[single_model]
        weights_c = weights[single_model]
    else:
        if split_models:  # Assign separate models to each slice
            pred, weights_c = slice_combine_split(pred_all, weights, central_model)
        else:  # Use predictions from all models for all slices
            pred, weights_c = slice_combine_all(pred_all, weights, central_model)

    ens_pred = dict()
    ens_axis = (0, 1) if use_all_pred else 0
    for i in range(1, 1 + len(pred), 2 * stride):
        slice_offset = i // 2
        slice_center = 0 if FOVEA_REL else 64

        range_pred = list()
        range_weights = list()
        for j in range(0, i, stride):
            slice = j - slice_offset + slice_center
            slice *= step_size
            range_pred.append(pred[slice]["pred"])
            range_weights.append(weights_c[slice])

        try:
            range_pred = np.stack(range_pred, axis=0)
        except ValueError:
            break

        if i > 1 or use_all_pred:
            range_weights = np.stack(range_weights, axis=0)
            range_weights -= np.amin(range_weights)
            range_weights /= np.amax(range_weights)
        else:
            range_weights = np.ones((1))

        if type == "mean":
            range_pred = np.mean(range_pred, axis=ens_axis)
        elif type == "weighted":
            if single_model is None and not split_models:
                range_pred = np.average(
                    range_pred.reshape(-1, range_pred.shape[-1]),
                    weights=range_weights.ravel(),
                    axis=0,
                )
            else:
                range_pred = np.average(range_pred, weights=range_weights, axis=0)
        else:
            raise ValueError("'type' does not match any known values")
        ens_pred[i] = {
            "pred": range_pred,
            "true": pred[0]["true"],
            "eye": pred[0]["eye"],
            "scanID": pred[0]["scanID"],
        }
    return ens_pred


def filter_unpaired_eyes(data):
    unpaired_data = dict()
    for slice, slice_data in data.items():
        eye_id = slice_data["scanID"]
        id_unique, id_count = np.unique(eye_id, return_counts=True)
        multi_match = eye_id[..., None] == id_unique[id_count != 2][None, ...]
        multi_idx = np.logical_not(multi_match.any(axis=1))

        slice_unpaired_data = dict()
        for key, value in slice_data.items():
            if len(value.shape) == 1:  # If using split models
                slice_unpaired_data[key] = value[multi_idx]
            else:
                slice_unpaired_data[key] = value[:, multi_idx]
        unpaired_data[slice] = slice_unpaired_data
    return unpaired_data


def ml_slice_train(
    train, test, stride, ens_name, model, split, pair, step_size, slice_count
):
    slice_offset = slice_count // 2
    slice_center = 0 if FOVEA_REL else 64

    x_data = {"train": list(), "test": list()}

    if not pair:
        train_y = train[0]["true"]
        test_y = test[0]["true"]
        test_scan_ids = test[0]["scanID"]
        for i in range(0, slice_count, stride):
            slice = i - slice_offset + slice_center
            slice *= step_size
            x_data["train"].append(train[slice]["pred"])
            x_data["test"].append(test[slice]["pred"])
    else:
        train_y = train[0]["true"][train[0]["eye"]]
        test_y = test[0]["true"][test[0]["eye"]]
        test_scan_ids = test[0]["scanID"][test[0]["eye"]]
        for i in range(0, slice_count, stride):
            slice = i - slice_offset + slice_center
            slice *= step_size
            for set_name in x_data.keys():
                c_data = train if set_name == "train" else test

                right_idx = c_data[slice]["eye"]
                left_idx = np.logical_not(c_data[slice]["eye"])

                if split:
                    x_data[set_name].append(c_data[slice]["pred"][right_idx])
                    x_data[set_name].append(c_data[slice]["pred"][left_idx])
                else:
                    x_data[set_name].append(c_data[slice]["pred"][:, right_idx])
                    x_data[set_name].append(c_data[slice]["pred"][:, left_idx])

    if train_y.shape[0] == 0 or test_y.shape[0] == 0:
        return None

    try:
        train_x = np.stack(x_data["train"], axis=1)
        test_x = np.stack(x_data["test"], axis=1)
    except ValueError:
        return None

    if len(train_x.shape) == 3:
        train_x = train_x.reshape(-1, train_y.shape[0]).transpose()
        test_x = test_x.reshape(-1, test_y.shape[0]).transpose()

    scaler = StandardScaler().fit(train_x)
    train_x: np.NDArray = scaler.transform(train_x)
    test_x: np.NDArray = scaler.transform(test_x)

    if model == "lasso":
        rgr = LassoCV(max_iter=10000, random_state=555, n_jobs=1)
    elif model == "ridge":
        rgr = RidgeCV(alphas=np.linspace(1, 1000, num=100), cv=5)
    elif model == "rf":
        rgr = RandomForestRegressor(n_estimators=10, n_jobs=-1, random_state=555)
    elif model == "gb":
        rgr = HistGradientBoostingRegressor(l2_regularization=1000.0, random_state=555)
    else:
        raise ValueError("The 'model' parameter has no match")

    rgr.fit(train_x, train_y)
    pred_y = rgr.predict(test_x)
    pred_out = {
        "pred": pred_y,
        "true": test_y,
        "eye": None if pair else test[0]["eye"],
        "scanID": test_scan_ids,
    }

    split_str = "split" if split else "all"
    pair_str = "_pair" if pair else ""
    stride_str = f"_{stride}s" if stride > 1 else ""
    with open(
        f"data/{ens_name}/params/{model}_{split_str}{pair_str}{stride_str}_{slice_count}.bin",
        "wb",
    ) as fh:
        pickle.dump({"model": rgr, "scaler": scaler}, fh)
    return pred_out


def ml_slice_ensemble(
    train_in,
    test_in,
    ens_name,
    model="lasso",
    stride=1,
    step_size=1,
    split=True,
    pair=False,
):
    central_model = "+0" if FOVEA_REL else "64"

    if split:
        train = slice_combine_split(train_in, central_model=central_model)
        test = slice_combine_split(test_in, central_model=central_model)
    else:
        train = slice_combine_all(train_in, central_model=central_model)
        test = slice_combine_all(test_in, central_model=central_model)

    if pair:
        train = filter_unpaired_eyes(train)
        test = filter_unpaired_eyes(test)

    ens_pred = dict()
    if split:
        first_slice = stride * 2 + 1
    else:
        first_slice = 1

    if model == "rf":
        for i in range(first_slice, 1 + len(train), 2 * stride):
            print("Progress:", i, "/", 1 + len(train))
            ml_pred = ml_slice_train(
                train, test, stride, ens_name, model, split, pair, step_size, i
            )
            if ml_pred is not None:
                ens_pred[i] = ml_pred
    else:
        pool = mp.Pool(NUM_THREADS)
        process_f = partial(
            ml_slice_train, train, test, stride, ens_name, model, split, pair, step_size
        )
        slice_range = range(first_slice, 1 + len(train), 2 * stride)
        chunksize = max(len(train) // NUM_THREADS, 1)
        slice_map = pool.map(process_f, slice_range, chunksize)
        for slice, slice_pred in zip(slice_range, slice_map):
            if slice_pred is not None:
                ens_pred[slice] = slice_pred
    return ens_pred


def plot_slice_ensemble(scores, metric="MAE (years)"):
    plot_df = scores.rename(columns={"Slice": "Slice count"})
    plt.figure(figsize=(8, 6))
    ax = sns.lineplot(plot_df, x="Slice count", y=metric, hue="Model")
    sns.move_legend(ax, "upper left")
    plt.tight_layout()


def save_results(data, ensemble, name):
    path = os.path.join(f"data/{ensemble}/bin", name + ".dat")
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_results(name, ensemble):
    path = os.path.join(f"data/{ensemble}/bin", name + ".dat")
    try:
        with open(path, "rb") as fh:
            data = pickle.load(fh)
    except FileNotFoundError:
        return None
    return data


def main():
    args = parse_args()
    type_u = f"{args.type[0].upper()}{args.type[1:]}"
    plot_eval_str = f"_{args.eval_dataset}" if args.eval_dataset is not None else ""
    plot_title = f"{args.model}{plot_eval_str}"
    data_path = f"data/{args.model}/slice_eval"
    os.makedirs(f"data/{args.model}/bin", exist_ok=True)
    os.makedirs(f"data/{args.model}/params", exist_ok=True)
    sns.set_theme()
    model_paths = parse_models(data_path, args.dataset, args.eval_dataset)

    if args.dataset.find("nuth") == -1:
        print("Loading test data...")
        targets = load_targets(data_path, fovea_rel=FOVEA_REL, set_name="test")
        pred = load_predictions(data_path, model_paths, targets, set_name="test")
        slice_step_size = 1

        if not args.plot and not args.slice_analysis:
            print("Loading training data...")
            targets_val = load_targets(data_path, fovea_rel=FOVEA_REL, set_name="val")
            pred_val = load_predictions(
                data_path, model_paths, targets_val, set_name="val"
            )
    else:
        print("Loading test data...")
        anno_data = load_anno_nuth(os.path.join(data_path, "anno.csv"))
        targets = load_targets_nuth(data_path, anno_data, set_name="test")
        pred = load_predictions_nuth(data_path, model_paths, targets, set_name="test")
        slice_step_size = NUTH_SLICE_STEP

        if not args.plot and not args.slice_analysis:
            print("Loading training data...")
            targets_val = load_targets_nuth(data_path, anno_data, set_name="val")
            pred_val = load_predictions_nuth(
                data_path, model_paths, targets_val, set_name="val"
            )

    if not args.plot and not args.slice_analysis and args.extra_train is not None:
        print("Loading extra training data...")
        model_paths_x = parse_models(data_path, args.dataset, args.extra_train)
        set_name_x = f"val_{args.extra_train}"
        targets_val_x = load_targets(
            data_path, fovea_rel=FOVEA_REL, set_name=set_name_x
        )
        pred_val_x = load_predictions(
            data_path, model_paths_x, targets_val_x, set_name="val"
        )
        pred_val_x = filter_missing_slices(pred_val_x, pred_val)
        pred_val = combine_predictions(pred_val, pred_val_x, ratio=1.0)

    if args.slice_analysis:
        # Generate per slice results
        scores = slice_score(pred)
        plt.ylim()
        plot_slice_scores(scores, fovea_rel=FOVEA_REL)
        if args.save:
            plt.savefig(f"plots/slice_analysis/{plot_title}_slices.png", dpi=350)
        else:
            plt.show()
        exit(0)

    ens_pred = dict()
    if not args.plot:  # Train ensemble models
        print("\nBeginning ensemble training...")
        # ML models
        for stride in [1]:  # , 2, 3
            for split_type in ["Split", "All"]:
                split = split_type == "Split"
                stride_str = "" if stride == 1 else f" {stride}s"
                model_title = f"{split_type} {type_u}{stride_str}"

                ens_pred[model_title] = ml_slice_ensemble(
                    pred_val,
                    pred,
                    args.model,
                    model=args.type,
                    stride=stride,
                    step_size=slice_step_size,
                    split=split,
                )

                if stride == 1:
                    model_str = f"{split_type.lower()}_{args.type}"
                else:
                    model_str = f"{split_type.lower()}_{args.type}_{stride}s"
                save_results(ens_pred[model_title], args.model, model_str)
                print(model_str, "done")

        # ML Models Paired Eyes
        for stride in [1]:
            for split_type in ["Split", "All"]:
                split = split_type == "Split"
                stride_str = "" if stride == 1 else f" {stride}s"
                model_title = f"{split_type} {type_u} Pair{stride_str}"

                ens_pred[model_title] = ml_slice_ensemble(
                    pred_val,
                    pred,
                    args.model,
                    model=args.type,
                    stride=stride,
                    step_size=slice_step_size,
                    split=split,
                    pair=True,
                )

                if stride == 1:
                    model_str = f"{split_type.lower()}_{args.type}_pair"
                else:
                    model_str = f"{split_type.lower()}_{args.type}_pair_{stride}s"
                save_results(ens_pred[model_title], args.model, model_str)
                print(model_str, "done")
        print("Ensemble training completed")
    else:
        if args.basic:
            weights = get_slice_weights(pred, inverted=True)
            ens_pred["+0"] = slice_ensemble(
                pred, weights, single_model="+0", type="mean", step_size=slice_step_size
            )
            ens_pred["Split"] = slice_ensemble(
                pred, weights, type="mean", split_models=True, step_size=slice_step_size
            )
            ens_pred["All"] = slice_ensemble(
                pred,
                weights,
                type="mean",
                split_models=False,
                step_size=slice_step_size,
            )

            # Weighted models
            # ens_pred["+0 W"] = slice_ensemble(pred, weights, single_model="+0", type="weighted")

            # Stride models
            # ens_pred["+0 2S"] = slice_ensemble(
            #     pred, weights, single_model="+0", stride=2, type="mean"
            # )

        # Load results
        ens_pred[f"Split {type_u}"] = load_results(f"split_{args.type}", args.model)
        ens_pred[f"Split {type_u} 2s"] = load_results(
            f"split_{args.type}_2s", args.model
        )
        ens_pred[f"Split {type_u} 3s"] = load_results(
            f"split_{args.type}_3s", args.model
        )
        ens_pred[f"All {type_u}"] = load_results(f"all_{args.type}", args.model)
        ens_pred[f"All {type_u} 2s"] = load_results(f"all_{args.type}_2s", args.model)
        ens_pred[f"All {type_u} 3s"] = load_results(f"all_{args.type}_3s", args.model)

        ens_pred[f"Split {type_u} Pair"] = load_results("split_ridge_pair", args.model)
        ens_pred[f"All {type_u} Pair"] = load_results("all_ridge_pair", args.model)

        # Plot final ensemble results
        ens_scores = slice_score(ens_pred)
        plot_slice_ensemble(ens_scores)
        if args.save:
            plt.savefig(
                f"plots/slice_analysis/{plot_title}_{args.type}_ens_mae.png", dpi=350
            )
        else:
            plt.show()

        plot_slice_ensemble(ens_scores, metric="R2")
        if args.save:
            plt.savefig(
                f"plots/slice_analysis/{plot_title}_{args.type}_ens_r2.png", dpi=350
            )
        else:
            plt.show()


if __name__ == "__main__":
    main()
