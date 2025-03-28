import os
import argparse
import time
import pickle
from collections import defaultdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMAGE_LIST = "biobank_images.txt"
FOVEA_SLICE_DATA = "fovea_slice_stats/fovea_slice_data.pkl"
FOVEA_RANGE = {"min": 54, "max": 74}
FILTER_LISTS = {
    "hc": ["IDs-OCT-HC-male.txt", "IDs-OCT-HC-female.txt"],
    "hcnq": ["IDs-OCT-HCNQ.txt"],
    "q": ["IDs-Patel-Normal-mapped.txt"],
}
BIRTH_YEAR_FILE = "biobank-YOB.csv"
BIRTH_MONTH_FILE = "biobank-MOB.csv"
VISIT_DATES_FILE = "biobank-assessment-visits.csv"
ID_REMOVE_LIST = "id_remove_list.csv"
MEAN_OCT_DATE = {"0": "2010-03", "1": "2013-02"}
YEAR_SECONDS = 31536000
SPLIT_RATIO = {"train": 7, "val": 1, "test": 2}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate age prediction dataset")
    parser.add_argument(
        "-t",
        "--tslice",
        default="0",
        dest="tslice_str",
        type=str,
        help="Training slice/range (int_int) or offset to fovea or 'all'",
    )
    parser.add_argument(
        "-e",
        "--eslice",
        default="0",
        dest="eslice_str",
        type=str,
        help="Evaluation slice number/range (int_int) or offset to fovea or 'all'",
    )
    parser.add_argument(
        "--exact",
        default=False,
        action="store_true",
        dest="exact_slice",
        help="Use exact slices rather than offsets to the fovea",
    )
    parser.add_argument(
        "--tfilter",
        default=None,
        dest="train_filter",
        help="Filter the training set",
    )
    parser.add_argument(
        "--efilter",
        default=None,
        dest="eval_filter",
        help="Filter the evaluation set",
    )
    parser.add_argument(
        "--pair",
        default=False,
        action="store_true",
        dest="pair",
        help="Combine pairs of eyes",
    )
    parser.add_argument(
        "-n",
        "--name",
        default="bb_age",
        dest="name",
        help="Dataset name",
    )
    parser.add_argument(
        "-m",
        "--data",
        default="data",
        dest="data_path",
        help="Data path",
    )
    args = vars(parser.parse_args())
    args["data_path"] = os.path.expanduser(args["data_path"])
    args["tslice"] = parse_slices(args["tslice_str"])
    args["eslice"] = parse_slices(args["eslice_str"])
    if not args["exact_slice"]:
        if (
            args["tslice_str"] != "all"
            and args["tslice_str"][0] != "-"
            and args["tslice_str"][0] != "+"
        ):
            args["tslice_str"] = "+" + args["tslice_str"]
        if (
            args["eslice_str"] != "all"
            and args["eslice_str"][0] != "-"
            and args["eslice_str"][0] != "+"
        ):
            args["eslice_str"] = "+" + args["eslice_str"]
    return args


def parse_slices(slice_str):
    if slice_str == "all":
        slice_list = None
    elif slice_str.find("|") != -1:
        slice_parse = slice_str.split("|")
        slice_list = set(range(int(slice_parse[0]), int(slice_parse[1] + 1)))
    elif slice_str.find(",") != -1:
        slice_list = set(map(int, slice_str.split(",")))
    else:
        slice_list = {int(slice_str)}
    return slice_list


def group_splitter(patient_labels):
    rng = np.random.default_rng(555)
    patient_ids = list(patient_labels.keys())
    rng.shuffle(patient_ids)
    label_values = [patient_labels[x] for x in patient_ids]
    bins = np.arange(
        np.floor(np.amin(label_values)), np.ceil(np.amax(label_values) + 1)
    )
    inds = np.digitize(label_values, bins)
    split_ratio_total = sum(SPLIT_RATIO.values())

    split_groups = defaultdict(list)
    for i in np.unique(inds):
        bin_matches = inds == i
        bin_indices = np.flatnonzero(bin_matches)
        bin_ids = [patient_ids[x] for x in bin_indices]
        ind_count = len(bin_ids)

        split_indices = [0]
        for i, split_ratio in enumerate(SPLIT_RATIO.values()):
            split_size = round(ind_count * (split_ratio / split_ratio_total))
            split_indices.append(split_indices[-1] + split_size)
        split_indices[-1] = ind_count

        for i, split_name in enumerate(SPLIT_RATIO.keys()):
            split_ids = bin_ids[split_indices[i] : split_indices[i + 1]]
            split_groups[split_name] += split_ids
    return split_groups


def load_birth_times(data_path):
    birth_year_csv = pd.read_csv(
        os.path.join(data_path, BIRTH_YEAR_FILE),
        header=0,
        dtype={"eid": str, "34-0.0": float},
    )
    birth_month_csv = pd.read_csv(
        os.path.join(data_path, BIRTH_MONTH_FILE),
        header=0,
        dtype={"eid": str, "52-0.0": float},
    )
    birth_year_csv = birth_year_csv.dropna()
    birth_years = dict(
        zip(birth_year_csv["eid"], map(str, map(round, birth_year_csv["34-0.0"])))
    )
    birth_month_csv = birth_month_csv.dropna()
    birth_months = dict(
        zip(birth_month_csv["eid"], map(str, map(round, birth_month_csv["52-0.0"])))
    )

    birth_times = dict()
    for patient_id in birth_years.keys():
        birth_str = birth_years[patient_id] + "-" + birth_months[patient_id]
        birth_times[patient_id] = time.mktime(time.strptime(birth_str, "%Y-%m"))
    return birth_times


def load_visit_times(data_path):
    visits_csv = pd.read_csv(
        os.path.join(data_path, VISIT_DATES_FILE),
        header=0,
        dtype=str,
    )
    visit_times = dict()
    for _, row in visits_csv.iterrows():
        patient_times = dict()
        for inst in range(len(row) - 1):
            inst_str = str(inst)
            inst_row = row[f"53-{inst}.0"]
            if type(inst_row) is not str:
                patient_times[inst_str] = None
            else:
                patient_times[inst_str] = time.mktime(
                    time.strptime(inst_row, "%Y-%m-%d")
                )
        visit_times[row["eid"]] = patient_times
    return visit_times


def load_fovea_slices(data_path):
    with open(os.path.join(data_path, FOVEA_SLICE_DATA), "rb") as fh:
        file_data = pickle.load(fh)
    return file_data["idx"]


def get_images(birth_times, visit_times, data_path, slice_range, exact_slice=False):
    mean_oct_time = dict()
    for oct_instance, oct_date in MEAN_OCT_DATE.items():
        mean_oct_time[oct_instance] = time.mktime(time.strptime(oct_date, "%Y-%m"))

    patient_files = defaultdict(dict)
    age_list = list()
    with open(os.path.join(data_path, IMAGE_LIST)) as fh:
        image_files = fh.readlines()

    if not exact_slice:
        fovea_slices = load_fovea_slices(data_path)
    else:
        fovea_slices = None

    image_count = 0
    missing_fovea_count = 0
    existing_slice_count = 0
    missing_visit_time_count = 0
    for fname in image_files:
        if fname == "" or fname == "\n":
            continue

        fname = fname.rstrip()
        fn_split = fname.split("_")
        image_slice = int(fn_split[4][: fn_split[4].find(".")])
        image_id = f"{fn_split[0]}_{fn_split[1]}_{fn_split[2]}_{fn_split[3]}"

        if fovea_slices is not None:
            if image_id in fovea_slices:
                fovea_slice = fovea_slices[image_id]
                existing_slice_count += 1
            else:
                missing_fovea_count += 1
                fovea_slice = 64

            if fovea_slice < FOVEA_RANGE["min"]:
                fovea_slice = 64
            elif fovea_slice > FOVEA_RANGE["max"]:
                fovea_slice = 64
            image_slice -= fovea_slice

        if slice_range is None or image_slice in slice_range:
            patient_id = fn_split[0]

            if image_id not in patient_files[patient_id]:
                inst_num = fn_split[2]

                if visit_times[patient_id][inst_num] is not None:
                    visit_time = visit_times[patient_id][inst_num]
                else:
                    visit_time = mean_oct_time[inst_num]
                    missing_visit_time_count += 1

                patient_age_seconds = visit_time - birth_times[patient_id]
                patient_age = patient_age_seconds / YEAR_SECONDS
                age_list.append(patient_age)
                image_data = {
                    "slices": [fname],
                    "slice_val": [image_slice],
                    "age": patient_age,
                }
                patient_files[patient_id][image_id] = image_data
            else:
                patient_files[patient_id][image_id]["slices"].append(fname)
                patient_files[patient_id][image_id]["slice_val"].append(image_slice)
            image_count += 1

    if fovea_slices is not None:
        print(
            "Missing fovea slice count:", missing_fovea_count, "/", existing_slice_count
        )
    print("Image count:", image_count)
    print("Missing visit time count:", missing_visit_time_count)
    print()
    return patient_files, age_list


def pair_eyes(patient_files):
    pair_files = defaultdict(dict)
    for patient_id, patient_data in patient_files.items():
        eye_pairs = defaultdict(dict)
        for image_id, image_data in patient_data.items():
            image_id_split = image_id.split("_")
            eye = image_id_split[1]
            scan_id = (
                image_id_split[0] + "_" + image_id_split[2] + "_" + image_id_split[3]
            )
            eye_pairs[scan_id][eye] = image_data

        for scan_id, scan_eyes in eye_pairs.items():
            if len(scan_eyes) == 2:
                assert round(scan_eyes["21017"]["age"]) == round(
                    scan_eyes["21018"]["age"]
                )
                slice_dict_left = dict(
                    zip(scan_eyes["21017"]["slice_val"], scan_eyes["21017"]["slices"])
                )
                slice_dict_right = dict(
                    zip(scan_eyes["21018"]["slice_val"], scan_eyes["21018"]["slices"])
                )
                scan_slices = list()
                for slice_num in slice_dict_left.keys():
                    if slice_num in slice_dict_right:
                        slice_pair = (
                            slice_dict_left[slice_num]
                            + ":"
                            + slice_dict_right[slice_num]
                        )
                        scan_slices.append(slice_pair)
                scan_files = {
                    "slices": scan_slices,
                    "slice_val": scan_eyes["21017"]["slice_val"],
                    "age": scan_eyes["21017"]["age"],
                }
                pair_files[patient_id][scan_id] = scan_files
    return pair_files


def save_dataset(
    set_splits,
    filter_set_train,
    filter_set_eval,
    remove_set,
    patient_files,
    out_dir,
    train_slice,
    eval_slice,
):
    tslice_filter = train_slice is not None
    eslice_filter = eval_slice is not None
    for split_name, split_group in set_splits.items():
        targets = {"file": list(), "target": list()}
        train_split = split_name == "train"

        for patient_id in split_group:
            if patient_id in remove_set:
                continue

            if train_split:
                if filter_set_train is not None and patient_id not in filter_set_train:
                    continue
            else:
                if filter_set_eval is not None and patient_id not in filter_set_eval:
                    continue

            for patient_image in patient_files[patient_id].values():
                for fname, fslice in zip(
                    patient_image["slices"], patient_image["slice_val"]
                ):
                    if train_split and tslice_filter and fslice not in train_slice:
                        continue
                    elif not train_split and eslice_filter and fslice not in eval_slice:
                        continue
                    targets["file"].append(fname)
                    targets["target"].append(patient_image["age"])

        print("Out", split_name, "count:", len(targets["file"]))
        targets = pd.DataFrame.from_dict(targets)
        targets.to_csv(os.path.join(out_dir, split_name + ".csv"))


def main():
    args = parse_args()
    train_filter_str = args["train_filter"] if args["train_filter"] is not None else ""
    eval_filter_str = args["eval_filter"] if args["eval_filter"] is not None else ""
    pair_str = "_pair" if args["pair"] else ""
    out_name = (
        args["name"]
        + "_"
        + args["tslice_str"]
        + train_filter_str
        + "_"
        + args["eslice_str"]
        + eval_filter_str
        + pair_str
    )
    out_dir = os.path.join(args["data_path"], "datasets", out_name)
    if os.path.isdir(out_dir):
        print("Warning: output directory already exists")
    os.makedirs(out_dir, exist_ok=True)

    print("Loading birth times...")
    birth_times = load_birth_times(args["data_path"])
    print("Loading visit times...")
    visit_times = load_visit_times(args["data_path"])

    print("Loading filters...")
    with open(os.path.join(args["data_path"], ID_REMOVE_LIST), "r") as fh:
        remove_set = set([x.rstrip() for x in fh.readlines()])

    if args["train_filter"]:
        # Generate a set of filtered lists (group of patients to keep) in the train set
        filter_set_train = list()
        for filter_list in FILTER_LISTS[args["train_filter"]]:
            with open(os.path.join(args["data_path"], filter_list), "r") as fh:
                filter_set_train += fh.read().split("\n")
        filter_set_train = set(filter_set_train)
    else:
        filter_set_train = None

    if args["eval_filter"]:
        # Generate a set of filtered lists (group of patients to keep) in the eval set
        filter_set_eval = list()
        for filter_list in FILTER_LISTS[args["eval_filter"]]:
            with open(os.path.join(args["data_path"], filter_list), "r") as fh:
                filter_set_eval += fh.read().split("\n")
        filter_set_eval = set(filter_set_eval)
    else:
        filter_set_eval = None

    # Find and group image files belonging to each patient
    if args["tslice"] is None or args["eslice"] is None:
        slice_range = None
    else:
        slice_range = args["tslice"].union("eslice")

    print("Loading image data...")
    patient_files, age_list = get_images(
        birth_times,
        visit_times,
        args["data_path"],
        slice_range,
        args["exact_slice"],
    )

    if args["pair"]:
        print("Pairing eyes...")
        patient_files = pair_eyes(patient_files)

    print("Splitting dataset...")
    patient_age_avg = dict()
    for patient_id, patient_data in patient_files.items():
        avg_age = np.mean([x["age"] for x in patient_data.values()])
        patient_age_avg[patient_id] = avg_age

    # Visualise class data
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(12, 8))
    plt.hist(age_list, bins=len(set(map(round, age_list))), align="left")
    plt.xlabel("Age (years)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_dist.png"))

    # Split the dataset
    set_splits = group_splitter(patient_age_avg)

    # Assert that there is no overlap between the split groups
    for i, split_a in enumerate(set_splits.values()):
        for j, split_b in enumerate(set_splits.values()):
            if i != j:
                assert len(set(split_a).intersection(set(split_b))) == 0

    # Visualise split distributions
    for split_name, split_ids in set_splits.items():
        split_ages = list()
        for patient_id in split_ids:
            for patient_image in patient_files[patient_id].values():
                split_ages.append(patient_image["age"])

        print()
        print("Split", split_name, "count:", len(split_ages))
        print("Split", split_name, "mean:", np.mean(split_ages))
        print("Split", split_name, "std:", np.std(split_ages))
        print()

        plt.figure(figsize=(12, 8))
        plt.hist(split_ages, bins=len(set(map(round, split_ages))), align="left")
        plt.xlabel("Age (years)")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, split_name + "_dist.png"))

    # Save the dataset files
    print("Saving dataset...")
    save_dataset(
        set_splits,
        filter_set_train,
        filter_set_eval,
        remove_set,
        patient_files,
        out_dir,
        args["tslice"],
        args["eslice"],
    )


if __name__ == "__main__":
    main()
