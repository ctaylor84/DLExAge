#!/usr/bin/env python3

## Attempts to find the fovea slice from a stack of OCT images

import os
import argparse
import zipfile
import pickle
import io

from PIL import Image, ImageDraw
import numpy as np
from scipy.signal import find_peaks
from scipy import ndimage
import skimage.morphology as skm
import matplotlib.pyplot as plt

BLUR_SIGMA = 8


def flip_image(ims, zipFile):
    # Flip left/right (cols) if '_21017_' (left eye)
    if zipFile.find("_21017_") != -1:
        print("Flipping %s" % zipFile)
        ims = np.flip(ims, 2)  # (cols are axis=2)
    return ims


def sd_filter(ims, layer_thickness, sd_threshold):
    # replace slices whos standard deviation is below threshold
    sdPerSlice = np.std(ims, axis=(1, 2))
    lowSD_slice_idx = sdPerSlice < sd_threshold

    if len(np.nonzero(lowSD_slice_idx)[0]) > (ims.shape[0] * 0.5):
        lowSD_slice_idx = None
    else:
        # If some slices are good, replace with mean of OK slices
        layer_thickness[lowSD_slice_idx, :] = np.mean(
            layer_thickness[~lowSD_slice_idx, :], axis=0
        )

    # ims[lowSD,:,:] = np.mean(ims[~lowSD,:,:],axis=0)
    # layer_thickness[lowSD,:] = None
    return lowSD_slice_idx


def get_images(zipFile, out_folder, skip_existing=False):
    zipFile = zipFile.rstrip()
    folderBase = zipFile.split(os.path.sep)[-1]
    imageID = folderBase[: folderBase.find(".")]

    if skip_existing:
        thickness_file = os.path.join(
            out_folder,
            "thickness_%s.txt" % (imageID),
        )
        if os.path.isfile(thickness_file):
            print("Skipping existing file", imageID)
            exit(0)

    zippedImgs = zipfile.ZipFile(zipFile)
    filenames = zippedImgs.namelist()

    for i, filename in enumerate(filenames):
        # print(filename)
        data = zippedImgs.read(filename)
        dataEnc = io.BytesIO(data)
        im = np.array(Image.open(dataEnc))
        im_idx = filename.split("_")[-1].split(".")[0]
        im_idx = int(im_idx) - 1
        ## init
        if i == 0:
            imsRaw = np.zeros((len(filenames), im.shape[0], im.shape[1]))
        imsRaw[im_idx, :, :] = im

    # ims = flip_image(imsRaw, zipFile)
    return imsRaw, imageID


def calcFoveaWidth(imArr):
    # Width of foveal boundaries:
    peaks_max = find_peaks(imArr)[0]
    peaks_vals_max = imArr[peaks_max]
    if len(peaks_vals_max) < 2:
        fov_width = -1.0
    else:
        # get max peak
        maxPeakIdx = np.argmax(peaks_vals_max)
        # save idx of max peak
        maxThicknessIdx1 = peaks_max[maxPeakIdx]
        # get idx of second highed peak
        peaks_vals_max[maxPeakIdx] = -1.0
        maxThicknessIdx2 = peaks_max[np.argmax(peaks_vals_max)]
        fov_width = np.abs(maxThicknessIdx1 - maxThicknessIdx2)
    return fov_width


def saveThickness(
    args,
    imageID,
    blurImage,
    lowSD_slice_idx,
    fovea_slice_idx,
    blur_col,
    is_low_quality,
    maxSD_col,
    using_default_slice,
):
    ## ----------------
    # Save thickness
    ## ----------------
    # Thickness over all slices:
    if lowSD_slice_idx is not None:
        allThickness_blurred = np.mean(blurImage[~lowSD_slice_idx, :], axis=(0, 1))
        allThickness_blurred_std = np.std(blurImage[~lowSD_slice_idx, :], axis=(0, 1))
    else:
        allThickness_blurred = -1
        allThickness_blurred_std = -1

    blur_OCT = blurImage[fovea_slice_idx, :]
    # Thickness of selected slice:
    sliceThickness_blurred = np.mean(blur_OCT)
    sliceThickness_blurred_std = np.std(blur_OCT)

    # Thickness of COL (across slices)
    sliceThickness_blurred_col = np.mean(blur_col)
    sliceThickness_blurred_col_std = np.std(blur_col)

    # Max/Min thickness of selected slice:
    sliceThicknessMIN_blurred = np.min(blur_OCT)
    sliceThicknessMAX_blurred = np.max(blur_OCT)

    # Max/min thickness of COL (across slices):
    sliceThicknessMIN_blurred_col = np.min(blur_col)
    sliceThicknessMAX_blurred_col = np.max(blur_col)

    # Fovea width (across OCT slice, and the 'col' slice)
    fovWidth = calcFoveaWidth(blur_OCT)
    fovWidth_col = calcFoveaWidth(blur_col)

    ## Save
    thickStats = np.array(
        [
            fovea_slice_idx,
            maxSD_col,
            is_low_quality,
            using_default_slice,
            allThickness_blurred,
            allThickness_blurred_std,
            sliceThickness_blurred,
            sliceThickness_blurred_std,
            sliceThickness_blurred_col,
            sliceThickness_blurred_col_std,
            sliceThicknessMIN_blurred,
            sliceThicknessMIN_blurred_col,
            sliceThicknessMAX_blurred,
            sliceThicknessMAX_blurred_col,
            fovWidth,
            fovWidth_col,
        ]
    )
    print("Saving thickness stats:")
    print(imageID)
    print("allThickness_blurred:" + str(allThickness_blurred))
    print("sliceThicknessMAX_blurred:" + str(sliceThicknessMAX_blurred))
    print("sliceThicknessMAX_blurred_col:" + str(sliceThicknessMAX_blurred_col))
    print("sliceThicknessMIN_blurred:" + str(sliceThicknessMIN_blurred))
    print("sliceThicknessMIN_blurred_col:" + str(sliceThicknessMIN_blurred_col))
    print("fovWidth:" + str(fovWidth))
    print("fovWidth_col:" + str(fovWidth_col))

    thickness_file = os.path.join(
        args.out_folder,
        "thickness_%s.txt" % (imageID),
    )
    # print(thickness_file)
    f = open(thickness_file, "w+")
    np.savetxt(
        f,
        np.reshape(thickStats, [1, len(thickStats)]),
        fmt="%d\t%d\t%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%d\t%d",
        header="foveaSliceIdx\t"
        + "maxSD_col\t"
        + "isLowQuality\t"
        + "defaultSliceUsed\t"
        + "allThickness\t"
        + "allThickness_std\t"
        + "sliceThickness\t"
        + "sliceThickness_std\t"
        + "sliceThickness_col\t"
        + "sliceThickness_col_std\t"
        + "sliceThicknessMIN\t"
        + "sliceThicknessMIN_col\t"
        + "sliceThicknessMAX\t"
        + "sliceThicknessMAX_col\t"
        + "fovWidth\t"
        + "fovWidth_col",
    )
    f.close()


def draw_area(first_layer, second_layer, shape):
    area = Image.new("1", (shape[1], shape[0]))
    draw = ImageDraw.Draw(area)
    draw.line(list(zip(range(shape[1]), first_layer)), fill=1, width=1)

    layer_stack = np.stack((first_layer, second_layer), axis=1)
    area_start = np.amin(np.around(layer_stack), axis=1).astype(int)
    area_end = np.amax(np.around(layer_stack), axis=1).astype(int)

    # Draw lines between each segment boundary
    for j in range(shape[1]):
        draw.line([(j, area_start[j]), (j, area_end[j])], fill=1, width=1)
    return np.asarray(area, dtype=bool)


def plot_retinal_boundaries(img, upper, lower):
    plt.imshow(img, cmap="gray")
    plt.plot(upper, c="red")
    plt.plot(lower, c="blue")
    ax = plt.gca()
    ax.axis("off")
    plt.tight_layout()


def plot_retinal_area(img, upper, lower):
    retinal_area = draw_area(upper, lower, shape=img.shape).astype(float)
    plt.imshow(img, cmap="gray")
    plt.imshow(retinal_area, cmap="winter", alpha=retinal_area)
    ax = plt.gca()
    ax.axis("off")
    plt.tight_layout()


def plot_retinal_distances(img, upper, lower, min_col, max_col):
    img_crop = img[int(img.shape[0] * 0.15) : int(img.shape[0] * 0.8)]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(img_crop, cmap="gray")

    distances = np.abs(upper - lower)
    distances /= np.amax(distances)
    ax2 = ax.twinx()
    # ax2.plot(distances, c="blue")
    ax2.plot(np.arange(min_col, max_col) + 1, distances[min_col:max_col], c="red")
    ax2.vlines(np.argmin(distances), ymin=[0], ymax=img_crop.shape[0], colors="blue")
    ax2.set_ylim(0, np.amax(distances) + np.amax(distances) * 0.05)
    ax2.set_ylabel("Normalized retinal thickness")
    ax2.yaxis.set_label_position("left")
    ax2.yaxis.tick_left()
    ax2.vlines([min_col, max_col], ymin=[0], ymax=img_crop.shape[0], colors="green")

    ax.set_xlabel("Pixel column")
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()


def plot_thickness_map(img, slice_idx, col_idx):
    plt.imshow(img, aspect="auto")
    plt.plot(col_idx, slice_idx, "r+", markersize=40.0)
    plt.xlabel("Pixel column")
    plt.ylabel("Slice")
    plt.tight_layout()


def plot_thickness_map_wocts(ims, thickness_map, slice_idx, col_idx, oct_count=5):
    fig = plt.figure(layout="constrained")
    subfigs = fig.subfigures(1, 2, width_ratios=[3, 1])

    axs0 = subfigs[0].subplots(1, 1)
    axs0.imshow(thickness_map, aspect="auto")
    axs0.plot(col_idx, slice_idx, "r+", markersize=40.0)
    axs0.set_ylabel("Slice")
    axs0.set_xlabel("Pixel column")

    crop_upper = int(ims.shape[1] * 0.25)
    crop_lower = int(ims.shape[1] * 0.65)
    oct_idx = np.round(np.linspace(0, ims.shape[0] - 1, oct_count)).astype(int)

    axs1 = subfigs[1].subplots(
        oct_count + 1,
        1,
        sharex=True,
        gridspec_kw=dict(height_ratios=[1] * oct_count + [0.4], hspace=0),
    )
    for i, idx in enumerate(oct_idx):
        axs1[i].imshow(ims[idx, crop_upper:crop_lower], cmap="gray")
        axs1[i].axis("off")
    axs1[oct_count].axis("off")


def parse_args():
    parser = argparse.ArgumentParser(description="Find fovea slice from OCT stack.")
    parser.add_argument("--image-dir", dest="image_dir", type=str)
    parser.add_argument("--sd-threshold", dest="sd_threshold", default=25, type=int)
    parser.add_argument("--out-dir", dest="out_folder", default="fovea_slices")
    parser.add_argument("--use-sd", dest="use_sd", default=False, action="store_true")
    parser.add_argument(
        "--skip-existing", dest="skip_existing", default=False, action="store_true"
    )
    parser.add_argument(
        "--slice-dist",
        dest="slice_dist",
        default=0.046875,
        type=float,
        help="Distance between slices, set as -1 to remove slice restriction",
    )
    parser.add_argument(
        "--slice-dist-file",
        dest="slice_dist_file",
        default=None,
        type=str,
        help="Path to file containing distances between slices as an imageID:distance dictionary",
    )
    parser.add_argument("--plot", dest="plot", default=False, action="store_true")
    args = parser.parse_args()
    args.image_dir = os.path.expanduser(args.image_dir)
    args.out_folder = os.path.expanduser(args.out_folder)

    print(args)
    return args


def main():
    args = parse_args()

    zipFile = args.image_dir
    ims, imageID = get_images(zipFile, args.out_folder, args.skip_existing)
    slice_count = ims.shape[0]

    if args.slice_dist_file is not None:
        with open(os.path.expanduser(args.slice_dist_file), "rb") as fh:
            slice_dist_dict = pickle.load(fh)
        args.slice_dist = slice_dist_dict[imageID]

    if args.slice_dist > 0.0:
        max_slice = int(ims.shape[0] // 2 + 0.5 // args.slice_dist)
        min_slice = int(ims.shape[0] // 2 - 0.5 // args.slice_dist)
    else:
        max_slice = ims.shape[0]
        min_slice = 0

    ## Save layer thickness from each column of each image
    layer_thickness = np.zeros((ims.shape[0], ims.shape[-1]))
    front_row_pos = np.zeros((ims.shape[0], ims.shape[-1]))
    back_row_pos = np.zeros((ims.shape[0], ims.shape[-1]))

    # For each slice
    for i in range(ims.shape[0]):
        # print(i)
        y = ims[i, :, :]
        # Across the columns of each slice
        for j in range(ims.shape[2]):
            # Get distance between front and rear boundaries
            yy = y[:, j]
            yy_peaks = find_peaks(yy)[0]

            # if the slice image is uniform, no peaks will be present.
            if len(yy_peaks) > 0:
                yy_peak_vals = yy[yy_peaks]
                yy_peak_high = yy_peaks[
                    (yy_peak_vals > np.percentile(yy_peak_vals, 95))
                ]
                yy_front = yy_peak_high[0]
                yy_back = yy_peak_high[-1]
                yy_dist = np.abs(yy_front - yy_back)

                # save the thickness
                layer_thickness[i, j] = yy_dist
                front_row_pos[i, j] = yy_front
                back_row_pos[i, j] = yy_back

    # Remove slices with low SDs
    lowSD_slice_idx = sd_filter(ims, layer_thickness, args.sd_threshold)

    # Blur the thickness array
    blurImage = ndimage.gaussian_filter(layer_thickness, sigma=BLUR_SIGMA)

    # blur the Y positions of the front and rear of the retinal boundary
    front_row_pos = ndimage.gaussian_filter(front_row_pos, sigma=BLUR_SIGMA)
    back_row_pos = ndimage.gaussian_filter(back_row_pos, sigma=BLUR_SIGMA)
    # blurImage[lowSD,:] = None

    min_col = round(blurImage.shape[1] * 0.4)
    max_col = round(blurImage.shape[1] * 0.6)

    if args.use_sd:
        # Find the fundus column which maximises the SD
        # Only consider the middle 50% of slices when maximising the SD
        sd_lower_slice = slice_count // 2 - slice_count // 4 - 1
        sd_upper_slice = slice_count // 2 + slice_count // 4
        maxSD_col = np.argmax(np.std(blurImage[sd_lower_slice:sd_upper_slice, :], 0))  # type: ignore

        # Limit the selected column to within the central 20% of the image
        if maxSD_col < min_col or maxSD_col > max_col:
            maxSD_col = blurImage.shape[1] // 2 - 1
        blur_col = blurImage[:, maxSD_col]

        # Find minima
        col_peaks = find_peaks(-blur_col)[0]  # type: ignore

        # exclude peaks outside of the central slices
        withinBounds_peaks = np.logical_and(
            col_peaks >= min_slice,
            col_peaks <= max_slice,
        )
        col_peaks = col_peaks[withinBounds_peaks]
        col_peaks_vals = blur_col[col_peaks]

        using_default_slice = False
        try:
            # Selected slice is that of the global minimum
            # This may fail if no minima are found
            fovea_slice_idx = col_peaks[np.argmin(col_peaks_vals)]
        except ValueError:
            print("Using default slice")
            fovea_slice_idx = slice_count // 2 - 1
            using_default_slice = True
    else:
        # Locate the minima directly using the 2D data
        minima = skm.local_minima(blurImage, indices=True, allow_borders=False)

        using_default_slice = False
        if minima[0].shape[0] > 0:
            slice_match_slice = np.logical_and(
                minima[0] >= min_slice, minima[0] <= max_slice
            )
            slice_match_col = np.logical_and(minima[1] >= min_col, minima[1] <= max_col)
            slice_match = np.logical_and(slice_match_slice, slice_match_col)

            if slice_match.any():
                fovea_slice_idx = minima[0][slice_match][0]
                maxSD_col = minima[1][slice_match][0]
            elif slice_match_slice.any():
                fovea_slice_idx = minima[0][slice_match_slice][0]
                maxSD_col = minima[1][slice_match_slice][0]
            else:
                print("Using default slice")
                fovea_slice_idx = slice_count // 2 - 1
                maxSD_col = blurImage.shape[1] // 2 - 1
                using_default_slice = True
        else:
            print("Using default slice")
            fovea_slice_idx = slice_count // 2 - 1
            maxSD_col = blurImage.shape[1] // 2 - 1
            using_default_slice = True
        blur_col = blurImage[:, maxSD_col]

    is_low_quality = lowSD_slice_idx is None
    os.makedirs(args.out_folder, exist_ok=True)
    print("Fovea slice index (one-based):", fovea_slice_idx + 1)
    saveThickness(
        args,
        imageID,
        blurImage,
        lowSD_slice_idx,
        fovea_slice_idx,
        blur_col,
        is_low_quality,
        maxSD_col,
        using_default_slice,
    )

    if args.plot:
        central_slice = ims.shape[0] // 2
        for slice_name, slice_idx in {
            "bottom": 0,
            "central": central_slice,
            "foveal": fovea_slice_idx,
            "top": ims.shape[0] - 1,
        }.items():
            plot_retinal_distances(
                ims[slice_idx],
                front_row_pos[slice_idx],
                back_row_pos[slice_idx],
                min_col,
                max_col,
            )
            plt.savefig(
                os.path.join(args.out_folder, slice_name) + "_distances.png",
                dpi=350,
                bbox_inches="tight",
            )
            plt.close()

            plot_retinal_boundaries(
                ims[slice_idx], front_row_pos[slice_idx], back_row_pos[slice_idx]
            )
            plt.savefig(
                os.path.join(args.out_folder, slice_name) + "_boundary.png",
                dpi=350,
                bbox_inches="tight",
            )
            plt.close()

        plot_thickness_map(blurImage, fovea_slice_idx, maxSD_col)
        plt.savefig(
            os.path.join(args.out_folder, "tk_map.png"), dpi=350, bbox_inches="tight"
        )
        plt.close()

        plot_thickness_map_wocts(ims, blurImage, fovea_slice_idx, maxSD_col)
        plt.savefig(
            os.path.join(args.out_folder, "tk_map_oct.png"),
            dpi=350,
            bbox_inches="tight",
        )
        plt.savefig(
            os.path.join(args.out_folder, "tk_map_oct.eps"),
            dpi=350,
            bbox_inches="tight",
        )
        plt.close()


if __name__ == "__main__":
    main()
