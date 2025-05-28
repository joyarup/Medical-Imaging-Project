import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import rotate, shift
from scipy.optimize import least_squares
from skimage import exposure
from utils import (
    load_config,
    load_dicom_series,
    parse_arguments,
)
from visualization import (
    load_segmentations,
    map_segmentation_to_slices,
    create_ordered_segmentation_mask)


cost_log = []  # For logging RMSE at each registration step

def min_max_normalization(vol: np.ndarray) -> np.ndarray:
    min_val = np.min(vol)
    max_val = np.max(vol)
    return (vol - min_val) / (max_val - min_val + 1e-6)

def clahe_3d(volume, clip_limit=0.03, nbins=256):
    """Apply CLAHE (adaptive histogram equalization) to a 3D volume, slice by slice."""
    enhanced = np.zeros_like(volume, dtype=np.float32)
    for i in range(volume.shape[0]):
        # Robustly rescale each slice before CLAHE
        p2, p98 = np.percentile(volume[i], (2, 98))
        img_rescaled = exposure.rescale_intensity(volume[i], in_range=(p2, p98), out_range=(0, 1))
        enhanced[i] = exposure.equalize_adapthist(img_rescaled, clip_limit=clip_limit, nbins=nbins)
    return enhanced

def display_planes(volume: np.ndarray, pixel_len_mm: list, title=''):
    depth, height, width = volume.shape
    axial_idx = depth // 2
    coronal_idx = height // 2
    sagittal_idx = width // 2
    plt.figure(figsize=(12, 5))
    plt.subplot(131)
    plt.imshow(volume[axial_idx, :, :], cmap='bone', aspect=pixel_len_mm[1]/pixel_len_mm[2])
    plt.title("Axial" + title)
    plt.subplot(132)
    plt.imshow(volume[:, coronal_idx, :], cmap='bone', aspect=pixel_len_mm[0]/pixel_len_mm[2])
    plt.title("Coronal" + title)
    plt.subplot(133)
    plt.imshow(volume[:, :, sagittal_idx], cmap='bone', aspect=pixel_len_mm[0]/pixel_len_mm[1])
    plt.title("Sagittal" + title)
    plt.tight_layout()
    plt.show()

def show_overlay(img, mask, pixel_len_mm, title=""):
    d, h, w = img.shape
    idx_ax, idx_cor, idx_sag = d // 2, h // 2, w // 2
    slices = [
        (img[idx_ax, :, :], mask[idx_ax, :, :], pixel_len_mm[1]/pixel_len_mm[2], "Axial"),
        (img[:, idx_cor, :], mask[:, idx_cor, :], pixel_len_mm[0]/pixel_len_mm[2], "Coronal"),
        (img[:, :, idx_sag], mask[:, :, idx_sag], pixel_len_mm[0]/pixel_len_mm[1], "Sagittal"),
    ]
    plt.figure(figsize=(12, 4))
    for i, (im, ma, asp, lab) in enumerate(slices):
        plt.subplot(1, 3, i+1)
        plt.imshow(im, cmap="gray", aspect=asp)
        plt.imshow(np.ma.masked_where(ma == 0, ma), cmap="spring", alpha=0.5, aspect=asp)
        plt.title(f"{lab} {title}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

def rigid_transform(volume, params):
    tx, ty, tz, rx, ry, rz = params
    shifted = shift(volume, shift=(tx, ty, tz), order=1, mode='nearest')
    rotated = rotate(shifted, angle=rx, axes=(1, 2), reshape=False, order=1, mode='nearest')
    rotated = rotate(rotated, angle=ry, axes=(0, 2), reshape=False, order=1, mode='nearest')
    rotated = rotate(rotated, angle=rz, axes=(0, 1), reshape=False, order=1, mode='nearest')
    return rotated

def ssd_cost(params, ref_volume, input_volume):
    transformed = rigid_transform(input_volume, params)
    min_shape = np.minimum(ref_volume.shape, transformed.shape)
    ref_cropped = ref_volume[:min_shape[0], :min_shape[1], :min_shape[2]]
    transf_cropped = transformed[:min_shape[0], :min_shape[1], :min_shape[2]]
    diff = ref_cropped - transf_cropped
    rmse = np.sqrt(np.mean(diff ** 2))
    cost_log.append(rmse)
    return diff.ravel()

def coregister_volumes(ref_volume, input_volume):
    init_params = np.zeros(6)
    result = least_squares(
        ssd_cost, init_params, args=(ref_volume, input_volume),
        method='lm', verbose=2, max_nfev=50
    )
    print(f"Optimized params: {result.x}")
    return result.x

def normalized_cross_correlation(vol1, vol2):
    v1 = vol1.flatten()
    v2 = vol2.flatten()
    v1_mean = v1.mean()
    v2_mean = v2.mean()
    numerator = np.sum((v1 - v1_mean) * (v2 - v2_mean))
    denominator = np.sqrt(np.sum((v1 - v1_mean)**2) * np.sum((v2 - v2_mean)**2))
    return numerator / (denominator + 1e-8)

def main():
    global cost_log
    cost_log = []

    args = parse_arguments()
    config = load_config(args.config)

    dataset_dir = args.dataset_dir or config.get("dataset_dir")
    input_dir = args.input_dir or config.get("input_dir")
    segment_dir = args.segment_dir or config.get("segment_dir")
    output_dir = args.output_dir or config.get("output_dir", "results/animation")
    visualize_flag = args.visualize or config.get("visualize", False)

    if not dataset_dir or not input_dir or not segment_dir:
        raise ValueError("Reference, input, and segment directories must be provided.")

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Load volumes
    ref_slices, ref_pixel_len_mm = load_dicom_series(dataset_dir, key_info=True)
    input_slices, input_pixel_len_mm = load_dicom_series(input_dir, key_info=True)
    ref_volume = np.array([s.pixel_array for s in ref_slices])
    input_volume = np.array([s.pixel_array for s in input_slices])

    print(f"Reference shape: {ref_volume.shape}")
    print(f"Input shape: {input_volume.shape}")

    # Load segmentation (liver mask) for the input
    seg_dicoms = load_segmentations(segment_dir)
    valid_masks = map_segmentation_to_slices(seg_dicoms)
    liver_mask = create_ordered_segmentation_mask(input_volume, seg_dicoms, valid_masks)

    # Normalize
    ref_volume_norm = min_max_normalization(ref_volume)
    input_volume_norm = min_max_normalization(input_volume)

    # CLAHE enhancement (for display)
    input_volume_clahe = clahe_3d(input_volume_norm, clip_limit=0.03)
    
    # Show input + mask (before registration, CLAHE)
    if visualize_flag:
        print("CLAHE-enhanced Input (with Liver Mask):")
        show_overlay(input_volume_clahe, liver_mask, input_pixel_len_mm, title="(Input, CLAHE Liver Mask)")

    # Registration
    print("Starting registration ...")
    optimized_params = coregister_volumes(ref_volume_norm, input_volume_norm)
    input_aligned = rigid_transform(input_volume_norm, optimized_params)

    # Align the liver mask
    liver_mask_aligned = rigid_transform(liver_mask.astype(float), optimized_params)
    liver_mask_aligned = (liver_mask_aligned > 0.5).astype(int)

    # CLAHE-enhanced registered input
    input_aligned_clahe = clahe_3d(input_aligned, clip_limit=0.03)
    
    # Show registered input + mask (CLAHE)
    if visualize_flag:
        print("CLAHE-enhanced Registered Input (with Liver Mask):")
        show_overlay(input_aligned_clahe, liver_mask_aligned, input_pixel_len_mm, title="(Registered Input, CLAHE Liver Mask)")

    # Numerically assess alignment
    min_shape = np.minimum(ref_volume_norm.shape, input_aligned.shape)
    ref_cropped = ref_volume_norm[:min_shape[0], :min_shape[1], :min_shape[2]]
    input_cropped = input_aligned[:min_shape[0], :min_shape[1], :min_shape[2]]
    mse = np.mean((ref_cropped - input_cropped) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(ref_cropped - input_cropped))
    ncc = normalized_cross_correlation(ref_cropped, input_cropped)
    print(f"Registration MSE: {mse:.5f}")
    print(f"Registration RMSE: {rmse:.5f}")
    print(f"Registration MAE: {mae:.5f}")
    print(f"Registration NCC: {ncc:.5f}")

    # Plot convergence (RMSE per iteration)
    plt.figure(figsize=(7, 4))
    plt.plot(cost_log, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel("RMSE (registration)")
    plt.title("Coregistration Convergence (RMSE per Step)")
    plt.grid()
    plt.show()

    # Plot histogram of intensity differences after registration
    diff = (ref_cropped - input_cropped).flatten()
    plt.figure(figsize=(7, 4))
    plt.hist(diff, bins=100, color='c', alpha=0.7)
    plt.xlabel("Intensity Difference (Registered)")
    plt.ylabel("Voxels")
    plt.title("Histogram of Intensity Differences After Registration")
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

