import os
import pydicom
import numpy as np
import argparse
import yaml
from utils import load_config, parse_arguments, load_dicom_series, alpha_fusion, visualize_MIP_per_plane, create_rotating_mip_gif
import matplotlib.pyplot as plt
from skimage import exposure

VISUALIZE = False 


def load_segmentations(segment_dir):
    """Loads segmentation data from DICOM files in a directory."""
    seg_dcms = []
    
    for f in os.listdir(segment_dir):
        file_path = os.path.join(segment_dir, f)
        seg_dcm = pydicom.dcmread(file_path)
        seg_dcms.append(seg_dcm)
                    
    if not seg_dcms:
        raise IOError(f"No valid segmentation files found in {segment_dir}")
        
    return seg_dcms


def map_segmentation_to_slices(seg_dicoms):
    valid_masks = {}
    for i, seg_dcm in enumerate(seg_dicoms):
        series_dt = seg_dcm.PerFrameFunctionalGroupsSequence
        for k in range(len(series_dt)):
            ref_ct_image = (
                series_dt[k]
                .DerivationImageSequence[0]
                .SourceImageSequence[0]
                .ReferencedSOPInstanceUID
            )
            frame_it_references = series_dt[k].DerivationImageSequence[0].SourceImageSequence[0].ReferencedFrameNumber
            if i not in valid_masks:
                valid_masks[i] = [{'seg_id': k, 'referenced_id': int(frame_it_references)}]
            else:
                valid_masks[i].append({'seg_id': k, 'referenced_id': int(frame_it_references)})
    return valid_masks
    
def create_ordered_segmentation_mask(image_3d, seg_dicoms, valid_masks):
    """Creates an ordered 3D mask volume of all segmentation masks corresponding to the CT slices."""
    reordered_seg = np.zeros_like(image_3d, dtype=np.uint8)
    print(reordered_seg.shape)
    
    for i, seg_dcm in enumerate(seg_dicoms):
        seg_3d = np.array(seg_dcm.pixel_array)
        mask = np.zeros(seg_3d[0].shape, dtype=np.uint8)
        
        for info in valid_masks[i]:
            ref_mask = seg_3d[info['seg_id']] 
            mask[ref_mask == 1] = int(i+1)
            #reordered_seg[info['referenced_id']] = mask
            current_slice = reordered_seg[info['referenced_id']]
            current_slice[ref_mask == 1] = mask[ref_mask == 1]
            reordered_seg[info['referenced_id']] = current_slice
            
    return reordered_seg


def create_fused_volume(
    img_3d_np,
    reordered_seg_np,
    objects,
    object_colors,
    pixel_len_mm,
    visualize_slices=False,
):
    """Creates a 3D volume of alpha-fused images."""
    img_segmented = []
    print(f"Number of scan images: {len(img_3d_np)}")
    print(f"Number of mask layers: {len(reordered_seg_np)}")

    for i in range(len(img_3d_np)):
        fused_slice = alpha_fusion(
            img_3d_np[i], reordered_seg_np[i], len(objects), object_colors
        )
        img_segmented.append(fused_slice)
        if visualize_slices and i % 10 == 0:  # Visualize every 10th slice
            plt.imshow(fused_slice, aspect=pixel_len_mm[1] / pixel_len_mm[2])
            plt.title(f"Alpha-fused Slice {i} on Axial Plane")
            plt.show()
    return np.array(img_segmented)

def normalize_volume(img_vol: np.ndarray, lower_pct=0, upper_pct=99.9) -> np.ndarray:
    p_low, p_high = np.percentile(img_vol, (lower_pct, upper_pct))
    img_vol_norm = exposure.rescale_intensity(img_vol, in_range=(p_low, p_high), out_range=(0, 1))
    return img_vol_norm


def apply_window(img_vol, center, width):
    """Apply a window/level filter to the 3D CT volume."""
    min_hu = center - width // 2
    max_hu = center + width // 2
    img_windowed = np.clip(img_vol, min_hu, max_hu)
    img_windowed = (img_windowed - min_hu) / (max_hu - min_hu)
    return img_windowed


def main():
    
    args = parse_arguments()
    config = load_config(args.config)

    dataset_dir = args.dataset_dir or config.get("dataset_dir")
    segment_dir = args.segment_dir or config.get("segment_dir")
    output_dir = args.output_dir or config.get(
        "output_dir", "results/animation"
    )  
    visualize_flag = args.visualize or config.get("visualize", VISUALIZE)

    if not dataset_dir or not segment_dir:
        raise ValueError(
            "Dataset directory and segmentation file path must be provided via command line or config file."
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Object names and color configuration
    objects = ["Liver", "Tumor"]
    object_colors = [(0, 1, 0), (0, 0, 1)]  # Red for Liver, Green for Tumor

    try:
        ct_slices, pixel_len_mm = load_dicom_series(dataset_dir)
        img_3d_np = np.array([f.pixel_array for f in ct_slices])
        img_3d_np = apply_window(img_3d_np, center=1000, width=3000)
        img_3d_np = normalize_volume(img_3d_np)
        if visualize_flag:
            print("Visualizing raw CT slices...")
            for i, img in enumerate(img_3d_np):
                if i % 10 == 0:  # Visualize every 10th slice
                    plt.imshow(
                        img, aspect=pixel_len_mm[1] / pixel_len_mm[2], cmap="bone"
                    )
                    plt.title(f"CT slice {i} on axial plane")
                    plt.show()

        # 2. Load Segmentation
        seg_dcm = load_segmentations(segment_dir)

        # 3. Map Segmentation to Series
        valid_masks = map_segmentation_to_slices(seg_dcm)

        # 4. Create Ordered Mask Volume
        reordered_seg = create_ordered_segmentation_mask(
            img_3d_np,
            seg_dcm,
            valid_masks
        )
        
        # 5. Create Fused Volume
        img_fused_3d = create_fused_volume(
            img_3d_np,
            reordered_seg,
            objects,
            object_colors,
            pixel_len_mm,
            visualize_slices=visualize_flag,
        )

        # 6. Visualize MIPs
        if visualize_flag:
            print("Visualizing MIPs...")
            visualize_MIP_per_plane(img_fused_3d, pixel_len_mm)
        

        # 7. Create Rotating MIP GIF for Coronal-Sagittal Planes
        print("Creating rotating MIP GIF for coronal-sagittal planes...")
        
        # Import the new function (add this to your imports at the top)
        from utils import create_rotating_mip_gif
        
        # Create the GIF using the fused volume
        gif_output_path = os.path.join(output_dir, "rotating_coronal_sagittal_mip.gif")
        
        create_rotating_mip_gif(
            img_fused_3d,
            pixel_len_mm,
            output_path=gif_output_path,
            n_frames=36,  # 36 frames for smooth rotation
            equalize=True,
            method='adaptive',
            clip_limit=0.03,
            interval=200  # 200ms per frame
        )

        

    except (FileNotFoundError, ValueError, IOError) as e:
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
