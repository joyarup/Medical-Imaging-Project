import os
import pydicom
import yaml
import argparse
from typing import List
import matplotlib
import numpy as np
from matplotlib import pyplot as plt, animation

# -------General Functions-------

def load_config(config_path="config.yaml"):
    """Loads configuration from a YAML file."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as exc:
                print(f"Error loading config file: {exc}")
                return {}
    return {}

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Process and visualize DICOM series with segmentations."
    )
    parser.add_argument(
        "--dataset_dir", type=str, help="Directory containing the DICOM series."
    )
    parser.add_argument(
        "--input_dir", type=str, help="Directory containing the DICOM series for the Input Image."
    )
    parser.add_argument(
        "--segment_dir", type=str, help="Path to the DICOM segmentation file."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the configuration YAML file.",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization of intermediate steps.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/animation",
        help="Directory to save the output animation.",
    )
    return parser.parse_args()



def load_dicom_series(dataset_dir, key_info=False):
    """Loads and sorts a DICOM series from a directory."""
    ct_slices = []
    count = 0
    pixel_len_mm = None
    num_acquisitions = 0 # Initialize num_acquisitions

    # To register the acquisition number we are using
    acquisition = -1

    for f in os.listdir(dataset_dir):
        img_path = os.path.join(dataset_dir, f)
        try:
            ct_dcm = pydicom.dcmread(img_path)
            if acquisition == -1 and hasattr(ct_dcm, "SliceLocation"): # Ensure first slice has SliceLocation
                acquisition = ct_dcm.AcquisitionNumber
                pixel_len_mm = [ct_dcm.SliceThickness] + list(ct_dcm.PixelSpacing)
                num_acquisitions = 1 # Start counting acquisitions

                if key_info:
                    print("\n--- DICOM Series Key Information ---")
                    print(f"Patient ID: {ct_dcm.get('PatientID', 'N/A')}")
                    print(f"Study Description: {ct_dcm.get('StudyDescription', 'N/A')}")
                    print(f"Series Description: {ct_dcm.get('SeriesDescription', 'N/A')}")
                    print(f"Modality: {ct_dcm.get('Modality', 'N/A')}")
                    print(f"Acquisition Number: {ct_dcm.get('AcquisitionNumber', 'N/A')}")
                    print(f"Pixel Spacing (mm): {ct_dcm.get('PixelSpacing', 'N/A')}")
                    print(f"Slice Thickness (mm): {ct_dcm.get('SliceThickness', 'N/A')}")
                    print(f"Rows: {ct_dcm.get('Rows', 'N/A')}")
                    print(f"Columns: {ct_dcm.get('Columns', 'N/A')}")
                    print("------------------------------------\n")

            slice_acquisition = ct_dcm.AcquisitionNumber
            if hasattr(ct_dcm, "SliceLocation") and slice_acquisition == acquisition:
                ct_slices.append(ct_dcm)
            elif slice_acquisition != acquisition and acquisition != -1: # Check if acquisition was set
                # Only count additional acquisitions if they differ from the first valid one
                if slice_acquisition not in [s.AcquisitionNumber for s in ct_slices]:
                     num_acquisitions += 1
                print(f"Skipping slice {f} due to different acquisition number ({slice_acquisition} vs {acquisition}).")
                count += 1
            elif not hasattr(ct_dcm, "SliceLocation"):
                 print(f"Skipping file {f} as it lacks SliceLocation attribute.")
                 count += 1


        except Exception as e:
            print(f"Could not read file {img_path}: {e}")
            count += 1

    # The sorting of the slices is based on the SliceLocation attribute
    ct_slices = sorted(ct_slices, key=lambda x: -x.SliceLocation)

    print(f"Loaded {len(ct_slices)} slices from acquisition {acquisition}.")
    if num_acquisitions > 1:
        print(f"Detected {num_acquisitions} total acquisitions in the directory.")
    print(f"Skipped {count} files.")
    return ct_slices, pixel_len_mm

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage import exposure
from typing import List

#------Visualization Functions------

def alpha_fusion(img: np.ndarray, mask: np.ndarray, n_objects: int, object_colors: List, alpha: float=0.5) -> np.ndarray:
    """ Visualize both image and mask in the same plot. """
    
    cmap = matplotlib.colormaps['bone']
    cmap2 = matplotlib.colormaps['Set1']
    norm = matplotlib.colors.Normalize(vmin=np.amin(img), vmax=np.amax(img))
    fused_slice = (
        (1-alpha) * cmap(norm(img)) +
        alpha * cmap2((mask/4)) * mask[..., np.newaxis].astype('bool')
    )

    return (fused_slice * 255).astype('uint8')


def MIP_per_plane(img_dcm: np.ndarray, axis: int = 2) -> np.ndarray:
    """ Compute the maximum intensity projection on the defined orientation. """
    return np.max(img_dcm, axis=axis)


def equalize_slice(img: np.ndarray,
                   method: str = 'adaptive',
                   clip_limit: float = 0.3,
                   nbins: int = 256) -> np.ndarray:
    """
    Apply global or adaptive histogram equalization to a single slice.

    method: 'global' or 'adaptive'
    clip_limit: only used for adaptive (CLAHE) - higher = more contrast
    nbins: number of bins for global equalization
    """
    # Robust intensity rescaling between 2nd and 98th percentiles
    p2, p98 = np.percentile(img, (2, 98))
    img_rescaled = exposure.rescale_intensity(img, in_range=(p2, p98), out_range=(0,1))
    
    if method == 'global':
        img_eq = exposure.equalize_hist(img_rescaled, nbins=nbins)
    else:
        img_eq = exposure.equalize_adapthist(img_rescaled,
                                             clip_limit=clip_limit,
                                             nbins=nbins)
    return img_eq


def visualize_MIP_per_plane(img_vol: np.ndarray,
                            pixel_len_mm: List[float],
                            equalize: bool = True,
                            method: str = 'adaptive',
                            clip_limit: float = 0.05):
    """
    Creates an MIP visualization for each of the three planes,
    optionally histogram-equalized for higher dynamic range.
    
    Parameters:
      img_vol      -- 3D volume array
      pixel_len_mm -- voxel spacing [dx, dy, dz]
      equalize     -- whether to perform histogram equalization
      method       -- 'global' or 'adaptive' (CLAHE)
      clip_limit   -- for CLAHE; try values between 0.01 and 0.1
    """
    labels = ['Axial Plane', 'Coronal Plane', 'Sagittal Plane']
    ars    = [(1,2), (0,2), (0,1)]
    
    for i in range(3):
        mip = MIP_per_plane(img_vol, axis=i).astype(np.float32)
        
        if equalize:
            mip = equalize_slice(mip, method=method, clip_limit=clip_limit)
            imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
            title_extra = ' (equalized)'
        else:
            imshow_kwargs = {'cmap': 'bone'}
            title_extra = ''
        
        plt.figure(figsize=(6,6))
        plt.imshow(mip,
                   aspect=pixel_len_mm[ars[i][0]] / pixel_len_mm[ars[i][1]],
                   **imshow_kwargs)
        plt.title(f'MIP for {labels[i]}{title_extra}')
        plt.axis('off')
        plt.show()
 
def create_rotating_mip_gif(img_vol: np.ndarray, 
                           pixel_len_mm: List[float],
                           output_path: str = "rotating_mip.gif",
                           n_frames: int = 36,
                           equalize: bool = True,
                           method: str = 'adaptive',
                           clip_limit: float = 0.03,
                           interval: int = 200):
    """
    Creates a rotating GIF animation between coronal and sagittal MIP projections.
    
    Parameters:
        img_vol: 3D volume array (Z, Y, X) - the fused volume with segmentations
        pixel_len_mm: voxel spacing [slice_thickness, pixel_spacing_y, pixel_spacing_x]
        output_path: path to save the GIF file
        n_frames: number of frames in the animation
        equalize: whether to apply histogram equalization
        method: 'global' or 'adaptive' for histogram equalization
        clip_limit: clip limit for CLAHE
        interval: interval between frames in milliseconds
    """
    try:
        from scipy.ndimage import rotate
        print("Using scipy.ndimage.rotate for 3D rotation")
        use_rotation = True
    except ImportError:
        print("scipy not available, using interpolation between views")
        use_rotation = False
    
    print(f"Creating rotating MIP animation with {n_frames} frames...")
    
    # MIP plane definitions from your code
    labels = ['Axial Plane', 'Coronal Plane', 'Sagittal Plane']
    axes = [0, 1, 2]  # Corresponding axes for MIP projection
    ars = [(1,2), (0,2), (0,1)]  # Aspect ratio indices
    
    frames = []
    
    if use_rotation:
        # Method 1: True 3D rotation (if scipy available)
        for i in range(n_frames):
            # Rotate between coronal (0°) and sagittal (180°) views
            angle = (i / n_frames) * 270  # 0 to 180 degrees
            
            # Rotate the volume around the Z-axis (axial rotation)
            rotated_vol = rotate(img_vol, angle, axes=(1, 2), reshape=False, order=1)
            
            # Create MIP projection on coronal plane (axis=1)
            mip = MIP_per_plane(rotated_vol, axis=1).astype(np.float32)
            
            # Apply histogram equalization if requested
            if equalize:
                mip = equalize_slice(mip, method=method, clip_limit=clip_limit)
                imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
            else:
                imshow_kwargs = {'cmap': 'bone' if len(mip.shape) == 2 else None}
            
            # Calculate aspect ratio for coronal view
            aspect_ratio = pixel_len_mm[0] / pixel_len_mm[2]  # slice_thickness / pixel_spacing_x
            
            # Create the frame
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            if len(mip.shape) == 3:  # RGB/RGBA image from fused volume
                ax.imshow(mip, aspect=aspect_ratio)
            else:  # Grayscale image
                ax.imshow(mip, aspect=aspect_ratio, **imshow_kwargs)
            
            ax.set_title(f'Rotating MIP - Frame {i+1}/{n_frames} (Angle: {angle:.1f}°)', 
                        fontsize=14, pad=20)
            plt.tight_layout()
            
            # Convert to image array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[:, :, :3])  # Convert RGBA to RGB
            
            plt.close(fig)
            
            if (i + 1) % 10 == 0:
                print(f"Generated frame {i+1}/{n_frames}")
    
    else:
        # Method 2: Interpolation between coronal and sagittal views
        # Pre-compute coronal and sagittal MIPs
        mip_coronal = MIP_per_plane(img_vol, axis=1).astype(np.float32)  # Coronal (Y-X)
        mip_sagittal = MIP_per_plane(img_vol, axis=2).astype(np.float32)  # Sagittal (Z-Y)
        
        # Apply equalization to both
        if equalize:
            mip_coronal = equalize_slice(mip_coronal, method=method, clip_limit=clip_limit)
            mip_sagittal = equalize_slice(mip_sagittal, method=method, clip_limit=clip_limit)
            imshow_kwargs = {'cmap': 'gray', 'vmin': 0, 'vmax': 1}
        else:
            imshow_kwargs = {'cmap': 'bone' if len(mip_coronal.shape) == 2 else None}
        
        for i in range(n_frames):
            # Smooth transition parameter (0 to 1 and back)
            t = (np.cos(2 * np.pi * i / n_frames) + 1) / 2
            
            if t > 0.5:
                # Show coronal view
                mip = mip_coronal
                aspect_ratio = pixel_len_mm[2] / pixel_len_mm[1]  # pixel_spacing_x / pixel_spacing_y
                view_name = "Coronal"
            else:
                # Show sagittal view
                mip = mip_sagittal
                aspect_ratio = pixel_len_mm[1] / pixel_len_mm[0]  # pixel_spacing_y / slice_thickness
                view_name = "Sagittal"
            
            # Create the frame
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.axis('off')
            
            if len(mip.shape) == 3:  # RGB/RGBA image from fused volume
                ax.imshow(mip, aspect=aspect_ratio)
            else:  # Grayscale image
                ax.imshow(mip, aspect=aspect_ratio, **imshow_kwargs)
            
            ax.set_title(f'{view_name} MIP - Frame {i+1}/{n_frames}', 
                        fontsize=14, pad=20)
            plt.tight_layout()
            
            # Convert to image array
            fig.canvas.draw()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            frames.append(buf[:, :, :3])  # Convert RGBA to RGB
            
            plt.close(fig)
            
            if (i + 1) % 10 == 0:
                print(f"Generated frame {i+1}/{n_frames}")
    
    # Create and save the GIF using matplotlib animation
    print(f"Saving GIF to {output_path}...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # Create animation data
    animation_data = []
    for frame in frames:
        im = ax.imshow(frame, animated=True)
        animation_data.append([im])
    
    # Create animation
    anim = animation.ArtistAnimation(fig, animation_data, interval=interval, 
                                   blit=True, repeat_delay=1000)
    
    # Save the animation
    anim.save(output_path, writer='pillow')
    plt.close(fig)
    
    print(f"Rotating MIP GIF saved successfully to {output_path}")
    return output_path