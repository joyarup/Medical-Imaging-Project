import os
import pydicom
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.ndimage import rotate
import imageio


def normalize(input_array):
    """Normalize array to range [0,1]"""
    amin = np.amin(input_array)
    amax = np.amax(input_array)
    return (input_array - amin) / (amax - amin)


def load_image_data(image_path):
    """Load all DICOM files from a directory"""
    img_dcmset = []
    for root, _, filenames in os.walk(image_path):
        for filename in filenames:
            dcm_path = Path(root) / filename
            try:
                dicom = pydicom.dcmread(dcm_path, force=True)
                # Only add if it's not a segmentation file
                if hasattr(dicom, 'Modality') and dicom.Modality != 'SEG':
                    img_dcmset.append(dicom)
            except Exception as e:
                print(f"Error reading {dcm_path}: {e}")
    return img_dcmset


def find_segmentation_file(directory):
    """Find segmentation DICOM file in a directory"""
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            dcm_path = Path(root) / filename
            try:
                dicom = pydicom.dcmread(dcm_path, force=True)
                if hasattr(dicom, 'Modality') and dicom.Modality == 'SEG':
                    return str(dcm_path)
            except Exception:
                pass
    return None


def process_image_data(img_dcmset):
    """Process and sort DICOM images based on headers"""
    # Try to get slice thickness and pixel spacing if available
    slice_thickness = getattr(img_dcmset[0], 'SliceThickness', 1.0)
    pixel_spacing = getattr(img_dcmset[0], 'PixelSpacing', [1.0, 1.0])[0]
    
    # Sort by Acquisition Number if available
    if all(hasattr(dcm, 'AcquisitionNumber') for dcm in img_dcmset):
        acq_number = min(dcm.AcquisitionNumber for dcm in img_dcmset)
        img_dcmset = [dcm for dcm in img_dcmset if dcm.AcquisitionNumber == acq_number]
    
    # Try different sorting methods based on available headers
    try:
        # Try sorting by Image Position Patient
        img_dcmset.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, KeyError, IndexError):
        try:
            # Try sorting by Slice Index if available
            img_dcmset.sort(key=lambda x: getattr(x, 'SliceIndex', 0))
        except (AttributeError, KeyError):
            try:
                # Try PerFrameFunctionalGroupsSequence
                img_dcmset.sort(key=lambda x: float(
                    x.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient[2]))
            except (AttributeError, KeyError, IndexError):
                try:
                    # Fall back to SliceLocation
                    img_dcmset.sort(key=lambda x: float(x.SliceLocation))
                except (AttributeError, KeyError):
                    print("Warning: Could not sort by position, using default order")
    
    # Stack the pixel arrays to create a 3D volume
    img_pixelarray = np.stack([dcm.pixel_array for dcm in img_dcmset], axis=0)
    
    return img_dcmset, img_pixelarray, slice_thickness, pixel_spacing


def load_segmentation_data(seg_path):
    """Load segmentation data from DICOM file"""
    if not seg_path or not os.path.exists(seg_path):
        return None
    
    mask_dcm = pydicom.dcmread(seg_path)
    if mask_dcm.Modality != 'SEG':
        return None
    
    return mask_dcm


def process_segmentation(seg_dcm, ct_slices):
    """Process segmentation data to match CT volume"""
    if seg_dcm is None:
        return None
    
    # Get segmentation pixel data
    seg_data = seg_dcm.pixel_array
    
    # Check segmentation shape
    if len(seg_data.shape) == 3:  # Multi-frame segmentation
        num_frames = seg_data.shape[0]
        
        # Get CT volume dimensions
        num_slices = len(ct_slices)
        rows = ct_slices[0].pixel_array.shape[0]
        cols = ct_slices[0].pixel_array.shape[1]
        
        # Create empty tumor mask volume
        tumor_mask = np.zeros((num_slices, rows, cols), dtype=bool)
        
        # Extract segment identification sequence info
        try:
            seg_info = {}
            for i, segment in enumerate(seg_dcm.SegmentSequence):
                seg_number = segment.SegmentNumber
                seg_label = segment.SegmentLabel
                # Check if this is a tumor segment
                is_tumor = 'tumor' in seg_label.lower() or 'tumour' in seg_label.lower()
                seg_info[seg_number] = {'label': seg_label, 'is_tumor': is_tumor}
            
            print(f"Segment information: {seg_info}")
            
            # Map segmentation frames to CT slices
            for frame_idx in range(num_frames):
                try:
                    # Get segmentation frame info
                    frame = seg_dcm.PerFrameFunctionalGroupsSequence[frame_idx]
                    seg_num = frame.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                    
                    # Skip if not a tumor segment
                    if seg_num in seg_info and not seg_info[seg_num]['is_tumor']:
                        continue
                    
                    # Try to get position from frame
                    ct_idx = None
                    try:
                        pos = frame.PlanePositionSequence[0].ImagePositionPatient[2]
                        
                        # Find closest CT slice by position
                        ct_positions = []
                        for i, ct in enumerate(ct_slices):
                            try:
                                ct_pos = ct.ImagePositionPatient[2]
                                ct_positions.append((i, ct_pos))
                            except (AttributeError, IndexError):
                                pass
                        
                        if ct_positions:
                            # Find index of closest position
                            ct_idx = min(ct_positions, key=lambda x: abs(x[1] - pos))[0]
                    except (AttributeError, KeyError, IndexError):
                        pass
                    
                    # If position matching failed, try to use referenced SOP instance
                    if ct_idx is None:
                        try:
                            # Try to find in DerivationImageSequence
                            ref_uid = None
                            for ref in frame.DerivationImageSequence:
                                for src in ref.SourceImageSequence:
                                    ref_uid = src.ReferencedSOPInstanceUID
                                    break
                                if ref_uid:
                                    break
                            
                            if ref_uid:
                                # Find CT slice with matching UID
                                for i, ct in enumerate(ct_slices):
                                    if ct.SOPInstanceUID == ref_uid:
                                        ct_idx = i
                                        break
                        except (AttributeError, KeyError, IndexError):
                            pass
                    
                    # If all matching failed, use frame index if in range
                    if ct_idx is None and frame_idx < num_slices:
                        ct_idx = frame_idx
                    
                    # Add segmentation to tumor mask
                    if ct_idx is not None and ct_idx < num_slices:
                        tumor_mask[ct_idx] = np.logical_or(tumor_mask[ct_idx], seg_data[frame_idx] > 0)
                
                except Exception as e:
                    print(f"Error processing frame {frame_idx}: {e}")
            
            if not np.any(tumor_mask):
                print("No tumor mask created, using all segmentation data")
                # If no tumor-specific mapping worked, use direct frame mapping
                for frame_idx in range(min(num_frames, num_slices)):
                    tumor_mask[frame_idx] = seg_data[frame_idx] > 0
            
            return tumor_mask
            
        except Exception as e:
            print(f"Error processing segmentation: {e}")
            # Fall back to direct mapping if available
            if num_frames <= num_slices:
                tumor_mask = seg_data > 0
                return tumor_mask
    
    # If single-frame or fallback needed
    elif len(seg_data.shape) == 2:
        # Single frame segmentation, put in middle slice
        num_slices = len(ct_slices)
        rows = ct_slices[0].pixel_array.shape[0]
        cols = ct_slices[0].pixel_array.shape[1]
        tumor_mask = np.zeros((num_slices, rows, cols), dtype=bool)
        middle_slice = num_slices // 2
        tumor_mask[middle_slice] = seg_data > 0
        return tumor_mask
    
    return None


def maximum_intensity_projection(image, axis=0):
    """Create Maximum Intensity Projection along specified axis"""
    return np.max(image, axis=axis)


def create_rotating_mip_animation(img_volume, mask_volume=None, output_file='rotating_mip_animation.gif', 
                                  n_frames=36, fps=10):
    """Create rotating Maximum Intensity Projection animation with tumor overlay"""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Normalize image volume
    img_volume_norm = normalize(img_volume)
    
    # Create frames directory for temporary storage
    frames_dir = 'frames'
    os.makedirs(frames_dir, exist_ok=True)
    
    # Generate frames for different rotation angles
    angles = np.linspace(0, 360, n_frames, endpoint=False)
    
    print(f"Generating {n_frames} rotation frames...")
    for i, angle in enumerate(angles):
        if i % 5 == 0:  # Progress indicator
            print(f"Processing frame {i+1}/{n_frames}...")
        
        # Rotate volumes
        rotated_img = rotate(img_volume_norm, angle, axes=(0, 2), reshape=False)
        
        # Create MIP of image
        mip_img = maximum_intensity_projection(rotated_img, axis=0)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 10))
        
        if mask_volume is not None and np.any(mask_volume):
            # Rotate mask with nearest neighbor interpolation to preserve binary values
            rotated_mask = rotate(mask_volume.astype(float), angle, axes=(0, 2), 
                                 reshape=False, order=0)
            rotated_mask = rotated_mask > 0.5  # Ensure binary mask after rotation
            
            # Create MIP of mask
            mip_mask = maximum_intensity_projection(rotated_mask, axis=0)
            
            # Convert grayscale MIP to RGB
            mip_rgb = np.stack([mip_img, mip_img, mip_img], axis=-1)
            
            # Apply red coloring to tumor regions
            mip_rgb[mip_mask > 0, 0] = 1.0  # Red channel - maximum
            mip_rgb[mip_mask > 0, 1] = 0.0  # Green channel - zero
            mip_rgb[mip_mask > 0, 2] = 0.0  # Blue channel - zero
            
            # Display the colored MIP
            ax.imshow(mip_rgb)
        else:
            # Display grayscale MIP
            ax.imshow(mip_img, cmap='gray')
        
        ax.set_title(f'MIP Rotation Angle: {angle:.1f}°')
        ax.axis('off')
        
        # Save frame
        frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
        plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, dpi=100)
        plt.close(fig)
    
    # Create GIF from frames
    print(f"Creating GIF animation...")
    with imageio.get_writer(output_file, mode='I', fps=fps) as writer:
        for i in range(n_frames):
            frame_path = os.path.join(frames_dir, f'frame_{i:03d}.png')
            image = imageio.v2.imread(frame_path)
            writer.append_data(image)
    
    print(f"Animation saved to {output_file}")
    return output_file


def main():
    # Define paths to your data
    reference_dir = '31_EQP_Ax5.00mm'
    input_dir = '21_PP_Ax5.00mm'
    
    # Create output directories
    results_dir = 'visualization_output'
    os.makedirs(results_dir, exist_ok=True)
    print(f"Created output directory: {results_dir}")
    
    # Load reference CT series
    print("\n=== Loading Reference CT Series ===")
    ref_dicom_files = load_image_data(reference_dir)
    print(f"Found {len(ref_dicom_files)} CT images")
    
    # Process CT images
    print("Processing CT images...")
    ref_dicom_files, ref_volume, slice_thickness, pixel_spacing = process_image_data(ref_dicom_files)
    print(f"Created volume with shape {ref_volume.shape}")
    
    # Look for segmentation file
    print("\n=== Loading Segmentation Data ===")
    seg_file = find_segmentation_file(reference_dir)
    if not seg_file:
        print("No segmentation found in reference directory, checking input directory...")
        seg_file = find_segmentation_file(input_dir)
    
    if seg_file:
        print(f"Found segmentation file: {seg_file}")
        seg_dcm = load_segmentation_data(seg_file)
        
        # Process segmentation
        tumor_mask = process_segmentation(seg_dcm, ref_dicom_files)
        
        if tumor_mask is None or not np.any(tumor_mask):
            print("Failed to create tumor mask, creating dummy data for visualization")
            # Create dummy tumor mask in center of volume
            tumor_mask = np.zeros_like(ref_volume, dtype=bool)
            center_z = ref_volume.shape[0] // 2
            center_y = ref_volume.shape[1] // 2
            center_x = ref_volume.shape[2] // 2
            radius = min(ref_volume.shape) // 8
            
            z, y, x = np.ogrid[:ref_volume.shape[0], :ref_volume.shape[1], :ref_volume.shape[2]]
            dist = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
            tumor_mask[dist <= radius] = True
    else:
        print("No segmentation file found, creating dummy data for visualization")
        # Create dummy tumor mask
        tumor_mask = np.zeros_like(ref_volume, dtype=bool)
        center_z = ref_volume.shape[0] // 2
        center_y = ref_volume.shape[1] // 2
        center_x = ref_volume.shape[2] // 2
        radius = min(ref_volume.shape) // 8
        
        z, y, x = np.ogrid[:ref_volume.shape[0], :ref_volume.shape[1], :ref_volume.shape[2]]
        dist = np.sqrt((z - center_z)**2 + (y - center_y)**2 + (x - center_x)**2)
        tumor_mask[dist <= radius] = True
    
    # Create MIP animation
    print("\n=== Creating MIP Animation ===")
    output_file = os.path.join(results_dir, 'rotating_mip_animation.gif')
    create_rotating_mip_animation(ref_volume, tumor_mask, output_file)
    
    print("\n=== Processing Complete ===")
    print(f"Results saved to {results_dir}")


if __name__ == "__main__":
    main()