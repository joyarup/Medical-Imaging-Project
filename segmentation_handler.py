import os
import numpy as np
import pydicom
import highdicom
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence
import matplotlib.pyplot as plt

def rearrange_segmentation(ct_series, seg_file):
    """
    Rearrange the segmentation to match the CT series based on headers
    
    Parameters:
    -----------
    ct_series : list
        List of DICOM datasets from CT series
    seg_file : str
        Path to the segmentation DICOM file
        
    Returns:
    --------
    numpy.ndarray
        3D array with segmentation aligned to CT volume
    """
    # Load segmentation file
    seg_dcm = pydicom.dcmread(seg_file)
    
    # Get CT volume dimensions
    ct_positions = []
    for ct in ct_series:
        try:
            # Try to get position from Image Position Patient
            pos = ct.ImagePositionPatient[2]
        except (AttributeError, KeyError, IndexError):
            try:
                # Try to get position from functional groups sequence
                pos = ct.PerFrameFunctionalGroupsSequence[0].PlanePositionSequence[0].ImagePositionPatient[2]
            except (AttributeError, KeyError, IndexError):
                try:
                    # Fall back to slice location
                    pos = ct.SliceLocation
                except (AttributeError, KeyError):
                    # If we can't determine position, use index as fallback
                    pos = ct_series.index(ct)
        
        ct_positions.append((ct, pos))
    
    # Sort CT series by position
    ct_positions.sort(key=lambda x: x[1])
    ct_series_sorted = [item[0] for item in ct_positions]
    
    # Create empty segmentation volume matched to CT dimensions
    rows = ct_series_sorted[0].Rows
    cols = ct_series_sorted[0].Columns
    seg_volume = np.zeros((len(ct_series_sorted), rows, cols), dtype=np.uint8)
    
    # Extract segmentation pixel data
    if seg_dcm.Modality == 'SEG':
        seg_array = seg_dcm.pixel_array
        
        # Check if this is a multi-frame segmentation
        if len(seg_array.shape) == 3:
            num_frames = seg_array.shape[0]
            
            # Map each segmentation frame to the correct CT slice
            for frame_idx in range(num_frames):
                try:
                    # Get frame data
                    frame_group = seg_dcm.PerFrameFunctionalGroupsSequence[frame_idx]
                    
                    # Try to get segment number
                    segment_number = frame_group.SegmentIdentificationSequence[0].ReferencedSegmentNumber
                    
                    # Get referenced SOP instance UID to map to correct CT slice
                    referenced_instance_uid = None
                    try:
                        # Try to get from derivation sequence
                        deriv_seq = frame_group.DerivationImageSequence
                        for deriv_item in deriv_seq:
                            source_seq = deriv_item.SourceImageSequence
                            for source_item in source_seq:
                                referenced_instance_uid = source_item.ReferencedSOPInstanceUID
                                break
                            if referenced_instance_uid:
                                break
                    except (AttributeError, KeyError, IndexError):
                        pass
                    
                    if not referenced_instance_uid:
                        try:
                            # Try to get from frame content sequence
                            referenced_instance_uid = frame_group.FrameContentSequence[0].ReferencedSOPInstanceUID
                        except (AttributeError, KeyError, IndexError):
                            pass
                    
                    # If we have a referenced UID, find matching CT slice
                    ct_idx = None
                    if referenced_instance_uid:
                        for i, ct in enumerate(ct_series_sorted):
                            if ct.SOPInstanceUID == referenced_instance_uid:
                                ct_idx = i
                                break
                    
                    # If we couldn't find by UID, try to match by position
                    if ct_idx is None:
                        try:
                            # Get position from segmentation frame
                            seg_pos = frame_group.PlanePositionSequence[0].ImagePositionPatient[2]
                            
                            # Find closest CT slice by position
                            positions = [pos for _, pos in ct_positions]
                            ct_idx = min(range(len(positions)), key=lambda i: abs(positions[i] - seg_pos))
                        except (AttributeError, KeyError, IndexError):
                            # If all else fails, use frame index if it's in range
                            if frame_idx < len(ct_series_sorted):
                                ct_idx = frame_idx
                    
                    # Place segmentation data in the correct slice
                    if ct_idx is not None and ct_idx < len(seg_volume):
                        # Check if this is a tumor segment
                        is_tumor = False
                        try:
                            for segment in seg_dcm.SegmentSequence:
                                if segment.SegmentNumber == segment_number:
                                    segment_label = segment.SegmentLabel.lower()
                                    if 'tumor' in segment_label or 'tumour' in segment_label:
                                        is_tumor = True
                                        break
                        except (AttributeError, KeyError):
                            # If we can't determine, assume it might be tumor
                            is_tumor = True
                        
                        if is_tumor:
                            seg_volume[ct_idx] = np.logical_or(seg_volume[ct_idx], seg_array[frame_idx] > 0)
                
                except (AttributeError, KeyError, IndexError) as e:
                    print(f"Error processing frame {frame_idx}: {e}")
                    continue
        
        # If single-frame segmentation, place in middle slice as fallback
        elif len(seg_array.shape) == 2:
            middle_idx = len(ct_series_sorted) // 2
            seg_volume[middle_idx] = seg_array > 0
    
    return seg_volume

def visualize_segmentation(ct_volume, seg_volume, output_dir='.'):
    """
    Visualize the CT and segmentation overlay
    
    Parameters:
    -----------
    ct_volume : numpy.ndarray
        3D array of CT data
    seg_volume : numpy.ndarray
        3D array of segmentation data
    output_dir : str
        Directory to save visualization images
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Normalize CT for display
    ct_norm = (ct_volume - ct_volume.min()) / (ct_volume.max() - ct_volume.min())
    
    # Create overlay images for each slice
    for z in range(ct_volume.shape[0]):
        if np.any(seg_volume[z]):  # Only create images for slices with segmentation
            fig, ax = plt.subplots(figsize=(8, 8))
            
            # Display CT slice
            ax.imshow(ct_norm[z], cmap='gray')
            
            # Create segmentation mask overlay
            mask = np.ma.masked_where(seg_volume[z] == 0, seg_volume[z])
            ax.imshow(mask, cmap='hot', alpha=0.5)
            
            ax.set_title(f'Slice {z}')
            ax.axis('off')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'slice_{z:03d}.png'), dpi=100)
            plt.close(fig)
    
    print(f"Saved visualization slices to {output_dir}")

def visualize_3d_mip(ct_volume, seg_volume, output_file='mip_visualization.png'):
    """
    Create a Maximum Intensity Projection visualization of CT with segmentation overlay
    
    Parameters:
    -----------
    ct_volume : numpy.ndarray
        3D array of CT data
    seg_volume : numpy.ndarray
        3D array of segmentation data
    output_file : str
        Path to save the visualization image
    """
    # Normalize CT for display
    ct_norm = (ct_volume - ct_volume.min()) / (ct_volume.max() - ct_volume.min())
    
    # Create MIP of CT data
    mip_ct = np.max(ct_norm, axis=0)
    
    # Create MIP of segmentation (any segmentation along projection)
    mip_seg = np.max(seg_volume, axis=0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Display CT MIP
    ax.imshow(mip_ct, cmap='gray')
    
    # Create segmentation mask overlay
    mask = np.ma.masked_where(mip_seg == 0, mip_seg)
    ax.imshow(mask, cmap='hot', alpha=0.5)
    
    ax.set_title('Maximum Intensity Projection with Tumor Overlay')
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_file, dpi=150)
    plt.close(fig)
    
    print(f"Saved MIP visualization to {output_file}")

def main():
    # Define paths to your data
    reference_dir = '31_EQP_Ax5.00mm'
    input_dir = '21_PP_Ax5.00mm'
    
    # Function to find and load DICOM files from a directory
    def load_dicoms(directory):
        dicom_files = []
        seg_file = None
        
        for filename in os.listdir(directory):
            if filename.endswith('.dcm'):
                file_path = os.path.join(directory, filename)
                try:
                    dcm = pydicom.dcmread(file_path)
                    if hasattr(dcm, 'Modality'):
                        if dcm.Modality == 'SEG':
                            seg_file = file_path
                        else:
                            dicom_files.append(dcm)
                except:
                    continue
        
        return dicom_files, seg_file
    
    # Load reference CT series
    print("Loading reference CT series...")
    ct_files, seg_file_ref = load_dicoms(reference_dir)
    
    # Sort CT files by position
    ct_positions = []
    for ct in ct_files:
        try:
            pos = ct.ImagePositionPatient[2]
        except (AttributeError, KeyError, IndexError):
            try:
                pos = ct.SliceLocation
            except (AttributeError, KeyError):
                pos = ct_files.index(ct)
        ct_positions.append((ct, pos))
    
    ct_positions.sort(key=lambda x: x[1])
    ct_series_sorted = [item[0] for item in ct_positions]
    
    # Extract CT pixel data
    ct_volume = np.stack([ct.pixel_array for ct in ct_series_sorted])
    
    # Load input series to find segmentation if not in reference
    if not seg_file_ref:
        print("No segmentation found in reference directory, checking input directory...")
        _, seg_file_input = load_dicoms(input_dir)
        seg_file = seg_file_input
    else:
        seg_file = seg_file_ref
    
    if seg_file:
        print(f"Found segmentation file: {seg_file}")
        
        # Rearrange segmentation to match CT volume
        seg_volume = rearrange_segmentation(ct_series_sorted, seg_file)
        
        # Create visualization output directory
        vis_dir = 'visualization_output'
        os.makedirs(vis_dir, exist_ok=True)
        
        # Generate slice visualizations
        visualize_segmentation(ct_volume, seg_volume, vis_dir)
        
        # Generate MIP visualization
        visualize_3d_mip(ct_volume, seg_volume, os.path.join(vis_dir, 'mip_visualization.png'))
    else:
        print("No segmentation file found.")

if __name__ == "__main__":
    main()