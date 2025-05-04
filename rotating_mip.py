import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio
from scipy.ndimage import rotate

def create_rotating_mip_animation(volume, mask=None, output_file='rotating_mip.gif', 
                                  num_frames=36, fps=10, dpi=100):
    """
    Create a rotating Maximum Intensity Projection (MIP) animation on coronal-sagittal planes
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D array of image data
    mask : numpy.ndarray or None
        3D array of segmentation mask (same shape as volume)
    output_file : str
        Path to save the animation file
    num_frames : int
        Number of frames in the animation
    fps : int
        Frames per second in the animation
    dpi : int
        Resolution of the output animation
    
    Returns:
    --------
    str
        Path to the saved animation file
    """
    # Ensure we're working with float data for MIP calculation
    volume = volume.astype(float)
    
    # Normalize volume to [0, 1] for visualization
    volume_min = volume.min()
    volume_max = volume.max()
    volume_norm = (volume - volume_min) / (volume_max - volume_min)
    
    # Ensure mask has same shape as volume if provided
    if mask is not None:
        if mask.shape != volume.shape:
            print(f"Warning: Mask shape {mask.shape} doesn't match volume shape {volume.shape}")
            print("Skipping mask overlay")
            mask = None
    
    # Create figure for animation
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Generate frames for different rotation angles
    frames = []
    angles = np.linspace(0, 360, num_frames, endpoint=False)
    
    print(f"Generating {num_frames} rotation frames...")
    for i, angle in enumerate(angles):
        if i % 5 == 0:  # Progress indicator
            print(f"Processing frame {i+1}/{num_frames}...")
        
        # Generate MIP for this angle
        frame = generate_mip_frame(volume_norm, mask, angle)
        frames.append(frame)
    
    # Create animation
    print("Creating animation...")
    
    # Function to update the plot for each frame
    def update(frame_idx):
        ax.clear()
        ax.imshow(frames[frame_idx], cmap='gray' if mask is None else None)
        ax.set_title(f'MIP Rotation Angle: {angles[frame_idx]:.1f}°')
        ax.axis('off')
        return ax,
    
    # Create the animation
    anim = FuncAnimation(fig, update, frames=range(len(frames)), interval=1000/fps)
    
    # Save as GIF
    print(f"Saving animation to {output_file}...")
    anim.save(output_file, writer='pillow', fps=fps, dpi=dpi)
    plt.close(fig)
    
    return output_file

def generate_mip_frame(volume, mask=None, angle=0):
    """
    Generate a single Maximum Intensity Projection frame at the specified rotation angle
    
    Parameters:
    -----------
    volume : numpy.ndarray
        3D normalized array of image data
    mask : numpy.ndarray or None
        3D array of segmentation mask
    angle : float
        Rotation angle in degrees
    
    Returns:
    --------
    numpy.ndarray
        2D MIP image
    """
    # Determine rotation axes (we want to rotate in coronal-sagittal plane)
    # For this we rotate around the vertical axis (axis=1)
    rotated_volume = rotate(volume, angle, axes=(0, 2), reshape=False, order=1, mode='constant')
    
    if mask is not None:
        # Rotate mask the same way
        rotated_mask = rotate(mask.astype(float), angle, axes=(0, 2), reshape=False, order=0, mode='constant')
        
        # Create colored volume
        colored_volume = np.stack([
            volume,  # Red channel - will be brightened for tumor
            volume,  # Green channel
            volume   # Blue channel
        ], axis=-1)
        
        # Highlight tumor regions in red
        colored_volume[mask > 0, 0] = 1.0  # Max red
        colored_volume[mask > 0, 1] = 0.2  # Low green
        colored_volume[mask > 0, 2] = 0.2  # Low blue
        
        # Rotate the colored volume
        rotated_colored = rotate(colored_volume, angle, axes=(0, 2), reshape=False, order=1, mode='constant')
        
        # Compute MIP on the colored volume
        mip = np.max(rotated_colored, axis=0)
    else:
        # Compute MIP on grayscale volume
        mip = np.max(rotated_volume, axis=0)
    
    return mip

def main():
    # Define paths to your DICOM data
    reference_dir = '31_EQP_Ax5.00mm'  # Contains reference CT images
    
    # Load all DICOM files from directory
    dicom_files = []
    seg_file = None
    
    print(f"Loading DICOM files from {reference_dir}...")
    for filename in os.listdir(reference_dir):
        if filename.endswith('.dcm'):
            file_path = os.path.join(reference_dir, filename)
            try:
                dcm = pydicom.dcmread(file_path)
                # Check if this is a segmentation file
                if hasattr(dcm, 'Modality') and dcm.Modality == 'SEG':
                    seg_file = file_path
                else:
                    dicom_files.append(dcm)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
    
    # Sort DICOM files by position
    try:
        # Try sorting by Image Position Patient
        dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except (AttributeError, KeyError, IndexError):
        try:
            # Try sorting by acquisition number and slice index
            dicom_files.sort(key=lambda x: (x.AcquisitionNumber, x.SliceIndex))
        except (AttributeError, KeyError):
            try:
                # Try sorting by slice location
                dicom_files.sort(key=lambda x: float(x.SliceLocation))
            except (AttributeError, KeyError):
                print("Warning: Could not sort DICOM files by position")
    
    # Extract pixel data
    if dicom_files:
        print(f"Found {len(dicom_files)} CT images")
        
        # Get volume dimensions
        rows = dicom_files[0].Rows
        cols = dicom_files[0].Columns
        slices = len(dicom_files)
        
        # Create volume array
        volume = np.zeros((slices, rows, cols))
        
        # Fill volume with pixel data
        for i, dcm in enumerate(dicom_files):
            # Apply rescale slope and intercept if available
            pixel_array = dcm.pixel_array.astype(float)
            
            if hasattr(dcm, 'RescaleSlope') and hasattr(dcm, 'RescaleIntercept'):
                pixel_array = pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
            
            volume[i] = pixel_array
        
        # Load segmentation if available
        mask = None
        if seg_file:
            print(f"Found segmentation file: {seg_file}")
            seg_dcm = pydicom.dcmread(seg_file)
            
            # Extract segmentation data
            if seg_dcm.Modality == 'SEG':
                seg_pixel_data = seg_dcm.pixel_array
                
                # Handle different segmentation formats
                if len(seg_pixel_data.shape) == 3:
                    # Multi-frame segmentation
                    mask = np.zeros_like(volume, dtype=bool)
                    
                    # Try to rearrange frames according to headers
                    try:
                        for frame_idx in range(seg_pixel_data.shape[0]):
                            # Get segment identification
                            seg_num = seg_dcm.PerFrameFunctionalGroupsSequence[frame_idx].SegmentIdentificationSequence[0].ReferencedSegmentNumber
                            
                            # Check if this is a tumor segment
                            is_tumor = False
                            for segment in seg_dcm.SegmentSequence:
                                if segment.SegmentNumber == seg_num:
                                    if 'tumor' in segment.SegmentLabel.lower() or 'tumour' in segment.SegmentLabel.lower():
                                        is_tumor = True
                                        break
                            
                            if is_tumor:
                                # Try to get position from headers
                                slice_pos = None
                                
                                try:
                                    # Try to get image position
                                    pos = seg_dcm.PerFrameFunctionalGroupsSequence[frame_idx].PlanePositionSequence[0].ImagePositionPatient[2]
                                    
                                    # Find closest CT slice
                                    ct_positions = [float(dcm.ImagePositionPatient[2]) if hasattr(dcm, 'ImagePositionPatient') else i 
                                                  for i, dcm in enumerate(dicom_files)]
                                    
                                    # Find index of closest position
                                    slice_idx = min(range(len(ct_positions)), key=lambda i: abs(ct_positions[i] - pos))
                                    
                                    # Place segmentation data in correct slice
                                    mask[slice_idx] = np.logical_or(mask[slice_idx], seg_pixel_data[frame_idx] > 0)
                                    
                                except (AttributeError, KeyError, IndexError):
                                    # If we can't match position, just use frame index if in range
                                    if frame_idx < slices:
                                        mask[frame_idx] = np.logical_or(mask[frame_idx], seg_pixel_data[frame_idx] > 0)
                                    
                    except (AttributeError, KeyError, IndexError) as e:
                        print(f"Error rearranging segmentation: {e}")
                        # If rearrangement fails, use original frames directly
                        if seg_pixel_data.shape[0] <= slices:
                            mask[:seg_pixel_data.shape[0]] = seg_pixel_data > 0
                        else:
                            # If too many frames, use first 'slices' frames
                            mask = seg_pixel_data[:slices] > 0
                
                elif len(seg_pixel_data.shape) == 2:
                    # Single frame segmentation, place in middle slice
                    mask = np.zeros_like(volume, dtype=bool)
                    middle_slice = slices // 2
                    mask[middle_slice] = seg_pixel_data > 0
            
            if mask is None:
                print("Could not extract segmentation mask")
        
        # Create output directory
        output_dir = 'mip_output'
        os.makedirs(output_dir, exist_ok=True)
        
        # Create MIP animation
        output_file = os.path.join(output_dir, 'rotating_mip_animation.gif')
        create_rotating_mip_animation(volume, mask, output_file)
        
        print(f"Animation saved to {output_file}")
    else:
        print("No DICOM images found")

if __name__ == "__main__":
    main()