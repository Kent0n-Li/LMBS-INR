"""
Gradio-based Landmark Matching Visualization Tool

This script provides an interactive web interface for:
1. Loading fixed and moving 3D medical images
2. Running feature-based landmark matching
3. Visualizing matched landmarks on different slices
"""

import os
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import gradio as gr
from PIL import Image
import io

# Import matcher loader
from matcher_loader import load_matcher

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Global variables to store loaded data and results
global_state = {
    'fixed_data': None,
    'moving_data': None,
    'matcher': None,
    'matcher_type': None,
    'device': None,
    'matches': None,  # Store matching results
}


def load_images(fixed_path, moving_path):
    """Load fixed and moving images"""
    try:
        # Load fixed image
        if fixed_path is None:
            fixed_path = os.path.join(SCRIPT_DIR, 'data', 'reference_image.nii.gz')

        fixed_nib = nib.load(fixed_path)
        fixed_data = fixed_nib.get_fdata()

        # Handle 4D images
        if len(fixed_data.shape) == 4:
            fixed_data = fixed_data[..., 0]

        # Load moving image
        if moving_path is None:
            moving_path = os.path.join(SCRIPT_DIR, 'data', 'AP_dwi.nii.gz')

        moving_nib = nib.load(moving_path)
        moving_data = moving_nib.get_fdata()

        # Handle 4D images
        if len(moving_data.shape) == 4:
            moving_data = moving_data[..., 0]

        # Normalize for visualization
        fixed_data = (fixed_data - np.min(fixed_data)) / (np.max(fixed_data) - np.min(fixed_data) + 1e-8)
        moving_data = (moving_data - np.min(moving_data)) / (np.max(moving_data) - np.min(moving_data) + 1e-8)

        # Store in global state
        global_state['fixed_data'] = fixed_data
        global_state['moving_data'] = moving_data
        global_state['matches'] = None  # Reset matches when loading new images

        info = f"Fixed image shape: {fixed_data.shape}\nMoving image shape: {moving_data.shape}"

        return info, gr.update(maximum=fixed_data.shape[2]-1, value=fixed_data.shape[2]//2)

    except Exception as e:
        return f"Error loading images: {str(e)}", gr.update()


def initialize_matcher(matcher_type, gpu_id, match_threshold):
    """Initialize the feature matcher"""
    try:
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        global_state['device'] = device

        # Only reload if matcher type changed
        if global_state['matcher'] is None or global_state['matcher_type'] != matcher_type:
            print(f"Loading {matcher_type} matcher on {device}...")
            matcher = load_matcher(matcher_type, device=device, thr=match_threshold)
            global_state['matcher'] = matcher
            global_state['matcher_type'] = matcher_type

        return f"Matcher '{matcher_type}' initialized on {device}"
    except Exception as e:
        return f"Error initializing matcher: {str(e)}"


def extract_2d_matches_single_slice(fixed_slice, moving_slice, matcher, conf_threshold):
    """Extract 2D matches from a single slice pair"""
    # Normalize to [0, 255] for matching
    fixed_slice = np.clip(fixed_slice * 255, 0, 255).astype(np.uint8)
    moving_slice = np.clip(moving_slice * 255, 0, 255).astype(np.uint8)

    with torch.no_grad():
        match_res = matcher(fixed_slice, moving_slice)

    mkpts0 = match_res['mkpts0']
    mkpts1 = match_res['mkpts1']
    mconf = match_res['mconf']

    if len(mkpts0) > 0:
        # Filter by confidence
        mask_conf = mconf > conf_threshold
        mkpts0 = mkpts0[mask_conf]
        mkpts1 = mkpts1[mask_conf]
        mconf = mconf[mask_conf]

    return mkpts0, mkpts1, mconf


def run_matching(matcher_type, gpu_id, match_threshold, conf_threshold, run_all_slices, orientations):
    """Run landmark matching on loaded images"""
    if global_state['fixed_data'] is None or global_state['moving_data'] is None:
        return "Please load images first!"

    # Initialize matcher
    init_msg = initialize_matcher(matcher_type, gpu_id, match_threshold)
    if "Error" in init_msg:
        return init_msg

    matcher = global_state['matcher']
    fixed_data = global_state['fixed_data']
    moving_data = global_state['moving_data']

    # Resize moving to match fixed for matching (simple approach)
    from scipy.ndimage import zoom as scipy_zoom

    zoom_factors = (
        fixed_data.shape[0] / moving_data.shape[0],
        fixed_data.shape[1] / moving_data.shape[1],
        fixed_data.shape[2] / moving_data.shape[2]
    )
    moving_resampled = scipy_zoom(moving_data, zoom_factors, order=1)

    x_dim, y_dim, z_dim = fixed_data.shape

    all_matches = {
        'axial': {'mkpts0': [], 'mkpts1': [], 'mconf': [], 'slice_idx': []},
        'coronal': {'mkpts0': [], 'mkpts1': [], 'mconf': [], 'slice_idx': []},
        'sagittal': {'mkpts0': [], 'mkpts1': [], 'mconf': [], 'slice_idx': []},
    }

    total_matches = 0
    total_slices = x_dim + y_dim + z_dim if run_all_slices else 0

    # 1. Axial slices (xy plane, varying z)
    if "Axial" in orientations:
        axial_step = 1 if run_all_slices else max(1, z_dim // 10)
        print(f"Processing Axial slices (step={axial_step})...")

        for slice_idx in range(0, z_dim, axial_step):
            fixed_slice = fixed_data[:, :, slice_idx]
            moving_slice = moving_resampled[:, :, slice_idx]

            if np.sum(fixed_slice) > 0:
                mkpts0, mkpts1, mconf = extract_2d_matches_single_slice(
                    fixed_slice, moving_slice, matcher, conf_threshold
                )
                if len(mkpts0) > 0:
                    all_matches['axial']['mkpts0'].append(mkpts0)
                    all_matches['axial']['mkpts1'].append(mkpts1)
                    all_matches['axial']['mconf'].append(mconf)
                    all_matches['axial']['slice_idx'].append(slice_idx)
                    total_matches += len(mkpts0)

    # 2. Coronal slices (xz plane, varying y)
    if "Coronal" in orientations:
        coronal_step = 1 if run_all_slices else max(1, y_dim // 10)
        print(f"Processing Coronal slices (step={coronal_step})...")

        for slice_idx in range(0, y_dim, coronal_step):
            fixed_slice = fixed_data[:, slice_idx, :]
            moving_slice = moving_resampled[:, slice_idx, :]

            if np.sum(fixed_slice) > 0:
                mkpts0, mkpts1, mconf = extract_2d_matches_single_slice(
                    fixed_slice, moving_slice, matcher, conf_threshold
                )
                if len(mkpts0) > 0:
                    all_matches['coronal']['mkpts0'].append(mkpts0)
                    all_matches['coronal']['mkpts1'].append(mkpts1)
                    all_matches['coronal']['mconf'].append(mconf)
                    all_matches['coronal']['slice_idx'].append(slice_idx)
                    total_matches += len(mkpts0)

    # 3. Sagittal slices (yz plane, varying x)
    if "Sagittal" in orientations:
        sagittal_step = 1 if run_all_slices else max(1, x_dim // 10)
        print(f"Processing Sagittal slices (step={sagittal_step})...")

        for slice_idx in range(0, x_dim, sagittal_step):
            fixed_slice = fixed_data[slice_idx, :, :]
            moving_slice = moving_resampled[slice_idx, :, :]

            if np.sum(fixed_slice) > 0:
                mkpts0, mkpts1, mconf = extract_2d_matches_single_slice(
                    fixed_slice, moving_slice, matcher, conf_threshold
                )
                if len(mkpts0) > 0:
                    all_matches['sagittal']['mkpts0'].append(mkpts0)
                    all_matches['sagittal']['mkpts1'].append(mkpts1)
                    all_matches['sagittal']['mconf'].append(mconf)
                    all_matches['sagittal']['slice_idx'].append(slice_idx)
                    total_matches += len(mkpts0)

    # Store results
    global_state['matches'] = all_matches
    global_state['moving_resampled'] = moving_resampled

    # Count matches per orientation
    axial_count = sum(len(m) for m in all_matches['axial']['mkpts0'])
    coronal_count = sum(len(m) for m in all_matches['coronal']['mkpts0'])
    sagittal_count = sum(len(m) for m in all_matches['sagittal']['mkpts0'])

    result = f"Matching completed!\n"
    result += f"Total matches: {total_matches}\n"
    result += f"  - Axial: {axial_count} matches in {len(all_matches['axial']['slice_idx'])} slices\n"
    result += f"  - Coronal: {coronal_count} matches in {len(all_matches['coronal']['slice_idx'])} slices\n"
    result += f"  - Sagittal: {sagittal_count} matches in {len(all_matches['sagittal']['slice_idx'])} slices"

    return result


def visualize_matches(orientation, slice_selector, show_lines):
    """Visualize matches on a specific slice"""
    if global_state['fixed_data'] is None:
        return None, "Please load images first!"

    if global_state['matches'] is None:
        return None, "Please run matching first!"

    fixed_data = global_state['fixed_data']
    moving_resampled = global_state.get('moving_resampled', global_state['moving_data'])
    matches = global_state['matches']

    slice_idx = int(slice_selector)

    # Get the appropriate slice based on orientation
    if orientation == "Axial":
        orient_key = 'axial'
        max_idx = fixed_data.shape[2] - 1
        slice_idx = min(slice_idx, max_idx)
        fixed_slice = fixed_data[:, :, slice_idx]
        moving_slice = moving_resampled[:, :, slice_idx]
    elif orientation == "Coronal":
        orient_key = 'coronal'
        max_idx = fixed_data.shape[1] - 1
        slice_idx = min(slice_idx, max_idx)
        fixed_slice = fixed_data[:, slice_idx, :]
        moving_slice = moving_resampled[:, slice_idx, :]
    else:  # Sagittal
        orient_key = 'sagittal'
        max_idx = fixed_data.shape[0] - 1
        slice_idx = min(slice_idx, max_idx)
        fixed_slice = fixed_data[slice_idx, :, :]
        moving_slice = moving_resampled[slice_idx, :, :]

    # Find matching results for this slice or closest slice
    orient_matches = matches[orient_key]
    slice_indices = orient_matches['slice_idx']

    mkpts0, mkpts1, mconf = None, None, None

    if len(slice_indices) > 0:
        # Find closest slice with matches
        closest_idx = min(range(len(slice_indices)), key=lambda i: abs(slice_indices[i] - slice_idx))
        if abs(slice_indices[closest_idx] - slice_idx) <= 5:  # Within 5 slices
            mkpts0 = orient_matches['mkpts0'][closest_idx]
            mkpts1 = orient_matches['mkpts1'][closest_idx]
            mconf = orient_matches['mconf'][closest_idx]

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Fixed image with keypoints
    # Note: imshow with .T transposes the image, so we need to swap x,y for scatter
    axes[0].imshow(fixed_slice.T, cmap='gray', origin='lower')
    axes[0].set_title(f'Fixed Image ({orientation} slice {slice_idx})')
    if mkpts0 is not None and len(mkpts0) > 0:
        colors = plt.cm.jet(mconf / mconf.max()) if len(mconf) > 0 else 'red'
        # Swap coordinates due to transpose: [:, 1] for x-axis, [:, 0] for y-axis
        axes[0].scatter(mkpts0[:, 1], mkpts0[:, 0], c=colors, s=20, marker='o')
    axes[0].axis('off')

    # Moving image with keypoints
    axes[1].imshow(moving_slice.T, cmap='gray', origin='lower')
    axes[1].set_title(f'Moving Image ({orientation} slice {slice_idx})')
    if mkpts1 is not None and len(mkpts1) > 0:
        colors = plt.cm.jet(mconf / mconf.max()) if len(mconf) > 0 else 'red'
        # Swap coordinates due to transpose: [:, 1] for x-axis, [:, 0] for y-axis
        axes[1].scatter(mkpts1[:, 1], mkpts1[:, 0], c=colors, s=20, marker='o')
    axes[1].axis('off')

    # Side-by-side comparison with lines
    # Transpose slices first to match the display in axes[0] and axes[1]
    fixed_slice_T = fixed_slice.T
    moving_slice_T = moving_slice.T

    h1, w1 = fixed_slice_T.shape
    h2, w2 = moving_slice_T.shape

    # Create combined image (horizontal concatenation after transpose)
    combined = np.zeros((max(h1, h2), w1 + w2))
    combined[:h1, :w1] = fixed_slice_T
    combined[:h2, w1:w1+w2] = moving_slice_T

    axes[2].imshow(combined, cmap='gray', origin='lower')
    axes[2].set_title(f'Matched Landmarks (n={len(mkpts0) if mkpts0 is not None else 0})')

    if mkpts0 is not None and len(mkpts0) > 0:
        # Draw keypoints (swap x,y because we transposed the image)
        colors = plt.cm.jet(mconf / mconf.max()) if len(mconf) > 0 else ['red'] * len(mkpts0)
        axes[2].scatter(mkpts0[:, 1], mkpts0[:, 0], c=colors, s=20, marker='o')
        axes[2].scatter(mkpts1[:, 1] + w1, mkpts1[:, 0], c=colors, s=20, marker='o')

        # Draw connecting lines
        if show_lines:
            for i in range(len(mkpts0)):
                color = plt.cm.jet(mconf[i] / mconf.max()) if len(mconf) > 0 else 'red'
                axes[2].plot([mkpts0[i, 1], mkpts1[i, 1] + w1],
                           [mkpts0[i, 0], mkpts1[i, 0]],
                           color=color, linewidth=0.5, alpha=0.7)

    axes[2].axis('off')

    plt.tight_layout()

    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    plt.close()

    img = Image.open(buf)

    info = f"Orientation: {orientation}, Slice: {slice_idx}"
    if mkpts0 is not None:
        info += f"\nMatches displayed: {len(mkpts0)}"
        if len(mconf) > 0:
            info += f"\nConfidence range: [{mconf.min():.3f}, {mconf.max():.3f}]"
    else:
        info += "\nNo matches found for this slice"

    return img, info


def update_slice_range(orientation):
    """Update slice slider range based on orientation"""
    if global_state['fixed_data'] is None:
        return gr.update()

    fixed_data = global_state['fixed_data']

    if orientation == "Axial":
        max_val = fixed_data.shape[2] - 1
        default_val = fixed_data.shape[2] // 2
    elif orientation == "Coronal":
        max_val = fixed_data.shape[1] - 1
        default_val = fixed_data.shape[1] // 2
    else:  # Sagittal
        max_val = fixed_data.shape[0] - 1
        default_val = fixed_data.shape[0] // 2

    return gr.update(maximum=max_val, value=default_val)


def create_demo():
    """Create the Gradio interface"""

    with gr.Blocks(title="Landmark Matching Visualization") as demo:
        gr.Markdown("# Landmark Matching Visualization Tool")
        gr.Markdown("Load 3D medical images and visualize feature-based landmark matching results.")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 1. Load Images")
                fixed_path = gr.Textbox(
                    label="Fixed Image Path",
                    value=os.path.join(SCRIPT_DIR, 'data', 'reference_image.nii.gz'),
                    placeholder="Path to fixed image (.nii.gz)"
                )
                moving_path = gr.Textbox(
                    label="Moving Image Path",
                    value=os.path.join(SCRIPT_DIR, 'data', 'AP_dwi.nii.gz'),
                    placeholder="Path to moving image (.nii.gz)"
                )
                load_btn = gr.Button("Load Images", variant="primary")
                load_info = gr.Textbox(label="Load Status", lines=2)

                gr.Markdown("### 2. Matching Settings")
                matcher_type = gr.Dropdown(
                    choices=['sp_lg', 'loftr', 'roma'],
                    value='sp_lg',
                    label="Matcher Type"
                )
                gpu_id = gr.Number(value=0, label="GPU ID", precision=0)
                match_threshold = gr.Slider(0.0, 1.0, value=0.2, label="Match Threshold")
                conf_threshold = gr.Slider(0.0, 1.0, value=0.4, label="Confidence Threshold")

                run_all_slices = gr.Checkbox(value=True, label="Run All Slices (every slice)")
                orientations = gr.CheckboxGroup(
                    choices=["Axial", "Coronal", "Sagittal"],
                    value=["Axial"],
                    label="Orientations to Match"
                )

                match_btn = gr.Button("Run Matching", variant="primary")
                match_info = gr.Textbox(label="Matching Results", lines=5)

            with gr.Column(scale=2):
                gr.Markdown("### 3. Visualization")
                with gr.Row():
                    orientation = gr.Radio(
                        choices=["Axial", "Coronal", "Sagittal"],
                        value="Axial",
                        label="Orientation"
                    )
                    show_lines = gr.Checkbox(value=True, label="Show Connecting Lines")

                slice_slider = gr.Slider(0, 100, value=50, step=1, label="Slice Index")

                visualize_btn = gr.Button("Update Visualization", variant="secondary")

                output_image = gr.Image(label="Landmark Visualization", type="pil")
                vis_info = gr.Textbox(label="Visualization Info", lines=3)

        # Event handlers
        load_btn.click(
            fn=load_images,
            inputs=[fixed_path, moving_path],
            outputs=[load_info, slice_slider]
        )

        match_btn.click(
            fn=run_matching,
            inputs=[matcher_type, gpu_id, match_threshold, conf_threshold, run_all_slices, orientations],
            outputs=[match_info]
        )

        visualize_btn.click(
            fn=visualize_matches,
            inputs=[orientation, slice_slider, show_lines],
            outputs=[output_image, vis_info]
        )

        # Auto-update slice range when orientation changes
        orientation.change(
            fn=update_slice_range,
            inputs=[orientation],
            outputs=[slice_slider]
        )

        # Auto-visualize when slice changes
        slice_slider.release(
            fn=visualize_matches,
            inputs=[orientation, slice_slider, show_lines],
            outputs=[output_image, vis_info]
        )

        # Auto-visualize when show_lines changes
        show_lines.change(
            fn=visualize_matches,
            inputs=[orientation, slice_slider, show_lines],
            outputs=[output_image, vis_info]
        )

    return demo


if __name__ == '__main__':
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
