"""
Gradio visualization interface for INR-based DWI distortion correction.

This app provides an interactive interface for:
1. Visualizing results (before/after, DVF)
2. Exploring slice-by-slice views
"""

import gradio as gr
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def load_nifti(filepath):
    """Load a NIfTI file and return the data array."""
    if filepath is None:
        return None
    nii = nib.load(filepath)
    data = nii.get_fdata()
    return data


def normalize_for_display(data):
    """Normalize data to 0-1 range for display."""
    if data is None:
        return None
    data_min = np.min(data)
    data_max = np.max(data)
    if data_max - data_min < 1e-10:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)


def create_slice_figure(moving, reference, corrected, slice_idx, orientation='axial'):
    """Create a comparison figure showing moving, reference, and corrected images."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Get slices based on orientation
    if orientation == 'axial':
        if moving is not None:
            moving_slice = moving[:, :, slice_idx] if slice_idx < moving.shape[2] else moving[:, :, moving.shape[2]//2]
        if reference is not None:
            reference_slice = reference[:, :, slice_idx] if slice_idx < reference.shape[2] else reference[:, :, reference.shape[2]//2]
        if corrected is not None:
            corrected_slice = corrected[:, :, slice_idx] if slice_idx < corrected.shape[2] else corrected[:, :, corrected.shape[2]//2]
    elif orientation == 'coronal':
        if moving is not None:
            moving_slice = moving[:, slice_idx, :] if slice_idx < moving.shape[1] else moving[:, moving.shape[1]//2, :]
        if reference is not None:
            reference_slice = reference[:, slice_idx, :] if slice_idx < reference.shape[1] else reference[:, reference.shape[1]//2, :]
        if corrected is not None:
            corrected_slice = corrected[:, slice_idx, :] if slice_idx < corrected.shape[1] else corrected[:, corrected.shape[1]//2, :]
    else:  # sagittal
        if moving is not None:
            moving_slice = moving[slice_idx, :, :] if slice_idx < moving.shape[0] else moving[moving.shape[0]//2, :, :]
        if reference is not None:
            reference_slice = reference[slice_idx, :, :] if slice_idx < reference.shape[0] else reference[reference.shape[0]//2, :, :]
        if corrected is not None:
            corrected_slice = corrected[slice_idx, :, :] if slice_idx < corrected.shape[0] else corrected[corrected.shape[0]//2, :, :]

    # Display slices
    if moving is not None:
        axes[0].imshow(normalize_for_display(moving_slice).T, cmap='gray', origin='lower')
        axes[0].set_title('Moving (AP DWI)', fontsize=12)
    else:
        axes[0].set_title('Moving (not loaded)', fontsize=12)
    axes[0].axis('off')

    if reference is not None:
        axes[1].imshow(normalize_for_display(reference_slice).T, cmap='gray', origin='lower')
        axes[1].set_title('Reference', fontsize=12)
    else:
        axes[1].set_title('Reference (not loaded)', fontsize=12)
    axes[1].axis('off')

    if corrected is not None:
        axes[2].imshow(normalize_for_display(corrected_slice).T, cmap='gray', origin='lower')
        axes[2].set_title('Corrected', fontsize=12)
    else:
        axes[2].set_title('Corrected (not available)', fontsize=12)
    axes[2].axis('off')

    plt.suptitle(f'{orientation.capitalize()} View - Slice {slice_idx}', fontsize=14)
    plt.tight_layout()

    return fig


def create_dvf_figure(dvf_data, magnitude_data, slice_idx, orientation='axial'):
    """Create a figure showing the deformation vector field."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dvf_data is None or magnitude_data is None:
        axes[0].text(0.5, 0.5, 'DVF not available', ha='center', va='center', transform=axes[0].transAxes)
        axes[1].text(0.5, 0.5, 'Magnitude not available', ha='center', va='center', transform=axes[1].transAxes)
        return fig

    # Get slices based on orientation
    if orientation == 'axial':
        if slice_idx < magnitude_data.shape[2]:
            mag_slice = magnitude_data[:, :, slice_idx]
            dvf_x = dvf_data[:, :, slice_idx, 0]
            dvf_y = dvf_data[:, :, slice_idx, 1]
        else:
            mid = magnitude_data.shape[2] // 2
            mag_slice = magnitude_data[:, :, mid]
            dvf_x = dvf_data[:, :, mid, 0]
            dvf_y = dvf_data[:, :, mid, 1]
    elif orientation == 'coronal':
        if slice_idx < magnitude_data.shape[1]:
            mag_slice = magnitude_data[:, slice_idx, :]
            dvf_x = dvf_data[:, slice_idx, :, 0]
            dvf_y = dvf_data[:, slice_idx, :, 2]
        else:
            mid = magnitude_data.shape[1] // 2
            mag_slice = magnitude_data[:, mid, :]
            dvf_x = dvf_data[:, mid, :, 0]
            dvf_y = dvf_data[:, mid, :, 2]
    else:  # sagittal
        if slice_idx < magnitude_data.shape[0]:
            mag_slice = magnitude_data[slice_idx, :, :]
            dvf_x = dvf_data[slice_idx, :, :, 1]
            dvf_y = dvf_data[slice_idx, :, :, 2]
        else:
            mid = magnitude_data.shape[0] // 2
            mag_slice = magnitude_data[mid, :, :]
            dvf_x = dvf_data[mid, :, :, 1]
            dvf_y = dvf_data[mid, :, :, 2]

    # Display magnitude
    im = axes[0].imshow(mag_slice.T, cmap='jet', origin='lower')
    axes[0].set_title('Deformation Magnitude', fontsize=12)
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Display quiver plot (subsample for clarity)
    step = max(1, min(mag_slice.shape) // 20)
    Y, X = np.mgrid[0:mag_slice.shape[0]:step, 0:mag_slice.shape[1]:step]
    U = dvf_x[::step, ::step]
    V = dvf_y[::step, ::step]

    axes[1].imshow(mag_slice.T, cmap='gray', origin='lower', alpha=0.5)
    axes[1].quiver(X, Y, U.T, V.T, color='red', scale=2, width=0.003)
    axes[1].set_title('Deformation Field (vectors)', fontsize=12)
    axes[1].axis('off')

    plt.suptitle(f'{orientation.capitalize()} View - Slice {slice_idx}', fontsize=14)
    plt.tight_layout()

    return fig


class INRVisualizer:
    """Class to manage data and visualization state."""

    def __init__(self):
        self.moving_data = None
        self.reference_data = None
        self.corrected_data = None
        self.dvf_data = None
        self.magnitude_data = None
        self.output_dir = None

    def load_default_data(self):
        """Load default data from the data directory."""
        base_dir = os.path.dirname(os.path.abspath(__file__))

        moving_path = os.path.join(base_dir, 'data', 'AP_dwi.nii.gz')
        reference_path = os.path.join(base_dir, 'data', 'reference_image.nii.gz')

        messages = []

        if os.path.exists(moving_path):
            self.moving_data = load_nifti(moving_path)
            messages.append(f"Loaded moving image: {self.moving_data.shape}")
        else:
            messages.append(f"Moving image not found: {moving_path}")

        if os.path.exists(reference_path):
            self.reference_data = load_nifti(reference_path)
            messages.append(f"Loaded reference image: {self.reference_data.shape}")
        else:
            messages.append(f"Reference image not found: {reference_path}")

        return "\n".join(messages)

    def load_results(self, output_dir):
        """Load results from the output directory."""
        self.output_dir = output_dir
        messages = []

        # Try to load corrected image
        corrected_path = os.path.join(output_dir, 'corrected_image.nii.gz')
        if os.path.exists(corrected_path):
            self.corrected_data = load_nifti(corrected_path)
            messages.append(f"Loaded corrected image: {self.corrected_data.shape}")
        else:
            messages.append("Corrected image not found")

        # Try to load DVF
        dvf_path = os.path.join(output_dir, 'deformation.nii.gz')
        if os.path.exists(dvf_path):
            self.dvf_data = load_nifti(dvf_path)
            messages.append(f"Loaded DVF: {self.dvf_data.shape}")
        else:
            messages.append("DVF not found")

        # Try to load magnitude
        magnitude_path = os.path.join(output_dir, 'deformation_magnitude.nii.gz')
        if os.path.exists(magnitude_path):
            self.magnitude_data = load_nifti(magnitude_path)
            messages.append(f"Loaded magnitude: {self.magnitude_data.shape}")
        else:
            messages.append("Magnitude not found")

        return "\n".join(messages)

    def get_max_slices(self, orientation='axial'):
        """Get the maximum number of slices for the given orientation."""
        if self.moving_data is not None:
            if orientation == 'axial':
                return self.moving_data.shape[2]
            elif orientation == 'coronal':
                return self.moving_data.shape[1]
            else:
                return self.moving_data.shape[0]
        return 100


# Global visualizer instance
visualizer = INRVisualizer()


def load_all_data_on_start():
    """Load default data and results on app start."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, 'output')

    # Load default data
    msg1 = visualizer.load_default_data()

    # Load results
    msg2 = visualizer.load_results(output_dir)

    # Get max slices and initial slice index
    max_slice = visualizer.get_max_slices('axial')
    initial_slice = max_slice // 2

    # Create initial figures
    slice_fig = create_slice_figure(
        visualizer.moving_data,
        visualizer.reference_data,
        visualizer.corrected_data,
        initial_slice,
        'axial'
    )
    dvf_fig = create_dvf_figure(
        visualizer.dvf_data,
        visualizer.magnitude_data,
        initial_slice,
        'axial'
    )

    status_msg = f"=== Default Data ===\n{msg1}\n\n=== Results ===\n{msg2}"

    return status_msg, gr.update(maximum=max_slice-1, value=initial_slice), slice_fig, dvf_fig


def update_slice_view(slice_idx, orientation):
    """Update the slice comparison view."""
    fig = create_slice_figure(
        visualizer.moving_data,
        visualizer.reference_data,
        visualizer.corrected_data,
        int(slice_idx),
        orientation
    )
    return fig


def update_dvf_view(slice_idx, orientation):
    """Update the DVF view."""
    fig = create_dvf_figure(
        visualizer.dvf_data,
        visualizer.magnitude_data,
        int(slice_idx),
        orientation
    )
    return fig


def update_all_views(slice_idx, orientation):
    """Update all views at once."""
    slice_fig = update_slice_view(slice_idx, orientation)
    dvf_fig = update_dvf_view(slice_idx, orientation)
    return slice_fig, dvf_fig


def on_orientation_change(orientation):
    """Handle orientation change."""
    max_slice = visualizer.get_max_slices(orientation)
    return gr.update(maximum=max_slice-1, value=max_slice//2)


# Create the Gradio interface
def create_interface():
    """Create the Gradio interface."""

    with gr.Blocks(title="INR DWI Distortion Correction", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # INR-based DWI Distortion Correction Visualization

        This tool provides visualization for Implicit Neural Representation (INR) based
        distortion correction for Diffusion-Weighted Imaging (DWI) data.
        """)

        # Status display
        load_status = gr.Textbox(label="Data Loading Status", lines=6, interactive=False)

        gr.Markdown("---")
        gr.Markdown("### Visualization Controls")

        with gr.Row():
            orientation_input = gr.Radio(
                choices=['axial', 'coronal', 'sagittal'],
                value='axial',
                label="Orientation"
            )
            slice_slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="Slice Index")
            update_btn = gr.Button("Update Views")

        with gr.Tabs():
            with gr.TabItem("Image Comparison"):
                slice_plot = gr.Plot(label="Moving / Reference / Corrected")

            with gr.TabItem("Deformation Field"):
                dvf_plot = gr.Plot(label="Deformation Vector Field")

        # Event handlers
        update_btn.click(
            fn=update_all_views,
            inputs=[slice_slider, orientation_input],
            outputs=[slice_plot, dvf_plot]
        )

        slice_slider.change(
            fn=update_all_views,
            inputs=[slice_slider, orientation_input],
            outputs=[slice_plot, dvf_plot]
        )

        orientation_input.change(
            fn=on_orientation_change,
            inputs=[orientation_input],
            outputs=[slice_slider]
        ).then(
            fn=update_all_views,
            inputs=[slice_slider, orientation_input],
            outputs=[slice_plot, dvf_plot]
        )

        # Auto-load data on page load
        demo.load(
            fn=load_all_data_on_start,
            outputs=[load_status, slice_slider, slice_plot, dvf_plot]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        inbrowser=True
    )
