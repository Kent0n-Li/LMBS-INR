"""
INR Inference Script for DWI Distortion Correction

This script loads pre-trained model weights and applies the learned deformation
field to correct geometric distortions in DWI images.
"""

import os
import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np

from deepali.core.bspline import evaluate_cubic_bspline
from deepali.spatial.bspline import FreeFormDeformation
from deepali.core import Grid


# ============= Model Definitions =============
class MLPv1(nn.Module):
    """Multi-layer perceptron for implicit neural representation"""
    def __init__(self, input_size=3, hidden_size=512, output_size=1, dropout=0, num_layers=5):
        super(MLPv1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout = dropout
        self.num_layers = num_layers

        # Build layers
        fc_blocks = []
        for _ in range(self.num_layers):
            fc_blocks.append(nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))

        self.fc_in = nn.Linear(self.input_size, self.hidden_size)
        self.fc_out = nn.Linear(self.hidden_size, self.output_size)
        self.fc = nn.Sequential(*fc_blocks)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = F.relu(self.fc_in(x))
        x = self.dropout_layer(x)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


class InputMapping(nn.Module):
    """Fourier feature mapping for positional encoding"""
    def __init__(self, B=None, factor=1.0):
        super(InputMapping, self).__init__()
        self.B = factor * B

    def forward(self, x):
        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


# ============= Utility Functions =============
def crop_with_roi(tensor, roi_bounds):
    """
    Crop tensor using pre-computed ROI bounds.

    Args:
        tensor: (B, C, D, H, W) tensor to crop
        roi_bounds: dict with start/end indices for each dimension

    Returns:
        Cropped tensor
    """
    return tensor[:, :,
                  roi_bounds['d_start']:roi_bounds['d_end'],
                  roi_bounds['h_start']:roi_bounds['h_end'],
                  roi_bounds['w_start']:roi_bounds['w_end']]


def load_model(weights_path, device):
    """
    Load pre-trained model weights.

    Args:
        weights_path: Path to the saved model weights (.pth file)
        device: torch device to load the model to

    Returns:
        model: Loaded MLPv1 model
        input_mapper: Loaded InputMapping module
        checkpoint: Full checkpoint dictionary with metadata
    """
    print(f"Loading model weights from: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)

    # Extract config from checkpoint
    config = checkpoint['config']

    # Initialize model with saved config
    model = MLPv1(
        input_size=config['input_size'],
        output_size=config['output_size'],
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Initialize input mapper with saved B matrix
    B_gauss = checkpoint['B_gauss'].to(device)
    input_mapper = InputMapping(B=B_gauss, factor=config['ff_factor']).to(device)

    print(f"  Model loaded successfully!")
    print(f"  - Hidden size: {config['hidden_size']}")
    print(f"  - Num layers: {config['num_layers']}")
    print(f"  - Mapping size: {config['mapping_size']}")

    return model, input_mapper, checkpoint


def run_inference(moving_path, weights_path, output_dir, device, affine_source='moving'):
    """
    Run inference using pre-trained model weights.

    Args:
        moving_path: Path to moving image (AP DWI) to be corrected
        weights_path: Path to pre-trained model weights (.pth file)
        output_dir: Path to output directory
        device: torch device
        affine_source: Source for the output affine matrix ('moving' or 'reference')

    Returns:
        result_dict: Dictionary containing paths to saved files
    """
    result_dict = {
        'success': False,
        'corrected_image_path': None,
        'dvf_path': None,
        'magnitude_path': None,
    }

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and checkpoint
    model, input_mapper, checkpoint = load_model(weights_path, device)

    # Extract saved parameters from checkpoint
    stride_size = checkpoint['stride_size']
    image_size = checkpoint['image_size']
    coordinates_normal = checkpoint['coordinates_normal']
    roi_bounds = checkpoint['roi_bounds']
    ap_dwi_mean = checkpoint['ap_dwi_mean']
    ap_dwi_std = checkpoint['ap_dwi_std']

    x_dim, y_dim, z_dim = image_size

    # Compute control point grid dimensions
    X = np.arange(0, x_dim, stride_size[0])
    Y = np.arange(0, y_dim, stride_size[1])
    Z = np.arange(0, z_dim, stride_size[2])
    x_coor_dim = len(X)
    y_coor_dim = len(Y)
    z_coor_dim = len(Z)

    # Load moving image
    print(f"Loading moving image: {moving_path}")
    ap_dwi_nib = nib.load(moving_path)
    ap_dwi_data = ap_dwi_nib.get_fdata()
    affine = ap_dwi_nib.affine

    # Handle 4D images
    if len(ap_dwi_data.shape) == 4:
        print(f"  Input is 4D with shape {ap_dwi_data.shape}, extracting first volume")
        ap_dwi_data = ap_dwi_data[..., 0]

    print(f"  Moving image shape: {ap_dwi_data.shape}")

    # Store original statistics for denormalization
    original_mean = np.mean(ap_dwi_data)
    original_std = np.std(ap_dwi_data)

    # Z-normalization (use same statistics as training for consistency)
    ap_dwi_normalized = (ap_dwi_data - ap_dwi_mean) / ap_dwi_std

    ap_dwi_torch = torch.from_numpy(ap_dwi_normalized).float().unsqueeze(0).unsqueeze(0).to(device)

    # Pad images (same as training)
    pad_size = [30, 30, 30]
    min_value_ap_dwi = ap_dwi_normalized.min()

    ap_dwi_pad = F.pad(ap_dwi_torch, (pad_size[0], pad_size[0], pad_size[1], pad_size[1],
                                       pad_size[2], pad_size[2]), mode='constant', value=min_value_ap_dwi)

    # Permute for correct orientation
    ap_dwi_pad = ap_dwi_pad.permute(0, 1, 4, 3, 2)

    # Setup B-spline transformation
    grid = Grid(image_size)
    kernel = FreeFormDeformation(grid, stride=stride_size).to(device).kernel()
    transformation = evaluate_cubic_bspline

    # Run inference
    print("\nRunning inference...")
    with torch.no_grad():
        # Forward pass through model
        coor_normal = torch.from_numpy(coordinates_normal).to(device)
        data = input_mapper(coor_normal)
        pred = model(data)

        # Compute deformation field
        def_coords = torch.add(pred, coor_normal)
        def_coords = def_coords.reshape(1, x_coor_dim, y_coor_dim, z_coor_dim, 3).permute(0, 4, 1, 2, 3)
        def_coords = transformation(def_coords, stride=stride_size, shape=image_size, kernel=kernel, transpose=False)
        def_field = def_coords
        def_coords = def_coords.permute(0, 2, 3, 4, 1)

        # Compute normal field (identity mapping)
        coor_normal_grid = coor_normal.reshape(1, x_coor_dim, y_coor_dim, z_coor_dim, 3).permute(0, 4, 1, 2, 3)
        coor_normal_grid = transformation(coor_normal_grid, stride=stride_size, shape=image_size, kernel=kernel, transpose=False)
        normal_field = coor_normal_grid

        # Warp image
        align_corners = True
        warped_ap_dwi = F.grid_sample(ap_dwi_pad, def_coords, align_corners=align_corners, mode='bilinear')

        # Crop using pre-computed ROI bounds
        warped_ap_dwi = crop_with_roi(warped_ap_dwi, roi_bounds)

        # Convert to numpy
        warped_ap_dwi_np = warped_ap_dwi.squeeze().detach().cpu().numpy()

        # Denormalize back to original intensity values
        warped_ap_dwi_np = warped_ap_dwi_np * ap_dwi_std + ap_dwi_mean

        # Clip negative values to 0
        warped_ap_dwi_np = np.clip(warped_ap_dwi_np, 0, None)

        # Save corrected image
        corrected_path = os.path.join(output_dir, 'corrected_image.nii.gz')
        nib.save(nib.Nifti1Image(warped_ap_dwi_np, affine), corrected_path)
        print(f'Saved corrected image to {corrected_path}')
        print(f'  Output shape: {warped_ap_dwi_np.shape}')
        result_dict['corrected_image_path'] = corrected_path

        # ============= Save Deformation Vector Field (DVF) =============
        # Crop DVF to valid region
        dvf_field_cropped = crop_with_roi(def_field, roi_bounds)  # [1, 3, D, H, W]
        normal_field_cropped = crop_with_roi(normal_field, roi_bounds)  # [1, 3, D, H, W]

        # Compute displacement: dvf = def_field - normal_field (in normalized coords)
        dvf_normalized = dvf_field_cropped - normal_field_cropped  # [1, 3, D, H, W]

        # Convert to numpy
        dvf_np = dvf_normalized.squeeze(0).detach().cpu().numpy()  # [3, D, H, W]

        # Transpose to [D, H, W, 3] for NIfTI format
        dvf_np = dvf_np.transpose(1, 2, 3, 0)  # [D, H, W, 3]

        # Save DVF as NIfTI file
        deformation_path = os.path.join(output_dir, 'deformation.nii.gz')
        nib.save(nib.Nifti1Image(dvf_np, affine), deformation_path)
        print(f'Saved deformation vector field to {deformation_path}')
        print(f'  DVF shape: {dvf_np.shape}')
        print(f'  DVF range - X: [{dvf_np[:,:,:,0].min():.4f}, {dvf_np[:,:,:,0].max():.4f}]')
        print(f'  DVF range - Y: [{dvf_np[:,:,:,1].min():.4f}, {dvf_np[:,:,:,1].max():.4f}]')
        print(f'  DVF range - Z: [{dvf_np[:,:,:,2].min():.4f}, {dvf_np[:,:,:,2].max():.4f}]')
        result_dict['dvf_path'] = deformation_path

        # Save deformation magnitude
        dvf_magnitude = np.sqrt(np.sum(dvf_np**2, axis=-1))
        magnitude_path = os.path.join(output_dir, 'deformation_magnitude.nii.gz')
        nib.save(nib.Nifti1Image(dvf_magnitude, affine), magnitude_path)
        print(f'Saved deformation magnitude to {magnitude_path}')
        print(f'  Magnitude range: [{dvf_magnitude.min():.4f}, {dvf_magnitude.max():.4f}]')
        result_dict['magnitude_path'] = magnitude_path

    # Clean up GPU memory
    del model, input_mapper
    del ap_dwi_pad, warped_ap_dwi
    del def_coords, def_field, coor_normal_grid, normal_field
    del dvf_field_cropped, normal_field_cropped, dvf_normalized

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_dict['success'] = True
    return result_dict


# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='INR Inference for DWI Distortion Correction')

    # Data paths - use relative paths based on script location
    parser.add_argument('--moving_image', type=str,
                        default=os.path.join(SCRIPT_DIR, 'data', 'AP_dwi.nii.gz'),
                        help='Path to moving image (AP DWI) to be corrected')
    parser.add_argument('--weights', type=str,
                        default=os.path.join(SCRIPT_DIR, 'output', 'model_weights.pth'),
                        help='Path to pre-trained model weights (.pth file)')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(SCRIPT_DIR, 'output'),
                        help='Output directory')

    # GPU device
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"\n{'='*60}")
    print(f"INR Inference - DWI Distortion Correction")
    print('='*60)
    print(f"Moving image: {args.moving_image}")
    print(f"Model weights: {args.weights}")
    print(f"Output directory: {args.output_dir}")
    print('='*60)

    # Check if files exist
    if not os.path.exists(args.moving_image):
        print(f"Error: Moving image not found: {args.moving_image}")
        return

    if not os.path.exists(args.weights):
        print(f"Error: Model weights not found: {args.weights}")
        return

    # Run inference
    result = run_inference(
        moving_path=args.moving_image,
        weights_path=args.weights,
        output_dir=args.output_dir,
        device=device,
    )

    print(f"\n{'='*60}")
    if result['success']:
        print("Inference completed successfully!")
        print(f"\nOutput files:")
        print(f"  - Corrected image: {result['corrected_image_path']}")
        print(f"  - DVF: {result['dvf_path']}")
        print(f"  - DVF magnitude: {result['magnitude_path']}")
    else:
        print("Inference failed!")
    print('='*60)

    return result


if __name__ == '__main__':
    main()
