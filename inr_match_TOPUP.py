import time
import os
import argparse
import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import math
import json

from deepali.core.bspline import evaluate_cubic_bspline
from deepali.spatial.bspline import FreeFormDeformation
from deepali.core import Grid
# Import matcher loader
from matcher_loader import load_matcher


def normalize_coordinates_with_padding(points, dims, pad_size, align_corners=True):
    """
    Normalize coordinates to [-1, 1] accounting for padding.

    Args:
        points: (N, 3) array of points in image coordinates
        dims: (x_dim, y_dim, z_dim) dimensions of padded image
        pad_size: (pad_x, pad_y, pad_z) padding applied to each side
        align_corners: if True, -1 and 1 correspond to corners, else to edges

    Returns:
        Normalized coordinates in [-1, 1]
    """
    points_normalized = np.empty_like(points, dtype=np.float32)

    for i in range(3):
        if align_corners:
            # With align_corners=True, corners are at -1 and 1
            points_normalized[:, i] = 2.0 * points[:, i] / (dims[i] - 1) - 1.0
        else:
            # With align_corners=False, -1 and 1 are half pixel outside corners
            points_normalized[:, i] = 2.0 * points[:, i] / dims[i] - 1.0 + 1.0 / dims[i]

    return points_normalized


# ============= Default Configuration =============
DEFAULT_CONFIG = {
    'MODEL': {
        'HIDDEN_CHANNELS': 512,
        'NUM_LAYERS': 5,
        'DROPOUT': 0.0,
        'USE_FF': True,  # Use Fourier Features
    },
    'FOURIER': {
        'MAPPING_SIZE': 128,
        'FF_SCALE': 10.0,
        'FF_FACTOR': 1.0,
    },
    'TRAINING': {
        'EPOCHS': 500,
        'LR': 0.0001,
        'LOSS': 'NMI',  # Options: NMI, MSELoss, L1Loss
        'OPTIM': 'Adam',
        'SEED': 42,
    },
    'SETTINGS': {
        'GPU_DEVICE': 0,
    },
    'NMI': {
        'SIGMA': 0.1,
        'NBINS': 64,
    },
    'REGULARIZATION': {
        'BENDING_WEIGHT': 0,
        'LANDMARK_WEIGHT': 100,
    }
}


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
        for i in range(self.num_layers):
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


# ============= Loss Functions =============
class NMI(nn.Module):
    """Normalized Mutual Information loss"""
    def __init__(self, intensity_range=None, nbins=64, sigma=0.1):
        super().__init__()
        self.intensity_range = intensity_range
        self.nbins = nbins
        self.sigma = sigma

    def forward(self, fixed, warped):
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed.min(), fixed.max()
                warped_range = warped.min(), warped.max()

        bins_fixed = torch.linspace(
            fixed_range[0], fixed_range[1], self.nbins,
            dtype=fixed.dtype, device=fixed.device
        )
        bins_warped = torch.linspace(
            warped_range[0], warped_range[1], self.nbins,
            dtype=fixed.dtype, device=fixed.device
        )

        return -self.nmi_gauss(fixed, warped, bins_fixed, bins_warped).mean()

    def forward_sampled(self, fixed, warped, sample_ratio=0.1):
        """Compute NMI using random sampling to save GPU memory"""
        B, C, D, H, W = fixed.shape

        # Calculate total number of voxels
        total_voxels = D * H * W
        sample_voxels = int(total_voxels * sample_ratio)

        # Ensure minimum number of samples
        sample_voxels = max(sample_voxels, 10000)  # At least 10k voxels
        sample_voxels = min(sample_voxels, total_voxels)  # But not more than total

        # Flatten the images
        fixed_flat = fixed.reshape(B, C, -1)
        warped_flat = warped.reshape(B, C, -1)

        # Random sampling of voxels
        indices = torch.randperm(total_voxels, device=fixed.device)[:sample_voxels]

        # Sample the voxels
        fixed_sampled = fixed_flat[:, :, indices]
        warped_sampled = warped_flat[:, :, indices]

        # Compute range for bins
        with torch.no_grad():
            if self.intensity_range:
                fixed_range = self.intensity_range
                warped_range = self.intensity_range
            else:
                fixed_range = fixed_sampled.min(), fixed_sampled.max()
                warped_range = warped_sampled.min(), warped_sampled.max()

        bins_fixed = torch.linspace(
            fixed_range[0], fixed_range[1], self.nbins,
            dtype=fixed.dtype, device=fixed.device
        )
        bins_warped = torch.linspace(
            warped_range[0], warped_range[1], self.nbins,
            dtype=fixed.dtype, device=fixed.device
        )

        # Flatten for NMI computation (already flattened)
        fixed_sampled_flat = fixed_sampled.flatten(0, 1)  # [B*C, sample_voxels]
        warped_sampled_flat = warped_sampled.flatten(0, 1)  # [B*C, sample_voxels]

        return -self.nmi_gauss(fixed_sampled_flat, warped_sampled_flat, bins_fixed, bins_warped).mean()

    def nmi_gauss(self, x1, x2, x1_bins, x2_bins):
        def gaussian_window(x, bins, sigma):
            assert x.ndim == 2, "Input tensor should be 2-dimensional."
            return torch.exp(
                -((x[:, None, :] - bins[None, :, None]) ** 2) / (2 * sigma ** 2)
            ) / (math.sqrt(2 * math.pi) * sigma)

        x1_windowed = gaussian_window(x1.flatten(1), x1_bins, self.sigma)
        x2_windowed = gaussian_window(x2.flatten(1), x2_bins, self.sigma)
        p_XY = torch.bmm(x1_windowed, x2_windowed.transpose(1, 2))
        p_XY = p_XY + 1e-10  # numerical stability

        p_XY = p_XY / p_XY.sum((1, 2))[:, None, None]
        p_X = p_XY.sum(1)
        p_Y = p_XY.sum(2)

        # Prevent log(0) by adding small epsilon
        eps = 1e-10
        p_X_safe = torch.clamp(p_X, min=eps)
        p_Y_safe = torch.clamp(p_Y, min=eps)

        I = (p_XY * torch.log(p_XY / (p_X_safe[:, None] * p_Y_safe[:, :, None]))).sum((1, 2))
        marg_ent_0 = (p_X_safe * torch.log(p_X_safe)).sum(1)
        marg_ent_1 = (p_Y_safe * torch.log(p_Y_safe)).sum(1)

        # Prevent division by zero
        normalized = -1 * 2 * I / (marg_ent_0 + marg_ent_1 + eps)
        return normalized


# ============= Utility Functions =============
class InputMapping(nn.Module):
    """Fourier feature mapping for positional encoding"""
    def __init__(self, B=None, factor=1.0):
        super(InputMapping, self).__init__()
        self.B = factor * B

    def forward(self, x):
        x_proj = (2. * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def dict2obj(dict1):
    """Convert dictionary to object recursively"""
    if isinstance(dict1, dict):
        class obj:
            def __init__(self, dict1):
                for key, value in dict1.items():
                    if isinstance(value, dict):
                        setattr(self, key, dict2obj(value))
                    else:
                        setattr(self, key, value)
        return obj(dict1)
    else:
        return dict1


def l2reg_loss(u):
    """L2 regularization loss for deformation field"""
    derives = []
    ndim = u.size()[1]
    for i in range(ndim):
        derives += [finite_diff(u, dim=i)]
    loss = torch.cat(derives, dim=1).pow(2).sum(dim=1).mean()
    return loss


def finite_diff(x, dim, mode="forward", boundary="Neumann"):
    """Compute finite differences"""
    ndim = x.ndim - 2
    sizes = x.shape[2:]

    paddings = [[0, 0] for _ in range(ndim)]
    if mode == "forward":
        paddings[dim][1] = 1
    elif mode == "backward":
        paddings[dim][0] = 1
    else:
        raise ValueError(f'Mode {mode} not recognised')

    paddings.reverse()
    paddings = [p for ppair in paddings for p in ppair]

    if boundary == "Neumann":
        x_pad = F.pad(x, paddings, mode='replicate')
    elif boundary == "Dirichlet":
        x_pad = F.pad(x, paddings, mode='constant')
    else:
        raise ValueError("Boundary condition not recognised.")

    x_diff = x_pad.index_select(dim + 2, torch.arange(1, sizes[dim] + 1).to(device=x.device)) \
             - x_pad.index_select(dim + 2, torch.arange(0, sizes[dim]).to(device=x.device))

    return x_diff


def find_exact_crop_roi(warped_mask):
    """
    Find the exact ROI (Region of Interest) from the B-spline transformed mask.
    The ROI is simply the bounding box of all pixels with value > 0.5 in the mask.

    Args:
        warped_mask: (B, C, D, H, W) warped binary mask after identity B-spline transform

    Returns:
        roi_bounds: dict with 'start' and 'end' indices for each dimension
    """
    # Convert to numpy for easier processing
    mask_np = warped_mask[0, 0].detach().cpu().numpy()

    # Find non-zero coordinates (valid region after B-spline)
    nonzero_coords = np.where(mask_np > 0.5)

    if len(nonzero_coords[0]) == 0:
        # Fallback: use full image if no valid region found
        print("Warning: No valid region found in B-spline mask, using full image")
        d, h, w = mask_np.shape
        return {
            'd_start': 0, 'd_end': d,
            'h_start': 0, 'h_end': h,
            'w_start': 0, 'w_end': w,
        }

    # Find bounding box of valid region - this IS our ROI
    min_d, max_d = int(nonzero_coords[0].min()), int(nonzero_coords[0].max()) + 1
    min_h, max_h = int(nonzero_coords[1].min()), int(nonzero_coords[1].max()) + 1
    min_w, max_w = int(nonzero_coords[2].min()), int(nonzero_coords[2].max()) + 1

    # Calculate valid region size
    valid_d = max_d - min_d
    valid_h = max_h - min_h
    valid_w = max_w - min_w

    roi_bounds = {
        'd_start': min_d, 'd_end': max_d,
        'h_start': min_h, 'h_end': max_h,
        'w_start': min_w, 'w_end': max_w,
        'size': (valid_d, valid_h, valid_w)
    }

    return roi_bounds


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


def extract_3d_matches(fixed_image, moving_image, matcher, args, image_dims):
    """
    Extract 2D matches from all three orientations (axial, coronal, sagittal) and combine into 3D landmarks.

    Args:
        fixed_image: 3D numpy array (normalized to [0, 255])
        moving_image: 3D numpy array (normalized to [0, 255])
        matcher: The feature matcher object
        args: Arguments containing conf_threshold, num_slices, etc.
        image_dims: (x_dim, y_dim, z_dim) dimensions of the image

    Returns:
        all_mkpts0_3D: 3D coordinates in fixed image
        all_mkpts1_3D: 3D coordinates in moving image
        all_mconf: Confidence scores
    """
    x_dim, y_dim, z_dim = image_dims

    all_mkpts0_3D = []
    all_mkpts1_3D = []
    all_mconf = []

    # 1. Axial slices (xy plane, varying z)
    num_axial_slices = min(args.num_slices, z_dim)
    axial_step = max(1, z_dim // num_axial_slices)

    print(f"Extracting axial matches...")
    for slice_idx in range(0, z_dim, axial_step):
        fixed_slice = fixed_image[:, :, slice_idx]
        moving_slice = moving_image[:, :, slice_idx]

        if np.sum(fixed_slice) > 0:
            with torch.no_grad():
                match_res = matcher(fixed_slice, moving_slice)

            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            mconf = match_res['mconf']

            if len(mkpts0) > 0:
                # Filter by confidence
                mask_conf = mconf > args.conf_threshold
                mkpts0 = mkpts0[mask_conf]
                mkpts1 = mkpts1[mask_conf]
                mconf = mconf[mask_conf]

                if len(mkpts0) > 0:
                    # Convert to 3D coordinates (x, y, z)
                    mkpts0_3D = np.column_stack((mkpts0[:, 0], mkpts0[:, 1], np.full(mkpts0.shape[0], slice_idx)))
                    mkpts1_3D = np.column_stack((mkpts1[:, 0], mkpts1[:, 1], np.full(mkpts1.shape[0], slice_idx)))

                    all_mkpts0_3D.append(mkpts0_3D)
                    all_mkpts1_3D.append(mkpts1_3D)
                    all_mconf.append(mconf)

    # 2. Coronal slices (xz plane, varying y)
    num_coronal_slices = min(args.num_slices, y_dim)
    coronal_step = max(1, y_dim // num_coronal_slices)

    print(f"Extracting coronal matches...")
    for slice_idx in range(0, y_dim, coronal_step):
        fixed_slice = fixed_image[:, slice_idx, :]
        moving_slice = moving_image[:, slice_idx, :]

        if np.sum(fixed_slice) > 0:
            with torch.no_grad():
                match_res = matcher(fixed_slice, moving_slice)

            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            mconf = match_res['mconf']

            if len(mkpts0) > 0:
                # Filter by confidence
                mask_conf = mconf > args.conf_threshold
                mkpts0 = mkpts0[mask_conf]
                mkpts1 = mkpts1[mask_conf]
                mconf = mconf[mask_conf]

                if len(mkpts0) > 0:
                    # Convert to 3D coordinates (x, y, z)
                    # mkpts0[:, 0] is x coordinate, mkpts0[:, 1] is z coordinate
                    mkpts0_3D = np.column_stack((mkpts0[:, 0], np.full(mkpts0.shape[0], slice_idx), mkpts0[:, 1]))
                    mkpts1_3D = np.column_stack((mkpts1[:, 0], np.full(mkpts1.shape[0], slice_idx), mkpts1[:, 1]))

                    all_mkpts0_3D.append(mkpts0_3D)
                    all_mkpts1_3D.append(mkpts1_3D)
                    all_mconf.append(mconf)

    # 3. Sagittal slices (yz plane, varying x)
    num_sagittal_slices = min(args.num_slices, x_dim)
    sagittal_step = max(1, x_dim // num_sagittal_slices)

    print(f"Extracting sagittal matches...")
    for slice_idx in range(0, x_dim, sagittal_step):
        fixed_slice = fixed_image[slice_idx, :, :]
        moving_slice = moving_image[slice_idx, :, :]

        if np.sum(fixed_slice) > 0:
            with torch.no_grad():
                match_res = matcher(fixed_slice, moving_slice)

            mkpts0 = match_res['mkpts0']
            mkpts1 = match_res['mkpts1']
            mconf = match_res['mconf']

            if len(mkpts0) > 0:
                # Filter by confidence
                mask_conf = mconf > args.conf_threshold
                mkpts0 = mkpts0[mask_conf]
                mkpts1 = mkpts1[mask_conf]
                mconf = mconf[mask_conf]

                if len(mkpts0) > 0:
                    # Convert to 3D coordinates (x, y, z)
                    # mkpts0[:, 0] is y coordinate, mkpts0[:, 1] is z coordinate
                    mkpts0_3D = np.column_stack((np.full(mkpts0.shape[0], slice_idx), mkpts0[:, 0], mkpts0[:, 1]))
                    mkpts1_3D = np.column_stack((np.full(mkpts1.shape[0], slice_idx), mkpts1[:, 0], mkpts1[:, 1]))

                    all_mkpts0_3D.append(mkpts0_3D)
                    all_mkpts1_3D.append(mkpts1_3D)
                    all_mconf.append(mconf)

    if len(all_mkpts0_3D) == 0:
        return None, None, None

    # Concatenate all matches
    all_mkpts0_3D = np.concatenate(all_mkpts0_3D, axis=0)
    all_mkpts1_3D = np.concatenate(all_mkpts1_3D, axis=0)
    all_mconf = np.concatenate(all_mconf, axis=0)

    print(f"Total 3D matches found: {len(all_mkpts0_3D)}")

    return all_mkpts0_3D, all_mkpts1_3D, all_mconf




# ============= Main Function =============
# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description='INR Matching for TOPUP DWI Correction')

    # Data paths - use relative paths based on script location
    parser.add_argument('--moving_image', type=str,
                       default=os.path.join(SCRIPT_DIR, 'data', 'AP_dwi.nii.gz'),
                       help='Path to moving image (AP DWI)')
    parser.add_argument('--reference_image', type=str,
                       default=os.path.join(SCRIPT_DIR, 'data', 'reference_image.nii.gz'),
                       help='Path to reference image')
    parser.add_argument('--output_dir', type=str,
                       default=os.path.join(SCRIPT_DIR, 'output'),
                       help='Output directory')

    # Model parameters
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--stride_size', type=int, nargs=3, default=[18, 18, 18])

    # Matcher arguments
    parser.add_argument('--matcher', type=str, default='sp_lg', choices=['sp_lg', 'loftr', 'roma'],
                        help='Feature matcher to use')
    parser.add_argument('--match_threshold', type=float, default=0.2, help='Matching threshold for LoFTR')
    parser.add_argument('--conf_threshold', type=float, default=0.4, help='Confidence threshold for matches')
    parser.add_argument('--num_slices', type=int, default=10, help='Number of slices to process for matching per orientation')

    # GPU device
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')

    return parser.parse_args()


def run_registration(moving_path, reference_path, output_dir, config, matcher, args):
    """
    Run INR registration on a pair of images.

    Args:
        moving_path: Path to moving image (AP DWI)
        reference_path: Path to reference image
        output_dir: Path to output directory
        config: Configuration object
        matcher: Feature matcher object
        args: Command line arguments

    Returns:
        result_dict: Dictionary containing paths to saved files and metadata
    """
    result_dict = {
        'success': False,
        'corrected_image_path': None,
        'model_weights_path': None,
        'dvf_path': None,
        'magnitude_path': None,
    }

    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load reference image
    print(f"Loading reference image: {reference_path}")
    ref_nib = nib.load(reference_path)
    ref_data = ref_nib.get_fdata()
    affine = ref_nib.affine

    print(f"  Reference shape: {ref_data.shape}")

    # Z-normalization
    ref_data = (ref_data - np.mean(ref_data)) / np.std(ref_data)

    ref_torch = torch.from_numpy(ref_data).float().unsqueeze(0).unsqueeze(0)

    # Set device
    device = torch.device(f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu')

    # Model configuration
    mapping_size = config.FOURIER.MAPPING_SIZE
    input_size = 2 * mapping_size
    output_size = 3
    hidden_size = config.MODEL.HIDDEN_CHANNELS
    num_layers = config.MODEL.NUM_LAYERS

    # Load moving image (AP DWI)
    print(f"Loading moving image: {moving_path}")
    ap_dwi_nib = nib.load(moving_path)
    ap_dwi_data = ap_dwi_nib.get_fdata()

    # Verify it's 3D
    if len(ap_dwi_data.shape) != 3:
        print(f"  Warning: Expected 3D image but got shape {ap_dwi_data.shape}")
        if len(ap_dwi_data.shape) == 4:
            print(f"  Extracting first volume")
            ap_dwi_data = ap_dwi_data[..., 0]
        else:
            print(f"  Unsupported image shape, skipping...")
            return result_dict

    print(f"  Moving image shape: {ap_dwi_data.shape}")

    # Initialize model
    model = MLPv1(
        input_size=input_size,
        output_size=output_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=config.MODEL.DROPOUT
    ).to(device)

    print(f'Number of MLP parameters: {sum(p.numel() for p in model.parameters())}')

    # Loss and optimizer
    criterion_nmi = NMI(sigma=config.NMI.SIGMA, nbins=config.NMI.NBINS)

    if config.TRAINING.OPTIM == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAINING.LR)
    else:
        raise ValueError(f'Optimizer {config.TRAINING.OPTIM} not supported')

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.TRAINING.EPOCHS)

    # Initialize Fourier feature mapping
    B_gauss = torch.randn(mapping_size, 3, dtype=torch.float32).to(device) * config.FOURIER.FF_SCALE
    input_mapper = InputMapping(B=B_gauss, factor=config.FOURIER.FF_FACTOR).to(device)

    # Store original statistics for denormalization
    ap_dwi_mean = np.mean(ap_dwi_data)
    ap_dwi_std = np.std(ap_dwi_data)

    # Z-normalization
    ap_dwi_data = (ap_dwi_data - ap_dwi_mean) / ap_dwi_std

    ap_dwi_torch = torch.from_numpy(ap_dwi_data).float().unsqueeze(0).unsqueeze(0).to(device)

    # Pad images
    pad_size = [30, 30, 30]
    min_value_ref = ref_data.min()
    min_value_ap_dwi = ap_dwi_data.min()

    ref_pad = F.pad(ref_torch.to(device), (pad_size[0], pad_size[0], pad_size[1], pad_size[1],
                               pad_size[2], pad_size[2]), mode='constant', value=min_value_ref)

    ap_dwi_pad = F.pad(ap_dwi_torch, (pad_size[0], pad_size[0], pad_size[1], pad_size[1],
                                           pad_size[2], pad_size[2]), mode='constant', value=min_value_ap_dwi)

    # Get dimensions (use moving image dimensions for consistent matching)
    x_dim, y_dim, z_dim = ap_dwi_pad.shape[2], ap_dwi_pad.shape[3], ap_dwi_pad.shape[4]
    image_size = (x_dim, y_dim, z_dim)

    # Match landmarks using 3D matching
    fixed_image_match = ref_pad.squeeze().detach().cpu().numpy()
    moving_image_match = ap_dwi_pad.squeeze().detach().cpu().numpy()

    # Resample fixed image to match moving image dimensions in XY, crop to min z
    from scipy.ndimage import zoom as scipy_zoom
    zoom_factors = (moving_image_match.shape[0] / fixed_image_match.shape[0],
                   moving_image_match.shape[1] / fixed_image_match.shape[1],
                   1.0)  # No z-axis resampling
    fixed_image_match = scipy_zoom(fixed_image_match, zoom_factors, order=1)

    # Crop to minimum z dimension for matching
    min_z = min(fixed_image_match.shape[2], moving_image_match.shape[2])
    fixed_image_match = fixed_image_match[:, :, :min_z]
    moving_image_match = moving_image_match[:, :, :min_z]

    # Update image_size for matching to use the cropped dimensions
    match_image_size = (moving_image_match.shape[0], moving_image_match.shape[1], min_z)

    # Normalize for matching
    fixed_image_match = (fixed_image_match - np.min(fixed_image_match)) / (np.max(fixed_image_match) - np.min(fixed_image_match))
    moving_image_match = (moving_image_match - np.min(moving_image_match)) / (np.max(moving_image_match) - np.min(moving_image_match))
    fixed_image_match = np.clip(fixed_image_match * 255, 0, 255).astype(np.uint8)
    moving_image_match = np.clip(moving_image_match * 255, 0, 255).astype(np.uint8)

    # Extract 3D matches from all orientations
    print("\nExtracting 3D landmark matches...")
    all_mkpts0_3D, all_mkpts1_3D, all_mconf = extract_3d_matches(
        fixed_image_match, moving_image_match, matcher, args, match_image_size
    )

    if all_mkpts0_3D is None:
        print(f"No matches found, skipping...")
        return result_dict

    # Use fixed normalization that properly handles align_corners
    align_corners = True  # Match grid_sample setting

    # Normalize coordinates to [-1, 1] using fixed function
    all_mkpts0_3D_norm = normalize_coordinates_with_padding(
        all_mkpts0_3D, (x_dim, y_dim, z_dim), pad_size, align_corners
    )
    all_mkpts1_3D_norm = normalize_coordinates_with_padding(
        all_mkpts1_3D, (x_dim, y_dim, z_dim), pad_size, align_corners
    )

    # Convert to torch tensors with correct ordering
    fixed_kpts_norm = torch.from_numpy(all_mkpts0_3D_norm[:, [2, 0, 1]]).float().to(device)
    moving_kpts_norm = torch.from_numpy(all_mkpts1_3D_norm[:, [2, 0, 1]]).float().to(device)
    kpts_conf = torch.from_numpy(all_mconf).float().to(device)

    # Create FFD grid
    stride_size = args.stride_size
    X = np.arange(0, x_dim, stride_size[0])
    Y = np.arange(0, y_dim, stride_size[1])
    Z = np.arange(0, z_dim, stride_size[2])

    x_coor_dim = len(X)
    y_coor_dim = len(Y)
    z_coor_dim = len(Z)

    points = np.meshgrid(X, Y, Z, indexing='ij')
    points = np.stack(points).transpose(1, 2, 3, 0).reshape(-1, 3)

    # Normalize grid points using fixed function
    coordinates_normal = normalize_coordinates_with_padding(
        points, (x_dim, y_dim, z_dim), [0, 0, 0], align_corners  # No padding for grid points
    )

    grid = Grid((x_dim, y_dim, z_dim))
    kernel = FreeFormDeformation(grid, stride=stride_size).cuda().kernel()
    transformation = evaluate_cubic_bspline

    # Permute for correct orientation
    ref_pad = ref_pad.permute(0, 1, 4, 3, 2)
    ap_dwi_pad = ap_dwi_pad.permute(0, 1, 4, 3, 2)

    # Keep original images for loss computation
    # Resample ref to match ap_dwi dimensions for loss computation
    ref_ori_resampled = F.interpolate(ref_torch.to(device), size=ap_dwi_data.shape, mode='trilinear', align_corners=False)
    ref_ori = ref_ori_resampled.detach()
    ap_dwi_ori = ap_dwi_torch.detach().to(device)

    # Create a binary mask for accurate crop region detection
    valid_mask = torch.ones(1, 1, ap_dwi_data.shape[0], ap_dwi_data.shape[1], ap_dwi_data.shape[2],
                       dtype=torch.float32, device=device)

    # Apply the same padding as the images (pad with 0s)
    valid_mask_pad = F.pad(valid_mask, (pad_size[0], pad_size[0], pad_size[1], pad_size[1],
                                   pad_size[2], pad_size[2]), mode='constant', value=0)

    # Permute to match image orientation
    valid_mask_pad = valid_mask_pad.permute(0, 1, 4, 3, 2)

    # Pre-compute the zero-displacement B-spline transformation for the mask
    with torch.no_grad():
        # Create zero displacement control points (identity mapping)
        coor_normal_torch = torch.from_numpy(coordinates_normal).to(device)

        # Use original implementation with zero displacement
        zero_cp = coor_normal_torch.reshape(1, x_coor_dim, y_coor_dim, z_coor_dim, 3).permute(0, 4, 1, 2, 3)
        identity_field = transformation(zero_cp, stride=stride_size, shape=image_size, kernel=kernel, transpose=False)
        identity_coords = identity_field.permute(0, 2, 3, 4, 1)

        # Apply the identity transformation to the mask
        mask_after_bspline = F.grid_sample(valid_mask_pad, identity_coords, align_corners=align_corners, mode='nearest')


        # Find exact ROI bounds using the mask after B-spline
        roi_bounds_precomputed = find_exact_crop_roi(mask_after_bspline)

    # Training loop
    roi_bounds = roi_bounds_precomputed  # Use the pre-computed ROI bounds

    print(f"\nStarting training for {config.TRAINING.EPOCHS} epochs...")
    best_loss = float('inf')

    for epoch in range(config.TRAINING.EPOCHS):
        model.train()

        # Use original implementation
        # Forward pass
        coor_normal = torch.from_numpy(coordinates_normal).to(device)
        data = input_mapper(coor_normal)
        pred = model(data)

        # Compute deformation field
        def_coords = torch.add(pred, coor_normal)
        def_coords = def_coords.reshape(1, x_coor_dim, y_coor_dim, z_coor_dim, 3).permute(0, 4, 1, 2, 3)
        def_coords = transformation(def_coords, stride=stride_size, shape=image_size, kernel=kernel, transpose=False)
        def_field = def_coords
        def_coords = def_coords.permute(0, 2, 3, 4, 1)

        # Compute normal field
        coor_normal_grid = coor_normal.reshape(1, x_coor_dim, y_coor_dim, z_coor_dim, 3).permute(0, 4, 1, 2, 3)
        coor_normal_grid = transformation(coor_normal_grid, stride=stride_size, shape=image_size, kernel=kernel, transpose=False)
        normal_field = coor_normal_grid
        coor_normal_grid = coor_normal_grid.permute(0, 2, 3, 4, 1)

        # Compute landmark loss
        fixed_sample_grid = fixed_kpts_norm.view(1, -1, 1, 1, 3)
        fixed_kpts = F.grid_sample(def_field, fixed_sample_grid, align_corners=align_corners, mode='bilinear').view(3, -1).t()

        moving_sample_grid = moving_kpts_norm.view(1, -1, 1, 1, 3)
        moving_kpts = F.grid_sample(normal_field, moving_sample_grid, align_corners=align_corners, mode='bilinear').view(3, -1).t()

        diff_sq = (fixed_kpts - moving_kpts).pow(2).sum(1)
        loss_kpts = (diff_sq * kpts_conf).mean()

        # Warp image
        warped_ap_dwi = F.grid_sample(ap_dwi_pad, def_coords, align_corners=align_corners)

        # Crop warped images using pre-computed ROI bounds
        warped_ap_dwi = crop_with_roi(warped_ap_dwi, roi_bounds)

        # Compute regularization
        output_rel = torch.subtract(def_coords, coor_normal_grid)
        bending_reg = l2reg_loss(output_rel.permute(0, 4, 1, 2, 3))

        # Compute NMI loss
        total_voxels = ref_ori.shape[2] * ref_ori.shape[3] * ref_ori.shape[4]
        max_voxels = 300 * 300 * 100

        if total_voxels > max_voxels:
            # Use sampling for very large images
            sample_ratio = min(0.1, max_voxels / total_voxels)
            loss_mutual = criterion_nmi.forward_sampled(ref_ori, warped_ap_dwi, sample_ratio=sample_ratio)
        else:
            # Use full resolution for small images
            loss_mutual = criterion_nmi(ref_ori, warped_ap_dwi)

        # Total loss
        loss = loss_mutual + loss_kpts * config.REGULARIZATION.LANDMARK_WEIGHT

        if epoch % 20 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.6f}, NMI: {loss_mutual.item():.6f}, '
                  f'Landmarks: {loss_kpts.item():.6f}')

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent gradient explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        # Track best model
        if loss.item() < best_loss:
            best_loss = loss.item()

        # Save final result
        if epoch == config.TRAINING.EPOCHS - 1:
            # Save model weights
            model_weights_path = os.path.join(output_dir, 'model_weights.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'B_gauss': B_gauss.cpu(),
                'config': {
                    'input_size': input_size,
                    'output_size': output_size,
                    'hidden_size': hidden_size,
                    'num_layers': num_layers,
                    'dropout': config.MODEL.DROPOUT,
                    'mapping_size': mapping_size,
                    'ff_scale': config.FOURIER.FF_SCALE,
                    'ff_factor': config.FOURIER.FF_FACTOR,
                },
                'stride_size': stride_size,
                'image_size': image_size,
                'coordinates_normal': coordinates_normal,
                'roi_bounds': roi_bounds,
                'ap_dwi_mean': ap_dwi_mean,
                'ap_dwi_std': ap_dwi_std,
            }, model_weights_path)
            print(f'Saved model weights to {model_weights_path}')
            result_dict['model_weights_path'] = model_weights_path

            # Re-warp with nearest neighbor for final output to avoid interpolation blur
            with torch.no_grad():
                # Recompute final deformation field to ensure consistency
                final_warped_ap_dwi = F.grid_sample(ap_dwi_pad, def_coords, align_corners=align_corners, mode='bilinear')

                # Crop using pre-computed ROI bounds
                final_warped_ap_dwi = crop_with_roi(final_warped_ap_dwi, roi_bounds)

                # Convert to numpy
                warped_ap_dwi_HR = final_warped_ap_dwi.squeeze().detach().cpu().numpy()

                # Denormalize back to original intensity values
                warped_ap_dwi_HR = warped_ap_dwi_HR * ap_dwi_std + ap_dwi_mean

                # Clip negative values to 0
                warped_ap_dwi_HR = np.clip(warped_ap_dwi_HR, 0, None)

                # Save corrected image
                corrected_path = os.path.join(output_dir, 'corrected_image.nii.gz')
                nib.save(nib.Nifti1Image(warped_ap_dwi_HR, affine), corrected_path)
                print(f'Saved corrected image to {corrected_path}')
                result_dict['corrected_image_path'] = corrected_path

                # ============= Save Deformation Vector Field (DVF) =============
                # Compute DVF: deformation field - normal field
                # DVF is saved in normalized coordinates [-1, 1]

                # Crop DVF to valid region
                dvf_field_cropped = crop_with_roi(def_field, roi_bounds)  # [1, 3, D, H, W]
                normal_field_cropped = crop_with_roi(normal_field, roi_bounds)  # [1, 3, D, H, W]

                # Compute displacement: dvf = def_field - normal_field (in normalized coords)
                dvf_normalized = dvf_field_cropped - normal_field_cropped  # [1, 3, D, H, W]

                # Convert to numpy
                dvf_np = dvf_normalized.squeeze(0).detach().cpu().numpy()  # [3, D, H, W]

                # Transpose to [D, H, W, 3] for NIfTI format
                dvf_np = dvf_np.transpose(1, 2, 3, 0)  # [D, H, W, 3]

                # Save DVF as NIfTI file (4D image with 3 components)
                deformation_path = os.path.join(output_dir, 'deformation.nii.gz')
                nib.save(nib.Nifti1Image(dvf_np, affine), deformation_path)
                print(f'Saved deformation vector field to {deformation_path}')
                print(f'  DVF shape: {dvf_np.shape}')
                print(f'  DVF range - X: [{dvf_np[:,:,:,0].min():.4f}, {dvf_np[:,:,:,0].max():.4f}]')
                print(f'  DVF range - Y: [{dvf_np[:,:,:,1].min():.4f}, {dvf_np[:,:,:,1].max():.4f}]')
                print(f'  DVF range - Z: [{dvf_np[:,:,:,2].min():.4f}, {dvf_np[:,:,:,2].max():.4f}]')
                result_dict['dvf_path'] = deformation_path

                # Also save the magnitude of deformation for easy visualization
                dvf_magnitude = np.sqrt(np.sum(dvf_np**2, axis=-1))
                magnitude_path = os.path.join(output_dir, 'deformation_magnitude.nii.gz')
                nib.save(nib.Nifti1Image(dvf_magnitude, affine), magnitude_path)
                print(f'Saved deformation magnitude to {magnitude_path}')
                print(f'  Magnitude range: [{dvf_magnitude.min():.4f}, {dvf_magnitude.max():.4f}] (normalized coords)')
                result_dict['magnitude_path'] = magnitude_path


    # Clean up GPU memory after processing each image
    del model, optimizer, scheduler, input_mapper, B_gauss
    del ref_pad, ap_dwi_pad, warped_ap_dwi
    del ref_ori, ap_dwi_ori
    del def_coords, def_field, coor_normal_grid, normal_field
    del fixed_kpts_norm, moving_kpts_norm, kpts_conf
    del valid_mask, valid_mask_pad, mask_after_bspline
    del criterion_nmi

    # Delete DVF related tensors
    if 'dvf_field_cropped' in locals():
        del dvf_field_cropped, normal_field_cropped, dvf_normalized

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    result_dict['success'] = True
    return result_dict


def main(args):
    # Load configuration
    config = dict2obj(DEFAULT_CONFIG)

    # Override config with command line arguments
    if args.lr is not None:
        config.TRAINING.LR = args.lr
    if args.epochs is not None:
        config.TRAINING.EPOCHS = args.epochs
    config.SETTINGS.GPU_DEVICE = args.gpu

    # Create output directory
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(f'cuda:{config.SETTINGS.GPU_DEVICE}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config.TRAINING.SEED)
    np.random.seed(config.TRAINING.SEED)

    # Initialize matcher (only once)
    print(f"Loading {args.matcher} matcher...")
    matcher = load_matcher(args.matcher, device=device, thr=args.match_threshold)

    print(f"\n{'='*60}")
    print(f"INR-based DWI Distortion Correction")
    print('='*60)
    print(f"Moving image: {args.moving_image}")
    print(f"Reference image: {args.reference_image}")
    print(f"Output directory: {args.output_dir}")
    print('='*60)

    # Run registration
    result = run_registration(
        args.moving_image,
        args.reference_image,
        args.output_dir,
        config,
        matcher,
        args
    )

    print(f"\n{'='*60}")
    if result['success']:
        print("Registration completed successfully!")
        print(f"\nOutput files:")
        print(f"  - Corrected image: {result['corrected_image_path']}")
        print(f"  - Model weights: {result['model_weights_path']}")
        print(f"  - DVF: {result['dvf_path']}")
        print(f"  - DVF magnitude: {result['magnitude_path']}")
    else:
        print("Registration failed!")
    print('='*60)

    return result


if __name__ == '__main__':
    args = parse_args()
    main(args)
