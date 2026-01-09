import logging
import torch
import numpy as np
import cv2
from copy import deepcopy
import sys
import os

# Add third_party to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'third_party'))


class BaseMatcherWrapper:
    """Base wrapper class for matchers to provide consistent interface"""
    def __init__(self, matcher):
        self.matcher = matcher
        
    def __call__(self, img0, img1):
        """Match two images and return keypoints and confidence scores"""
        raise NotImplementedError


class SuperPointLightGlueWrapper(BaseMatcherWrapper):
    """Wrapper for SuperPoint + LightGlue matcher"""
    def __init__(self, device='cuda'):
        sys.path.append("./third_party/LightGlue/")
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import rbd
        self.rbd = rbd  # Store the function as instance variable
        
        self.device = device
        self.extractor = SuperPoint(
            descriptor_dim=256,
            nms_radius=4,
            max_num_keypoints=2048,
            detection_threshold=0.0005,
            remove_borders=4,
        ).eval().to(device)
        
        self.matcher = LightGlue(
            features='superpoint',
            name="lightglue",
            input_dim=256,
            descriptor_dim=256,
            add_scale_ori=False,
            n_layers=9,
            num_heads=4,
            flash=True,
            mp=False,
            depth_confidence=0.95,
            width_confidence=0.99,
            filter_threshold=0.1,
        ).eval().to(device)
        
        # Load weights if available
        ckpt_path = "./weights/minima_lightglue.pth"
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location=device)
            # Rename old state dict entries
            for i in range(9):  # n_layers
                pattern = f"self_attn.{i}", f"transformers.{i}.self_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
                pattern = f"cross_attn.{i}", f"transformers.{i}.cross_attn"
                state_dict = {k.replace(*pattern): v for k, v in state_dict.items()}
            self.matcher.load_state_dict(state_dict, strict=False)
            print("Loaded SuperPoint+LightGlue weights")
        else:
            print("Warning: SuperPoint+LightGlue weights not found, using random initialization")
    
    def __call__(self, img0, img1):
        """Match two grayscale images"""
        # Convert to torch tensors
        if isinstance(img0, np.ndarray):
            img0 = torch.from_numpy(img0).float() / 255.0
            img1 = torch.from_numpy(img1).float() / 255.0
        
        # Add batch and channel dimensions if needed
        if img0.dim() == 2:
            img0 = img0.unsqueeze(0).unsqueeze(0)
            img1 = img1.unsqueeze(0).unsqueeze(0)
        
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        
        # Extract features
        with torch.no_grad():
            feats0 = self.extractor.extract(img0)
            feats1 = self.extractor.extract(img1)
            
            # Match features
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            
            # Remove batch dimension
            feats0, feats1, matches01 = [self.rbd(x) for x in [feats0, feats1, matches01]]
            
            # Get matched keypoints
            matches = matches01['matches']
            if len(matches) == 0:
                return {
                    'mkpts0': np.array([]),
                    'mkpts1': np.array([]),
                    'mconf': np.array([])
                }
            
            mkpts0 = feats0['keypoints'][matches[..., 0]].cpu().numpy()
            mkpts1 = feats1['keypoints'][matches[..., 1]].cpu().numpy()
            mconf = matches01['matching_scores0'][matches[..., 0]].cpu().numpy()
            
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf
        }


class LoFTRWrapper(BaseMatcherWrapper):
    """Wrapper for LoFTR matcher"""
    def __init__(self, device='cuda', thr=0.2):
        sys.path.append("./third_party/LoFTR/")
        from src.loftr import LoFTR, default_cfg
        
        self.device = device
        self.thr = thr
        
        _default_cfg = deepcopy(default_cfg)
        _default_cfg['coarse']['temp_bug_fix'] = True
        _default_cfg['match_coarse']['thr'] = thr
        
        self.matcher = LoFTR(config=_default_cfg)
        
        # Load weights if available
        ckpt_path = "./weights/minima_loftr.ckpt"
        if os.path.exists(ckpt_path):
            self.matcher.load_state_dict(torch.load(ckpt_path, map_location=device)['state_dict'], strict=True)
            print("Loaded LoFTR weights")
        else:
            print("Warning: LoFTR weights not found, using random initialization")
        
        self.matcher = self.matcher.eval().to(device)
    
    def __call__(self, img0, img1):
        """Match two grayscale images"""
        # Convert to torch tensors and normalize
        if isinstance(img0, np.ndarray):
            img0 = torch.from_numpy(img0).float() / 255.0
            img1 = torch.from_numpy(img1).float() / 255.0
        
        # Add batch and channel dimensions if needed
        if img0.dim() == 2:
            img0 = img0.unsqueeze(0).unsqueeze(0)
            img1 = img1.unsqueeze(0).unsqueeze(0)
        
        img0 = img0.to(self.device)
        img1 = img1.to(self.device)
        
        batch = {
            'image0': img0,
            'image1': img1,
        }
        
        with torch.no_grad():
            self.matcher(batch)
            
            # Check if any matches were found
            if 'mkpts0_f' in batch and len(batch['mkpts0_f']) > 0:
                mkpts0 = batch['mkpts0_f'].cpu().numpy()
                mkpts1 = batch['mkpts1_f'].cpu().numpy()
                mconf = batch['mconf'].cpu().numpy()
            else:
                mkpts0 = np.array([])
                mkpts1 = np.array([])
                mconf = np.array([])
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf
        }


class RoMaWrapper(BaseMatcherWrapper):
    """Wrapper for RoMa matcher"""
    def __init__(self, device='cuda', model_type='large'):
        sys.path.append("./third_party/RoMa/")
        from romatch import roma_outdoor, tiny_roma_v1_outdoor
        from PIL import Image
        
        self.device = device
        self.Image = Image  # Store Image class
        
        if model_type == 'large':
            ckpt_path = "./weights/minima_roma.pth"
            if os.path.exists(ckpt_path):
                state_dict = torch.load(ckpt_path, map_location=device)
                self.matcher = roma_outdoor(device=device, weights=state_dict)
                print("Loaded RoMa weights")
            else:
                self.matcher = roma_outdoor(device=device)
                print("Warning: RoMa weights not found, using pretrained model")
        else:
            self.matcher = tiny_roma_v1_outdoor(device=device)
    
    def __call__(self, img0, img1):
        """Match two grayscale images"""
        # Ensure images are numpy arrays
        if isinstance(img0, torch.Tensor):
            img0 = img0.cpu().numpy()
        if isinstance(img1, torch.Tensor):
            img1 = img1.cpu().numpy()
        
        # Ensure uint8 type
        if img0.dtype != np.uint8:
            img0 = (img0 * 255).astype(np.uint8) if img0.max() <= 1 else img0.astype(np.uint8)
        if img1.dtype != np.uint8:
            img1 = (img1 * 255).astype(np.uint8) if img1.max() <= 1 else img1.astype(np.uint8)
        
        # Convert to RGB if grayscale
        if len(img0.shape) == 2:
            img0 = cv2.cvtColor(img0, cv2.COLOR_GRAY2RGB)
            img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
        
        # Convert numpy arrays to PIL images (RoMa expects PIL images)
        img0_pil = self.Image.fromarray(img0)
        img1_pil = self.Image.fromarray(img1)
        
        with torch.no_grad():
            # Use RoMa's match method with PIL images
            dense_matches, dense_certainty = self.matcher.match(img0_pil, img1_pil)
            
            # Sample sparse matches
            sparse_matches, sparse_certainty = self.matcher.sample(
                dense_matches, dense_certainty, num=2000
            )
            
            # Extract keypoints
            if len(sparse_matches.shape) == 3:
                # Remove batch dimension if present
                sparse_matches = sparse_matches[0]
                sparse_certainty = sparse_certainty[0]
            
            # Get image dimensions
            height, width = img0.shape[:2]
            
            # Extract and denormalize keypoints
            kpts0 = sparse_matches[:, :2].clone()
            kpts1 = sparse_matches[:, 2:].clone()
            
            # Denormalize from [-1, 1] to pixel coordinates
            kpts0[:, 0] = (kpts0[:, 0] + 1) * (width - 1) / 2
            kpts0[:, 1] = (kpts0[:, 1] + 1) * (height - 1) / 2
            kpts1[:, 0] = (kpts1[:, 0] + 1) * (width - 1) / 2
            kpts1[:, 1] = (kpts1[:, 1] + 1) * (height - 1) / 2
            
            mkpts0 = kpts0.cpu().numpy()
            mkpts1 = kpts1.cpu().numpy()
            mconf = sparse_certainty.cpu().numpy()
        
        return {
            'mkpts0': mkpts0,
            'mkpts1': mkpts1,
            'mconf': mconf
        }


def load_matcher(method='sp_lg', device='cuda', **kwargs):
    """Load a specific matcher
    
    Args:
        method: Matcher method ('sp_lg', 'loftr', 'roma')
        device: Device to run on
        **kwargs: Additional arguments for specific matchers
    
    Returns:
        Matcher wrapper instance
    """
    if method == 'sp_lg':
        return SuperPointLightGlueWrapper(device=device)
    elif method == 'loftr':
        thr = kwargs.get('thr', 0.2)
        return LoFTRWrapper(device=device, thr=thr)
    elif method == 'roma':
        model_type = kwargs.get('model_type', 'large')
        return RoMaWrapper(device=device, model_type=model_type)
    else:
        raise ValueError(f"Unknown matcher method: {method}")