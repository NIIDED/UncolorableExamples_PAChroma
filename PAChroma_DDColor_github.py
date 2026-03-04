#!/usr/bin/env python
# coding: utf-8

import os
#!git clone https://github.com/piddnad/DDColor.git
os.chdir("DDColor")

import numpy as np
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image, to_tensor
from PIL import Image
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Resize, Compose
import torch.nn.functional as F
from torchvision.transforms import Resize
import csv
from torchvision.utils import save_image
from tqdm import tqdm
import lpips
import pandas as pd
import kornia.color as kc
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import random
import json
import glob
from torchvision.utils import make_grid
import torchvision.utils as vutils
import io
from skimage.metrics import structural_similarity as ssim
import torchvision.transforms as T
from torchvision import transforms
import math
import gc


# Load DDColor model
from ddcolor_model import DDColor

resize = Resize((256, 256))  # Define globally (so both grayscale & RGB use same)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_ground_truth_rgb(image_path):
    """
    Loads a ground truth RGB image as a tensor in [0, 1].

    Args:
        image_path (str): Path to the ground truth color image (RGB).

    Returns:
        torch.Tensor: Tensor of shape [3, H, W] in range [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(image)
    return tensor


def load_grayscale_tensor(img_path, size=(256, 256)):
    """
    Load grayscale image as a tensor [1, 1, H, W] in [0,1] using torchvision.io
    """
    img = read_image(img_path)  # [3, H, W], uint8 [0,255]
    img = TF.rgb_to_grayscale(img.float() / 255.0)  # → [1, H, W] in [0,1]
    img = F.interpolate(img.unsqueeze(0), size=size, mode='bilinear', align_corners=False)  # [1, 1, H, W]
    return img
    
def load_ground_truth_rgb_resize(image_path):
    """
    Loads a ground truth RGB image, resizes it, and returns as a tensor in [0, 1].

    Args:
        image_path (str): Path to the ground truth color image (RGB).

    Returns:
        torch.Tensor: Tensor of shape [3, 256, 256] in range [0, 1]
    """
    image = Image.open(image_path).convert('RGB')
    tensor = ToTensor()(image)
    tensor = resize(tensor)
    return tensor


def lab_to_rgb_torch(lab: torch.Tensor) -> torch.Tensor:
    """
    Numerically stable LAB to RGB conversion.
    Input: LAB [B,3,H,W], where L ∈ [0,100], a,b ∈ [-128,127]
    Output: RGB [B,3,H,W] in [0,1]
    """
    L, a, b = lab[:, 0:1], lab[:, 1:2], lab[:, 2:3]

    y = (L + 16.) / 116.
    x = a / 500. + y
    z = y - b / 200.

    xyz = torch.cat([x, y, z], dim=1)

    mask = xyz > 0.2068966
    xyz = torch.where(mask, xyz**3, (xyz - 16./116.) / 7.787)

    xyz[:, 0, :, :] *= 0.95047  # X
    xyz[:, 2, :, :] *= 1.08883  # Z

    matrix = torch.tensor([[ 3.2406, -1.5372, -0.4986],
                           [-0.9689,  1.8758,  0.0415],
                           [ 0.0557, -0.2040,  1.0570]], 
                           device=lab.device, dtype=lab.dtype)

    B, C, H, W = xyz.shape
    xyz_flat = xyz.permute(0, 2, 3, 1).reshape(-1, 3)  # [B*H*W, 3]
    rgb_flat = torch.matmul(xyz_flat, matrix.T)       # [B*H*W, 3]
    rgb = rgb_flat.reshape(B, H, W, 3).permute(0, 3, 1, 2)  # [B, 3, H, W]


    # Clamp before gamma correction
    rgb = torch.clamp(rgb, min=1e-5)

    # Apply gamma correction
    rgb = torch.where(rgb > 0.0031308,
                      1.055 * torch.pow(rgb, 1/2.4) - 0.055,
                      12.92 * rgb)

    return torch.clamp(rgb, 0, 1)


class DDColorWrapperLAB(torch.nn.Module):
    def __init__(self, model, input_size=256,return_ab=False):
        super().__init__()
        self.model = model.eval()
        self.model.requires_grad_(True)
        self.input_size = input_size
        self.return_ab = return_ab
        self.resize = Resize((input_size, input_size), antialias=True)

    def forward(self, x_gray):
        B, C, H, W = x_gray.shape
        assert C == 1, "Expected input of shape [B, 1, H, W]"

        # Save original L for fusion later
        L_orig = x_gray

        # Create fake RGB for input
        x_rgb_fake = x_gray.repeat(1, 3, 1, 1)
        x_rgb_resized = self.resize(x_rgb_fake)

        # Forward model to get ab
        ab = self.model(x_rgb_resized)  # shape: [B, 2, 256, 256]
        ab = F.interpolate(ab, size=(H, W), mode='bilinear', align_corners=False)  # match original H,W

        if self.return_ab:
            return ab  # <-- Directly expose ab for use in loss

        # Scale ranges for LAB
        L_scaled = L_orig * 100
        #ab_scaled = ab * 110

        #lab = torch.cat([L_scaled, ab_scaled], dim=1)  # shape: [B, 3, H, W]
        lab = torch.cat([L_scaled, ab], dim=1)

        #rgb = kc.lab_to_rgb(lab)
        rgb = lab_to_rgb_torch(lab)
        

        return rgb.clamp(0, 1)

def differentiable_colorfulness(rgb_tensor):
    # Assumes input in [0,1], shape [B,3,H,W]
    # Scale the tensor to [0, 255] inside differentiable_colorfulness
    rgb_tensor = (rgb_tensor * 255.0).clamp(0, 255)

    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    rg = torch.abs(r - g)
    yb = torch.abs(0.5 * (r + g) - b)

    std_rg = torch.std(rg, dim=[1,2])
    std_yb = torch.std(yb, dim=[1,2])
    mean_rg = torch.mean(rg, dim=[1,2])
    mean_yb = torch.mean(yb, dim=[1,2])

    colorfulness = torch.sqrt(std_rg**2 + std_yb**2) + 0.3 * torch.sqrt(mean_rg**2 + mean_yb**2)
    return colorfulness.mean()



def apply_jpeg_compression(tensor, quality=75):
    """
    Apply JPEG compression to a grayscale tensor [1,1,H,W] using Pillow.
    """
    pil_image = to_pil_image(tensor.squeeze(0).cpu())  # [1,1,H,W] → PIL
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality)
    buffer.seek(0)
    compressed = Image.open(buffer).convert('L')
    return to_tensor(compressed).unsqueeze(0).to(tensor.device)  # back to [1,1,H,W]

def visualize_jpeg_transferability(
    original_L,
    perturbed_L,
    model,
    jpeg_q1=90,
    jpeg_q2=70,
    save_path=None,
    show=False,
    evaluate = False 
):
    device = original_L.device
    model.eval()

    # Apply JPEG compression
    jpeg1 = apply_jpeg_compression(perturbed_L, quality=jpeg_q1)
    jpeg2 = apply_jpeg_compression(perturbed_L, quality=jpeg_q2)

    # Colorize
    with torch.no_grad():
        clean_color = model(original_L)
        perurb_color = model(perturbed_L)
        color_origin = differentiable_colorfulness(clean_color).item()
        color1 = model(jpeg1)
        color1_perturb_out = differentiable_colorfulness(color1).item()
        color2 = model(jpeg2)
        color2_perturb_out = differentiable_colorfulness(color2).item()
        

    # Clamp outputs
    color1 = torch.clamp(color1, 0, 1)
    color2 = torch.clamp(color2, 0, 1)
    

    # Prepare tensors
    imgs = [
        original_L.squeeze(0).squeeze(0).cpu().numpy(),
        clean_color.squeeze(0).cpu().permute(1, 2, 0).numpy(),
        # perturbed_L.squeeze(0).squeeze(0).cpu().numpy(),
        perturbed_L.squeeze(0).squeeze(0).detach().cpu().numpy(),
        perurb_color.squeeze(0).cpu().permute(1, 2, 0).numpy(),
        jpeg1.squeeze(0).squeeze(0).cpu().numpy(),
        color1.squeeze(0).cpu().permute(1, 2, 0).numpy(),
        jpeg2.squeeze(0).squeeze(0).cpu().numpy(),
        color2.squeeze(0).cpu().permute(1, 2, 0).numpy()
    ]

    titles = [
        "Original Input",
        "Clean Output",
        "Perturbed Input",
        "Perturbed Output",
        f"Quality={jpeg_q1} Perturbed Input",
        f"Quality={jpeg_q1} Perturbed Output",
        f"Quality={jpeg_q2} Perturbed Input",
        f"Quality={jpeg_q2} Perturbed Output",     
    ]

    # Plot
    plt.figure(figsize=(24, 5))
    for i, (img, title) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, 8, i)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved JPEG transferability visualization to {save_path}")
    if show:
        plt.show()   
        print(f"Clean output : {color_origin} ")
        print(f"jpeg_ql{jpeg_q1} : {color1_perturb_out} ")
        print(f"jpeg_ql{jpeg_q2} : {color2_perturb_out} ")
    plt.close()
    
    if evaluate: 
        return color1, color2





def visualize_pgd_attack(
    original_L, perturbation, perturbed_L, perturbed_output_rgb, ground_truth_rgb, 
    save_path=None, show=True
):
    """
    Visualize PGD Attack Results:
    1. Clean Grayscale
    2. Perturbation
    3. Perturbed Grayscale
    4. Perturbed Output (Colorized RGB)
    5. Ground Truth RGB
    """

    # Prepare tensors
    original_gray = original_L.squeeze(0).squeeze(0).detach().cpu().numpy()
    perturbed_gray = perturbed_L.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Normalize perturbation to [0,1] for visualization
    perturbation_vis = perturbation.squeeze(0).squeeze(0).detach().cpu().numpy()
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min() + 1e-8)

    perturbed_output_rgb = perturbed_output_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    ground_truth_rgb = ground_truth_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

    # Plot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(original_gray, cmap='gray')
    plt.title("Clean Input L")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(perturbation_vis, cmap='gray')
    plt.title("Perturbation")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(perturbed_gray, cmap='gray')
    plt.title("Perturbed L")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(perturbed_output_rgb)
    plt.title("Perturbed Output RGB")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(ground_truth_rgb)
    plt.title("Ground Truth RGB")
    plt.axis('off')

    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        # print(f"Saved visualization to {save_path}")

    if show:
        plt.show()

    plt.close()




def save_individual_images(
    original_L,
    perturbed_L,
    perturbation,
    perturbed_output_rgb,
    ground_truth_rgb,
    save_dir
):
    """
    Save each image individually:
      - original_L
      - perturbed_L
      - perturbation (normalized)
      - perturbed_output_rgb
      - ground_truth_rgb
    """

    def save_gray(tensor, filename):
        # Tensor: [1,1,H,W] → [1,H,W] → normalize to 0-1
        img = tensor.squeeze(0)  # [1,H,W]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        save_image(img, filename)

    def save_rgb(tensor, filename):
        img = tensor.squeeze(0)  # [3,H,W]
        img = torch.clamp(img, 0, 1)
        save_image(img, filename)

    # Save each
    save_gray(original_L, os.path.join(save_dir, "original_L.png"))
    save_gray(perturbed_L, os.path.join(save_dir, "perturbed_L.png"))

    # Normalize perturbation separately for visibility
    perturbation_vis = perturbation.squeeze(0)  # [1,H,W]
    perturbation_vis = (perturbation_vis - perturbation_vis.min()) / (perturbation_vis.max() - perturbation_vis.min() + 1e-8)
    save_image(perturbation_vis, os.path.join(save_dir, "perturbation.png"))

    save_rgb(perturbed_output_rgb, os.path.join(save_dir, "perturbed_output_rgb.png"))
    save_rgb(ground_truth_rgb, os.path.join(save_dir, "ground_truth_rgb.png"))

    torch.save(original_L.detach().cpu(), os.path.join(save_dir, "original_L.pt"))
    torch.save(perturbed_L.detach().cpu(), os.path.join(save_dir, "perturbed_L.pt"))


def PGD_colorfulness(
    grayscale_tensor, 
    model, 
    ground_truth_rgb_tensor,
    attack_name="pic_name",
    perturbation=None, 
    epsilon=0.05, 
    alpha=0.01, 
    num_iterations=100, 
    visualize_every=10,
    show=False,
    verbose=False,
    log_path=None,
    best_log_path=None,
    ground_truth_rgb_for_best=None,
    visualization_name = 'visualization',
    crop_params=None
):

    model.eval()

    # Ensure ab output mode
    model.return_ab = False
    
    device = grayscale_tensor.device
    save_root_dir = os.path.join(f"{visualization_name}/pgd_colorfulness",  attack_name,f"epsilon_{epsilon:.4f}", f"num_iteration_{num_iterations}")
    os.makedirs(save_root_dir, exist_ok=True)

    if perturbation is None:
        perturbation = torch.zeros_like(grayscale_tensor).to(device)

    best_loss = float('-inf')
    best_epoch = 0
    best_perturbed_L = None
    best_output_rgb = None
    best_perturbation = None
    epoch_logs = []

    for i in range(num_iterations):
        torch.cuda.empty_cache()
        perturbation = perturbation.detach().requires_grad_(True)
        perturbed_L = (grayscale_tensor + perturbation).clamp(0, 1)
        output_rgb = model(perturbed_L).to(perturbed_L.dtype)
        colorfulness_loss = differentiable_colorfulness(output_rgb)

        if -colorfulness_loss.item() > best_loss:
            best_loss = -colorfulness_loss.item()
            best_perturbed_L = perturbed_L.detach().clone()
            best_output_rgb = output_rgb.detach().clone()
            best_perturbation = perturbation.detach().clone()
            best_epoch = i

        with torch.no_grad():
            color_clean = differentiable_colorfulness(model(grayscale_tensor)).item()
            color_perturb = colorfulness_loss.item()
            delta_color = color_clean - color_perturb
            lpips_val = lpips_model(to_3ch(grayscale_tensor), to_3ch(perturbed_L)).mean().item()
            psnr = calculate_psnr_01(grayscale_tensor, perturbed_L)
            ssim_val = calculate_ssim_01(grayscale_tensor, perturbed_L)

        # epoch_logs.append([attack_name, i, epsilon, colorfulness_loss.item(), color_perturb, delta_color, lpips_val, psnr, ssim_val])

        grad = torch.autograd.grad(-colorfulness_loss, perturbation)[0]
        with torch.no_grad():
            perturbation += alpha * grad.sign()
            perturbation = perturbation.clamp(-epsilon, epsilon)

    best_folder = os.path.join(save_root_dir, f"best_attack_epoch_{best_epoch}")
    os.makedirs(best_folder, exist_ok=True)
    visualize_pgd_attack(grayscale_tensor, best_perturbation, best_perturbed_L, best_output_rgb, ground_truth_rgb_tensor, os.path.join(best_folder, 'visualization.png'), show)
    save_individual_images(grayscale_tensor, best_perturbed_L, best_perturbation, best_output_rgb, ground_truth_rgb_tensor, best_folder)
    jpeg_q1_output_best , jpeg_q2_output_best = visualize_jpeg_transferability(
        original_L=grayscale_tensor,
        perturbed_L=best_perturbed_L,
        model=model,
        jpeg_q1=75,
        jpeg_q2=50,
        save_path= os.path.join(best_folder, 'jpeg_transfer.png'),
        show=False,
        evaluate = True
    )
    resized_cf_best =visualize_randomresizecrop_transferability(
        original_L=grayscale_tensor,
        perturbed_L=best_perturbed_L,
        model=model,
        scale=(0.8, 1.0),
        save_path=os.path.join(best_folder, 'rrc_transferability.png'),
        show=False,
        evaluate=True,
        crop_params=crop_params
    )


    final_folder = os.path.join(save_root_dir, "final_attack")
    os.makedirs(final_folder, exist_ok=True)
    visualize_pgd_attack(grayscale_tensor, perturbation, perturbed_L, output_rgb, ground_truth_rgb_tensor, os.path.join(final_folder, 'visualization.png'), show)
    save_individual_images(grayscale_tensor, perturbed_L, perturbation, output_rgb, ground_truth_rgb_tensor, final_folder)
    jpeg_q1_output_fin , jpeg_q2_output_fin = visualize_jpeg_transferability(
        original_L=grayscale_tensor,
        perturbed_L=perturbed_L,
        model=model,
        jpeg_q1=75,
        jpeg_q2=50,
        save_path= os.path.join(final_folder, 'jpeg_transfer.png'),
        show=False,
        evaluate = True
    )
    resized_cf_final = visualize_randomresizecrop_transferability(
        original_L=grayscale_tensor,
        perturbed_L=perturbed_L,
        model=model,
        scale=(0.8, 1.0),
        save_path=os.path.join(final_folder, 'rrc_transferability.png'),
        show=False,
        evaluate=True,
        crop_params=crop_params
    )

    
    final_clean_output = model(grayscale_tensor).detach()
    
    if ground_truth_rgb_for_best is None:
        ground_truth_rgb_for_best = ground_truth_rgb_tensor
    
    final_metrics = {
        "image_name": attack_name,
        "epsilon": epsilon,
        "final_epoch": num_iterations - 1,
        "clean_colorfulness": differentiable_colorfulness(final_clean_output).item(),
        "perturb_colorfulness": differentiable_colorfulness(output_rgb).item(),
        "delta_colorfulness": differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item(),
        "colorfulness_reduction": (differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item()) / differentiable_colorfulness(final_clean_output).item() * 100,
        "mean_ab_clean": compute_mean_ab_magnitude(final_clean_output),
        "mean_ab_perturb": compute_mean_ab_magnitude(output_rgb),
        "LPIPS_vs_gray": lpips_model(to_3ch(grayscale_tensor), to_3ch(perturbed_L)).mean().item(),
        "PSNR_L_vs_perturbed": calculate_psnr_01(grayscale_tensor, perturbed_L),
        "SSIM_L_vs_perturbed": calculate_ssim_01(grayscale_tensor, perturbed_L),
        "LPIPS_clean_vs_GT": lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "LPIPS_perturb_vs_GT": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "delta_LPIPS": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "PSNR_clean_vs_GT": calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
        "PSNR_perturb_vs_GT": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best),
        "delta_PSNR": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
        "SSIM_clean_vs_GT": calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
        "SSIM_perturb_vs_GT": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best),
        "delta_SSIM": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
        "PSNR_clean_vs_perturb": calculate_psnr_rgb(final_clean_output, output_rgb),
        "SSIM_clean_vs_perturb": calculate_ssim_rgb(final_clean_output, output_rgb),
        "jpeg_75_colorfulness": differentiable_colorfulness(jpeg_q1_output_fin).item(),
        "jpeg_50_colorfulness": differentiable_colorfulness(jpeg_q2_output_fin).item(),
        "resized_cf" : resized_cf_final
    }

    final_log_path = best_log_path.replace("best_scores", "final_scores")
    df_final = pd.DataFrame([final_metrics])
    df_final.to_csv(final_log_path, mode='a', header=not os.path.exists(final_log_path), index=False)



    
    
    best_clean_output = model(grayscale_tensor).detach()
    best_metrics = {
        "image_name": attack_name,
        "epsilon": epsilon,
        "best_epoch": best_epoch,
        "clean_colorfulness": differentiable_colorfulness(best_clean_output).item(),
        "perturb_colorfulness": differentiable_colorfulness(best_output_rgb).item(),
        "delta_colorfulness": differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item(),
        "colorfulness_reduction": (differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item()) / differentiable_colorfulness(best_clean_output).item() * 100,
        "mean_ab_clean": compute_mean_ab_magnitude(best_clean_output),
        "mean_ab_perturb": compute_mean_ab_magnitude(best_output_rgb),
        "LPIPS_vs_gray": lpips_model(to_3ch(grayscale_tensor), to_3ch(best_perturbed_L)).mean().item(),
        "PSNR_L_vs_perturbed": calculate_psnr_01(grayscale_tensor, best_perturbed_L),
        "SSIM_L_vs_perturbed": calculate_ssim_01(grayscale_tensor, best_perturbed_L),
        "LPIPS_clean_vs_GT": lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "LPIPS_perturb_vs_GT": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "delta_LPIPS": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "PSNR_clean_vs_GT": calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
        "PSNR_perturb_vs_GT": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best),
        "delta_PSNR": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
        "SSIM_clean_vs_GT": calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
        "SSIM_perturb_vs_GT": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best),
        "delta_SSIM": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
        "PSNR_clean_vs_perturb": calculate_psnr_rgb(best_clean_output, best_output_rgb),
        "SSIM_clean_vs_perturb": calculate_ssim_rgb(best_clean_output, best_output_rgb),
        "jpeg_75_colorfulness" : differentiable_colorfulness(jpeg_q1_output_best).item(),
        "jpeg_50_colorfulness" : differentiable_colorfulness(jpeg_q2_output_best).item(),
        "resized_cf" : resized_cf_best

    }
    df_best = pd.DataFrame([best_metrics])
    df_best.to_csv(best_log_path, mode='a', header=not os.path.exists(best_log_path), index=False)    

    return -best_loss, best_perturbed_L, best_perturbation







# SIA using Multi-copy transformation
# SIA using MI-FGSM


def ensure_4d(fn):
    def wrapper(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        return fn(self, x)
    return wrapper


class SIA(torch.nn.Module):
    def __init__(self, model, epsilon=16/255, alpha=1.6/255, epoch=10, decay=1., num_copies=20, num_block=3,
                 random_start=False, laplacian_on=False, mask_transform=False, visualize=False, device=None):
        super().__init__()
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.epoch = epoch
        self.decay = decay
        self.num_copies = num_copies
        self.num_block = num_block
        self.device = device if device else next(model.parameters()).device
        self.random_start = random_start
        self.laplacian_on = laplacian_on
        self.mask_transform = mask_transform
        self.visualize = visualize
        self.loss_fn = torch.nn.MSELoss()

        self.op = [self.resize, self.vertical_shift, self.horizontal_shift,
                   self.vertical_flip, self.horizontal_flip, self.rotate180,
                   self.scale, self.add_noise, self.dct, self.drop_out]

        self.colorfulnesss = []
        self.psnrs = []
        self.ssims = []
        self.lpips_scores = []
        self.delta_colorfulnesss = []

    def forward_in_chunks(self, input_tensor, chunk_size=1):
        outputs = []
        for i in range(0, input_tensor.size(0), chunk_size):
            chunk = input_tensor[i:i+chunk_size]
            #with autocast():
            out = self.model(chunk)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

    # -------------------------------
    # Block-wise random transformation
    # -------------------------------
    def split_blocks(self, x):
        B, C, H, W = x.shape
        nb = self.num_block
        h_step = H // nb
        w_step = W // nb
        blocks = []
        for i in range(nb):
            for j in range(nb):
                blocks.append(x[:, :, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step])
        return blocks

    def combine_blocks(self, blocks, x_shape):
        B, C, H, W = x_shape
        nb = self.num_block
        h_step = H // nb
        w_step = W // nb
        out = torch.zeros((B, C, H, W), device=blocks[0].device, dtype=blocks[0].dtype)
        idx = 0
        for i in range(nb):
            for j in range(nb):
                out[:, :, i*h_step:(i+1)*h_step, j*w_step:(j+1)*w_step] = blocks[idx]
                idx += 1
        return out

    def block_transform(self, x):
        blocks = self.split_blocks(x)
        transformed_blocks = []
        for block in blocks:
            op = np.random.choice(self.op)
            transformed_blocks.append(op(block))
        return self.combine_blocks(transformed_blocks, x.shape)

    # -------------------------------
    # Transformation Operations
    # -------------------------------
    @ensure_4d
    def vertical_shift(self, x):
        step = np.random.randint(low=0, high=x.shape[2])
        return x.roll(step, dims=2)

    @ensure_4d
    def horizontal_shift(self, x):
        step = np.random.randint(low=0, high=x.shape[3])
        return x.roll(step, dims=3)

    @ensure_4d
    def vertical_flip(self, x):
        return x.flip(dims=(2,))

    @ensure_4d
    def horizontal_flip(self, x):
        return x.flip(dims=(3,))

    @ensure_4d
    def rotate180(self, x):
        return x.rot90(k=2, dims=(2, 3))

    @ensure_4d
    def scale(self, x):
        return torch.rand(1, device=x.device)[0] * x

    @ensure_4d
    def resize(self, x):
        _, _, w, h = x.shape
        new_w, new_h = int(w * 0.8) + 1, int(h * 0.8) + 1
        x = F.interpolate(x, size=(new_w, new_h), mode='bilinear', align_corners=False)
        x = F.interpolate(x, size=(w, h), mode='bilinear', align_corners=False).clamp(-1, 1)
        return x

    @ensure_4d
    def dct(self, x):
        dctx = dct_2d(x)
        _, _, w, h = dctx.shape
        dctx[:, :, -int(w*0.4):, :] = 0
        dctx[:, :, :, -int(h*0.4):] = 0
        return idct_2d(dctx)

    @ensure_4d
    def add_noise(self, x):
        return torch.clip(x + torch.zeros_like(x).uniform_(-16/255, 16/255), -1, 1)

    @ensure_4d
    def drop_out(self, x):
        return F.dropout2d(x, p=0.1, training=True)

    # -------------------------------
    # Attack Utilities
    # -------------------------------
    def laplacian_mask(self, x):
        kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=x.device).unsqueeze(0).unsqueeze(0)
        edge = F.conv2d(x, kernel, padding=1).abs()
        edge = (edge - edge.amin(dim=(2, 3), keepdim=True)) / (edge.amax(dim=(2, 3), keepdim=True) - edge.amin(dim=(2, 3), keepdim=True) + 1e-8)
        return edge

    def calculate_psnr(self, L, perturbed_L, max_value=255.0):
        original = (L.detach().cpu().numpy()[0, 0] * 255).astype(np.float32)
        perturbed = (perturbed_L.detach().cpu().numpy()[0, 0] * 255).astype(np.float32)
        mse = np.mean((original - perturbed) ** 2)
        return float('inf') if mse == 0 else 10 * np.log10((max_value ** 2) / mse)

    def calculate_ssim(self, L, perturbed_L):
        original = (L.detach().cpu().numpy()[0, 0] * 255).astype(np.float32)
        perturbed = (perturbed_L.detach().cpu().numpy()[0, 0] * 255).astype(np.float32)
        return ssim(original, perturbed, data_range=255)

    def calculate_psnr_rgb(self, img1, img2):
        mse = F.mse_loss(img1, img2).item()
        return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)

    def calculate_ssim_rgb(self, img1, img2):
        img1_np = img1.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        img2_np = img2.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
        return ssim(img1_np, img2_np, data_range=1.0, channel_axis=-1)


    
    def attack(
        self,
        L,
        ground_truth_rgb_tensor=None,
        save_name="sia_attack",
        save_dir="visualizations",
        attack_name="00000",
        final_log_path=None, 
        best_log_path=None,
        crop_params=None
    ):
        torch.cuda.empty_cache()
        gc.collect()

        self.model.return_ab = False
        L = L.clone().detach().to(self.device)
        delta = torch.zeros_like(L).to(self.device)
        if self.random_start:
            delta.uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad_()
    
        best_colorfulness = float('inf')
        best_perturbed_L, best_output_rgb, best_perturbation = None, None, None
        epoch_logs = []
        save_root_dir = os.path.join(save_dir, save_name, f"{attack_name}",  f"epsilon_{self.epsilon:.4f}", f"num_iteration_{self.epoch}")
        os.makedirs(save_root_dir, exist_ok=True)

        # Initialize outside loop once:
        momentum = torch.zeros_like(delta)
    
        for epoch_num in tqdm(range(self.epoch), desc='SIA Epochs'):
            torch.cuda.empty_cache()
            gc.collect()

            # Generate transformed inputs
            transformed_copies = [self.block_transform(L + delta) for _ in range(self.num_copies)]
            transformed_L_copies = torch.cat(transformed_copies, dim=0)
            output_rgb_copies = self.forward_in_chunks(transformed_L_copies, chunk_size=10)
            loss = -differentiable_colorfulness(output_rgb_copies)
            
            # Compute gradient w.r.t. delta (requires trick to ensure delta influences all copies)
            grad = torch.autograd.grad(loss, delta, retain_graph=False, create_graph=False)[0]
            
            # Normalize gradient
            grad = grad / (grad.abs().mean(dim=(1, 2, 3), keepdim=True) + 1e-8)
            
            # Compute soft Laplacian mask (no thresholding, keep continuous values)
            if self.laplacian_on:
                mask = self.laplacian_mask(L if not self.mask_transform else self.block_transform(L))
            else:
                mask = torch.ones_like(grad)
            
            # Accumulate momentum
            momentum = self.decay * momentum + grad
            
            # Update delta with edge-weighted direction
            delta = delta + self.alpha * momentum.sign() * (mask + 1e-6)  # 💡 continuous mask controls update strength
            delta = torch.clamp(delta, -self.epsilon, self.epsilon)
            perturbed_L = torch.clamp(L + delta, 0, 1)
        
            # Output for evaluation/logging
            output_rgb = self.model(perturbed_L).clamp(0, 1)
            color_out = differentiable_colorfulness(output_rgb).item()

    
            to_3ch = lambda x: x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x
            lpips_val = lpips_model(to_3ch(L), to_3ch(perturbed_L)).mean().item()
            psnr = self.calculate_psnr(L, perturbed_L)
            ssim_val = self.calculate_ssim(L, perturbed_L)
    
            # Save best result
            if color_out < best_colorfulness:
                best_colorfulness = color_out
                best_perturbed_L = perturbed_L.clone().detach()
                best_output_rgb = output_rgb.clone().detach()
                best_perturbation = delta.clone().detach()
                best_epoch = epoch_num
            gc.collect()
            torch.cuda.empty_cache()
    

        # Save best result and best metrics
        if best_perturbed_L is not None:
            best_folder = os.path.join(save_root_dir, f"best_attack_epoch_{best_epoch}")
            os.makedirs(best_folder, exist_ok=True)
            visualize_pgd_attack(L, best_perturbation, best_perturbed_L, best_output_rgb, ground_truth_rgb_tensor,
                                 os.path.join(best_folder, "visualization.png"), show=self.visualize)
            save_individual_images(L, best_perturbed_L, best_perturbation, best_output_rgb, ground_truth_rgb_tensor, best_folder)
            jpeg_q1_output_best , jpeg_q2_output_best = visualize_jpeg_transferability(
                original_L=L,
                perturbed_L=best_perturbed_L,
                model=self.model,
                jpeg_q1=75,
                jpeg_q2=50,
                save_path= os.path.join(best_folder, 'jpeg_transfer.png'),
                show=False,
                evaluate = True
            )
            resized_cf_best = visualize_randomresizecrop_transferability(
                original_L=L,
                perturbed_L=best_perturbed_L,
                model=self.model,
                scale=(0.8, 1.0),
                save_path=os.path.join(best_folder, 'rrc_transferability.png'),
                show=False,
                evaluate=True,
                crop_params=crop_params
            )
            final_folder = os.path.join(save_root_dir, "final_attack")
            os.makedirs(final_folder, exist_ok=True)
            visualize_pgd_attack(L, delta, perturbed_L, output_rgb, ground_truth_rgb_tensor, os.path.join(final_folder, 'visualization.png'), show=self.visualize)
            save_individual_images(L, perturbed_L, delta, output_rgb, ground_truth_rgb_tensor, final_folder)
            jpeg_q1_output_fin , jpeg_q2_output_fin = visualize_jpeg_transferability(
                original_L=L,
                perturbed_L=perturbed_L,
                model=self.model,
                jpeg_q1=75,
                jpeg_q2=50,
                save_path= os.path.join(final_folder, 'jpeg_transfer.png'),
                show=False,
                evaluate = True
            )
            resized_cf_final = visualize_randomresizecrop_transferability(
                original_L=L,
                perturbed_L=perturbed_L,
                model=self.model,
                scale=(0.8, 1.0),
                save_path=os.path.join(final_folder, 'rrc_transferability.png'),
                show=False,
                evaluate=True,
                crop_params=crop_params
            )
            
            
            # if ground_truth_rgb_for_best is None:
            ground_truth_rgb_for_best = ground_truth_rgb_tensor
                
            self.model.return_ab = False
            best_clean_output = self.model(L).detach()
            final_clean_output = self.model(L).detach()
            clean_color = differentiable_colorfulness(best_clean_output).item()
            perturb_color = differentiable_colorfulness(best_output_rgb).item()
            delta_color = clean_color - perturb_color
            
            final_metrics = {
                "image_name": attack_name,
                "epsilon": self.epsilon,
                "final_epoch": self.epoch - 1,
                "clean_colorfulness": differentiable_colorfulness(final_clean_output).item(),
                "perturb_colorfulness": differentiable_colorfulness(output_rgb).item(),
                "delta_colorfulness": differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item(),
                "colorfulness_reduction": (differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item()) / differentiable_colorfulness(final_clean_output).item() * 100,
                "mean_ab_clean": compute_mean_ab_magnitude(final_clean_output),
                "mean_ab_perturb": compute_mean_ab_magnitude(output_rgb),
                "LPIPS_vs_gray": lpips_model(to_3ch(L), to_3ch(perturbed_L)).mean().item(),
                "PSNR_L_vs_perturbed": self.calculate_psnr(L, perturbed_L),
                "SSIM_L_vs_perturbed": self.calculate_ssim(L, perturbed_L),
                "LPIPS_clean_vs_GT": lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "LPIPS_perturb_vs_GT": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "delta_LPIPS": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "PSNR_clean_vs_GT": calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
                "PSNR_perturb_vs_GT": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best),
                "delta_PSNR": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
                "SSIM_clean_vs_GT": calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
                "SSIM_perturb_vs_GT": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best),
                "delta_SSIM": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
                "PSNR_clean_vs_perturb": self.calculate_psnr_rgb(final_clean_output, output_rgb),
                "SSIM_clean_vs_perturb": self.calculate_ssim_rgb(final_clean_output, output_rgb),
                "jpeg_75_colorfulness": differentiable_colorfulness(jpeg_q1_output_fin).item(),
                "jpeg_50_colorfulness": differentiable_colorfulness(jpeg_q2_output_fin).item(),
                "resized_cf" : resized_cf_final
            }
        
            df_final = pd.DataFrame([final_metrics])
            df_final.to_csv(final_log_path, mode='a', header=not os.path.exists(final_log_path), index=False)
        
            
            
            best_metrics = {
                "image_name": attack_name,
                "epsilon": self.epsilon,
                "best_epoch": best_epoch,
                "clean_colorfulness": differentiable_colorfulness(best_clean_output).item(),
                "perturb_colorfulness": differentiable_colorfulness(best_output_rgb).item(),
                "delta_colorfulness": differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item(),
                "colorfulness_reduction": (differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item()) / differentiable_colorfulness(best_clean_output).item() * 100,
                "mean_ab_clean": compute_mean_ab_magnitude(best_clean_output),
                "mean_ab_perturb": compute_mean_ab_magnitude(best_output_rgb),
                "LPIPS_vs_gray": lpips_model(to_3ch(L), to_3ch(best_perturbed_L)).mean().item(),
                "PSNR_L_vs_perturbed": self.calculate_psnr(L, best_perturbed_L),
                "SSIM_L_vs_perturbed": self.calculate_ssim(L, best_perturbed_L),
                "LPIPS_clean_vs_GT": lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "LPIPS_perturb_vs_GT": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "delta_LPIPS": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
                "PSNR_clean_vs_GT": calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
                "PSNR_perturb_vs_GT": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best),
                "delta_PSNR": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
                "SSIM_clean_vs_GT": calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
                "SSIM_perturb_vs_GT": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best),
                "delta_SSIM": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
                "PSNR_clean_vs_perturb": self.calculate_psnr_rgb(best_clean_output, best_output_rgb),
                "SSIM_clean_vs_perturb": self.calculate_ssim_rgb(best_clean_output, best_output_rgb),
                "jpeg_75_colorfulness" : differentiable_colorfulness(jpeg_q1_output_best).item(),
                "jpeg_50_colorfulness" : differentiable_colorfulness(jpeg_q2_output_best).item(),
                "resized_cf" : resized_cf_best
        
            }
            df_best = pd.DataFrame([best_metrics])
            df_best.to_csv(best_log_path, mode='a', header=not os.path.exists(best_log_path), index=False)    
        
            return best_perturbed_L, best_output_rgb, best_perturbation
        
def calculate_psnr_rgb(img1, img2):
    if img1.ndim == 4:
        img1_np = (img1[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img1_np = (img1.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    if img2.ndim == 4:
        img2_np = (img2[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img2_np = (img2.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return peak_signal_noise_ratio(img1_np, img2_np, data_range=255.0)

def calculate_ssim_rgb(img1, img2):
    if img1.ndim == 4:
        img1_np = (img1[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img1_np = (img1.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    if img2.ndim == 4:
        img2_np = (img2[0].detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    else:
        img2_np = (img2.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)

    return structural_similarity(img1_np, img2_np, channel_axis=2, data_range=255.0)

# -------------------------------
# DCT Utilities (you already have)
# -------------------------------
def create_dct_matrix(N, device='cpu'):
    mat = torch.zeros(N, N, device=device)
    for k in range(N):
        for n in range(N):
            alpha = math.sqrt(1 / N) if k == 0 else math.sqrt(2 / N)
            mat[k, n] = alpha * math.cos(math.pi * (2 * n + 1) * k / (2 * N))
    return mat

def dct_2d(x):
    B, C, H, W = x.shape
    dct_mat_h = create_dct_matrix(H, device=x.device)
    dct_mat_w = create_dct_matrix(W, device=x.device)
    x = x.view(B * C, H, W)
    x = torch.matmul(dct_mat_h, x)
    x = torch.matmul(x, dct_mat_w.T)
    return x.view(B, C, H, W)

def idct_2d(X):
    B, C, H, W = X.shape
    dct_mat_h = create_dct_matrix(H, device=X.device)
    dct_mat_w = create_dct_matrix(W, device=X.device)
    X = X.view(B * C, H, W)
    X = torch.matmul(dct_mat_h.T, X)
    X = torch.matmul(X, dct_mat_w)
    return X.view(B, C, H, W)
 





def run_sia_batch_attack(
    image_folder,
    sia_attack_instance,  # Initialized instance of SIA class
    final_log_path,
    best_log_path,
    max_images=None,
    save_name="sia_attack",
    save_dir="visualizations",
    image_paths = None,
    crop_params=None
):
    os.makedirs(os.path.dirname(best_log_path), exist_ok=True)
    os.makedirs(os.path.dirname(final_log_path), exist_ok=True)
    
    if not os.path.exists(best_log_path):
        pd.DataFrame(columns=[
            "image_name", "epsilon", "best_epoch",
            "clean_colorfulness", "perturb_colorfulness", "delta_colorfulness", "colorfulness_reduction",
            "mean_ab_clean", "mean_ab_perturb",
            "LPIPS_vs_gray", "PSNR_L_vs_perturbed", "SSIM_L_vs_perturbed",
            "LPIPS_clean_vs_GT", "LPIPS_perturb_vs_GT", "delta_LPIPS",
            "PSNR_clean_vs_GT", "PSNR_perturb_vs_GT", "delta_PSNR",
            "SSIM_clean_vs_GT", "SSIM_perturb_vs_GT", "delta_SSIM",
            "PSNR_clean_vs_perturb", "SSIM_clean_vs_perturb",
            "jpeg_75_colorfulness", "jpeg_50_colorfulness","resized_cf"
        ]).to_csv(best_log_path, index=False)

    if not os.path.exists(final_log_path):
        pd.DataFrame(columns=[
            "image_name", "epsilon", "final_epoch",
            "clean_colorfulness", "perturb_colorfulness", "delta_colorfulness", "colorfulness_reduction",
            "mean_ab_clean", "mean_ab_perturb",
            "LPIPS_vs_gray", "PSNR_L_vs_perturbed", "SSIM_L_vs_perturbed",
            "LPIPS_clean_vs_GT", "LPIPS_perturb_vs_GT", "delta_LPIPS",
            "PSNR_clean_vs_GT", "PSNR_perturb_vs_GT", "delta_PSNR",
            "SSIM_clean_vs_GT", "SSIM_perturb_vs_GT", "delta_SSIM",
            "PSNR_clean_vs_perturb", "SSIM_clean_vs_perturb",
            "jpeg_75_colorfulness", "jpeg_50_colorfulness","resized_cf"
        ]).to_csv(final_log_path, index=False)
   
    for img_path in tqdm(image_paths, desc="Running SIA for all images"):
        try:
            image_name = os.path.basename(img_path).rsplit(".", 1)[0]

            # Load grayscale and GT color (implement these according to your pipeline)
            grayscale_tensor = load_grayscale_tensor(img_path).to(sia_attack_instance.device)
            ground_truth_rgb_tensor = load_ground_truth_rgb_resize(img_path).to(sia_attack_instance.device)

            torch.cuda.empty_cache()
            gc.collect()

            # Run the attack
            sia_attack_instance.attack(
                L=grayscale_tensor,
                ground_truth_rgb_tensor=ground_truth_rgb_tensor,
                save_name=save_name,
                save_dir=save_dir,
                attack_name=image_name,
                final_log_path=final_log_path,
                best_log_path=best_log_path,
                crop_params=crop_params
            )

        except Exception as e:
            print(f"Failed on {img_path}: {e}")
            with open(os.path.join(save_dir, "sia_failed_images.txt"), 'a') as err_log:
                err_log.write(f"{img_path}: {str(e)}\n")




def visualize_randomresizecrop_transferability(
    original_L,
    perturbed_L,
    model,
    size=256,
    scale=(0.8, 1.0),
    save_path=None,
    show=False,
    evaluate=False,
    ratio = (3/4, 4/3),
    crop_params=None
):
    device = original_L.device
    model.eval()

    # Apply same crop to both clean and perturbed L
    def apply_fixed_crop(tensor, crop_params):
        i, j, h, w = crop_params
        to_pil = T.ToPILImage()
        to_tensor = T.ToTensor()
        pil_img = to_pil(tensor.squeeze(0).cpu())
        cropped = TF.resized_crop(pil_img, i, j, h, w, size=(size, size))
        return to_tensor(cropped).unsqueeze(0).to(device)

    cropped_L = apply_fixed_crop(perturbed_L, crop_params)
    resized_back = F.interpolate(cropped_L, size=perturbed_L.shape[-2:], mode='bilinear', align_corners=False)

    with torch.no_grad():
        clean_color = model(original_L)
        perturbed_color = model(perturbed_L)
        resized_color = model(resized_back)

        clean_cf = differentiable_colorfulness(clean_color).item()
        perturbed_cf = differentiable_colorfulness(perturbed_color).item()
        resized_cf = differentiable_colorfulness(resized_color).item()



    imgs = [
        original_L.squeeze().detach().cpu().numpy(),
        clean_color.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
        perturbed_L.squeeze().detach().cpu().numpy(),
        perturbed_color.squeeze().detach().cpu().permute(1, 2, 0).numpy(),
        resized_back.squeeze().detach().cpu().numpy(),
        resized_color.squeeze().detach().cpu().permute(1, 2, 0).numpy()
    ]

    titles = [
        "Original Input",
        "Clean Output",
        "Perturbed Input",
        "Perturbed Output",
        f"RandomResizedCrop (resized back)",
        "Output after RRC"
    ]

    plt.figure(figsize=(20, 5))
    for i, (img, title) in enumerate(zip(imgs, titles), 1):
        plt.subplot(1, 6, i)
        if img.ndim == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved RandomResizedCrop transferability visualization to {save_path}")
    if show:
        plt.show()
        print(f"Clean output colorfulness: {clean_cf}")
        print(f"Perturbed output colorfulness: {perturbed_cf}")
        print(f"After RRC colorfulness: {resized_cf}")
    plt.close()

    if evaluate:
        return resized_cf
    else:
        return None






def PGD_colorfulness_lap(
    grayscale_tensor, 
    model, 
    ground_truth_rgb_tensor,
    laplacian_map_torch,
    attack_name="pic_name",
    perturbation=None, 
    epsilon=0.05, 
    alpha=0.01, 
    num_iterations=100, 
    visualize_every=10,
    show = False,
    verbose = False,
    log_path=None,
    best_log_path=None,
    ground_truth_rgb_for_best=None,
    visualization_name = "visualization",
    crop_params=None
):
    device = grayscale_tensor.device
    model.eval()

    # Ensure ab output mode
    model.return_ab = False


    # Save root folder: DeOldify/visualizations/pgd_rgb/{attack_name}/epsilon_{epsilon}/
    save_root_dir = os.path.join(f"{visualization_name}/pgd_colorfulness_lap",  attack_name,f"epsilon_{epsilon:.4f}", f"num_iteration_{num_iterations}")
    os.makedirs(save_root_dir, exist_ok=True)

    if perturbation is None:
        perturbation = torch.zeros_like(grayscale_tensor).to(device)

    best_loss = float('-inf')
    best_epoch = 0
    best_perturbed_L = None
    best_output_rgb = None
    best_perturbation = None
    epoch_logs = []
    
    # Normalize Laplacian map to [0,1] for scaling
    laplacian_map_torch = laplacian_map_torch.to(device)
    laplacian_map_torch = (laplacian_map_torch - laplacian_map_torch.min()) / (laplacian_map_torch.max() - laplacian_map_torch.min())


    for i in range(num_iterations):
    #for i in range(num_iterations):
        torch.cuda.empty_cache()
        perturbation = perturbation.detach().requires_grad_(True)
        perturbed_L = (grayscale_tensor + perturbation).clamp(0, 1)
        output_rgb = model(perturbed_L)
        output_rgb = output_rgb.to(perturbed_L.dtype)
        colorfulness_loss = differentiable_colorfulness(output_rgb)

        # Track best attack
        if -colorfulness_loss.item() > best_loss:
            best_loss = -colorfulness_loss.item()
            best_perturbed_L = perturbed_L.detach().clone()
            best_output_rgb = output_rgb.detach().clone()
            best_perturbation = perturbation.detach().clone()
            best_epoch = i

        with torch.no_grad():
            color_clean = differentiable_colorfulness(model(grayscale_tensor)).item()
            color_perturb = colorfulness_loss.item()
            delta_color = color_clean - color_perturb
            lpips_val = lpips_model(to_3ch(grayscale_tensor), to_3ch(perturbed_L)).mean().item()
            psnr = calculate_psnr_01(grayscale_tensor, perturbed_L)
            ssim_val = calculate_ssim_01(grayscale_tensor, perturbed_L)

        # epoch_logs.append([attack_name, i, epsilon, colorfulness_loss.item(), color_perturb, delta_color, lpips_val, psnr, ssim_val])

        grad = torch.autograd.grad(-colorfulness_loss, perturbation)[0]
        with torch.no_grad():
            perturbation += alpha * grad.sign() * (laplacian_map_torch + 1e-6)  # Avoid zero division
            perturbation = perturbation.clamp(-epsilon, epsilon)

    best_folder = os.path.join(save_root_dir, f"best_attack_epoch_{best_epoch}")
    os.makedirs(best_folder, exist_ok=True)
    visualize_pgd_attack(grayscale_tensor, best_perturbation, best_perturbed_L, best_output_rgb, ground_truth_rgb_tensor, os.path.join(best_folder, 'visualization.png'), show)
    save_individual_images(grayscale_tensor, best_perturbed_L, best_perturbation, best_output_rgb, ground_truth_rgb_tensor, best_folder)
    jpeg_q1_output_best , jpeg_q2_output_best = visualize_jpeg_transferability(
        original_L=grayscale_tensor,
        perturbed_L=best_perturbed_L,
        model=model,
        jpeg_q1=75,
        jpeg_q2=50,
        save_path= os.path.join(best_folder, 'jpeg_transfer.png'),
        show=False,
        evaluate = True
    )
    resized_cf_best = visualize_randomresizecrop_transferability(
        original_L=grayscale_tensor,
        perturbed_L=best_perturbed_L,
        model=model,
        scale=(0.8, 1.0),
        save_path=os.path.join(best_folder, 'rrc_transferability.png'),
        show=False,
        evaluate=True,
        crop_params=crop_params
    )
    final_folder = os.path.join(save_root_dir, "final_attack")
    os.makedirs(final_folder, exist_ok=True)
    visualize_pgd_attack(grayscale_tensor, perturbation, perturbed_L, output_rgb, ground_truth_rgb_tensor, os.path.join(final_folder, 'visualization.png'), show)
    save_individual_images(grayscale_tensor, perturbed_L, perturbation, output_rgb, ground_truth_rgb_tensor, final_folder)
    jpeg_q1_output_fin , jpeg_q2_output_fin = visualize_jpeg_transferability(
        original_L=grayscale_tensor,
        perturbed_L=perturbed_L,
        model=model,
        jpeg_q1=75,
        jpeg_q2=50,
        save_path= os.path.join(final_folder, 'jpeg_transfer.png'),
        show=False,
        evaluate = True
    )
    resized_cf_final = visualize_randomresizecrop_transferability(
        original_L=grayscale_tensor,
        perturbed_L=perturbed_L,
        model=model,
        scale=(0.8, 1.0),
        save_path=os.path.join(final_folder, 'rrc_transferability.png'),
        show=False,
        evaluate=True,
        crop_params=crop_params
    )

    final_clean_output = model(grayscale_tensor).detach()
    
    if ground_truth_rgb_for_best is None:
        ground_truth_rgb_for_best = ground_truth_rgb_tensor
    
    final_metrics = {
        "image_name": attack_name,
        "epsilon": epsilon,
        "final_epoch": num_iterations - 1,
        "clean_colorfulness": differentiable_colorfulness(final_clean_output).item(),
        "perturb_colorfulness": differentiable_colorfulness(output_rgb).item(),
        "delta_colorfulness": differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item(),
        "colorfulness_reduction": (differentiable_colorfulness(final_clean_output).item() - differentiable_colorfulness(output_rgb).item()) / differentiable_colorfulness(final_clean_output).item() * 100,
        "mean_ab_clean": compute_mean_ab_magnitude(final_clean_output),
        "mean_ab_perturb": compute_mean_ab_magnitude(output_rgb),
        "LPIPS_vs_gray": lpips_model(to_3ch(grayscale_tensor), to_3ch(perturbed_L)).mean().item(),
        "PSNR_L_vs_perturbed": calculate_psnr_01(grayscale_tensor, perturbed_L),
        "SSIM_L_vs_perturbed": calculate_ssim_01(grayscale_tensor, perturbed_L),
        "LPIPS_clean_vs_GT": lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "LPIPS_perturb_vs_GT": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "delta_LPIPS": lpips_model(to_3ch(output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(final_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "PSNR_clean_vs_GT": calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
        "PSNR_perturb_vs_GT": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best),
        "delta_PSNR": calculate_psnr_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(final_clean_output, ground_truth_rgb_for_best),
        "SSIM_clean_vs_GT": calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
        "SSIM_perturb_vs_GT": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best),
        "delta_SSIM": calculate_ssim_rgb(output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(final_clean_output, ground_truth_rgb_for_best),
        "PSNR_clean_vs_perturb": calculate_psnr_rgb(final_clean_output, output_rgb),
        "SSIM_clean_vs_perturb": calculate_ssim_rgb(final_clean_output, output_rgb),
        "jpeg_75_colorfulness": differentiable_colorfulness(jpeg_q1_output_fin).item(),
        "jpeg_50_colorfulness": differentiable_colorfulness(jpeg_q2_output_fin).item(),
        "resized_cf" : resized_cf_final
    }

    final_log_path = best_log_path.replace("best_scores", "final_scores")
    df_final = pd.DataFrame([final_metrics])
    df_final.to_csv(final_log_path, mode='a', header=not os.path.exists(final_log_path), index=False)



    
    
    best_clean_output = model(grayscale_tensor).detach()
    best_metrics = {
        "image_name": attack_name,
        "epsilon": epsilon,
        "best_epoch": best_epoch,
        "clean_colorfulness": differentiable_colorfulness(best_clean_output).item(),
        "perturb_colorfulness": differentiable_colorfulness(best_output_rgb).item(),
        "delta_colorfulness": differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item(),
        "colorfulness_reduction": (differentiable_colorfulness(best_clean_output).item() - differentiable_colorfulness(best_output_rgb).item()) / differentiable_colorfulness(best_clean_output).item() * 100,
        "mean_ab_clean": compute_mean_ab_magnitude(best_clean_output),
        "mean_ab_perturb": compute_mean_ab_magnitude(best_output_rgb),
        "LPIPS_vs_gray": lpips_model(to_3ch(grayscale_tensor), to_3ch(best_perturbed_L)).mean().item(),
        "PSNR_L_vs_perturbed": calculate_psnr_01(grayscale_tensor, best_perturbed_L),
        "SSIM_L_vs_perturbed": calculate_ssim_01(grayscale_tensor, best_perturbed_L),
        "LPIPS_clean_vs_GT": lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "LPIPS_perturb_vs_GT": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "delta_LPIPS": lpips_model(to_3ch(best_output_rgb), to_3ch(ground_truth_rgb_for_best)).mean().item() - lpips_model(to_3ch(best_clean_output), to_3ch(ground_truth_rgb_for_best)).mean().item(),
        "PSNR_clean_vs_GT": calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
        "PSNR_perturb_vs_GT": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best),
        "delta_PSNR": calculate_psnr_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_psnr_rgb(best_clean_output, ground_truth_rgb_for_best),
        "SSIM_clean_vs_GT": calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
        "SSIM_perturb_vs_GT": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best),
        "delta_SSIM": calculate_ssim_rgb(best_output_rgb, ground_truth_rgb_for_best) - calculate_ssim_rgb(best_clean_output, ground_truth_rgb_for_best),
        "PSNR_clean_vs_perturb": calculate_psnr_rgb(best_clean_output, best_output_rgb),
        "SSIM_clean_vs_perturb": calculate_ssim_rgb(best_clean_output, best_output_rgb),
        "jpeg_75_colorfulness" : differentiable_colorfulness(jpeg_q1_output_best).item(),
        "jpeg_50_colorfulness" : differentiable_colorfulness(jpeg_q2_output_best).item(),
        "resized_cf" : resized_cf_best

    }
    df_best = pd.DataFrame([best_metrics])
    df_best.to_csv(best_log_path, mode='a', header=not os.path.exists(best_log_path), index=False)    

    return -best_loss, best_perturbed_L, best_perturbation





def compute_mean_ab_magnitude(tensor, input_color_space="rgb"):
    """
    Compute mean absolute value of ab channels in LAB.
    Args:
        tensor: [B, 3, H, W] - RGB or LAB tensor
        input_color_space: 'rgb', 'lab', or 'auto' to infer
    Returns:
        mean_ab: scalar float
    """
    if input_color_space == "rgb":
        tensor = kc.rgb_to_lab(tensor.clamp(0, 1))  # convert RGB to LAB

    ab = tensor[:, 1:, :, :]  # take a and b channels
    mean_ab = ab.abs().mean()

    return mean_ab.item()

def calculate_psnr_minus11(L, perturbed_L, max_value=255.0):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Args:
    original_image (torch.Tensor): The original image tensor of shape (C, H, W) or (H, W).
    distorted_image (torch.Tensor): The distorted image tensor of shape (C, H, W) or (H, W).
    max_value (float): The maximum possible pixel value in the image. For images in range [0, 1], max_value = 1.0.
    
    Returns:
    float: The PSNR value in dB.
    """

    perturbed_gray = perturbed_L.detach()
    original_gray = L.detach()
    
    original_gray = ((original_gray + 1) * 127.5).cpu().numpy()[0, 0]
    perturbed_gray = ((perturbed_gray + 1) * 127.5).cpu().numpy()[0, 0]
    
    # Ensure the images are in the correct format (C, H, W)
    if original_gray.shape != perturbed_gray.shape:
        raise ValueError("The original and distorted images must have the same shape.")
    
    # Compute Mean Squared Error (MSE)
    mse = np.mean((perturbed_gray - original_gray) ** 2)
    
    # If MSE is 0 (images are identical), PSNR is infinite
    if mse == 0:
        return float('inf')
    
    # Calculate PSNR using the formula
    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr.item()


def calculate_psnr_01(L, perturbed_L, max_value=255.0):
    """
    Calculate PSNR between two grayscale tensors in [0, 1] range.
    """
    original_gray = (L.detach() * 255).cpu().numpy()[0, 0]
    perturbed_gray = (perturbed_L.detach() * 255).cpu().numpy()[0, 0]

    if original_gray.shape != perturbed_gray.shape:
        raise ValueError("Shapes must match for PSNR.")

    mse = np.mean((perturbed_gray - original_gray) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 10 * np.log10((max_value ** 2) / mse)
    return psnr

def calculate_ssim_minus11(L, perturbed_L):
    """
    Calculate the Structural Similarity Index (SSIM) between two images.
    
    Returns:
    float: The SSIM score (between -1 and 1), where 1 means the images are identical.
    """
    perturbed_gray = perturbed_L.detach()
    original_gray = L.detach()
    
    original_gray = ((original_gray + 1) * 127.5).cpu().numpy()[0, 0]
    perturbed_gray = ((perturbed_gray + 1) * 127.5).cpu().numpy()[0, 0]
    
    # Ensure the images are in the correct format (C, H, W)
    if original_gray.shape != perturbed_gray.shape:
        raise ValueError("The original and distorted images must have the same shape.")
        
    # Compute SSIM
    ssim_value = ssim(original_gray, perturbed_gray, data_range=perturbed_gray.max() - perturbed_gray.min())

    return ssim_value


def calculate_ssim_01(L, perturbed_L):
    """
    Calculate SSIM between two grayscale images in [0, 1] range.
    Returns SSIM value between -1 and 1.
    """
    original_gray = (L.detach() * 255).cpu().numpy()[0, 0]
    perturbed_gray = (perturbed_L.detach() * 255).cpu().numpy()[0, 0]

    if original_gray.shape != perturbed_gray.shape:
        raise ValueError("Shapes must match for SSIM.")

    return ssim(original_gray, perturbed_gray, data_range=255.0)


def to_3ch(x):
    return x.expand(-1, 3, -1, -1) if x.shape[1] == 1 else x

def normalize_for_lpips(x):
    return x * 2 - 1  # convert from [0, 1] → [-1, 1]


    
    # Convert HWC numpy image [0,1] → PyTorch tensor [1, 3, H, W]
def to_tensor_batch(np_img):
    tensor = TF.to_tensor(np_img).unsqueeze(0)  # [1, 3, H, W]


def to_float(x):
    if isinstance(x, torch.Tensor):
        return x.item()
    elif isinstance(x, np.ndarray):
        return float(x)
    else:
        return x  # assume it's already float


def visualize_transferability(
    grayscale_tensor,
    perturbed_gray,
    model,
    ground_truth_rgb,
    attack_name="pic_name",
    epsilon=0.05,
    attack_model_name="DeOldify",
    colorization_model_name="DDColor",
    pgd_types = 'pgd_lab_lap',
    iteration=None,  # can pass int like 10, 20 etc
    save_root="visualizations",
    show=True,
    return_metrics=False
):
    """
    Visualize transferability between models and save consistently.
    """
    model.eval()
    device = grayscale_tensor.device
    model.return_ab = False

    
    # Colorize
    with torch.no_grad():
        perturbed_rgb = model(perturbed_gray.to(device))
        clean_rgb = model(grayscale_tensor.to(device))

    perturbed_rgb = torch.clamp(perturbed_rgb, 0, 1)
    clean_rgb = torch.clamp(clean_rgb, 0, 1)
    
    # Create save root path
    save_root_dir = os.path.join(save_root,pgd_types, attack_name, f"epsilon_{epsilon:.4f}")
    os.makedirs(save_root_dir, exist_ok=True)

    if iteration is not None:
        iter_folder = os.path.join(save_root_dir, f"iter_{iteration:03d}")
    else:
        iter_folder = save_root_dir

    os.makedirs(iter_folder, exist_ok=True)

    save_path = os.path.join(iter_folder, f"transferability_{attack_model_name}2{colorization_model_name}_{pgd_types}.png")

    # Prepare tensors
    grayscale_tensor_tensor = grayscale_tensor.clone().detach()
    grayscale_tensor = grayscale_tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    perturbed_gray_tensor = perturbed_gray.clone().detach()
    perturbed_gray = perturbed_gray.squeeze(0).squeeze(0).detach().cpu().numpy()

    perturbed_rgb_tensor = perturbed_rgb.clone().detach()
    clean_rgb_tensor = clean_rgb.clone().detach()
    perturbed_rgb = perturbed_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    clean_rgb = clean_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()
    ground_truth_rgb = ground_truth_rgb.squeeze(0).detach().cpu().permute(1, 2, 0).numpy()

    
    # Apply the fix
    perturbed_rgb_tensor = to_tensor_batch(perturbed_rgb)
    clean_rgb_tensor = to_tensor_batch(clean_rgb)


###########evaluation##############

    print("perturbed_rgb range:", perturbed_rgb.min(), perturbed_rgb.max())


    print("Measures how little color is left in the colorized output.")
    # Now compute differentiable colorfulness
    color_perturb_out = differentiable_colorfulness(perturbed_rgb_tensor).item()
    color_clean_out = differentiable_colorfulness(clean_rgb_tensor).item()

    print ("Colorfulness perturb_out :", color_perturb_out) 
    print ("Colorfulness clean_out :", color_clean_out) 
    print ("Colorfulness delta :",  color_clean_out-color_perturb_out) 
    
    colorfulness_loss = ((color_clean_out - color_perturb_out) / color_clean_out) * 100
    print(f"Colorfulness Reduction: { colorfulness_loss:.2f}%")


    mean_ab_perturb = to_float(compute_mean_ab_magnitude(perturbed_rgb_tensor, input_color_space='rgb'))
    mean_ab_clean = to_float(compute_mean_ab_magnitude(clean_rgb_tensor, input_color_space='rgb'))
    print("mean_ab_perturb :",mean_ab_perturb)
    print("mean_ab_clean :",mean_ab_clean)
    print ("mean_ab delta :",  mean_ab_clean-mean_ab_perturb) 


    save_path_jpeg = os.path.join(iter_folder, f"jpeg_transfer_{attack_model_name}2{colorization_model_name}_{pgd_types}.png")
    save_path_rrc = os.path.join(iter_folder, f"jpeg_transfer_{attack_model_name}2{colorization_model_name}_{pgd_types}.png")
    
    jpeg_q1_output , jpeg_q2_output = visualize_jpeg_transferability(
        original_L=grayscale_tensor_tensor,
        perturbed_L=perturbed_gray_tensor,
        model=model,
        jpeg_q1=75,
        jpeg_q2=50,
        save_path= os.path.join(save_path_jpeg),
        show=False,
        evaluate = True
    )

    resized_cf = visualize_randomresizecrop_transferability(
        original_L=grayscale_tensor_tensor,
        perturbed_L=perturbed_gray_tensor,
        model=model,
        scale=(0.8, 1.0),
        save_path=os.path.join(save_path_rrc),
        show=False,
        evaluate=True,
        crop_params=crop_params
    )


##########################3##############
    

    # Plot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 5, 1)
    plt.imshow(grayscale_tensor, cmap='gray')
    plt.title("Original Grayscale")
    plt.axis('off')

    plt.subplot(1, 5, 2)
    plt.imshow(perturbed_gray, cmap='gray')
    plt.title(f"Perturbed Grayscale {pgd_types}")
    plt.axis('off')

    plt.subplot(1, 5, 3)
    plt.imshow(perturbed_rgb)
    plt.title(f"{attack_model_name} Attack → {colorization_model_name} Colorized")
    plt.axis('off')

    plt.subplot(1, 5, 4)
    plt.imshow(clean_rgb)
    plt.title(f"{colorization_model_name} Clean Colorized")
    plt.axis('off')

    plt.subplot(1, 5, 5)
    plt.imshow(ground_truth_rgb)
    plt.title("Ground Truth RGB")
    plt.axis('off')

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Saved transferability visualization to {save_path}")

    if show:
        plt.show()

    plt.close()

    if return_metrics:
        return {
            "colorfulness_clean": color_clean_out,
            "colorfulness_perturb": color_perturb_out,
            "colorfulness_delta": color_clean_out - color_perturb_out,
            "colorfulness_reduction": colorfulness_loss,
            "mean_ab_clean": mean_ab_clean,
            "mean_ab_perturb": mean_ab_perturb,
            "mean_ab_delta": mean_ab_clean - mean_ab_perturb,
            "PSNR_clean_vs_perturb": calculate_psnr_rgb(clean_rgb_tensor, perturbed_rgb_tensor),
            "SSIM_clean_vs_perturb": calculate_ssim_rgb(clean_rgb_tensor, perturbed_rgb_tensor),
            "LPIPS_vs_gray": lpips_model(to_3ch(grayscale_tensor_tensor), to_3ch(perturbed_gray_tensor)).mean().item(),
            "PSNR_L_vs_perturbed": calculate_psnr_01(grayscale_tensor_tensor, perturbed_gray_tensor),
            "SSIM_L_vs_perturbed": calculate_ssim_01(grayscale_tensor_tensor, perturbed_gray_tensor),
            "jpeg_75_colorfulness" : differentiable_colorfulness(jpeg_q1_output).item(),
            "jpeg_50_colorfulness" : differentiable_colorfulness(jpeg_q2_output).item()
        }

        



################################################ main ############################################################


######################  initialization  ########################## 

lpips_model = lpips.LPIPS(net='alex').to(device).eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open("pretrain/config.json", "r") as f:
    config = json.load(f)

ddcolor = DDColor(
    encoder_name=config.get("encoder_name", "convnext-l"),
    decoder_name=config.get("decoder_name", "MultiScaleColorDecoder"),
    num_input_channels=config.get("num_input_channels", 3),
    input_size=(256, 256),
    nf=config.get("nf", 512),
    num_output_channels=config.get("num_output_channels", 2),
    last_norm=config.get("last_norm", "Spectral"),
    do_normalize=config.get("do_normalize", False),
    num_queries=config.get("num_queries", 100),
    num_scales=config.get("num_scales", 3),
    dec_layers=config.get("dec_layers", 9)
)
ddcolor.load_state_dict(torch.load("pretrain/pytorch_model.bin", map_location="cpu"))
ddcolor = ddcolor.to("cuda")

# Wrap and run
colorization_model = DDColorWrapperLAB(ddcolor).to("cuda")

max_image = 100
num_epoch_list = [100]


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--eps_size", type=int, default=16)
args = parser.parse_args()

seed = args.seed
eps_size = args.eps_size
epsilon = eps_size / 255.0

print("seed:", seed, "eps_size:", eps_size, "epsilon:", epsilon)

alpha = epsilon / 10.0
print("alpha:", alpha)

seeds = [seed,]

for seed in seeds:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)  
    
    # ImageNet Eval index list (00001.jpg ~ 50000.jpg)
    # all_indices = list(range(1, 50001))
    
    # random 100
    # selected_indices = random.sample(all_indices, 100)

    # for i in selected_indices:
    #    img_num = f"{i:05d}"
    #    image_paths = f"/home_fmg2/v-yuki/test/imageColorization/datasets/input_image/{img_num}.jpg"
    
    image_paths = f"/home_fmg2/v-yuki/test/imageColorization/Input/09547.jpg"
    print(f"\n===== Processing {image_paths} =====")
    image_paths = [image_paths]

    # Get deterministic crop parameters
    rrc_img = torch.ones(1,1, 255, 255)
    crop_params = T.RandomResizedCrop.get_params(
        rrc_img.squeeze(0), scale=(0.8, 1.0), ratio = (3/4, 4/3)
    )
    
    ################################## PAChroma ###############################################################################################

    for num_epochs in num_epoch_list:
            
        torch.cuda.empty_cache()
        gc.collect()
        
        sia_attack_zeroStart_withMASK = SIA(
            model=colorization_model,
            epsilon=epsilon,
            alpha=alpha,
            epoch=num_epochs,
            random_start=False,
            num_copies=20,
            visualize=False,
            laplacian_on=True,
            device="cuda"
        )
        
        run_sia_batch_attack(
            image_folder="/home_fmg2/v-yuki/test/imageColorization/Input/",
            sia_attack_instance=sia_attack_zeroStart_withMASK,
            final_log_path=f"Output/final_log_iteration{num_epochs}.csv",
            best_log_path=f"Output/best_log_iteration{num_epochs}.csv",
            save_name=f"PAChroma_{seed}",
            save_dir="Output",
            max_images=max_image,
            image_paths = image_paths,
            crop_params=crop_params
        )
