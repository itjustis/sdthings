from IPython import display as disp
import os

import gc, math, os, pathlib, subprocess, sys, time
import cv2
import numpy as np
import pandas as pd
import random
import requests
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import urllib
from urllib.parse import urlparse

from contextlib import contextmanager, nullcontext
from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from pytorch_lightning import seed_everything
from skimage.exposure import match_histograms
from torchvision.utils import make_grid
from tqdm import tqdm, trange
from types import SimpleNamespace
from torch import autocast

from helpers import DepthModel, sampler_fn, CFGDenoiserWithGrad
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
#from ldm.models.diffusion.dpm_solver import DPMSolverSampler

import clip
from torchvision.transforms import Normalize as Normalize
from torch.nn import functional as F

device='cuda'

def basename (url):
    return os.path.basename( urlparse(url).path)

## CLIP -----------------------------------------

class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1.):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            cutouts.append(F.adaptive_avg_pool2d(cutout, self.cut_size))
        return torch.cat(cutouts)


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)

def make_clip_loss_fn(clip_model, clip_prompts, cutn=1, cut_pow=1):
    clip_size = clip_model.visual.input_resolution # for openslip: clip_model.visual.image_size

    def parse_prompt(prompt):
        if prompt.startswith('http://') or prompt.startswith('https://'):
            vals = prompt.rsplit(':', 2)
            vals = [vals[0] + ':' + vals[1], *vals[2:]]
        else:
            vals = prompt.rsplit(':', 1)
        vals = vals + ['', '1'][len(vals):]
        return vals[0], float(vals[1])

    def parse_clip_prompts(clip_prompts):
        target_embeds, weights = [], []
        for prompt in clip_prompts:
            txt, weight = parse_prompt(prompt)
            target_embeds.append(clip_model.encode_text(clip.tokenize(txt).to(device)).float())
            weights.append(weight)
        target_embeds = torch.cat(target_embeds)
        weights = torch.tensor(weights, device=device)
        if weights.sum().abs() < 1e-3:
            raise RuntimeError('Clip prompt weights must not sum to 0.')
        weights /= weights.sum().abs()
        return target_embeds, weights

    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                          std=[0.26862954, 0.26130258, 0.27577711])

    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)
    target_embeds, weights = parse_clip_prompts(clip_prompts)

    def clip_loss_fn(x, sigma, **kwargs):
        nonlocal target_embeds, weights, make_cutouts, normalize
        clip_in = normalize(make_cutouts(x.add(1).div(2)))
        image_embeds = clip_model.encode_image(clip_in).float()
        dists = spherical_dist_loss(image_embeds[:, None], target_embeds[None])
        dists = dists.view([cutn, 1, -1])
        losses = dists.mul(weights).sum(2).mean(0)
        return losses.sum()
    
    return clip_loss_fn

## end CLIP -----------------------------------------
def slice_imgs(imgs, count, size=224, transform=None, align='uniform', macro=0.):
    def map(x, a, b):
        return x * (b-a) + a

    rnd_size = torch.rand(count)
    if align == 'central': # normal around center
        rnd_offx = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
        rnd_offy = torch.clip(torch.randn(count) * 0.2 + 0.5, 0., 1.)
    else: # uniform
        rnd_offx = torch.rand(count)
        rnd_offy = torch.rand(count)
    
    sz = [img.shape[2:] for img in imgs]
    sz_max = [torch.min(torch.tensor(s)) for s in sz]
    if 'over' in align: # expand frame to sample outside 
        if align == 'overmax':
            sz = [[2*s[0], 2*s[1]] for s in list(sz)]
        else:
            sz = [[int(1.5*s[0]), int(1.5*s[1])] for s in list(sz)]
        imgs = [pad_up_to(imgs[i], sz[i], type='centr') for i in range(len(imgs))]

    sliced = []
    for i, img in enumerate(imgs):
        cuts = []
        sz_max_i = sz_max[i]
        for c in range(count):
            sz_min_i = 0.9*sz_max[i] if torch.rand(1) < macro else size
            csize = map(rnd_size[c], sz_min_i, sz_max_i).int()
            offsetx = map(rnd_offx[c], 0, sz[i][1] - csize).int()
            offsety = map(rnd_offy[c], 0, sz[i][0] - csize).int()
            cut = img[:, :, offsety:offsety + csize, offsetx:offsetx + csize]
            cut = F.interpolate(cut, (size,size), mode='bicubic', align_corners=True) # bilinear
            if transform is not None: 
                cut = transform(cut)
            cuts.append(cut)
        sliced.append(torch.cat(cuts, 0))
    return sliced
import torch.nn.functional as F


def enc_image(model_clip,in_img,cutn,clip_size,normalize):
    img_in = torch.from_numpy(cv2.imread(in_img).astype(np.float32)/255.).unsqueeze(0).permute(0,3,1,2).cuda()[:,:3,:,:]

    r_size = [512, 512]
    img_in = F.interpolate(img_in, r_size).float()
   
    in_sliced = slice_imgs([img_in], cutn, clip_size, normalize, 'uniform')[0]
    
    #in_sliced = slice_imgs([img_in], samples, modsize, T.normalize(), args.align)[0]
    img_enc = model_clip.encode_image(in_sliced).detach().clone()
    return img_enc
    

def make_clip_loss_fn_img(clip_model, in_img, cutn=1, cut_pow=1):
    normalize = Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                          std=[0.26862954, 0.26130258, 0.27577711])
    clip_size = clip_model.visual.input_resolution # for openslip: clip_model.visual.image_size
    
    #target_embeds = enc_image(clip_model,in_img,cutn,clip_size,normalize)
    weights=1.0
    
    img_in = torch.from_numpy(cv2.imread(in_img).astype(np.float32)/255.).unsqueeze(0).permute(0,3,1,2).cuda()[:,:3,:,:]

    r_size = [512, 512]
    img_in = F.interpolate(img_in, r_size).float()
    
    make_cutouts = MakeCutouts(clip_size, cutn, cut_pow)  
    
    target_embeds = clip_model.encode_image(normalize(make_cutouts(img_in))).float()
    
    #dists =  torch.cosine_similarity(img_enc, out_enc, dim=-1).mean()
    def clip_loss_fn_img(x, sigma, **kwargs):
        nonlocal target_embeds, weights, make_cutouts, normalize
        clip_in = normalize(make_cutouts(x.add(1).div(2)))
        image_embeds = clip_model.encode_image(clip_in).float()
        #dists = spherical_dist_loss(image_embeds[:, None], target_embeds[None])
        
        #dists = dists.view([cutn, 1, -1])
        #losses = dists.mul(weights).sum(2).mean(0)
        losses = torch.cosine_similarity(target_embeds, image_embeds, dim=-1).mean()
        print(losses.sum())
        return losses.sum()
    
    return clip_loss_fn_img

def sanitize(prompt):
    whitelist = set('abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ')
    tmp = ''.join(filter(whitelist.__contains__, prompt))
    return tmp.replace(' ', '_')

def add_noise(sample: torch.Tensor, noise_amt: float) -> torch.Tensor:
    return sample + torch.randn(sample.shape, device=sample.device) * noise_amt

def get_output_folder(output_path, batch_folder):
    out_path = os.path.join(output_path,time.strftime('%Y-%m'))
    if batch_folder != "":
        out_path = os.path.join(out_path, batch_folder)
    os.makedirs(out_path, exist_ok=True)
    return out_path

def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    try:
        if path.startswith('http://') or path.startswith('https://'):
            image = Image.open(requests.get(path, stream=True).raw)
        else:
            image = Image.open(path)
    except:
        image = path
        

    if use_alpha_as_mask:
        image = image.convert('RGBA')
    else:
        image = image.convert('RGB')

    image = image.resize(shape, resample=Image.LANCZOS)

    mask_image = None
    if use_alpha_as_mask:
      # Split alpha channel into a mask_image
      red, green, blue, alpha = Image.Image.split(image)
      mask_image = alpha.convert('L')
      image = image.convert('RGB')

    image = np.array(image).astype(np.float16) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.

    return image, mask_image

def load_mask_latent(mask_input, shape):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape

    if isinstance(mask_input, str): # mask input is probably a file name
        if mask_input.startswith('http://') or mask_input.startswith('https://'):
            mask_image = Image.open(requests.get(mask_input, stream=True).raw).convert('RGBA')
        else:
            mask_image = Image.open(mask_input).convert('RGBA')
    elif isinstance(mask_input, Image.Image):
        mask_image = mask_input
    else:
        raise Exception("mask_input must be a PIL image or a file name")

    mask_w_h = (shape[-1], shape[-2])
    mask = mask_image.resize(mask_w_h, resample=Image.LANCZOS)
    mask = mask.convert("L")
    return mask

def prepare_mask(mask_input, mask_shape, mask_brightness_adjust=1.0, mask_contrast_adjust=1.0):
    # mask_input (str or PIL Image.Image): Path to the mask image or a PIL Image object
    # shape (list-like len(4)): shape of the image to match, usually latent_image.shape
    # mask_brightness_adjust (non-negative float): amount to adjust brightness of the iamge,
    #     0 is black, 1 is no adjustment, >1 is brighter
    # mask_contrast_adjust (non-negative float): amount to adjust contrast of the image,
    #     0 is a flat grey image, 1 is no adjustment, >1 is more contrast

    mask = load_mask_latent(mask_input, mask_shape)

    # Mask brightness/contrast adjustments
    if mask_brightness_adjust != 1:
        mask = TF.adjust_brightness(mask, mask_brightness_adjust)
    if mask_contrast_adjust != 1:
        mask = TF.adjust_contrast(mask, mask_contrast_adjust)

    # Mask image to array
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask,(4,1,1))
    mask = np.expand_dims(mask,axis=0)
    mask = torch.from_numpy(mask)

    if args.invert_mask:
        mask = ( (mask - 0.5) * -1) + 0.5

    mask = np.clip(mask,0,1)
    return mask

def maintain_colors(prev_img, color_match_sample, mode):
    if mode == 'Match Frame 0 RGB':
        return match_histograms(prev_img, color_match_sample, multichannel=True)
    elif mode == 'Match Frame 0 HSV':
        prev_img_hsv = cv2.cvtColor(prev_img, cv2.COLOR_RGB2HSV)
        color_match_hsv = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2HSV)
        matched_hsv = match_histograms(prev_img_hsv, color_match_hsv, multichannel=True)
        return cv2.cvtColor(matched_hsv, cv2.COLOR_HSV2RGB)
    else: # Match Frame 0 LAB
        prev_img_lab = cv2.cvtColor(prev_img, cv2.COLOR_RGB2LAB)
        color_match_lab = cv2.cvtColor(color_match_sample, cv2.COLOR_RGB2LAB)
        matched_lab = match_histograms(prev_img_lab, color_match_lab, multichannel=True)
        return cv2.cvtColor(matched_lab, cv2.COLOR_LAB2RGB)


def make_callback(sampler_name, dynamic_threshold=None, static_threshold=None, mask=None, init_latent=None, sigmas=None, sampler=None, masked_noise_modifier=1.0):
    # Creates the callback function to be passed into the samplers
    # The callback function is applied to the image at each step
    def dynamic_thresholding_(img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(args_dict):
        if dynamic_threshold is not None:
            dynamic_thresholding_(args_dict['x'], dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*static_threshold, static_threshold)
        if mask is not None:
            init_noise = init_latent + noise * args_dict['sigma']
            is_masked = torch.logical_and(mask >= mask_schedule[args_dict['i']], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
            args_dict['x'].copy_(new_img)

    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(img, i):
        # Thresholding functions
        if dynamic_threshold is not None:
            dynamic_thresholding_(img, dynamic_threshold)
        if static_threshold is not None:
            torch.clamp_(img, -1*static_threshold, static_threshold)
        if mask is not None:
            i_inv = len(sigmas) - i - 1
            init_noise = sampler.stochastic_encode(init_latent, torch.tensor([i_inv]*batch_size).to(device), noise=noise)
            is_masked = torch.logical_and(mask >= mask_schedule[i], mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
            img.copy_(new_img)

    if init_latent is not None:
        noise = torch.randn_like(init_latent, device=device) * masked_noise_modifier
    if sigmas is not None and len(sigmas) > 0:
        mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
    elif len(sigmas) == 0:
        mask = None # no mask needed if no steps (usually happens because strength==1.0)
    if sampler_name in ["plms","ddim"]:
        # Callback function formated for compvis latent diffusion samplers
        if mask is not None:
            assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"
            batch_size = init_latent.shape[0]

        callback = img_callback_
    else:
        # Default callback function uses k-diffusion sampler variables
        callback = k_callback_

    return callback

def sample_from_cv2(sample: np.ndarray) -> torch.Tensor:
    sample = ((sample.astype(float) / 255.0) * 2) - 1
    sample = sample[None].transpose(0, 3, 1, 2).astype(np.float16)
    sample = torch.from_numpy(sample)
    return sample

def sample_to_cv2(sample: torch.Tensor, type=np.uint8) -> np.ndarray:
    sample_f32 = rearrange(sample.squeeze().cpu().numpy(), "c h w -> h w c").astype(np.float32)
    sample_f32 = ((sample_f32 * 0.5) + 0.5).clip(0, 1)
    sample_int8 = (sample_f32 * 255)
    return sample_int8.astype(type)


import cv2

def imgtobytes(image):
  success, encoded_image = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  content2 = encoded_image.tobytes()
  return content2


def load_model(model_checkpoint =  "sd-v1-4.ckpt", basedir = '/workspace/'):
    from transformers import logging
    logging.set_verbosity_error()
    model_checkpoint = basename(model_checkpoint)
    models_path = os.path.join(basedir,'models')
    #@markdown **Select and Load Model**

    model_config = "v1-inference.yaml"
    # config path
    ckpt_config_path = custom_config_path if model_config == "custom" else os.path.join(models_path, model_config)
    if os.path.exists(ckpt_config_path):
        print(f"{ckpt_config_path} exists")
    else:
        ckpt_config_path = os.path.join(os.path.join(basedir,'packages'),"stable-diffusion/configs/stable-diffusion/v1-inference.yaml")
    if (model_checkpoint == '768-v-ema.ckpt'):
        ckpt_config_path = os.path.join(os.path.join(basedir,'packages'),'stablediffusion/configs/stable-diffusion/v2-inference-v.yaml')
    
    print(f"Using config: {ckpt_config_path}")

    # checkpoint path or download
    ckpt_path = custom_checkpoint_path if model_checkpoint == "custom" else os.path.join(models_path, model_checkpoint)
    print(f"Using ckpt: {ckpt_path}")
    
    config = OmegaConf.load(f"{ckpt_config_path}")
    model = load_model_from_config(config, f"{ckpt_path}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    
    def make_linear_decode(model_version, device='cuda:0'):
        v1_4_rgb_latent_factors = [
            #   R       G       B
            [ 0.298,  0.207,  0.208],  # L1
            [ 0.187,  0.286,  0.173],  # L2
            [-0.158,  0.189,  0.264],  # L3
            [-0.184, -0.271, -0.473],  # L4
        ]

        if model_version[:5] == "sd-v1":
            rgb_latent_factors = torch.Tensor(v1_4_rgb_latent_factors).to(device)
        else:
            raise Exception(f"Model name {model_version} not recognized.")

        def linear_decode(latent):
            latent_image = latent.permute(0, 2, 3, 1) @ rgb_latent_factors
            latent_image = latent_image.permute(0, 3, 1, 2)
            return latent_image

        return linear_decode
    autoencoder_version = "sd-v1" #TODO this will be different for different models
    model.linear_decode = make_linear_decode(autoencoder_version, 'cuda')


    return model

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

###
# Thresholding functions for grad
###
def threshold_by(threshold, threshold_type):

  def dynamic_thresholding(vals, sigma):
      # Dynamic thresholding from Imagen paper (May 2022)
      s = np.percentile(np.abs(vals.cpu()), threshold, axis=tuple(range(1,vals.ndim)))
      s = np.max(np.append(s,1.0))
      vals = torch.clamp(vals, -1*s, s)
      vals = torch.FloatTensor.div(vals, s)
      return vals

  def static_thresholding(vals, sigma):
      vals = torch.clamp(vals, -1*threshold, threshold)
      return vals

  def mean_thresholding(vals, sigma): # Thresholding that appears in Jax and Disco
      magnitude = vals.square().mean(axis=(1,2,3),keepdims=True).sqrt()
      vals = vals * torch.where(magnitude > threshold, threshold / magnitude, 1.0)
      return vals

  if threshold_type == 'dynamic':
      return dynamic_thresholding
  elif threshold_type == 'static':
      return static_thresholding
  elif threshold_type == 'mean':
      return mean_thresholding
  else:
      raise Exception(f"Thresholding type {threshold_type} not supported")



#
# Callback functions
#
class SamplerCallback(object):
    # Creates the callback function to be passed into the samplers for each step
    def __init__(self, args, mask=None, init_latent=None, sigmas=None, sampler=None,
                  verbose=False):
        self.sampler_name = args.sampler
        self.dynamic_threshold = args.dynamic_threshold
        self.static_threshold = args.static_threshold
        self.mask = mask
        self.init_latent = init_latent 
        self.sigmas = sigmas
        self.sampler = sampler
        self.verbose = verbose

        self.batch_size = args.n_samples
        self.save_sample_per_step = args.save_sample_per_step
        self.show_sample_per_step = args.show_sample_per_step
        self.paths_to_image_steps = [os.path.join( args.outdir, f"{args.timestring}_{index:02}_{args.seed}") for index in range(args.n_samples) ]

        if self.save_sample_per_step:
            for path in self.paths_to_image_steps:
                os.makedirs(path, exist_ok=True)

        self.step_index = 0

        self.noise = None
        if init_latent is not None:
            self.noise = torch.randn_like(init_latent, device=device)

        self.mask_schedule = None
        if sigmas is not None and len(sigmas) > 0:
            self.mask_schedule, _ = torch.sort(sigmas/torch.max(sigmas))
        elif len(sigmas) == 0:
            self.mask = None # no mask needed if no steps (usually happens because strength==1.0)

        if self.sampler_name in ["plms","ddim"]: 
            if mask is not None:
                assert sampler is not None, "Callback function for stable-diffusion samplers requires sampler variable"

        if self.sampler_name in ["plms","ddim"]: 
            # Callback function formated for compvis latent diffusion samplers
            self.callback = self.img_callback_
        else: 
            # Default callback function uses k-diffusion sampler variables
            self.callback = self.k_callback_

        self.verbose_print = print if verbose else lambda *args, **kwargs: None

    def view_sample_step(self, latents, path_name_modifier=''):
        if self.save_sample_per_step:
            samples = model.decode_first_stage(latents)
            fname = f'{path_name_modifier}_{self.step_index:05}.png'
            for i, sample in enumerate(samples):
                sample = sample.double().cpu().add(1).div(2).clamp(0, 1)
                sample = torch.tensor(np.array(sample))
                grid = make_grid(sample, 4).cpu()
                TF.to_pil_image(grid).save(os.path.join(self.paths_to_image_steps[i], fname))
        if self.show_sample_per_step:
            samples = model.linear_decode(latents)
            print(path_name_modifier)
            self.display_images(samples)
        return

    def display_images(self, images):
        images = images.double().cpu().add(1).div(2).clamp(0, 1)
        images = torch.tensor(np.array(images))
        grid = make_grid(images, 4).cpu()
        display.display(TF.to_pil_image(grid))
        return

    # The callback function is applied to the image at each step
    def dynamic_thresholding_(self, img, threshold):
        # Dynamic thresholding from Imagen paper (May 2022)
        s = np.percentile(np.abs(img.cpu()), threshold, axis=tuple(range(1,img.ndim)))
        s = np.max(np.append(s,1.0))
        torch.clamp_(img, -1*s, s)
        torch.FloatTensor.div_(img, s)

    # Callback for samplers in the k-diffusion repo, called thus:
    #   callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
    def k_callback_(self, args_dict):
        self.step_index = args_dict['i']
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(args_dict['x'], self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(args_dict['x'], -1*self.static_threshold, self.static_threshold)
        if self.mask is not None:
            init_noise = self.init_latent + self.noise * args_dict['sigma']
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[args_dict['i']], self.mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + args_dict['x'] * torch.where(is_masked,0,1)
            args_dict['x'].copy_(new_img)

        self.view_sample_step(args_dict['denoised'], "x0_pred")
        self.view_sample_step(args_dict['x'], "x")

    # Callback for Compvis samplers
    # Function that is called on the image (img) and step (i) at each step
    def img_callback_(self, img, pred_x0, i):
        self.step_index = i
        # Thresholding functions
        if self.dynamic_threshold is not None:
            self.dynamic_thresholding_(img, self.dynamic_threshold)
        if self.static_threshold is not None:
            torch.clamp_(img, -1*self.static_threshold, self.static_threshold)
        if self.mask is not None:
            i_inv = len(self.sigmas) - i - 1
            init_noise = self.sampler.stochastic_encode(self.init_latent, torch.tensor([i_inv]*self.batch_size).to(device), noise=self.noise)
            is_masked = torch.logical_and(self.mask >= self.mask_schedule[i], self.mask != 0 )
            new_img = init_noise * torch.where(is_masked,1,0) + img * torch.where(is_masked,0,1)
            img.copy_(new_img)

        self.view_sample_step(pred_x0, "x0_pred")
        self.view_sample_step(img, "x")


import cv2
from PIL import ImageOps



def imgtobytes(image):
  success, encoded_image = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
  content2 = encoded_image.tobytes()
  return content2


def generate(model, clip_model, args, return_latent=False, return_sample=False, return_c=False):
    device = 'cuda'
    seed_everything(args.seed)
    os.makedirs(args.outdir, exist_ok=True)

    sampler = PLMSSampler(model) if args.sampler == 'plms' else DDIMSampler(model)
    model_wrap = CompVisDenoiser(model)
    batch_size = args.n_samples
    prompt = args.prompt
    assert prompt is not None
    data = [batch_size * [prompt]]
    precision_scope = autocast if args.precision == "autocast" else nullcontext

    init_latent = None
    mask_image = None
    init_image = None
    if args.init_latent is not None:
        init_latent = args.init_latent
    elif args.init_sample is not None:
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(args.init_sample))
    elif args.use_init and args.init_image != None and args.init_image != '':
        init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=args.use_alpha_as_mask)
        init_image = init_image.to(device)
        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
        with precision_scope("cuda"):
            init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space

    if not args.use_init and args.strength > 0 and args.strength_0_no_init:
        print("\nNo init image, but strength > 0. Strength has been auto set to 0, since use_init is False.")
        print("If you want to force strength > 0 with no init, please set strength_0_no_init to False.\n")
        args.strength = 0

    # Mask functions
    if args.use_mask:
        assert args.mask_file is not None or mask_image is not None, "use_mask==True: An mask image is required for a mask. Please enter a mask_file or use an init image with an alpha channel"
        assert args.use_init, "use_mask==True: use_init is required for a mask"
        assert init_latent is not None, "use_mask==True: An latent init image is required for a mask"

        mask = prepare_mask(args.mask_file if mask_image is None else mask_image,
                            init_latent.shape,
                            args.mask_contrast_adjust,
                            args.mask_brightness_adjust)

        if (torch.all(mask == 0) or torch.all(mask == 1)) and args.use_alpha_as_mask:
            raise Warning("use_alpha_as_mask==True: Using the alpha channel from the init image as a mask, but the alpha channel is blank.")

        #mask = mask.to(device)
        #mask = repeat(mask, '1 ... -> b ...', b=batch_size)

        init_mask = mask_image
        latmask = init_mask.convert('RGB').resize((init_latent.shape[3], init_latent.shape[2]))
        latmask = np.moveaxis(np.array(latmask, dtype=np.float32), 2, 0) / 255
        latmask = latmask[0]
        latmask = np.around(latmask)
        latmask = np.tile(latmask[None], (4, 1, 1))

        mask = torch.asarray(1.0 - latmask).to(device).type(model.dtype)
        nmask = torch.asarray(latmask).to(device).type(model.dtype)

        init_latent = init_latent * mask + torch.randn_like(init_latent, device=device) * 1.0 * nmask
    else:
        mask = None

    t_enc = int((1.0-args.strength) * args.steps)

    # Noise schedule for the k-diffusion samplers (used for masking)
    k_sigmas = model_wrap.get_sigmas(args.steps)
    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]

    if args.sampler in ['plms','ddim']:
        sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta, verbose=False)
        #sampler.make_schedule(ddim_num_steps=args.steps, ddim_eta=args.ddim_eta,  verbose=False)
        
    if args.clip_loss_scale != 0:
        clip_loss_fn = make_clip_loss_fn(clip_model=clip_model, 
                                        clip_prompts=args.clip_prompts, 
                                        cutn=args.cutn, 
                                        cut_pow=args.cut_pow)
    else:
        clip_loss_fn = None
    
    if args.clip_loss_img_scale != 0:
        clip_loss_fn_img = make_clip_loss_fn_img(clip_model=clip_model, 
                                        in_img=args.in_img, 
                                        cutn=args.cutn, 
                                        cut_pow=args.cut_pow)
    else:
        clip_loss_fn_img = None

    loss_fns_scales = [
        [clip_loss_fn,args.clip_loss_scale],
        [clip_loss_fn_img,args.clip_loss_img_scale]
    ]

    callback = SamplerCallback(args=args,
                            mask=mask, 
                            init_latent=init_latent,
                            sigmas=k_sigmas,
                            sampler=sampler,
                            verbose=False).callback 

    clamp_fn = threshold_by(threshold=args.clamp_grad_threshold, threshold_type=args.grad_threshold_type)

    cfg_model = CFGDenoiserWithGrad(model_wrap, 
                                    loss_fns_scales, 
                                    clamp_fn, 
                                    args.gradient_wrt, 
                                    args.gradient_add_to, 
                                    args.cond_uncond_sync,
                                    decode_method=args.decode_method,
                                    verbose=False)

    results = []
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for prompts in data:
                    uc = None
                    if args.scale != 1.0:
                        uc = model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = model.get_learned_conditioning(prompts)

                    if args.init_c != None:
                        c = args.init_c

                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        samples = sampler_fn(
                            c=c, 
                            uc=uc, 
                            args=args, 
                            model_wrap=cfg_model, 
                            init_latent=init_latent, 
                            t_enc=t_enc, 
                            device=device, 
                            cb=callback,
                            verbose=False)
                    else:
                        # args.sampler == 'plms' or args.sampler == 'ddim':
                        if init_latent is not None and args.strength > 0:
                            z_enc = sampler.stochastic_encode(init_latent, torch.tensor([t_enc]*batch_size).to(device))
                        else:
                            z_enc = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)
                        if args.sampler == 'ddim':
                            samples = sampler.decode(z_enc,
                                                     c,
                                                     t_enc,
                                                     unconditional_guidance_scale=args.scale,
                                                     unconditional_conditioning=uc,
                                                     img_callback=callback)
                        elif args.sampler == 'plms': # no "decode" function in plms, so use "sample"
                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=args.steps,
                                                            conditioning=c,
                                                            batch_size=args.n_samples,
                                                            shape=shape,
                                                            verbose=False,
                                                            unconditional_guidance_scale=args.scale,
                                                            unconditional_conditioning=uc,
                                                            eta=args.ddim_eta,
                                                            x_T=z_enc,
                                                            img_callback=callback)
                        else:
                            raise Exception(f"Sampler {args.sampler} not recognised.")

                    if return_latent:
                        results.append(samples.clone())

                    x_samples = model.decode_first_stage(samples)
                    if return_sample:
                        results.append(x_samples.clone())

                    x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                    if return_c:
                        results.append(c.clone())

                    for x_sample in x_samples:
                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                        image = Image.fromarray(x_sample.astype(np.uint8))
                        results.append(image)
    return results
