import os
import random
from things import generate
from PIL import Image
import numpy as np
from einops import rearrange, repeat
import math

import torch
from torch import autocast
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddim import DDIMSampler
from types import SimpleNamespace

device = 'cuda'

##### defs

def ParametricBlend( t):
  sqt = t * t
  return (sqt / (2.0 * (sqt - t) + 1.0))

def CustomBlend( x):
  r=0
  if x >= 0.5:
    r =  x * (1 - x) *-2+1
  else:
    r =  x * (1 - x) *2
  return r


def BezierBlend( t):
  return t * t * (3.0 - 2.0 * t)

def blend(t,ip):
  if ip=='bezier':
    return BezierBlend(t)
  elif ip=='parametric':
    return ParametricBlend(t)
  elif ip=='inbetween':
    return CustomBlend(t)
  else:
    return t

def slerp(low, high,val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
def slerp_theta(z1, z2,theta): return math.cos(theta) * z1 + math.sin(theta) * z2


def slerp2( v0, v1, t, DOT_THRESHOLD=0.9995):
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    # Copy the vectors to reuse them later
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    # Normalize the vectors to get the directions and angles
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    # Dot product with the normalized vectors (can't use np.dot in W)
    dot = np.sum(v0 * v1)
    # If absolute value of dot product is almost 1, vectors are ~colineal, so use lerp
    if np.abs(dot) > DOT_THRESHOLD:
        return torch.lerp(t, v0_copy, v1_copy)
    # Calculate initial angle between v0 and v1
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
    # Finish the slerp algorithm
    s0 = np.sin(theta_0 - theta_t) / sin_theta_0
    s1 = sin_theta_t / sin_theta_0
    v2 = s0 * v0_copy + s1 * v1_copy
    if c:
        res = torch.from_numpy(v2).to("cuda")
    else:
        res = v2
    return res
def slerp2(v0, v1, t, DOT_THRESHOLD=0.9995):
    """helper function to spherically interpolate two arrays v1 v2"""

    if not isinstance(v0, np.ndarray):
        inputs_are_torch = True
        input_device = v0.device
        v0 = v0.cpu().numpy()
        v1 = v1.cpu().numpy()

    dot = np.sum(v0 * v1 / (np.linalg.norm(v0) * np.linalg.norm(v1)))
    if np.abs(dot) > DOT_THRESHOLD:
        v2 = (1 - t) * v0 + t * v1
    else:
        theta_0 = np.arccos(dot)
        sin_theta_0 = np.sin(theta_0)
        theta_t = theta_0 * t
        sin_theta_t = np.sin(theta_t)
        s0 = np.sin(theta_0 - theta_t) / sin_theta_0
        s1 = sin_theta_t / sin_theta_0
        v2 = s0 * v0 + s1 * v1

    if inputs_are_torch:
        v2 = torch.from_numpy(v2).to(input_device)

    return v2

def slerp3(low, high, val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res



def show(previews):
  lenp=len(previews)
  if lenp<6:
    lll2=lenp
    lll=1
  else:
    lll2=5
    lll=lenp//5+1
  _, axs = plt.subplots(lll, lll2, figsize=(20, lll*10/2))
  axs = axs.flatten()
  index=0
  for img, ax in zip(previews, axs):
      ax.imshow(img)
      ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)


      #plt.title(index)
      ax.set_xlabel('keyframe index: '+str(index))

      index+=1
  plt.show()
def ret_lat(model, args,di=False):
    results = generate(model, args, return_latent=True, return_sample=False)
    latent, image = results[0], results[1]
    if di:
      display.display(image)
    return latent, image

def lats_all(model, all_args):
  all_lats = []
  imgs=[]
  for args in all_args:
    lat,img = ret_lat(model, args, di=False)
    imgs.append( img )
    all_lats.append(lat)

  return all_lats , imgs
import copy


def load_img(path, shape, use_alpha_as_mask=False):
    # use_alpha_as_mask: Read the alpha channel of the image as the mask image
    if path.startswith('http://') or path.startswith('https://'):
        image = Image.open(requests.get(path, stream=True).raw)
    else:
        image = Image.open(path)

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

    return image, mask_image


import copy
def makeKeyframes(model, frames_folder, args):

    init_frames_files=sorted(os.listdir(frames_folder))



    all_args=[]
    for file in init_frames_files:

        k_args = copy.deepcopy(args)

        f = os.path.join(frames_folder,file)

        print ( f )

        k_args.init_image=f

        if k_args.seed == -1:
            k_args.seed = random.randint(0, 2**32)

        all_args.append( k_args )

    all_lats, previews = lats_all( model, all_args )

    return all_args, all_lats, previews


import os, shutil

def animate ( model, args, keyframes_args, keyframes_lats, basedir, output_path , batchname='randomtest', video_steps=25,keyframes_strength=0.4,total_frames=60,easing='linear', interpolation='slerp3', videoseed = -1 ):

  frames_temp_dir = os.path.join ( basedir , 'temp_frames')

  if os.path.isdir(frames_temp_dir):
      for filename in os.listdir(frames_temp_dir):
        file_path = os.path.join(frames_temp_dir, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

  os.makedirs(frames_temp_dir, exist_ok=True)

  precision_scope = autocast if args.precision == "autocast" else nullcontext
  results = []
  sampler = DDIMSampler(model)
  sampler.make_schedule(ddim_num_steps=video_steps, ddim_eta=0, verbose=False)


  frames=total_frames
  length=len(keyframes_args)
  idx1=0
  idx2=1
  batch_size=1

  index=0
  frames=frames//(length)


  with torch.no_grad():
    with precision_scope("cuda"):
        with model.ema_scope():
          t_enc = int((1.0-keyframes_strength) * video_steps)

          uc = model.get_learned_conditioning( [""])
          all_enc=[]


          for lat in keyframes_lats:
              all_enc.append (sampler.stochastic_encode(lat, torch.tensor([t_enc]).to(device)) )

          for idx1 in range (length):

              if idx2>length-2:
                idx2=0
              else:
                idx2=idx1+1

              prompts=[keyframes_args[idx1].prompt,keyframes_args[idx2].prompt]
              seed_everything(videoseed)


              z_enc_1 = all_enc[idx1]
              z_enc_2 = all_enc[idx2]

              c1 = model.get_learned_conditioning(prompts[0])
              c2 = model.get_learned_conditioning(prompts[1])

              for i in range(frames):

                  t = blend((i/frames),easing)

                  if interpolation=='slerp':
                    interpolate = slerp
                  elif interpolation=='slerp2':
                    interpolate = slerp2
                  elif interpolation=='slerp3':
                    interpolate = slerp3
                  else:
                    interpolate = slerp_theta


                  c = torch.lerp(  c1, c2, t )
                  q = interpolate(  z_enc_1, z_enc_2, t )

                  samples = sampler.decode(q, c, t_enc, unconditional_guidance_scale=args.scale,unconditional_conditioning=uc,)

                  x_samples = model.decode_first_stage(samples)
                  x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                  for x_sample in x_samples:
                      x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                      image = Image.fromarray(x_sample.astype(np.uint8))
                      results.append(image)
                      #if display_frames:
                      #  display.display(image)
                      filename = f"{index:04}.png"
                      image.save(os.path.join(frames_temp_dir, filename))
                  index+=1
