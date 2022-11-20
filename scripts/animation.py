import os, shutil
import random
from things import generate
from PIL import Image
import numpy as np
from einops import rearrange, repeat

import math
import torch
import copy
import time
from torch import autocast
from pytorch_lightning import seed_everything
from ldm.models.diffusion.ddim import DDIMSampler
from types import SimpleNamespace
import subprocess, time, requests

from IPython import display as disp

def clear():
    disp.clear_output()

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

def slerp4(t, v0, v1, DOT_THRESHOLD=0.9995):
    """ helper function to spherically interpolate two arrays v1 v2 """

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

def slerp2( v0, v1, t, DOT_THRESHOLD=0.9995):
    c = False
    if not isinstance(v0,np.ndarray):
        c = True
        v0 = v0.detach().cpu().numpy()
    if not isinstance(v1,np.ndarray):
        c = True
        v1 = v1.detach().cpu().numpy()
    v0_copy = np.copy(v0)
    v1_copy = np.copy(v1)
    v0 = v0 / np.linalg.norm(v0)
    v1 = v1 / np.linalg.norm(v1)
    dot = np.sum(v0 * v1)
    if np.abs(dot) > DOT_THRESHOLD:
        return torch.lerp(t, v0_copy, v1_copy)
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    # Angle at timestep t
    theta_t = theta_0 * t
    sin_theta_t = np.sin(theta_t)
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
    #results = generate(model, args, return_latent=True, return_sample=False)
    init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=False)
    init_image = init_image.to(device)
    init_image = repeat(init_image, '1 ... -> b ...', b=1)
    init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
    
    
    latent = init_latent
    image = ''
    #if di:
    #  display.display(image)
    return latent, image

def lats_all(model, all_args):
  all_lats = []
  imgs=[]
  for args in all_args:
    lat,img = ret_lat(model, args, di=False)
    imgs.append( img )
    all_lats.append(lat)

  return all_lats , imgs


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
        
    n = Image.open('/workspace/unifier.png')
    n = n.convert('RGB')
    n = n.resize(shape, resample=Image.LANCZOS)
    
    n = np.array(n).astype(np.float16)

    image = (np.array(image).astype(np.float16)*.97+np.array(n).astype(np.float16)*.03) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    image = 2.*image - 1.

    return image, mask_image

#####
def makeKeyframes(model, frames_folder, args):

    init_frames_files=sorted(os.listdir(frames_folder))

    all_args=[]
    for file in init_frames_files:
        
        if (file.endswith('.jpg') or file.endswith('.png')):
            k_args = copy.deepcopy(args)
            f = os.path.join(frames_folder,file)
            print ( f )
            k_args.init_image=f
            if k_args.seed == -1:
                k_args.seed = random.randint(0, 2**32)
            all_args.append( k_args )
    all_lats, previews = lats_all( model, all_args )
    return all_args, all_lats, previews

#####
def makeloop(frames_folder, model, videoargs, args, outfolder, ffmpeg):
    video_steps = videoargs.video_steps
    keyframes_strength = videoargs.keyframes_strength
    vseed = videoargs.vseed
    total_frames = videoargs.total_frames
    easing = videoargs.easing
    interpolation = videoargs.interpolation
    isdisplay = videoargs.isdisplay
    vscale = videoargs.vscale
    trunc = videoargs.truncation
    idx = videoargs.index
    
    fps = videoargs.fps
    eta = videoargs.eta
    
    basedir = args.basedir

    
    timestring = time.strftime('%Y%m%d%H%M%S')
    
    filename = str(timestring)+' n_vseed_'+str(vseed)+' vs_'+str(video_steps)+' ks_'+str(keyframes_strength)+' interp_'+str(interpolation)+' vscale_'+str(vscale)+' easing_'+str(easing)+' iter_'+'000'
    
    keyframes_args, keyframes_lats, keyframes_previews = makeKeyframes(model, frames_folder, args)
    
    animate(idx, trunc, eta, isdisplay, vscale, model, args, keyframes_args, keyframes_lats, basedir, video_steps, keyframes_strength, total_frames, easing, interpolation, vseed )
    print ('render frames – done')
    print ('making video')
    
    output_path = os.path.join(basedir, outfolder)
    
    os.makedirs(output_path, exist_ok=True)
    outfile = os.path.join(output_path,filename)
 
    clear()

    mp4_path = str(outfile)+'.mp4'
    frames_temp_dir = os.path.join ( basedir , 'temp_frames')
    image_path = os.path.join(frames_temp_dir, "%04d.png")
    cmd = [
        ffmpeg,
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '7',
        '-preset', 'slow',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
        
    return(mp4_path)

    
#####
def animate (idx, trunc, eta, isdisplay, vscale, model, args, keyframes_args, keyframes_lats, basedir, video_steps=25,keyframes_strength=0.4,total_frames=60,easing='linear', interpolation='slerp3', videoseed = -1 ):

  torch.cuda.empty_cache()

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
  sampler.make_schedule(ddim_num_steps=video_steps, ddim_eta=eta, verbose=False)


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
              if idx1 == length - 1:
                idx2=0
              else:
                idx2=idx1+1

              prompts=[keyframes_args[idx1].prompt,keyframes_args[idx2].prompt]
            
              print(idx1,idx2)
              seed_everything(videoseed)


              z_enc_1 = all_enc[idx1]
              z_enc_2 = all_enc[idx2]

              c1 = model.get_learned_conditioning(prompts[0])
              #c2 = model.get_learned_conditioning(prompts[1])
            
              

              for i in range(frames):
                
                  t = blend((i/frames),easing)
                  dynkey = False                  
                  if dynkey:
                    #keyframes_strength = (math.sin(i/frames*math.pi/2+(math.pi*shift))/2+0.5)*amp+basestr
                    t_enc = int((1.0-keyframes_strength) * video_steps)
                  else:
                    t_enc = int((1.0-keyframes_strength) * video_steps)
                    
                  #print(t_enc, t, '_')

                  if interpolation=='slerp':
                    interpolate = slerp
                  elif interpolation=='slerp2':
                    interpolate = slerp2
                  elif interpolation=='slerp3':
                    interpolate = slerp3
                  elif interpolation=='slerp4':
                    interpolate = slerp4
                  else:
                    interpolate = slerp_theta
                    
                


                  #c = torch.lerp(  c1, c2, t )
                  c = c1
                  
                  if interpolation == 'mix':
                    q = (slerp(z_enc_1, z_enc_2, t) + slerp2(z_enc_1, z_enc_2, t) + slerp3(z_enc_1, z_enc_2, t) )/3
                  elif interpolation == 'exp':
                    tt = i / frames
                    #xc = sinh(a * (t * 2.0 - 1.0)) / sinh(a) / 2.0 + 0.5
                    xn = 2.0 * tt**2 if tt < 0.5 else 1.0 - 2.0 * (1.0 - tt) ** 2
                    q = z_enc_1 * math.sqrt(1.0 - xn) + z_enc_2 * math.sqrt(xn)
                  else:
                    q = interpolate(  z_enc_1, z_enc_2, t )
                    

                
                  #q = slerp2( q_m, q, trunc )
                    


                  #truncn = 2.0 * trunc**2 if trunc < 0.5 else 1.0 - 2.0 * (1.0 - trunc) ** 2
                  #q = q_m * math.sqrt(1.0 - truncn) + q * math.sqrt(truncn)
              
                  samples = sampler.decode(q, c, t_enc, unconditional_guidance_scale=vscale,unconditional_conditioning=uc,)
                  
                  x_samples = model.decode_first_stage(samples)
                  x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
                    
                  clear()

                  for x_sample in x_samples:
                      x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                      image = Image.fromarray(x_sample.astype(np.uint8))
                      results.append(image)
                      if isdisplay:
                        disp.display(image)
                      filename = f"{index:04}.png"
                      image.save(os.path.join(frames_temp_dir, filename))
                  index+=1
                
                
