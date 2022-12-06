import os, shutil
import random
from scripts.things import generate
from PIL import Image
import numpy as np
from einops import rearrange, repeat

import math
import torch
import copy
import time
from torch import autocast
from pytorch_lightning import seed_everything

from helpers import DepthModel, sampler_fn, CFGDenoiserWithGrad
from k_diffusion.external import CompVisDenoiser
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
#from ldm.models.diffusion.dpm_solver import DPMSolverSampler


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
    seed_everything(args.seed)
    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope(): 
                init_image = init_image.to(device)
                init_image = repeat(init_image, '1 ... -> b ...', b=1)
                init_latent = model.get_first_stage_encoding(model.encode_first_stage(init_image))  # move to latent space
                latent = init_latent
                image = ''

    
    #results = generate(model,args, return_latent=True, return_sample=False)
    #latent, image = results[0], results[1]
    
    
    #if di:
    #  display.display(image)
    return latent, image

def ret_lat_rec(model, args, prev, di=False):
    #results = generate(model, args, return_latent=True, return_sample=False)
    init_image, mask_image = load_img(args.init_image,
                                          shape=(args.W, args.H),
                                          use_alpha_as_mask=False)
    seed_everything(args.seed)
    
    if prev == None:
        args.init_latent = None
    else:
        args.init_latent = prev
    results = generate(model,None, args, return_latent=True, return_sample=False)
    latent, image = results[0], results[1]
    
    
    #if di:
    disp.display(image)
    return latent, image
def lats_all(model, all_args):
  all_lats = []
  imgs=[]
  lat = None
  for args in all_args:
    lat,img = ret_lat(model, args, di=False)
    #lat,img = ret_lat_rec(model, args, lat, di=False)
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
        
    n = Image.open('sdthings/misc/unifier.png')
    n = n.convert('RGB')
    n = n.resize(shape, resample=Image.LANCZOS)
    
    n = np.array(n).astype(np.float16)
    

    image = (np.array(image).astype(np.float16)*.9+np.array(n).astype(np.float16)*.1) / 255.0
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

  if args.vsampler == 'plms':
    sampler = PLMSSampler(model)
  elif args.vsampler == 'dpm':
    sampler = DPMSolverSampler(model)
  else:
    sampler = DDIMSampler(model)
  
  if (args.vsampler=='ddim'):
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
            c = model.get_learned_conditioning(keyframes_args[0].prompt)
            all_enc=[]
            
            if not args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                for lat in keyframes_lats:
                    all_enc.append (sampler.stochastic_encode(lat, torch.tensor([t_enc]).to(device)) )
            model_wrap = CompVisDenoiser(model)
            

            for idx1 in range (length):
                if idx1 == length - 1:
                    idx2=0
                else:
                    idx2=idx1+1

                #prompts=[keyframes_args[idx1].prompt,keyframes_args[idx2].prompt]
            
                print(idx1,idx2)
              
                
                
                if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","dpm"]:
                    z_enc_1 = keyframes_lats[idx1]
                    z_enc_2 = keyframes_lats[idx2]
                else:
                    z_enc_1 = all_enc[idx1]
                    z_enc_2 = all_enc[idx2]
                    

                c1 = c
                #c2 = model.get_learned_conditioning(prompts[1])
                
                for i in range(frames):
                    seed_everything(videoseed)
                    
                
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
                    loss_fns_scales = [[None,0]]
                    
                    
                    k_sigmas = model_wrap.get_sigmas(args.steps)
                    k_sigmas = k_sigmas[len(k_sigmas)-t_enc-1:]
                    
                    mask = None
                    
                    q =  q*0.9+(torch.randn_like(q, device=device)*0.1)
                    
                    callback = SamplerCallback(args=args,
                            mask=mask, 
                            init_latent=q,
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
                    
                    if args.sampler in ["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral"]:
                        samples = sampler_fn(
                            c=c, 
                            uc=uc, 
                            args=args, 
                            model_wrap=cfg_model, 
                            init_latent=q, 
                            t_enc=t_enc, 
                            device=device, 
                            cb=callback,
                            verbose=False)
                    else:
                        if (args.sampler!="ddim"):

                            shape = [args.C, args.H // args.f, args.W // args.f]
                            samples, _ = sampler.sample(S=video_steps,
                                                         conditioning=c,
                                                         batch_size=1,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=vscale,
                                                         unconditional_conditioning=uc,
                                                         eta=args.ddim_eta,
                                                         x_T=q)
                        else:                   
                    
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

