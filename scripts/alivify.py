from torch import autocast
from einops import rearrange, repeat
from PIL import Image
import subprocess , os , glob , gc , torch , time , copy , random
from IPython import display
import numpy as np
import math
from IPython import display as disp

from .vhelpers import process_frames, randomframes, interpolate

import os
from sdthings.scripts.misc import interpolate_folder


def itsalive(sd, args_int, baseargs):
    setup_folders(sd, args_int)

    print('Pre-interpolation pass. It will take a while...')
    interpolate_folder(args_int.inputs_folder, args_int.outputs_folder, args_int.iter1, args_int.sz)

    sd.load(args_int.model_url)

    print('Conditioning keyframes...')
    conditions, seeds, frames = process_frames(sd, args_int)
    
    print ('conditions:', len(conditions),'seeds', len(seeds),'frames',len(frames))
    print ('seeds are',seeds)

    print('Interpolating!')
    image_path, fps, mp4_path = alivify(sd, args_int, baseargs, conditions, seeds, frames)

    if args_int.iter2 > 0:
        temp_folder = os.path.join(sd.basedir, 'iter2temp')
        print('Post-interpolation pass. It will take a while...')
        interpolate_folder(os.path.join(sd.basedir, 'frames'), temp_folder, args_int.iter2, args_int.sz)
        image_path = os.path.join(temp_folder, "%05d.png")

    make_video(image_path, fps, mp4_path)
    return mp4_path


def setup_folders(sd, args_int):
    indir = os.path.join(sd.basedir, 'inputs')
    args_int.inputs_folder = os.path.join(indir, args_int.frames_folder)
    args_int.outputs_folder = os.path.join(indir, 'interpolated', args_int.frames_folder)



def alivify(sd, args_int, baseargs, conditions, seeds, frames):
    keyframes = len(frames)
    setup_directories(sd)

    args = copy.deepcopy(baseargs)
    print ('seeds are',seeds)
    all_z, all_c, all_i, atz = prepare_all_variables(sd, args_int, baseargs, conditions, seeds, frames)

    frame = 0
    cleanup_frames_folder(sd)

    all_c.append(all_c[0])
    all_z.append(all_z[0])

    c_i = copy.deepcopy(all_c[0])
    z_i = copy.deepcopy(all_z[0])

    for k in range(keyframes):
        i1 = k
        i2 = k + 1 if k < keyframes - 1 else 0

        for f in range(int(args_int.duration / keyframes)):
            handle_cache_cleanup()

            t = blend(f / (args_int.duration / keyframes - 0.4), 'linear')
            tLin = atz[i1]

            args.ddim_eta = 0

            z, c = interpolate_z_and_c(all_z, all_c, i1, i2, t)

            if args_int.dynamicprompt is not None:
                cd = args_int.dynamicprompt(frame, int(keyframes * args_int.duration))

                cd = sd.root.model.get_learned_conditioning(cd)
                c = torch.lerp(c, cd, args_int.dmix)

            z = interpolate(z_i, z, args_int.zamp)
            c = torch.lerp(c_i, c, args_int.camp)

            if frames[0] is None:
                args.use_init = True

            args.init_c = c

            if args_int.dynamicstrength:
                strength = DynStrength2(tLin, args_int.smin, args_int.smax)

            args.seed = seeds[0]
            img = sd.lat2img(args, z, args_int.strength)[0]

            display.display(img)
            filename = f"{frame:04}.png"
            img.save(os.path.join(sd.basedir, 'frames', filename))
            frame += 1

    timestring = time.strftime('%Y%m%d%H%M%S')
    filename = str(timestring) + '.mp4'

    outfile = os.path.join(sd.basedir, 'interpolations', filename)

    mp4_path = outfile
    image_path = os.path.join(sd.basedir, 'frames', "%05d.png")

    return image_path, args_int.fps, mp4_path



def setup_directories(sd):
    os.makedirs(os.path.join(sd.basedir, 'frames'), exist_ok=True)
    os.makedirs(os.path.join(sd.basedir, 'interpolations'), exist_ok=True)


def prepare_all_variables(sd, args_int, baseargs, conditions, seeds, frames):
    all_z = []
    all_c = []
    all_i = []
    atz = []

    conditions_length = len(conditions)
    frames_length = len(frames)
    inbetweens = int(args_int.duration / conditions_length)
    conditions.append(conditions[0])

    conditions_inbetweens = int(frames_length / (conditions_length - 1))
    conditions_inbetweens2 = conditions_inbetweens

    addict = (frames_length - (conditions_length - 1) * conditions_inbetweens)

    for k in range(conditions_length - 1):
        spirit = -addict
        i1 = k
        i2 = k + 1
        for f in range(conditions_inbetweens):
            t = blend(f / conditions_inbetweens, 'linear')
            c = torch.lerp(conditions[i1], conditions[i2], t)

            if args_int.stylemix > 0.:
                stylec = sd.autoc(args_int.styleimage, 0.)
                c = torch.lerp(c, stylec, args_int.stylemix)

            all_c.append(c)
            atz.append(blend(f / conditions_inbetweens, 'linear'))
            if addict > 0:
                conditions_inbetweens = conditions_inbetweens2 + 1
                addict -= 1

    kiki = 0
    for seed, frame in zip(seeds, frames):
        args = copy.deepcopy(baseargs)
        args.prompt = ''
        c = copy.deepcopy(all_c[kiki])

        if args_int.stylemix > 0.:
            c = (c * (1. - args_int.stylemix)) + (args_int.stylemix * stylec)

        args.init_c = c
        args.seed = seed

        if kiki == 0:
            scale = args.scale

        args.init_image = frame

        if args.init_image is not None:
            z, img = sd.img2img(args, args.init_image, args.strength, return_latent=True, return_c=False)
        else:
            z, img = sd.txt2img(args, return_latent=True, return_c=False)

        all_z.append(z)
        kiki += 1

    return all_z, all_c, all_i, atz

def cleanup_frames_folder(sd):
    files = glob.glob(os.path.join(sd.basedir, 'frames', '*'))
    for f in files:
        os.remove(f)

def handle_cache_cleanup():
    gc.collect()
    torch.cuda.empty_cache()

def interpolate_z_and_c(all_z, all_c, i1, i2, t):
    z = interpolate(all_z[i1], all_z[i2], t)
    c = torch.lerp(all_c[i1], all_c[i2], t)
    return z, c



#####################
def DynStrength2(t, tmin,tmax):
    if (t>0.5):
        t=0.5+(0.5-t)
    t=1-(t*2)
    return (tmin+ (t*(tmax-tmin)))


def make_video(image_path,fps,mp4_path):
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '10',
        '-preset', 'slow',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
        
        
    return mp4_path
    


def interpolate_prompts( sd,baseargs,duration,fps,zamp,camp,strength,blendmode, prompts_list):
    
    keyframes=len(prompts_list)

    #my_strength = strength * .5
    
    args = copy.deepcopy(baseargs)
    
    all_z=[]
    all_c=[]
    all_i=[]
    framesfolder = os.path.join(sd.basedir ,'frames')
    os.makedirs(framesfolder, exist_ok=True)
    outfolder = os.path.join(sd.basedir ,'interpolations')
    os.makedirs(outfolder, exist_ok=True)
    seed=1    
    kiki=0    
    seeds=[]
    prompts=[]
    
    random.seed()
    
    for prompt in prompts_list:

        if type(prompt[1])==str:        
          args.prompt = prompt[1]
        else:
          args.prompt = ''
          args.init_c = prompt[1]

        args.seed=prompt[0]
            
        if kiki==0:
            seed=args.seed            
            scale=args.scale
            
        if len(prompt)>2:
            args.init_image = prompt[2]
        
        
        prompts.append(args.prompt)
        
        seeds.append(args.seed)

        
            
        print(args.prompt)
        if args.init_image!= None:
            z, c, img = sd.img2img(args,args.init_image,args.strength, return_latent=True, return_c=True)
        else:
            z, c, img = sd.txt2img(args, return_latent=True, return_c=True)

        all_z.append(z)
        all_c.append(c)
        display.display(img)
        kiki+=1
        
        
    frame = 0
    
    i1=0
    i2=1
    
    inbetweens = int(duration/keyframes)
    files = glob.glob(framesfolder+'/*')
    for f in files:
        os.remove(f)
    
    c1=all_c[0]
    z1=all_z[0]
    
    c_i = c1
    z_i = z1
    
    for k in range(keyframes):
        
        i1=k
        i2=k+1
        if i2>keyframes-1:
            i2=0
            
        
        z1=all_z[i1]
  
        c1=all_c[i1]
        if k>0:
            c1=c2
            z1=z2
            
        z2=all_z[i2]
        c2=all_c[i2]
        
        if i2!=0:
            if zamp<1.:
                z2=interpolate(z1,z2,zamp)
            if camp<1.:
                c2=interpolate(c1,c2,camp)
        
        
        for f in range(inbetweens):
            gc.collect()
            torch.cuda.empty_cache() 
            t=blend(f/inbetweens,blendmode)
            tLin = (f/inbetweens)            
            args.ddim_eta=0
            c = torch.lerp(c1,c2,t)
            z = interpolate(z1,z2,t)
            
            tf = blend(frame/(inbetweens*keyframes),'linear')
            
            print (f,t,'-',tf)
            
            if args.smoothinterp:
                c = interpolate(c,c_i,tf)
                z = interpolate(z,z_i,tf)

            args.init_c=c

            if args.dynamicstrength:
                dynStrength = DynStrength(tLin, strength, args.smin,args.smax)
            else:
                dynStrength= strength
               
            img = sd.lat2img(args,z,dynStrength)[0]
            
            display.display(img)
            filename = f"{frame:04}.png"
            img.save(os.path.join(framesfolder,filename))
            frame+=1
            
        z2 = interpolate(z1,z2,1.0)
        c2 = torch.lerp(c1,c2,1.0)
        if args.smoothinterp:
            c2 = interpolate(c2,c_i,tf)
            z2 = interpolate(z2,z_i,tf)
        
            
    timestring = time.strftime('%Y%m%d%H%M%S')
    filename = str(timestring)+'.mp4'

    outfile = os.path.join(outfolder,filename)
    
    with open(os.path.join(outfolder, str(timestring)+'.txt'), 'w') as f:
        f.write(str(prompts)+'_'+str(seeds)+'_'+str(args.scale)+'_'+str(strength)+'_'+str(args.sampler)+'_'+str(args.steps)+'_'+str(duration)+'_'+str(zamp)+'_'+str(camp))
    
    mp4_path = outfile

    image_path = os.path.join(framesfolder, "%05d.png")
    #!ffmpeg -y -vcodec png -r {fps} -start_number 0 -i {image_path} -c:v libx264 -vf fps={fps} -pix_fmt yuv420p -crf 7 -preset slow -pattern_type sequence {mp4_path}
    cmd = [
        'ffmpeg',
        '-y',
        '-vcodec', 'png',
        '-r', str(fps),
        '-start_number', str(0),
        '-i', image_path,
        '-c:v', 'libx264',
        '-vf',
        f'fps={fps}',
        '-pix_fmt', 'yuv420p',
        '-crf', '10',
        '-preset', 'slow',
        '-pattern_type', 'sequence',
        mp4_path
    ]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(stderr)
        raise RuntimeError(stderr)
        
        
    return mp4_path



def ParametricBlend( t):
  sqt = t * t
  return (sqt / (2.0 * (sqt - t) + 1.0))

def DynStrength(t, strength, tmin,tmax):
  t = 1 - 2 * abs(.5 - t)
  return abs(1 - t ** 1.5 / (t ** 1.5 + (1 - t) ** 2.5))*(tmax-tmin)+tmin

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

def slerp4(v0, v1, t, eps=1e-6):
    v0_norm = v0 / torch.norm(v0)
    v1_norm = v1 / torch.norm(v1)
    dot = torch.dot(v0_norm.flatten(), v1_norm.flatten())
    omega = torch.acos(torch.clamp(dot, -1, 1))

    if omega < eps:
        return (1 - t) * v0 + t * v1

    sin_omega = torch.sin(omega)
    s0 = torch.sin((1.0 - t) * omega) / sin_omega
    s1 = torch.sin(t * omega) / sin_omega
    return s0[:, None, None] * v0 + s1[:, None, None] * v1



def slerpe(z_enc_1,z_enc_2,tt):
    #xc = sinh(a * (t * 2.0 - 1.0)) / sinh(a) / 2.0 + 0.5
    xn = 2.0 * tt**2 if tt < 0.5 else 1.0 - 2.0 * (1.0 - tt) ** 2
    return z_enc_1 * math.sqrt(1.0 - xn) + z_enc_2 * math.sqrt(xn)
def clear():
    disp.clear_output()
def slerp(low, high,val):
    low_norm = low/torch.norm(low, dim=1, keepdim=True)
    high_norm = high/torch.norm(high, dim=1, keepdim=True)
    omega = torch.acos((low_norm*high_norm).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-val)*omega)/so).unsqueeze(1)*low + (torch.sin(val*omega)/so).unsqueeze(1) * high
    return res
