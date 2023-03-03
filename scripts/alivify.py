from torch import autocast
from einops import rearrange, repeat
from PIL import Image
import subprocess , os , glob , gc , torch , time , copy , random
from IPython import display
import numpy as np
import math
from IPython import display as disp

from .vhelpers import processframes, randomframes, interpolate


def itsalive(randomseed, styleimage, stylemix,isimg, sz,imix,frames_folder, basedir, sd, model_url ,args, duration,fps,zamp,camp,strength,blendmode,iter1=2,iter2=2,doublefilm=True,dynamicprompt=None,dmix=0.7,cmix=0.,promptgen=None):
    from sdthings.scripts.misc import interpolate_folder
    
    indir = os.path.join(basedir,'inputs')

    inputs_folder = os.path.join(indir,frames_folder)
    outputs_folder = os.path.join(indir,'interpolated/'+frames_folder)
    
    print('pre-interpolation pass. it will take a while..')
    interpolate_folder(inputs_folder,outputs_folder,iter1,sz)
    
    try:
        sd.root.model
    except:
        sd.load(model_url)
        
    print('conditioning keyframes..')
    conditions, seeds, frames = processframes(sd, isimg,imix,inputs_folder,outputs_folder,randomseed,cmix,promptgen)
    #clear()
    print('interpolating!')
    image_path,fps,mp4_path = alivify(styleimage, stylemix, sd,args,duration,fps,zamp,camp,strength,blendmode, conditions, seeds, frames ,dynamicprompt,dmix)

    if(doublefilm):
        
        tempfld = os.path.join(basedir,'iter2temp')

        print('post-interpolation pass. it will take a while....')

        interpolate_folder(os.path.join(basedir,'frames'),tempfld,iter2,sz)
   
                
        image_path = os.path.join(tempfld, "%05d.png")
        makevideo(image_path,fps,mp4_path)
    else:
        makevideo(image_path,fps,mp4_path)

    #clear()
    return mp4_path


def alivify(styleimage, stylemix, sd,baseargs,duration,fps,zamp,camp,strength,blendmode, conditions, seeds, frames, dynamicprompt=None, dmix=0.5):
    keyframes=len(frames)

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
   
    random.seed()
    
    frames_length = len(frames)
    inbetweens = int(duration/keyframes)
    
    conditions_length = len(conditions)
    conditions_inbetweens = int((frames_length)/(conditions_length-1))
    conditions_inbetweens2 = conditions_inbetweens
    
    all_c=[]
    
    frame = 0
    
    i1=0
    i2=1    
    
    c_i = copy.deepcopy(conditions[0])
    atz=[]
    
    if stylemix>0.:
            stylec = sd.autoc(styleimage,0.)
    
    addict = (frames_length - (conditions_length-1)*conditions_inbetweens)
    
    for k in range(conditions_length-1):
        spirit = -addict        
        i1=k
        i2=k+1

        for f in range(conditions_inbetweens):
            t=blend(f/conditions_inbetweens,'bezier')
            c = interpolate(conditions[i1],conditions[i2],t)

            if stylemix>0.:           
                c = torch.lerp(c,stylec,stylemix)

            all_c.append(c)
            atz.append(blend(f/conditions_inbetweens,'parametric'))
            if addict>0:
                conditions_inbetweens=conditions_inbetweens2+1
                addict-=1


    print ('all_c len is :', len(all_c),'frames len is',frames_length)
    
    ########
    
    kiki = 0
    for seed,frame in zip(seeds,frames):
        
        args.prompt = ''
        c = copy.deepcopy(all_c[kiki])
        
        if stylemix>0.:
            c = (c * (1.-stylemix)) + (stylemix*stylec)
            
        args.init_c = c           

        args.seed=seed
            
        if kiki==0:          
            scale=args.scale
            
        args.init_image = frame
        
        if args.init_image!= None:
            z, img = sd.img2img(args,args.init_image,args.strength, return_latent=True, return_c=False)
        else:
            z, img = sd.txt2img(args, return_latent=True, return_c=False)

        all_z.append(z)
        
        display.display(img)
        kiki+=1
        
        
    frame = 0
    
    i1=0
    i2=1
    
    files = glob.glob(framesfolder+'/*')
    for f in files:
        os.remove(f)
        
        
    all_c.append(all_c[0])
    all_z.append(all_z[0])
    
    c_i = copy.deepcopy(all_c[0])
    z_i = copy.deepcopy(all_z[0])
    
    
    print('clen is',len(all_c))
    print('zlen is',len(all_z))
    
    for k in range(keyframes):
        
        i1=k
        i2=k+1
        
        if i2>keyframes-1:
            i2=0
            
            
        
        for f in range(inbetweens):
            gc.collect()
            torch.cuda.empty_cache() 
            
            t=blend(f/(inbetweens-.4),'linear')
            
            tLin = (f/(inbetweens-.4))  
            #print(tLin, atz[i1])
            tLin = atz[i1]
            
            args.ddim_eta=0

            z = interpolate(all_z[i1],all_z[i2],t)
            c = torch.lerp(all_c[i1],all_c[i2],t)            
            
            tf = blend(frame/(inbetweens*keyframes),'parametric')
                
            if dynamicprompt != None:
                cd = dynamicprompt(frame,int( keyframes*inbetweens ))
                cd = sd.root.model.get_learned_conditioning(cd)
                c = torch.lerp(c,cd,dmix)
        
            z=interpolate(z_i,z,zamp)
            c=torch.lerp(c_i,c,camp)
            
            if frames[0]==None:
                args.use_init=True
            
            args.init_c=c         

            if args.dynamicstrength:
                dynStrength = DynStrength(tLin, strength, args.smin,args.smax)
            else:
                dynStrength= strength
               
            img = sd.lat2img(args,z,dynStrength)[0]
            
            print (f+1,'/',inbetweens,'–',k+1,'/',keyframes,'–––','i1:',i1,'i2:',i2)
            
            display.display(img)
            filename = f"{frame:04}.png"
            img.save(os.path.join(framesfolder,filename))
            frame+=1
            
            
    timestring = time.strftime('%Y%m%d%H%M%S')
    filename = str(timestring)+'.mp4'

    outfile = os.path.join(outfolder,filename)
    
    mp4_path = outfile

    image_path = os.path.join(framesfolder, "%05d.png")
    
    return image_path,fps,mp4_path
    
#####################

def makevideo(image_path,fps,mp4_path):
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
    interpolate=slerp2
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
                c2=torch.lerp(c1,c2,camp)
        
        
        for f in range(inbetweens):
            gc.collect()
            torch.cuda.empty_cache() 
            t=blend(f/inbetweens,blendmode)
            tLin = (f/inbetweens)            
            args.ddim_eta=0
            c = torch.lerp(c1,c2,t)
            z = interpolate(z1,z2,t)
            
            tf = blend(frame/(inbetweens*keyframes),'parametric')
            
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
