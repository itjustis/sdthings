from torch import autocast
from einops import rearrange, repeat
from PIL import Image
import subprocess , os , glob , gc , torch , time , copy , random
from IPython import display
import numpy as np

def alivify( sd,baseargs,keyframes,duration,fps,zamp,camp,strength,blendmode, genxprompt):
    interpolate=slerp2
    
    args = copy.deepcopy(baseargs)
    
    all_z=[]
    all_c=[]
    all_i=[]
    framesfolder = os.path.join(sd.basedir ,'frames')
    os.makedirs(framesfolder, exist_ok=True)
    outfolder = os.path.join(sd.basedir ,'alivify')
    os.makedirs(outfolder, exist_ok=True)
    seed=1    
    kiki=0    
    seeds=[]
    prompts=[]
    
    random.seed()
    
    for k in range(keyframes):
        
        if (random.randint(0,100)>0):
            args.prompt = genxprompt()
            
        if kiki==0:
            seed=args.seed            
            scale=args.scale
        
        args.seed=random.randint(0,4294967295)        
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
    
    kf = int(duration/keyframes)
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
  
        с1=all_c[i1]
        if k>0:
            c1=c2
            z1=z2
        
        z2=interpolate(z1,all_z[i2],zamp)
            
        c2=interpolate(c1,all_c[i2],camp)
        
        
        for f in range(kf):
            gc.collect()
            torch.cuda.empty_cache() 
            t=blend(f/kf,blendmode)            
            args.ddim_eta=0
            c = interpolate(c1,c2,t)
            z = interpolate(z1,z2,t)
            
            tf = frame/(kf*keyframes)
            
            c = interpolate(c,c_i,tf*0.9)
            z = interpolate(z,z_i,tf*0.9)

            args.init_c=c
            
           
            img = sd.lat2img(args,z,strength)[0]
            
            display.display(img)
            filename = f"{frame:04}.png"
            img.save(os.path.join(framesfolder,filename))
            frame+=1
        z2 = interpolate(z1,z2,1.0)
        c2 = interpolate(c1,c2,1.0)

        c2 = interpolate(c2,c_i,tf*0.9)

        z2 = interpolate(z2,z_i,tf*0.9)
            
    timestring = time.strftime('%Y%m%d%H%M%S')
    filename = str(timestring)+'.mp4'

    outfile = os.path.join(outfolder,filename)
    
    with open(os.path.join(outfolder, str(timestring)+'.txt'), 'w') as f:
        f.write(str(prompts)+'_'+str(seeds)+'_'+str(args.scale)+'_'+str(strength)+'_'+str(args.sampler)+'_'+str(args.steps)+'_'+str(duration)+'_'+str(zamp)+'_'+str(camp))
    
    mp4_path = outfile

    image_path = os.path.join(framesfolder, "%04d.png")
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

def interpolate_prompts( sd,baseargs,duration,fps,zamp,camp,strength,blendmode, prompts_list):
    interpolate=slerp2
    keyframes=len(prompts_list)
    
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
        
        args.prompt = prompt[1]
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
    
    kf = int(duration/keyframes)
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
  
        с1=all_c[i1]
        if k>0:
            c1=c2
            z1=z2
            
        z2=all_z[i2]
        c2=all_c[i2]

            
        if zamp<1.:
            z2=interpolate(z1,z2,zamp)
        if camp<1.:
            c2=interpolate(c1,c2,camp)
        
        
        for f in range(kf):
            gc.collect()
            torch.cuda.empty_cache() 
            t=blend(f/kf,blendmode)            
            args.ddim_eta=0
            c = interpolate(c1,c2,t)
            z = interpolate(z1,z2,t)
            
            tf = frame/(kf*keyframes)
            
            if args.smoothinterp:
                c = interpolate(c,c_i,tf*0.9)
            if args.smoothinterp:
                z = interpolate(z,z_i,tf*0.9)

            args.init_c=c
           
            img = sd.lat2img(args,z,strength)[0]
            
            display.display(img)
            filename = f"{frame:04}.png"
            img.save(os.path.join(framesfolder,filename))
            frame+=1
            
        z2 = interpolate(z1,z2,1.0)
        c2 = interpolate(c1,c2,1.0)
        if args.smoothinterp:
            c2 = interpolate(c2,c_i,tf*0.9)
        if args.smoothinterp:
            z2 = interpolate(z2,z_i,tf*0.9)
        
            
    timestring = time.strftime('%Y%m%d%H%M%S')
    filename = str(timestring)+'.mp4'

    outfile = os.path.join(outfolder,filename)
    
    with open(os.path.join(outfolder, str(timestring)+'.txt'), 'w') as f:
        f.write(str(prompts)+'_'+str(seeds)+'_'+str(args.scale)+'_'+str(strength)+'_'+str(args.sampler)+'_'+str(args.steps)+'_'+str(duration)+'_'+str(zamp)+'_'+str(camp))
    
    mp4_path = outfile

    image_path = os.path.join(framesfolder, "%04d.png")
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
