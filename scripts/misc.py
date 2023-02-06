import zipfile, os, cv2, random

import subprocess
from subprocess import Popen, PIPE
from PIL import Image

from sldl.video.video_interpolation import VideoInterpolation
vi = VideoInterpolation()
ifrmodel = vi.model.to('cuda')
from sldl.video.ifrnet import IFRNet, ifnet_inference
import torch
from PIL import Image
from torchvision import transforms
device = torch.device('cuda')
precision = torch.float16
model_path='/workspace/film_net_fp16.pt'

model = torch.jit.load(model_path, map_location='cpu')
model.eval().to(device=device, dtype=precision)




def gen_prompt(vars,au):
    prompts=[]
    for i in range(au):
        random.seed()
        p = ''
        for v in vars:
            p+=random.choice(v)
        prompts.append(p)
    return (prompts)

def interpolate_keyframes(odir,basedir,d,FILM=True):
    film_models_folder = os.path.join(basedir,'packages/film_models')
    frames = []
    for frame in sorted(os.listdir(odir)):
        if frame.endswith('.png'):
            frames.append(int(frame.split('.')[0]))
    
    frames = sorted(frames)
    print(frames)
    
    nf=[]
    for f in frames: 
      nf.append(f)
    i=0
    print(len(frames)*d)
    md=2
    for z in range(len(frames)+1):
            print('___')
            iters=d
            i=0

            mm=d//2

            while len(nf)<len(frames)*d+1:
              i2=i+1

              if i2==len(nf)or i2>len(nf):
                i2=0

              if abs(sorted(nf)[i2]-sorted(nf)[i])>1:
                f1, f2 = sorted(nf)[i],sorted(nf)[i2]
                if ( abs(sorted(nf)[i2]-sorted(nf)[i])>d*2-1):
                  mx=d//md
                  if mx==0:
                    mx+=1
                  f3 = mx+f1
                  md+=2
                else:
                  f3 = abs(f1-f2)//2+f1

                print(f1,f2,'>',f3)
                interpolate_frames("%05d" % (f1,),"%05d" % (f2,),"%05d" % (f3,),odir,film_models_folder,FILM)
                
                nf.append(f3)
              else:
                i+=1    

    print(sorted(nf))



def interpolate_frames(f1,f2,f3,odir,film_models_folder,FILM):
    

    
    if FILM:
        
        img1 = transforms.ToTensor()(Image.open(os.path.join(odir,f1+'.png'))).unsqueeze_(0).to(precision).to(device)
        img3 = transforms.ToTensor()(Image.open(os.path.join(odir,f2+'.png'))).unsqueeze_(0).to(precision).to(device)
        img1 = img1-torch.min(img1)
        img1 = img1/torch.max(img1)
        img3 = img3-torch.min(img3)
        img3 = img3/torch.max(img3)
        dt = img1.new_full((1, 1), .5)
        
        with torch.no_grad():
            img2 = model(img1, img3, dt) 

        img2 = img2-torch.min(img2)
        img2 = img2/torch.max(img2)

        im = transforms.ToPILImage()(img2[0]).convert("RGB") 
        
        im.save(os.path.join(odir,f3+'.png'))
  
    else:
        pred_frame = ifnet_inference(ifrmodel, Image.open(os.path.join(odir,f1+'.png')), Image.open(os.path.join(odir,f2+'.png')), 'cuda')
       
        pred_frame.save(os.path.join(odir,f3+'.png'))
    
    return

def prepare_frames(indir, outdir, sz, d):
    i=1
    ki=1
    z = []
    #sdir = os.path.join(inputs_folder,folder)
    #odir = os.path.join(inputs_folder,'interpolated/'+folder)
    sdir = indir
    odir = outdir

    os.makedirs(odir, exist_ok=True)
    for file in os.listdir(odir):
        file = os.path.join(odir,file)
        os.remove(file)
    for image_file_name in sorted(os.listdir(sdir)):
        if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg") or image_file_name.endswith(".gif") or image_file_name.endswith(".png") or image_file_name.endswith(".bmp"):
            p  = os.path.join( sdir, image_file_name)
            t  = os.path.join( odir, "%05d" % (i,)+'.png') 
            img = cv2.imread(p)
            img = cv2.resize(img, (sz[0], sz[1]))
            cv2.imwrite(t, img)
            #print (i, ki)
            z.append(i)
            i+=d
            ki+=1

            

def unzip_inputs(folder):
  for z in os.listdir(folder):
      if z.endswith('.zip'):
          
          pp = os.path.join(folder,z)
          of = folder 

          with zipfile.ZipFile(pp, 'r') as zip_ref:
              zip_ref.extractall(folder)
          f = os.path.join(of,z)
          os.remove(f)
          clean_up_inputs(folder)
      
def clean_up_inputs(inputs_folder):
  files = []
  for z in os.listdir(inputs_folder):
    if z.endswith('.png'):
      files.append(z)
  if len(files)>0:
    i=0
    newdir = os.path.join(inputs_folder, 'input_'+str(i) )
    while os.path.isdir(newdir):
      i+=1
      newdir = os.path.join(inputs_folder, 'input_'+str(i) )
    os.makedirs(newdir, exist_ok=True)

    for f in files:
      os.rename(os.path.join(inputs_folder, f), os.path.join(newdir, f))

def resize(img,sz):
    height = np.size(img, 0)
    width = np.size(img, 1)
    
    if width > height:
        ratio = width / height
        nw = int(sz* ratio)
        nh = sz
    else:
        ratio =  height / width
        nh = int(sz* ratio)
        nw = sz
        
    return(cv2.resize(img, (nw, nh)))
    
