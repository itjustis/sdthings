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
from skimage.exposure import match_histograms
import cv2,bisect,os
import numpy as np
from tqdm import tqdm
from PIL import Image


device = torch.device('cuda')
precision = torch.float16
model_path='/workspace/film_net_fp16.pt'

model = torch.jit.load(model_path, map_location='cpu')
model.eval().to(device=device, dtype=precision)



    
def interpolate_folder(inf,outf,iters):
    keyframes = []
    for frame in sorted(os.listdir(inf)):
        if frame.endswith('.png') or frame.endswith('.jpg') or frame.endswith('.jpeg'):
            keyframes.append(os.path.join(inf,frame))
            
    index = 0
    
    for i in range(len(keyframes)):
            i2 = i+1
            if i2==len(keyframes):
                i2 = 0
            #print(i,'>',i2)
            img1 = keyframes[i]
            img2 = keyframes[i2]
            
            interpolate_frames(index,img1,img2,iters,outf)
            index=index+iters
        

    

def interpolate_frames(index, img1, img2, inter_frames, save_path, sz=[768,768], half = True):
    img_batch_1, crop_region_1 = load_image(img1,sz)
    img_batch_2, crop_region_2 = load_image(img2,sz)

    img_batch_1 = torch.from_numpy(img_batch_1).permute(0, 3, 1, 2)
    img_batch_2 = torch.from_numpy(img_batch_2).permute(0, 3, 1, 2)

    results = [
      img_batch_1,
      img_batch_2
    ]

    idxes = [0, inter_frames + 1]
    remains = list(range(1, inter_frames + 1))

    splits = torch.linspace(0, 1, inter_frames + 2)

    for _ in tqdm(range(len(remains)), 'Generating in-between frames'):
        starts = splits[idxes[:-1]]
        ends = splits[idxes[1:]]
        distances = ((splits[None, remains] - starts[:, None]) / (ends[:, None] - starts[:, None]) - .5).abs()
        matrix = torch.argmin(distances).item()
        start_i, step = np.unravel_index(matrix, distances.shape)
        end_i = start_i + 1

        x0 = results[start_i]
        x1 = results[end_i]

        if torch.cuda.is_available():
            if half:
              x0 = x0.half()
              x1 = x1.half()
            x0 = x0.cuda()
            x1 = x1.cuda()

        dt = x0.new_full((1, 1), (splits[remains[step]] - splits[idxes[start_i]])) / (splits[idxes[end_i]] - splits[idxes[start_i]])

        with torch.no_grad():
            prediction = model(x0, x1, dt)
        insert_position = bisect.bisect_left(idxes, remains[step])
        idxes.insert(insert_position, remains[step])
        results.insert(insert_position, prediction.clamp(0, 1).cpu().float())
        del remains[step]
        os.makedirs(save_path, exist_ok=True)

        y1, x1, y2, x2 = crop_region_1
        frames = [(tensor[0] * 255).byte().flip(0).permute(1, 2, 0).numpy()[y1:y2, x1:x2].copy() for tensor in results]

        w, h = frames[0].shape[1::-1]
        
        

        for i in range(len(frames)):
            if i != len(frames)-1:
                frame = frames[i]
                img =  Image.fromarray(frame)
                #display(img)
                filename = "%05d" % (i+index,)+".png"
                #print('index:',index,'saving to:',filename)
                img.save(os.path.join(save_path,filename))

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
    
    
    

def pad_batch(batch, align):
    height, width = batch.shape[1:3]
    height_to_pad = (align - height % align) if height % align != 0 else 0
    width_to_pad = (align - width % align) if width % align != 0 else 0

    crop_region = [height_to_pad >> 1, width_to_pad >> 1, height + (height_to_pad >> 1), width + (width_to_pad >> 1)]
    batch = np.pad(batch, ((0, 0), (height_to_pad >> 1, height_to_pad - (height_to_pad >> 1)),
                           (width_to_pad >> 1, width_to_pad - (width_to_pad >> 1)), (0, 0)), mode='constant')
    return batch, crop_region
def load_image(path, sz, align=384):
#    print( 'loading image',path)
    image = cv2.cvtColor(cv2.resize(cv2.imread(path),(sz[0],sz[1]), interpolation = cv2.INTER_AREA), cv2.COLOR_BGR2RGB).astype(np.float32) / np.float32(255)
    image_batch, crop_region = pad_batch(np.expand_dims(image, axis=0), align)
    return image_batch, crop_region



def gen_prompt(vars,au):
    prompts=[]
    for i in range(au):
        random.seed()
        p = ''
        for v in vars:
            p+=random.choice(v)
        prompts.append(p)
    return (prompts)
