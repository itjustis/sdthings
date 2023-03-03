import random, os, shutil
from IPython import display as disp
import torch
import numpy as np
#from deforum-stable-diffusion.helpers.prompt import get_uc_and_c

def interpolate(v0, v1, t, DOT_THRESHOLD=0.9995):
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

def processframes(sd, isimg, imix, indir, outputs_folder, randomseed=True, cmix=0., promptgen=None):
    random.seed()
    images = sorted(os.listdir(outputs_folder))  
    sd.args.icmix=0.5
    sd.args.cmix = 0.0
    prompts=[]
    seed= random.randint(0,4294967295)
    inimages = sorted(os.listdir(indir))

    conditions, seeds, frames = [],[],[]

    for img in inimages:
        if img.endswith('.png') or img.endswith('.jpg'):
            image = os.path.join(indir,img)
            c = sd.autoc(image,imix)
            if cmix>0. and promptgen != None:
                ppp = promptgen()
                print(ppp)
                #uc, c2 = get_uc_and_c(promptgen(), sd.root.model, args, frame)
                c2 = sd.root.model.get_learned_conditioning( promptgen() )
                #print(c.shape,c2.shape,'kek')
                c = torch.lerp (c.half(),c2.half(),cmix)
            conditions.append(c)

    for img in images:
        if img.endswith('.png') or img.endswith('.jpg'):
            random.seed()
            img = os.path.join(outputs_folder,img)
            if randomseed:
                seed= random.randint(0,4294967295)
            seeds.append(seed)
            if isimg:
                frames.append(img)
            else:
                frames.append(None)

    return conditions, seeds, frames


def randomselect(fld,keyframes):
    allimages = []
    selected =[]
    for img in  os.listdir(fld):
        imgfile = os.path.join( fld, img )
        if img.endswith('.png') or img.endswith('.jpg'):
            allimages.append(imgfile)
    
    if len(allimages)>keyframes:
        while len(selected)<keyframes+1:
            img = random.choice(allimages)
            if img.endswith('.png') or img.endswith('.jpg'):
                #imgfile = os.path.join( fld, img )
                imgfile=img
                if not imgfile in selected:
                    selected.append( imgfile )

    else:
        print('not enough images found in folder')
    
    return selected



def randomframes(fold,keyframes):
    fld=os.path.join('inputs/',fold)
    import random 
    random.seed()

    selected = sorted(randomselect(fld,keyframes))

    tempinp = 'inputs/tempfam'
    tempinpint = 'inputs/interpolated/tempfam'
    
    os.makedirs(tempinp, exist_ok=True)
    os.makedirs(tempinpint, exist_ok=True)
    
    cleanfolder(tempinpint)
    cleanfolder(tempinp)


    for img in selected:
        infile = img
        outfile = os.path.join(tempinp,os.path.basename(img))
        print(infile,outfile)
        shutil.copyfile(infile,outfile)
  
    print(selected)
    return keyframes


def cleanfolder(folder):
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))
