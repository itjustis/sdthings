

def interpolate_frames_deprecated(f1,f2,f3,odir,film_models_folder,FILM):

    
    if FILM:
        img1_i = Image.open(os.path.join(odir,f1+'.png'))
        img3_i = Image.open(os.path.join(odir,f2+'.png'))
        
        img1 = transforms.ToTensor()(img1_i).unsqueeze_(0).to(precision).to(device)
        img3 = transforms.ToTensor()(img3_i).unsqueeze_(0).to(precision).to(device)
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
        #im = match_histograms(im, img1_i)
        
        im.save(os.path.join(odir,f3+'.png'))
  
    else:
        pred_frame = ifnet_inference(ifrmodel, Image.open(os.path.join(odir,f1+'.png')), Image.open(os.path.join(odir,f2+'.png')), 'cuda')
       
        pred_frame.save(os.path.join(odir,f3+'.png'))
    
    return



def prepare(sd,iters,indir,outdir,sz):
    
#    from sdthings.scripts.misc import prepare_frames, interpolate_keyframes

    
    film_models_folder = os.path.join(basedir,'packages/film_models')

    if not os.path.exists(outdir):
        try:
            del sd.root.model
        except:
            print('')
        import torch
        torch.cuda.empty_cache()

        os.makedirs(indir, exist_ok=True)
        prepare_frames(indir,outdir, sz, iters)
        interpolate_keyframes(outdir,basedir,iters,True)
    else:
        'skipping. found existing interpolated frames'


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
