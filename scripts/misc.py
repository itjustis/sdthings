import zipfile, os, cv2, random

import subprocess
from subprocess import Popen, PIPE

def gen_prompt(vars,au):
    prompts=[]
    for i in range(au):
        random.seed()
        p = ''
        for v in vars:
            p+=random.choice(v)
        prompts.append(p)
    return (prompts)

def interpolate_keyframes(odir,basedir,d,debug=True):
    film_models_folder = os.path.join(basedir,'packages/film_models')
    frames = []
    for frame in sorted(os.listdir(odir)):
        if frame.endswith('.png'):
            print(int(frame.split('.')[0]))
            frames.append(int(frame.split('.')[0]))
    print(frames)

    nf=frames
    for i in range(len(frames)):
        print('___')

        i2=i+1
        if i2==len(frames):
            i2=0
            
        f1, f2 = frames[i],frames[i2]
        

        f3 = f1 +( d // 2 )
        print(f1,f2,'>',f3,'______1')
        nf.append(f3)
        output, err = (interpolate_frames("%02d" % (f1,),"%02d" % (f2,),"%02d" % (f3,),odir,film_models_folder))
        if (debug):
            print (err)
        
        if d>3:
            
            f4 = f1 + ( d // 4 )
            print(f1,f3,'>',f4,'______2')
            nf.append(f4)
            output, err = (interpolate_frames("%02d" % (f1,),"%02d" % (f3,),"%02d" % (f4,),odir,film_models_folder))
            if (debug):
                print (err)
        
            f5 = f3 + ( d // 4 )
            print(f3,f2,'>',f5,'______3')
            nf.append(f5)
            output, err = (interpolate_frames("%02d" % (f3,),"%02d" % (f2,),"%02d" % (f5,),odir,film_models_folder))
            if (debug):
                print (err)
            
            
        if d>5:
            
            f6 = f4 + ( d // 6 )
            print(f4,f5,'>',f6,'______4')
            nf.append(f6)
            output, err = (interpolate_frames("%02d" % (f4,),"%02d" % (f5,),"%02d" % (f6,),odir,film_models_folder))
            if (debug):
                print (err)
        
            f7 = f5 + ( d // 6 )
            print(f5,f2,'>',f7,'______5')
            nf.append(f7)
            output, err = (interpolate_frames("%02d" % (f5,),"%02d" % (f2,),"%02d" % (f7,),odir,film_models_folder))
            if (debug):
                print (err)
        
        if d>7:
            
            f8 = f3 + ( d // 8 )
            print(f3,f7,'>',f8,'______6')
            nf.append(f8)
            output, err = (interpolate_frames("%02d" % (f3,),"%02d" % (f7,),"%02d" % (f8,),odir,film_models_folder))
            if (debug):
                print (err)
        
            f9 = f1 + ( d // 8 )
            print(f1,f4,'>',f9,'______7')
            nf.append(f9)
            output, err = (interpolate_frames("%02d" % (f1,),"%02d" % (f4,),"%02d" % (f9,),odir,film_models_folder))
            if (debug):
                print (err)
            
        
            
        print('FILM pre-interpolating done')

            
        print('______')



def interpolate_frames(f1,f2,f3,odir,film_models_folder):
  model_path=os.path.join(film_models_folder,'film_net/Style/saved_model')
  command = ['python3', 
       '-m', 
       'packages.frame_interpolation.eval.interpolator_test',
       '--frame1',
       os.path.join(odir,f1+'.png'),
       '--frame2',
       os.path.join(odir,f2+'.png'),
       '--model_path',
       model_path,
       '--output_frame',
       os.path.join(odir,f3+'.png'),
       ]
  process = Popen(command, stdout=PIPE, stderr=PIPE)
  output, err = process.communicate()
  return(output, err)

def prepare_frames(inputs_folder,folder, sz, d):
    i=1
    ki=1
    z = []
    sdir = os.path.join(inputs_folder,folder)
    odir = os.path.join(inputs_folder,'interpolated/'+folder)

    os.makedirs(odir, exist_ok=True)
    for file in os.listdir(odir):
        file = os.path.join(odir,file)
        os.remove(file)
    for image_file_name in os.listdir(sdir):
        if image_file_name.endswith(".jpg") or image_file_name.endswith(".jpeg") or image_file_name.endswith(".gif") or image_file_name.endswith(".png") or image_file_name.endswith(".bmp"):
            p  = os.path.join( sdir, image_file_name)
            t  = os.path.join( odir, "%02d" % (i,)+'.png') 
            img = cv2.imread(p)
            img = cv2.resize(img, (sz[0], sz[1]))
            cv2.imwrite(t, img)
            print (i, ki)
            z.append(i)
            i+=d
            ki+=1

    for k in range(ki):
        f = k
    


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
    
