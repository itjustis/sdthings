import zipfile, os

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
    