import json
from IPython import display as disp
import argparse, sys
import os, shutil
import subprocess, time, requests
from urllib.parse import urlparse


def basename (url):
    return os.path.basename( urlparse(url).path)


#os.makedirs(output_path, exist_ok=True)

model_sha256 = 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
model_url = 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt'
model_url_runway_1_5 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt'
model_url_v2 = 'https://huggingface.co/stabilityai/stable-diffusion-2/resolve/main/768-v-ema.ckpt'
####
def useembedding(model,embedding_path):
    embedding_type = ".pt" #@param [".bin",".pt"]

    originalpt = r'/workspace/packages/stable-diffusion/ldm/modules/embedding_managerpt.py'

    originalbin = r'/workspace/packages/stable-diffusion/ldm/modules/embedding_managerbin.py'

    if embedding_type == ".pt":
      file_path = "/workspace/packages/stable-diffusion/ldm/modules/embedding_manager.py"
      if os.path.isfile(file_path):
        os.remove(file_path)
        shutil.copyfile(originalpt, file_path)
      print('using .pt embedding')
    elif embedding_type == ".bin":
      file_path = "/workspace/packages/stable-diffusion/ldm/modules/embedding_manager.py"
      if os.path.isfile(file_path):
        os.remove(file_path)
        shutil.copyfile(originalbin, file_path)
      print('using .bin embedding')
    

        #embedding_path= "/workspace/overprettified.pt"
    model.embedding_manager.load(embedding_path)
    
    return model

def setup(hf='none',model='sd-1.4', basedir = '/workspace/'):
    
  global model_url, model_url_runway_1_5, model_sha256
  if model=='sd-v1-4.ckpt':
      model_f='sd-v1-4.ckpt'
  elif model=='v1-5-pruned-emaonly.ckpt':
      model_f='v1-5-pruned-emaonly.ckpt'
      model_url = model_url_runway_1_5
  elif model=='768-v-ema.ckpt':
      model_url = model_url_v2
      model_f = '768-v-ema.ckpt'
  else:
      model_url = model

      model_f = basename(model_url)
    
  print ('using model', model_f, model_url)
    


  models_path = os.path.join(basedir,'models')

  deps_path = os.path.join(basedir,'packages')

  os.makedirs(deps_path, exist_ok=True)
  os.makedirs(models_path, exist_ok=True)

  if not os.path.exists(os.path.join(models_path, model_f)):

      url = model_url
      token=hf

      headers = {"Authorization": "Bearer "+token}

      # contact server for model
      print(f"Attempting to download model...this may take a while")
      ckpt_request = requests.get(model_url, headers=headers)
      request_status = ckpt_request.status_code

      # inform user of errors
      if request_status == 403:

        raise ConnectionRefusedError("You have not accepted the license for this model.")
      elif request_status == 404:
        raise ConnectionError("Could not make contact with server")
      elif request_status != 200:
        raise ConnectionError(f"Some other error has ocurred - response code: {request_status}")

      if request_status != 200:
          print(' downloading error : request_status')

      # write to model path
      if request_status == 200:
          print('model downloaded!')
          with open(os.path.join(models_path, model_f), 'wb') as model_file:
              model_file.write(ckpt_request.content)
      print('saved to', os.path.join(models_path, model_f))

  if not os.path.exists(os.path.join(deps_path,'k-diffusion/k_diffusion/__init__.py')):


          setup_environment = True
          print_subprocess = False
          if setup_environment:

              print("Setting up environment...")
              start_time = time.time()
            # ['git', 'clone', 'https://github.com/facebookresearch/xformers', os.path.join(deps_path,'xformers')],
            # ['git', 'clone', 'https://github.com/deforum/stable-diffusion', os.path.join(deps_path,'stable-diffusion')],
#                  
              all_process = [
                  
                  ['git', 'clone', 'https://github.com/Stability-AI/stablediffusion', os.path.join(deps_path,'stablediffusion')],
                  ['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git', os.path.join(deps_path,'AdaBins')],
                  ['git', 'clone', 'https://github.com/isl-org/MiDaS.git', os.path.join(deps_path,'MiDaS')],
                  ['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git', os.path.join(deps_path,'pytorch3d-lite')],
                  ['git', 'clone', 'https://github.com/google-research/frame-interpolation.git', os.path.join(deps_path,'frame_interpolation')],
              ]
              for process in all_process:
                  running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                  if print_subprocess:
                      print(running)

              print(subprocess.run(['git', 'clone', 'https://github.com/crowsonkb/k-diffusion/', os.path.join(deps_path,'k-diffusion')], stdout=subprocess.PIPE).stdout.decode('utf-8'))
              with open(os.path.join(deps_path,'k-diffusion/k_diffusion/__init__.py'), 'w') as f:
                  f.write('')
              end_time = time.time()
              print(f"Environment set up in {end_time-start_time:.0f} seconds")

  if not os.path.exists(os.path.join(basedir,'temp.temp')):
      print('packages setups...')
      p_i=0
#    ['pip','install','-qq','https://github.com/camenduru/stable-diffusion-webui-colab/releases/download/0.0.14/xformers-0.0.14.dev0-cp37-cp37m-linux_x86_64.whl'],
      
      all_process = [
                  ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
                  ['pip', 'install', 'tensorflow==2.8.0'],
                  ['pip', 'install', 'open_clip_torch','torchsde','clean-fid','gdown','pandas', 'scikit-image', 'opencv-python', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq'],
                  ['pip', 'install', 'flask_cors', 'flask_ngrok', 'pyngrok==4.1.1', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'kornia==0.6.7'],
                  ['pip', 'install', '-e', 'git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers','--src',os.path.join(deps_path,'src')],
                  ['pip', 'install', '-e', 'git+https://github.com/openai/CLIP.git@main#egg=clip','--src',os.path.join(deps_path,'src')],         
                  ['apt-get', 'update'],
                  ['apt-get', 'install', '-y', 'python3-opencv']
              ]

      for process in all_process:
          running = subprocess.run(process,stdout=subprocess.PIPE, ).stdout.decode('utf-8')
          print(running)
          disp.clear_output(wait=True)
          p_i += 1
          print('please wait...',p_i,'/',8)

  
      with open(os.path.join(basedir,'temp.temp'), 'w') as f:
          f.write('temp')

  film_models_folder = os.path.join(basedir,'packages/film_models')
  if not os.path.exists(film_models_folder):  
    os.makedirs(deps_path, exist_ok=True) 
    import gdown
    gdown.download_folder('https://drive.google.com/drive/folders/1q8110-qp225asX3DQvZnfLfJPkCHmDpy', output=film_models_folder, quiet=False, use_cookies=False)




def sys_extend(basedir):
  deps_path = os.path.join(basedir,'packages')
  sys.path.extend([
    deps_path,
    os.path.join(deps_path,'src/taming-transformers'),
    os.path.join(deps_path,'src/clip'),
    os.path.join(os.path.join(basedir,'sdthings'),'stable-diffusion/'),
    os.path.join(deps_path,'k-diffusion'),
    os.path.join(deps_path,'pytorch3d-lite'),
    os.path.join(deps_path,'AdaBins'),
    os.path.join(deps_path,'MiDaS'),
  ])

