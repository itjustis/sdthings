import json
from IPython import display as disp
import argparse, sys
import os
import subprocess, time, requests


#os.makedirs(output_path, exist_ok=True)

model_sha256 = 'fe4efff1e174c627256e44ec2991ba279b3816e364b49f9be2abc0b3ff3f8556'
model_url = 'https://huggingface.co/CompVis/stable-diffusion-v-1-4-original/resolve/main/sd-v1-4.ckpt'
model_url_runway_1_5 = 'https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt'

def setup(hf='none',model='sd-1.4', basedir = '/workspace/'):
    
  global model_url, model_url_runway_1_5, model_sha256
  if model=='sd-v1-4.ckpt':
      model_f='sd-v1-4.ckpt'
  if model=='v1-5-pruned-emaonly.ckpt':
      model_f='v1-5-pruned-emaonly.ckpt'
      model_url = model_url_runway_1_5
  else:
      model_f='sd-v1-4.ckpt'


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


              all_process = [

                  ['git', 'clone', 'https://github.com/deforum/stable-diffusion', os.path.join(deps_path,'stable-diffusion')],
                  ['git', 'clone', 'https://github.com/shariqfarooq123/AdaBins.git', os.path.join(deps_path,'AdaBins')],
                  ['git', 'clone', 'https://github.com/isl-org/MiDaS.git', os.path.join(deps_path,'MiDaS')],
                  ['git', 'clone', 'https://github.com/MSFTserver/pytorch3d-lite.git', os.path.join(deps_path,'pytorch3d-lite')],

              ]
              for process in all_process:
                  running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                  if print_subprocess:
                      print(running)

              print(subprocess.run(['git', 'clone', 'https://github.com/deforum/k-diffusion/', os.path.join(deps_path,'k-diffusion')], stdout=subprocess.PIPE).stdout.decode('utf-8'))
              with open(os.path.join(deps_path,'k-diffusion/k_diffusion/__init__.py'), 'w') as f:
                  f.write('')
              end_time = time.time()
              print(f"Environment set up in {end_time-start_time:.0f} seconds")

  if not os.path.exists(os.path.join(basedir,'/temp.temp')):
      print('packages setups...')
      p_i=0
      all_process = [
                  ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
                  ['pip', 'install', 'pandas', 'scikit-image', 'opencv-python', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq'],
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
          print('please wait...',p_i,'/',7)

      with open(os.path.join(basedir,'temp.temp'), 'w') as f:
          f.write('temp')


def sys_extend(basedir):
  deps_path = os.path.join(basedir,'packages')
  sys.path.extend([
    deps_path,
    os.path.join(deps_path,'src/taming-transformers'),
    os.path.join(deps_path,'src/clip'),
    os.path.join(deps_path,'stable-diffusion/'),
    os.path.join(deps_path,'k-diffusion'),
    os.path.join(deps_path,'pytorch3d-lite'),
    os.path.join(deps_path,'AdaBins'),
    os.path.join(deps_path,'MiDaS'),
  ])

