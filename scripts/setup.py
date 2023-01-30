import subprocess, time, gc, os, sys, requests
from urllib.parse import urlparse

def basename (url):
    return os.path.basename( urlparse(url).path)

def download_model(root,model_url,token=''):
  models_path = root.models_path
  model_f = basename(model_url)
  if not os.path.exists(os.path.join(models_path, model_f)):

      os.makedirs(models_path, exist_ok=True)

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


def load_model(root,model_f):
  from helpers.model_load import load_model
  if model_f.startswith('http'):
    model_url = model_f
    model_f = basename(model_url)
    download_model(root,model_url)
  
  root.model_checkpoint = model_f
  #root.models_path = os.path.join (root.models_path,model_f)
  root.model, root.device = load_model(root,load_on_run_all=True, check_sha256=False)
  


def setup_environment(print_subprocess=True):
    start_time = time.time()
    use_xformers_for_colab = True
    installit=True
    
    if installit:
        import torch
        
        all_process = [
            ['pip', 'install', 'omegaconf', 'einops==0.4.1', 'pytorch-lightning==1.7.7', 'torchmetrics', 'transformers', 'safetensors', 'kornia'],
            ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
            ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open-clip-torch','numpngw','numexpr','opencv-python'],
            ['apt-get', 'update'],
            ['apt-get', 'install', '-y', 'python3-opencv']
        ]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)
        with open('deforum-stable-diffusion/src/k_diffusion/__init__.py', 'w') as f:
            f.write('')
        sys.path.extend([
            'deforum-stable-diffusion/',
            'deforum-stable-diffusion/src',
        ])
        if use_xformers_for_colab:

            print("..installing triton and xformers")

            all_process = [['pip', 'install', 'triton==2.0.0.dev20221202', 'xformers==0.0.16rc424']]
            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")
    return
