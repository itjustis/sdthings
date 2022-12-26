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
    #try:
    #    ipy = get_ipython()
    #except:
    #    ipy = 'could not get_ipython'
    #if 'google.colab' in str(ipy):
    print("..setting up environment")
    #['git', 'clone', '-b', 'dev', 'https://github.com/deforum-art/deforum-stable-diffusion'],
    all_process = [
        ['pip', 'install', 'torch==1.12.1+cu113', 'torchvision==0.13.1+cu113', '--extra-index-url', 'https://download.pytorch.org/whl/cu113'],
        ['pip', 'install', 'omegaconf==2.2.3', 'einops==0.4.1', 'pytorch-lightning==1.7.4', 'torchmetrics==0.9.3', 'torchtext==0.13.1', 'transformers==4.21.2', 'safetensors', 'kornia==0.6.7'],
        ['git', 'clone', 'https://github.com/deforum-art/deforum-stable-diffusion'],
        ['pip', 'install', 'accelerate', 'ftfy', 'jsonmerge', 'matplotlib', 'resize-right', 'timm', 'torchdiffeq','scikit-learn','torchsde','open_clip_torch'],
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

        print("..installing xformers")

        all_process = [['pip', 'install', 'triton==2.0.0.dev20220701']]
        for process in all_process:
            running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
            if print_subprocess:
                print(running)

        v_card_name = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], stdout=subprocess.PIPE).stdout.decode('utf-8')
        supported = True
        if 't4' in v_card_name.lower():
            name_to_download = 'T4'
        elif 'v100' in v_card_name.lower():
            name_to_download = 'V100'
        elif 'a100' in v_card_name.lower():
            name_to_download = 'A100'
        elif 'p100' in v_card_name.lower():
            name_to_download = 'P100'
        elif 'a4000' in v_card_name.lower():
            name_to_download = 'Non-Colab/Paperspace/A4000'
        elif 'p5000' in v_card_name.lower():
            name_to_download = 'Non-Colab/Paperspace/P5000'
        elif 'quadro m4000' in v_card_name.lower():
            name_to_download = 'Non-Colab/Paperspace/Quadro M4000'
        elif 'rtx 4000' in v_card_name.lower():
            name_to_download = 'Non-Colab/Paperspace/RTX 4000'
        elif 'rtx 5000' in v_card_name.lower():
            name_to_download = 'Non-Colab/Paperspace/RTX 5000'
        else:
            supported = False
            print(v_card_name + ' is currently not supported with xformers flash attention in deforum!')
        if supported:
            if 'Non-Colab' in name_to_download:
                x_ver = 'xformers-0.0.14.dev0-cp39-cp39-linux_x86_64.whl'
            else:
                x_ver = 'xformers-0.0.13.dev0-py3-none-any.whl'

            x_link = 'https://github.com/TheLastBen/fast-stable-diffusion/raw/main/precompiled/' + name_to_download + '/' + x_ver

            all_process = [
                ['wget', '--no-verbose', '--no-clobber', x_link],
                ['pip', 'install', x_ver],
            ]

            for process in all_process:
                running = subprocess.run(process,stdout=subprocess.PIPE).stdout.decode('utf-8')
                if print_subprocess:
                    print(running)
    else:
        sys.path.extend([
            'src'
        ])
    end_time = time.time()
    print(f"..environment set up in {end_time-start_time:.0f} seconds")
    return
