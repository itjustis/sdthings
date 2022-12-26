import subprocess, time, gc, os, sys

def setup_environment():
    start_time = time.time()
    print_subprocess = False
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
            print(v_card_name + ' is currently not supported with xformers flash attention in deforum!')

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
