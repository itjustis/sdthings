from types import SimpleNamespace
def defArgs(basedir):
    dynamic_threshold = None
    static_threshold = None
    save_samples = False
    save_settings = False
    display_samples = False
    n_batch = 1
    batch_name = "xxz"
    filename_format = "{timestring}_{index}_{prompt}.png"
    seed_behavior = "iter"
    make_grid = False
    grid_rows = 2
    outdir = '/content/kek/'
    use_init = False
    strength = 0.0
    strength_0_no_init = True
    init_image = "https://cdn.pixabay.com/photo/2022/07/30/13/10/green-longhorn-beetle-7353749_1280.jpg"

    use_mask = False
    use_alpha_as_mask = False
    mask_file = "https://www.filterforge.com/wiki/images/archive/b/b7/20080927223728%21Polygonal_gradient_thumb.jpg"
    invert_mask = False

    mask_brightness_adjust = 1.0
    mask_contrast_adjust = 1.0

    n_samples = 1
    precision = 'autocast'
    C = 4
    f = 8

    prompt = ""
    timestring = ""
    init_latent = None
    init_sample = None
    init_c = None

    seed = -1
    sampler = 'euler'
    steps = 12
    scale = 9.3
    midas_weight = 0.3
    
    olormatch_loss_scale = 5000 #@param {type:"number"}
    colormatch_image = "https://www.saasdesign.io/wp-content/uploads/2021/02/palette-3-min-980x588.png" #@param {type:"string"}
    colormatch_n_colors = 4 #@param {type:"number"}
    clamp_grad_threshold = 1. #@param {type:"number"}
    ignore_sat_scale = 5 #@param 

    blue_loss_scale = 0
    init_mse_scale = 0 #@param {type:"number"}
    clip_loss_scale = 0 #@param {type:"number"}

    cutn = 1 #@param {type:"number"}
    cut_pow = 64 #@param {type:"number"}


    grad_threshold_type = 'dynamic' #@param ["dynamic", "static", "mean"]
    gradient_wrt = 'x' #@param ["x", "x0_pred"]
    gradient_add_to = 'both' #@param ["cond", "uncond", "both"]
    decode_method = 'linear' #@param ["autoencoder","linear"]

    cond_uncond_sync = True #@param {type:"boolean"}
    save_sample_per_step = False
    show_sample_per_step = False
    
    clip_prompts=['']
    clip_loss_img_scale = 0
    
    

    W = 512
    H = 512
    prompt='dog'
    
    basedir = basedir

    return locals()


def defVideoArgs():
    video_steps=70
    keyframes_strength=0.8
    vseed = 1
    total_frames=90
    easing='bezier'
    interpolation='slerp2'
    isdisplay = True
    vscale = 60
    truncation = 1.
    fps = 15
    eta = 0.0
    
    return locals()

def makeVideoArgs():
    return SimpleNamespace(**defVideoArgs())
    
def makeArgs(basedir):
    return SimpleNamespace(**defArgs(basedir))
    
    
