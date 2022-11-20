from types import SimpleNamespace
def defArgs():
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

    W = 512
    H = 512
    prompt='dog'

    return locals()

def makeArgs():
    return SimpleNamespace(**defArgs())
