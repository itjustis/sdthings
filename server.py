
from flask import Flask
from flask import Flask, Response, request, send_file, abort, stream_with_context

import numpy as np


import urllib, base64, random, threading, time


from PIL import Image
from io import BytesIO


from IPython import display as disp
import os, sys, random, shutil, json
from sdthings.scripts.setup import setup,sys_extend, useembedding

samplers_list=["klms","dpm2","dpm2_ancestral","heun","euler","euler_ancestral","plms", "ddim"]
fail_res = Response(
            json.dumps({"message": 'error', "code": 400, "status": "FAIL"}),
            mimetype="application/json",
            status=400,
        )

def imgtobytes(image):
    import cv2
    success, encoded_image = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    content2 = encoded_image.tobytes()
    return content2


def start_runner():
    def installation():
        global model,status
        print('installing')
        status = 'setups'
        hugging_face_token = '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'

        setup(hf = hugging_face_token, model = model_v , basedir = basedir )
        status = 'loading'
        setmodel(model_v)
        status = 'ready'
        print(status)

    print('starting runner')
    thread = threading.Thread(target=installation)
    thread.start()

def create_app():
    app = Flask(__name__)
    def run_on_start(*args, **argv):
        print ("function before start")
        start_runner()
    run_on_start()
    return app

app = create_app()

loaded = False
basedir = '/workspace/'
sys_extend(basedir)


model_v = 'sd-v1-4.ckpt'
#model_v = 'v1-5-pruned-emaonly.ckpt'
#model_v = '768-v-ema.ckpt'
#model_v = 'https://huggingface.co/doohickey/doohickey-mega/resolve/main/v3-8000.ckpt'
#model_v = 'https://huggingface.co/ShinCore/MMDv1-18/resolve/main/MMD%20V1-18%20MODEL%20MERGE%20(TONED%20DOWN)%20ALPHA.ckpt'
#model_v = 'https://huggingface.co/jinofcoolnes/sammod/resolve/main/samdoartsultmerge.ckpt'
#model_v = 'https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/redshift-diffusion-v1.ckpt'

@app.route("/api/check", methods=["POST"])
def check():
    try:
      r = request
      headers = r.headers
      if headers["message"] == "hello":
        return Response(response="{}", status=200)
      else:      
        return abort(fail_res)
    except:
      return abort(fail_res)

status = 'init'

model = None
clip_model = None

def setmodel(model_v):
    global model,status
    
    from sdthings.scripts.things import load_model
    model = load_model( model_checkpoint =  model_v ,  basedir = basedir )



@app.route("/api/img2img", methods=["POST"])
def img2img():
    from sdthings.scripts.things import generate
    return

@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    global model,status
    if model != None:
        from sdthings.scripts.things import generate
        from sdthings.scripts.modelargs import makeArgs
        args = makeArgs(basedir)
        
        args = parseHeaders(args,request.headers)
        args.use_mask = False
        
        results = generate(model,clip_model,args)
        
        newsize = (args.W_in, args.H_in)
        imgs=[]
        #for result in results:
        img = results[0]
        img = img.resize(newsize)
        #    imgs.append(img)

        #return_images=''
        #for img in imgs:
        #    return_images=return_images+'_'+imgtobytes(np.asarray(img))
        return_image = imgtobytes(np.asarray(img))

        return Response(response=return_image, status=200, mimetype="image/png")
    else:
        result = 'model not loaded, current status: '+status
        print (result)
        
    return status


    

def inpaint():
    return

def getstatus(userid):
    return status

def parseHeaders(args, headers):
    
    try:
      n_samples = int(headers["n_samples"])
    except:
      n_samples = 1
    args.n_samples = n_samples

    args.W_in, args.H_in = int(headers["W"]), int(headers["H"])
    W, H = map(lambda x: x - x % 64, (args.W_in, args.H_in))
    args.W = W
    args.H = H

    if not  headers['sampler'] in samplers_list:
      args.sampler = 'ddim'
    else:
      args.sampler =  headers['sampler']

    if args.sampler == 'ddim':
      args.ddim_eta = float(headers['ddim_eta'])
    else:
      args.ddim_eta = 0

    ######
    args.use_init=False
    args.seed = int(headers["seed"])
    args.prompt = urllib.parse.unquote(headers['prompt'])
    args.strength = 0
    args.steps = int(headers['steps']) 
    args.scale = float(headers['scale'])
    #########
    return args


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == '__main__': 
    app.run(host='0.0.0.0', port=80)