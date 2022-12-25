

from flask import Flask
from flask import Flask,json, Response, request, send_file, abort, stream_with_context
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-nk", "--ngrok", action="store_true")
parser.add_argument("-b", "--basedir",  default='/workspace/')
app_args = parser.parse_args()

if app_args.ngrok:
  from flask_ngrok import run_with_ngrok

import numpy as np
import urllib, base64, random, threading, time


from PIL import Image
from io import BytesIO


from IPython import display as disp
import os, sys, random, shutil, json
basedir = app_args.basedir
print('basedir',basedir)

sys.path.extend([basedir])
import random, os
from sdthings.scripts import tools
from sdthings.scripts.modelargs import makeArgs
from IPython import display as disp

def clear():
    disp.clear_output()
  

sd=None

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




model = None
clip_model = None

loaded = False
#sys_extend(basedir)


model_v = 'sd-v1-4.ckpt'
#model_v = 'v1-5-pruned-emaonly.ckpt'
#model_v = '768-v-ema.ckpt'
#model_v = 'https://huggingface.co/doohickey/doohickey-mega/resolve/main/v3-8000.ckpt'
#model_v = 'https://huggingface.co/ShinCore/MMDv1-18/resolve/main/MMD%20V1-18%20MODEL%20MERGE%20(TONED%20DOWN)%20ALPHA.ckpt'
#model_v = 'https://huggingface.co/jinofcoolnes/sammod/resolve/main/samdoartsultmerge.ckpt'
#model_v = 'https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/redshift-diffusion-v1.ckpt'
status = 'init'



def start_runner():
    global sd,status
    def installation():
        global sd,status
        print('installing')
        status = 'setups'
        hugging_face_token = ''
        model_checkpoint = 'https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/redshift-diffusion-v1.ckpt'
        sd = tools.Sd(
            model_checkpoint = model_checkpoint,
            hugging_face_token = hugging_face_token,
            basedir = basedir
            )
        print('model loaded',sd)

   
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

@app.route("/api/check", methods=["POST"])
def check():
    try:
      r = request
      headers = r.headers
      if headers["message"] == "hello":
        response = app.response_class(
        response=json.dumps({'status': status}),
        status=200,
        mimetype='application/json'
        )
        return response
      else:      
        return abort(fail_res)
    except:
      return abort(fail_res)



def setmodel(model_v):
    global sd,status
    
    from sdthings.scripts.things import load_model
    sd.model = load_model( model_checkpoint =  model_v ,  basedir = basedir )
    print('model loaded')
    print(sd.model)

    

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

    args.seed = int(headers["seed"])
    args.prompt = urllib.parse.unquote(headers['prompt'])
    args.strength = float(headers['strength'])
    args.steps = int(headers['steps']) 
    args.scale = float(headers['scale'])
    #########
    return args

img = ''
@app.route("/api/img2img", methods=["POST"])
def img2img():
    global sd,status,img
    if sd != None:
        from sdthings.scripts.things import generate
        from sdthings.scripts.modelargs import makeArgs
        args = makeArgs(basedir)
        
        args = parseHeaders(args,request.headers)
        args.use_mask = False
        
        if not args.sampler in samplers_list:
            args.sampler = 'euler'
        inpaint = request.headers["inpaint"]
        if inpaint=="true":
          args.use_alpha_as_mask = True
          args.use_mask = True
          args.strength = 0.2
        else:
          args.use_alpha_as_mask = False
          args.use_mask = False

            
        data = request.data
        variation = int(request.headers['variation'])+1
        print('variation', variation)
        if variation == 1:
            print(data)
            f = BytesIO()
            f.write(base64.b64decode(data))
            f.seek(0)
            if inpaint=="true":
                img = Image.open(f)
            else:
                img = Image.open(f).convert("RGB")

            newsize = (args.W, args.H)

            img = img.resize(newsize)

        args.init_image = img
        args.use_init=True

        results = sd.gen(args)
        
        newsize = (args.W_in, args.H_in)
        imgs=[]

        img = results[0]
        img = img.resize(newsize)
        
        return_image = imgtobytes(np.asarray(img))

        return Response(response=return_image, status=200, mimetype="image/png")
    else:
        result = 'model not loaded, current status: '+status
        return status

@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    global sd, status
    print('txt2img')
    if sd != None:
        from sdthings.scripts.things import generate
        from sdthings.scripts.modelargs import makeArgs
        args = makeArgs(basedir)
        
        
        args = parseHeaders(args,request.headers)

        print(args.prompt)
        
        args.use_mask = False
        args.use_init=False
        
        results = sd.gen(args)
        print('generating')
        
        newsize = (args.W_in, args.H_in)
        imgs=[]

        img = results[0]
        img = img.resize(newsize)
        
        return_image = imgtobytes(np.asarray(img))

        return Response(response=return_image, status=200, mimetype="image/png")
    else:
        
        result = 'model not loaded, current status: '+status
        print(result)
        return Response(response="{'status': '"+status+"'}", status=300)


if __name__ == '__main__':
      if app_args.ngrok:
        print('running with ngrok')
        run_with_ngrok(app)
        app.run()
      else:
        app.run(host='0.0.0.0', port=80)
