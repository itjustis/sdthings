####
####
####
from flask import Flask
from flask import Flask,json, Response, request, send_file, abort, stream_with_context
import argparse
parser = argparse.ArgumentParser()

parser.add_argument("-nk", "--ngrok", action="store_true")
parser.add_argument("-b", "--basedir",  default='/workspace/')
parser.add_argument("-m", "--model", default=None)

app_args = parser.parse_args()

if app_args.ngrok:
  from flask_ngrok import run_with_ngrok

import numpy as np
import urllib, base64, random, threading, time


from PIL import Image
from io import BytesIO
from skimage.exposure import match_histograms

from IPython import display as disp
import os, sys, random, shutil, json,random
basedir = app_args.basedir
print('basedir',basedir)

sys.path.extend([basedir])

def clear():
    disp.clear_output()
    
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

########## inits
loaded = False
status = 'init'
sd = None
sessions = []
########## temp fld
tempfolder=os.path.join(basedir,'temp')
os.makedirs(tempfolder, exist_ok = True)
for f in tempfolder:
  if f.endswith('.jpg') or f.endswith('.png'):
    os.remove(f)

from sdthings.scripts.ie import CLIPConverter, ImageEncoder

def iencloader(basedir='/workspace/'):
        from sdthings.scripts.ie import CLIPConverter, ImageEncoder
        ie = ImageEncoder(basedir,'ien3.pth')
        
        
        from clip_interrogator import Config, Interrogator
        
        clip_model_name = 'ViT-L-14/openai' #@param ["ViT-L-14/openai", "ViT-H-14/laion2b_s32b_b79k"]

        config = Config()
        config.blip_num_beams = 64
        config.blip_offload = False
        config.clip_model_name = clip_model_name
        print(config)
        interrogator = Interrogator(config)
        
        return ie, interrogator
    
########## all the fun
def start_runner():
    global sd,status
    def installation():
        global sd,status
        print('installing')
        status = 'setups'
        hugging_face_token = ''
        #model_checkpoint = 'https://huggingface.co/nitrosocke/redshift-diffusion/resolve/main/redshift-diffusion-v1.ckpt'
        from sdthings.scripts import tools
       
       
        sd = tools.Sd(basedir,False)
        sd.ie, sd.interrogator = iencloader(basedir)
        
        args = sd.makeargs()
        if app_args.model:
          import torch,gc
          gc.collect()
          torch.cuda.empty_cache()
          sd.load(app_args.model)      
          print('model loaded')
        else:
          sd.load()      
          print('model loaded')
        status = 'ready'
        print(status)

    print('starting runner')
    thread = threading.Thread(target=installation)
    thread.start()
    
def create_app():
    app = Flask(__name__)
    
    def run_on_start(*args, **argv):
        start_runner()
    run_on_start()
    return app

app = create_app()

@app.route("/api/loadmodel", methods=["POST"])
def load_model():
  global sd,status
  try:
    import torch,gc
    del sd.model
    gc.collect()
    torch.cuda.empty_cache()
    ##loading
    headers = request.headers
    #
    modelname = headers["modelname"]
    #
    sd.load(modelname)
    
    status = ('model loaded')
    status = ('model not found')
    status = ('model loaded')
    print(status)
  except Exception as e:
    status = e.message
    print(status)
    return Response(response="{'status': '"+status+"'}", status=300)
    ##return status 

#def setmodel(model_v):
#    global sd,status
#    
#    print('model loaded')
#    print(sd.model)


@app.route("/api/check", methods=["POST"])
def check():
    #print(request.headers)
    try:
      headers = request.headers
      if headers["message"] == "hello":
        if int(headers["session"]) == 0 or not headers["session"] in sessions:
          session = random.randint(0, 2**32 - 1)
          sessions.append(session)
          return Response(response="{'status': '"+status+"','session':'"+str(session)+"'}", status=200)
        else:
          return Response(response="{'status': '"+status+"'}", status=200)
      else:      
        return abort(fail_res)
    except Exception as e:
      
      print(e)
      return Response(response="{'status': '"+status+"'}", status=400)
      #return abort(fail_res)

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
    import gc,torch
    global sd,status,img
    try:
      if sd != None:
          from sdthings.scripts.modelargs import defArgs
          args = defArgs()
          headers = request.headers

          args = parseHeaders(args,headers)
          try:
            sessionid = request.headers['sessionid'] 
            jobid = request.headers['jobid']       
          except:
            sessionid = 0
            jobid = 0
          
          args.use_mask = False

          inpaint = headers["inpaint"]

          #process            
          data = request.data
          variation = int(headers['variation'])+1
          print('variation', variation)
          fn = f'{sessionid}_{jobid}'
          fn = os.path.join(tempfolder,fn+'.png')
          if variation == 1:
              #decode
              #print(data)
              f = BytesIO()
              f.write(base64.b64decode(data))
              f.seek(0)
              if inpaint=="true":
                  img = Image.open(f)
              else:
                  img = Image.open(f).convert("RGB")

              newsize = (args.W, args.H)

              print ('resizing to ',newsize,'generation size', args.W,args.H)

              img = img.resize(newsize)
              
              img.save(fn)
          img = fn
          #####
          #inpaint?        

          if inpaint=="true":
            args.use_alpha_as_mask = True
            args.use_mask = True
            args.invert_mask = False
            args.strength = 0.
            #red, green, blue, alpha = Image.open(img).split()
            args.mask_file= img
            print('trying to inpaint')
          else:
            args.use_alpha_as_mask = False
            args.use_mask = False
          
          if args.prompt == 'auto' or args.prompt == 'auto ':
                print('auto prompt')
                args.c2 = sd.autoc(img)
                args.cmix = 0.0
          else:
                print('prompt is:',args.prompt)
                
          status='busy'
          results = sd.img2img(args, image=img, strength=args.strength)
          status='ready'  

          print(fn)
                 
          
          newsize = (args.W_in, args.H_in)
          imgs=[]
          imgo= Image.open(fn)
          disp.display(img)

          img = results[0]
          img = img.resize(newsize)
          print('resized')

         

          gc.collect()
          torch.cuda.empty_cache()
          

          #color correct
          print('correcting',img)
         
          x = np.asarray(img).astype('float32')
          print('opening',fn)
          y = np.asarray(Image.open(fn).convert('RGB')).astype('float32')
          print('matching')
          img = match_histograms(x, y,multichannel=True)
          print('correction done')

       

          return_image = imgtobytes(np.asarray(img))

          return Response(response=return_image, status=200, mimetype="image/png")
      else:
          result = 'model not loaded, current status: '+status
          return Response(response="{'status': '"+result+"'}", status=300)
    except Exception as e:
      error = e
      print(status,error)
      return Response(response="{'status': '"+status+"','error':'"+error+"'}", status=666)

@app.route("/api/txt2img", methods=["POST"])
def txt2img():
    global sd, status
    print('txt2img')
    try:
      if sd != None:
          from sdthings.scripts.modelargs import defArgs

          args = defArgs()
          args = parseHeaders(args,request.headers)
          #####
          status='busy'
          results = sd.txt2img(args)
          status='ready'
          #####
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
        
    except Exception as e:
      error = e.message
      print(status,error)
      return Response(response="{'status': '"+status+"','error':'"+error+"'}", status=666)



if __name__ == '__main__':
      app.debug = False
      if app_args.ngrok:
        print('running with ngrok')
        run_with_ngrok(app)
        app.run()
      else:
        app.run(host='0.0.0.0', port=80)
