
from flask import Flask

from IPython import display as disp
import os, sys, random, shutil
from sdthings.scripts.setup import setup,sys_extend, useembedding

app = Flask(__name__)

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



status = 'init'

model = None
clip_model = None

def setmodel(model_v):
    global model,status
    
    from sdthings.scripts.things import load_model
    model = load_model( model_checkpoint =  model_v ,  basedir = basedir )

def installation():
    global model,status
    status = 'setups'
    hugging_face_token = ''

    setup(hf = hugging_face_token, model = model_v , basedir = basedir )
    status = 'loading'
    setmodel(model_v)
    status = 'ready'
    print(status)
    

def img2img():
    from sdthings.scripts.things import generate
    return

@app.route("/api/txt2img")
def txt2img():
    global model,status
    from sdthings.scripts.things import generate
    from sdthings.scripts.modelargs import makeArgs
    if model != None:
        args = makeArgs(basedir)
        image = generate(model,clip_model,args)[0]
        disp.display(image)
    else:
        result = 'model not loaded, current status: '+status
        print (result)
        if status == 'init':
            installation()
    return status
    
def inpaint():
    return

def getstatus(userid):
    if status == 'init':
        installation()

       
        
    

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

if __name__ == '__main__':
      app.run(host='0.0.0.0', port=80)