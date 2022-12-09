import numpy as np
import urllib, base64, random, time, pickle
from PIL import Image
from io import BytesIO
from IPython import display as disp
import os, sys, random, shutil, json
from sdthings.scripts.setup import setup, sys_extend, useembedding


class Sd:
    def __init__(self, model_checkpoint='sd-v1-4.ckpt',hugging_face_token='', basedir='/workspace/'):
        self.model_checkpoint = model_checkpoint
        try:
            if model_checkpoint.endswith('v2-1_768-ema-pruned.ckpt'):
                from sdthings.scripts.things2 import load_model
            else:
                from sdthings.scripts.things import load_model
            self.model = load_model( model_checkpoint =  model_checkpoint,  basedir = basedir )
        except:
            setup(hf = hugging_face_token, model = model_checkpoint , basedir = basedir )
            sys_extend(basedir)
            if model_checkpoint.endswith('v2-1_768-ema-pruned.ckpt'):
                from sdthings.scripts.things2 import load_model
            else:
                from sdthings.scripts.things import load_model
            self.model = load_model( model_checkpoint =  model_checkpoint,  basedir = basedir )
                       
    def gen(self,args='', return_latent=False, return_c=False):
        if self.model_checkpoint.endswith('v2-1_768-ema-pruned.ckpt'):
            from sdthings.scripts.things2 import generate
        else:
            from sdthings.scripts.things import generate
            
        self.clip_model=None
        results = generate(self.model,self.clip_model,args,return_latent,return_c)
        return results

        

class Dictionary:
    def __init__(self, basedir='/workspace/', folder='dictionary'):
        self.dictionary_folder=os.path.join(basedir,folder)
        dictionary_folder = self.dictionary_folder

        artists_filepath = os.path.join(dictionary_folder, 'ViT-L-14_openai_artists.pkl')
        flavors_filepath = os.path.join(dictionary_folder, 'ViT-L-14_openai_flavors.pkl')
        mediums_filepath = os.path.join(dictionary_folder, 'ViT-L-14_openai_mediums.pkl')
        movements_filepath = os.path.join(dictionary_folder, 'ViT-L-14_openai_movements.pkl')

        with open(artists_filepath, 'rb') as f:
            data = pickle.load(f)
            self.artists = data['labels']
        with open(flavors_filepath, 'rb') as f:
            data = pickle.load(f)
            self.flavors = data['labels']
        with open(mediums_filepath, 'rb') as f:
            data = pickle.load(f)
            self.mediums = data['labels']
        with open(movements_filepath, 'rb') as f:
            data = pickle.load(f)
            self.movements = data['labels']
                       
        self.words = []
        
    def gen(self,z,choices):
        prompt=''
        for x in range(z):
            library = random.choice(choices)
            prompt+=random.choice(library)+', '
        return prompt
      

    def load(self,wordsfile,split=True):
        self.words = []
        with open(os.path.join(self.dictionary_folder,wordsfile), 'r', encoding='UTF-8') as file:
                for line in file:

                       if split:
                           line = line.rstrip('\n').split(',')
                           for word in line:
                                  if len(word)>1:
                                        self.words.append(word)
                       else:
                            if len(line)>1:
                                        self.words.append(line)
                       
              
                       
                       
                       