import numpy as np
import urllib, base64, random, time, pickle
from PIL import Image
from io import BytesIO
from IPython import display as disp
import os, sys, random, shutil, json
import copy

def clear():
    disp.clear_output()
class Sd:
    def __init__(self,basedir='/workspace/',print_subprocess=True, hugging_face_token=''):
      from sdthings.scripts.setup import setup_environment
      self.basedir = basedir
      setup_environment(print_subprocess)
      
      from helpers.generate import generate
      self.generate = generate

    def makeargs(self):
      from sdthings.scripts.modelargs import makeArgs
      self.root, self.args = makeArgs(self.basedir)
      return self.args
    
    def load(self,model_checkpoint='https://huggingface.co/dreamlike-art/dreamlike-diffusion-1.0/resolve/main/dreamlike-diffusion-1.0.ckpt'):
      from sdthings.scripts.setup import load_model
      load_model(self.root,model_checkpoint)
        
    def txt2img(self,input_args, return_latent=False, return_c=False):
        args = copy.deepcopy(input_args)
        args.init_latent = None
        args.strength = 0
        args.use_init = False
        args.init_image = None
        
        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
            
        if args.c2!= None:
            if args.init_c == None:
                c1=self.root.model.get_learned_conditioning(args.prompt)
            else:
                c1 = args.init_c
            cmix=c1*(args.cmix)+args.c2*(1.-args.cmix)
            args.init_c=cmix

        results = self.generate(args, self.root,0, return_latent, False, return_c)
        return results
    
    def lat2img(self,input_args,latent=None,strength=0.3):
        args = copy.deepcopy(input_args)
        args.init_latent = latent
        args.strength = strength
        args.use_init = True
        args.init_image = None
        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
        
        if args.c2!= None:
            if args.init_c == None:
                c1=self.root.model.get_learned_conditioning(args.prompt)
            else:
                c1 = args.init_c
            cmix=c1*(args.cmix)+args.c2*(1.-args.cmix)
            args.init_c=cmix
            
        results = self.generate(args, self.root,0, False, False, False)
        return results
    
    def img2img(self,input_args, image=None, strength=0.5, return_latent=False, return_c=False):
        args = copy.deepcopy(input_args)
        args.init_latent = None
        args.strength = strength
        args.use_init = True
        args.init_image = image
        if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)
        
        if args.c2!= None:
            if args.init_c == None:
                c1=self.root.model.get_learned_conditioning(args.prompt)
            else:
                c1 = args.init_c
            cmix=c1*(args.cmix)+args.c2*(1.-args.cmix)
            args.init_c=cmix
            
        results = self.generate(args, self.root,0, return_latent, False, return_c)
        return results
      
                       
    def gen(self,input_args='', return_latent=False, return_c=False):
          args = copy.deepcopy(input_args)
          if args.seed == -1:
            args.seed = random.randint(0, 2**32 - 1)

          results = self.generate(args, self.root,0, return_latent, False, return_c)

          #results = generate(self.model,self.clip_model,args,return_latent,return_c)
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
        self.choices = [self.artists,self.flavors,self.mediums,self.movements]
        
    def gen(self,z=1):
        prompt=''
        for x in range(z):
            library = random.choice(self.choices)
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
                       
              
                       
                       
                       
