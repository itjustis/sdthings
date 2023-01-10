import torch,os

class CLIPConverter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Define the layers of the model
        self.linear1 = torch.nn.Linear(768, 512)
        self.linear2 = torch.nn.Linear(512, 77 * 768)

    def forward(self, x):
        # Perform the forward pass of the model
        x = self.linear1(x)
        x = self.linear2(x)
        # Reshape the output to [batch_size, 77, 768]
        return x.view(-1, 77, 768)

class ImageEncoder:
    def __init__(self,basedir='/workspace/',model_file='training_v1_4_temp.pth'):
        self.model = CLIPConverter()
        model_path = os.path.join(os.path.join(basedir,'models'),model_file)
        self.model = torch.load(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        from transformers import CLIPFeatureExtractor, CLIPModel
        self.image_processor = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-large-patch14")
        self.image_encoder = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
        self.device='cuda'
    def encode_image(self,image):
        input_image = self.image_processor(images=image, return_tensors="pt").to(self.device)
        return self.image_encoder.to(self.device).get_image_features(**input_image)
    def enc(self,image):
        image = self.encode_image(image)
        with torch.no_grad():
            x = self.model(image) 
        return x

