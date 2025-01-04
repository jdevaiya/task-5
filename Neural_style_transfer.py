import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(image_path, max_size=400, shape=None):
    image = Image.open(image_path).convert('RGB')
    
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape:
        size = shape
    
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image

# Define the model
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.features = models.vgg19(pretrained=True).features[:21]
        
    def forward(self, x):
        layers = {
            '0': 'conv1_1',
            '5': 'conv2_1',
            '10': 'conv3_1',
            '19': 'conv4_1',
            '21': 'conv4_2',  # Content layer
        }
        features = {}
        for name, layer in self.features._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        return features

# Content and style loss functions
def content_loss(content, target):
    return torch.mean((content - target) ** 2)

def gram_matrix(tensor):
    _, c, h, w = tensor.size()
    tensor = tensor.view(c, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram / (c * h * w)

def style_loss(style, target):
    return torch.mean((gram_matrix(style) - gram_matrix(target)) ** 2)

# Perform style transfer
def transfer_style(content_img, style_img, model, iterations=300, alpha=1, beta=1e6):
    target = content_img.clone().requires_grad_(True)
    optimizer = optim.Adam([target], lr=0.003)
    
    for i in range(iterations):
        target_features = model(target)
        content_features = model(content_img)
        style_features = model(style_img)
        
        c_loss = content_loss(target_features['conv4_2'], content_features['conv4_2'])
        s_loss = sum(style_loss(target_features[layer], style_features[layer]) for layer in style_features)
        
        total_loss = alpha * c_loss + beta * s_loss
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        if i % 50 == 0:
            print(f"Iteration {i}, Total Loss: {total_loss.item()}")
            
    return target

# Convert tensor to image
def im_convert(tensor):
    image = tensor.clone().detach()
    image = image.squeeze(0)
    image = image * torch.tensor((0.5, 0.5, 0.5)) + torch.tensor((0.5, 0.5, 0.5))
    image = image.clamp(0, 1)
    return transforms.ToPILImage()(image)

# Main script
if __name__ == "__main__":
    # Load images
    content_path = 'path_to_content.jpg'  # Replace with your content image path
    style_path = 'path_to_style.jpg'      # Replace with your style image path
    
    content = load_image(content_path)
    style = load_image(style_path, shape=content.shape[-2:])
    
    # Load the model
    model = VGG().eval()
    
    # Apply style transfer
    output = transfer_style(content, style, model)
    
    # Display results
    plt.figure(figsize=(15, 10))
    plt.subplot(1, 3, 1)
    plt.title("Content Image")
    plt.imshow(im_convert(content))
    
    plt.subplot(1, 3, 2)
    plt.title("Style Image")
    plt.imshow(im_convert(style))
    
    plt.subplot(1, 3, 3)
    plt.title("Stylized Image")
    plt.imshow(im_convert(output))
    
    plt.show()
