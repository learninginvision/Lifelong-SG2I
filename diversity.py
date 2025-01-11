import torch
import sys
sys.path.append('stable-diffusion/src')
from clip import clip
from PIL import Image
import os
import numpy as np


def variance_diversity(features):
    variances = features.var(dim=0)
    return variances.mean().item()

def parse_args():
    parser = argparse.ArgumentParser(description='Train model with specified dataset')
    parser.add_argument('--path', required=True, help='Your image folder path')


concepts = ['dog','duck_toy','cat','backpack','teddybear','car','flower','lamp','shoes','bike']

if __name__ == '__main__':
# 加载 CLIP 模型
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    image_folder_list = []
    # set image_folder_list to the paths of the image folders you want to analyze
    for concept in concepts:
        image_folder_list.append(f"{args.path}/{concept}/samples")
    batch_size = 100  
    features_list = []

    average_diversity = 0
    for image_folder in image_folder_list:
        image_paths = [
            os.path.join(image_folder, fname)
            for fname in os.listdir(image_folder)
            if fname.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

        batched_images = []  
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    image = Image.open(path)
                    images.append(preprocess(image).unsqueeze(0))  
                except Exception as e:
                    print(f"Error processing {path}: {e}")
            
            if images:
                batched_images.append(torch.cat(images).to(device))

        for batch in batched_images:
            with torch.no_grad():
                features = model.encode_image(batch).cpu() 
                features_list.append(features)


                diversity_var = variance_diversity(features)
                # print(f'variance diversity:', diversity_var)
                # print(diversity_var)
        average_diversity += diversity_var/len(image_folder_list)    
    print('average diversity:', average_diversity)    
