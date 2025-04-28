import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import pairwise_distances
import os
import random
from tqdm import tqdm 

dataset_path = r"C:\Users\mathe\Desktop\CSCI-44800-Biometrics-Projects\face-recognition\archive"
image_folder = os.path.join(dataset_path, "lfw-deepfunneled")
people_df = pd.read_csv(os.path.join(dataset_path, "people.csv"))
model = InceptionResnetV1(pretrained='vggface2').eval()

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def get_image_paths(max_images=13223):
    image_paths = []
    for root, _, files in os.walk(image_folder): 
        for img_file in files:
            if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):  
                image_paths.append(os.path.join(root, img_file))
                if max_images and len(image_paths) >= max_images:  
                    return image_paths
    return image_paths

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def extract_features(image_path):
    img_tensor = load_image(image_path)
    with torch.no_grad():
        embedding = model(img_tensor)
    return embedding.squeeze(0).numpy() 

def compute_cmc(query_features, gallery_features):
    distances = pairwise_distances(query_features, gallery_features, metric='cosine')
    ranks = np.argsort(distances, axis=1)
    cmc_curve = np.mean(ranks < np.arange(1, 11)[:, None], axis=1)
    return cmc_curve, ranks

def plot_cmc(cmc_curve):
    plt.plot(range(1, len(cmc_curve) + 1), cmc_curve, marker='o')
    plt.xlabel("Rank")
    plt.ylabel("Recognition Rate")
    plt.title("CMC Curve")
    plt.grid()
    plt.show()

def verification_mode(query_images, gallery_images):
    print("Extracting features for query images...")
    query_features = np.array([extract_features(img) for img in tqdm(query_images, desc="Query Images")])
    
    print("Extracting features for gallery images...")
    gallery_features = np.array([extract_features(img) for img in tqdm(gallery_images, desc="Gallery Images")])
    
    cmc_curve, ranks = compute_cmc(query_features, gallery_features)
    
    print(f"Query features shape: {query_features.shape}")
    print(f"Gallery features shape: {gallery_features.shape}")

    print("Verification Scores:", cmc_curve)
    plot_cmc(cmc_curve)

def single_query_matching(query_image, gallery_images):
    print("Extracting features for the query image...")
    query_feature = extract_features(query_image).reshape(1, -1)
    
    print("Extracting features for gallery images...")
    gallery_features = np.array([extract_features(img) for img in tqdm(gallery_images, desc="Gallery Images")])

    distances = pairwise_distances(query_feature, gallery_features, metric='cosine')
    top_matches = np.argsort(distances[0])[:3]

    print(f"Top 3 matches for {query_image}:")
    for idx in top_matches:
        print(f"  Match {idx + 1}: {gallery_images[idx]}")

    fig, axes = plt.subplots(1, 4, figsize=(15, 5)) 
    axes[0].imshow(Image.open(query_image)) 
    axes[0].set_title("Query Image")
    axes[0].axis("off")

    for i, idx in enumerate(top_matches):
        axes[i + 1].imshow(Image.open(gallery_images[idx])) 
        axes[i + 1].set_title(f"Match {i + 1}")
        axes[i + 1].axis("off")

    plt.tight_layout()
    plt.show()

all_faces = get_image_paths(max_images=10000) 

query_images = random.sample(all_faces, 10)
gallery_images = [img for img in all_faces if img not in query_images]

verification_mode(query_images, gallery_images)
single_query_matching(query_images[0], gallery_images)
