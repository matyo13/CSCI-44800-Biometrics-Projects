import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Step 1: Load image into variable X and resize
def load_and_resize_image(image_path, scale_percent):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or unable to load.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    resized_image = resized_image.astype(np.float64) / 255.0  
    # print(f"Loaded and resized image with shape: {resized_image.shape}")
    return resized_image

# Step 2: Randomly choose k pixels for initialization
def choose_initial_centroids(image, k):
    h, w, _ = image.shape
    centroids = []
    for _ in range(k):
        i = np.random.randint(0, h)
        j = np.random.randint(0, w)
        centroids.append(image[i, j])
    # print(f"Initial centroids for k={k}: {centroids}")
    return np.array(centroids)

# Step 3: Execute K-means using sklearn's KMeans
def perform_kmeans(image, k, initial_centroids):
    h, w, _ = image.shape
    pixels = image.reshape(-1, 3)
    kmeans = KMeans(n_clusters=k, init=initial_centroids, n_init=1, max_iter=100, random_state=None)
    kmeans.fit(pixels)
    labels = kmeans.labels_.reshape(h, w)
    centroids = kmeans.cluster_centers_
    # print(f"Cluster centroids for k={k}: {centroids}")
    return labels, centroids

# Step 3.5: Reassign clusters
def reassign_clusters(centroids, labels):
    reference_color = centroids[:, 0]  
    sorted_indices = np.argsort(reference_color)
    reassigned_labels = np.zeros_like(labels)
    for new_label, index in enumerate(sorted_indices):
        reassigned_labels[labels == index] = new_label
    # print(f"Reassigned cluster labels based on centroids: {sorted_indices}")
    return reassigned_labels

# Step 5: Calculate frequency of cluster assignments
def calculate_cluster_frequencies(M, h, w, k):
    frequencies = np.zeros((h, w, k))
    for cluster in range(k):
        frequencies[:, :, cluster] = np.sum(M == cluster, axis=2)
    print(f"Calculated cluster frequencies for k={k}")
    return frequencies / M.shape[2]

# Display probability maps and skin region map
def display_results(image, frequencies, skin_region_map, k, img_num):
    fig, axes = plt.subplots(1, k + 2, figsize=(15, 5))
    fig.suptitle(f'Results for Image #{img_num}, k={k}', fontsize=16)

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    for cluster in range(k):
        ax = axes[cluster + 1]
        im = ax.imshow(frequencies[:, :, cluster], cmap='jet', interpolation='nearest')
        ax.set_title(f'Cluster {cluster + 1}')
        fig.colorbar(im, ax=ax)
        ax.axis('off')
    
    axes[k + 1].imshow(skin_region_map, cmap='gray', interpolation='nearest')
    axes[k + 1].set_title('Skin Region Map')
    axes[k + 1].axis('off')
    
    plt.show()

# Threshold the probabilities to detect skin regions
def threshold_skin_regions(frequencies, skin_cluster, threshold):
    h, w, k = frequencies.shape
    skin_region_map = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            if frequencies[i, j, skin_cluster] > threshold:
                skin_region_map[i, j] = 1  
    print(f"Thresholded skin regions with threshold={threshold}")
    return skin_region_map

# Step 0: Initialization
image_path1 = 'C:\\Users\\mathe\\Desktop\\CSCI-44800-segmentation-with-K-means-assignment\\HandImage1.jpg'
image_path2 = 'C:\\Users\\mathe\\Desktop\\CSCI-44800-segmentation-with-K-means-assignment\\HandImage2.jpg'
scale_percent = 50
X1 = load_and_resize_image(image_path1, scale_percent)
X2 = load_and_resize_image(image_path2, scale_percent)
k_values = [2, 3, 5]

# Implement repetitive k-means
for img_num, X in enumerate([X1, X2], start=1):
    print(f"Processing image #{img_num}")
    h, w, _ = X.shape  
    for k in k_values:
        print(f"Executing repetitive k-means for k={k}")
        M = np.zeros((h, w, 100))
        for numofexec in range(100):
            print(f"Repetition {numofexec + 1}")
            initial_centroids = choose_initial_centroids(X, k)
            labels, centroids = perform_kmeans(X, k, initial_centroids)
            labels = reassign_clusters(centroids, labels)
            M[:, :, numofexec] = labels
        
        frequencies = calculate_cluster_frequencies(M, h, w, k)
        skin_cluster = np.argmax(np.mean(frequencies, axis=(0, 1)))
        skin_region_map = threshold_skin_regions(frequencies, skin_cluster, threshold=0.50)
        display_results(X, frequencies, skin_region_map, k, img_num)
