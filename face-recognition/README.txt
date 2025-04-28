Face Recognition Project
========================

Overview
--------
This project implements a face recognition system using the `facenet-pytorch` library and the `InceptionResnetV1` model pre-trained on the VGGFace2 dataset. The system includes two modes:
1. **Verification Mode**: Computes the Cumulative Match Characteristic (CMC) curve to evaluate recognition performance.
2. **Single Query Matching**: Displays the query image and its top 3 matches from the gallery.

Sources
-------

### Code and Libraries
1. **PyTorch**:  
   - Website: [https://pytorch.org/](https://pytorch.org/)  
   - Used for deep learning operations and model handling.

2. **Facenet-PyTorch**:  
   - GitHub Repository: [https://github.com/timesler/facenet-pytorch](https://github.com/timesler/facenet-pytorch)  
   - Provides the pre-trained `InceptionResnetV1` model used for feature extraction.

3. **Scikit-learn**:  
   - Website: [https://scikit-learn.org/](https://scikit-learn.org/)  
   - Used for computing pairwise distances and evaluating recognition performance.

4. **Matplotlib**:  
   - Website: [https://matplotlib.org/](https://matplotlib.org/)  
   - Used for plotting the CMC curve and displaying images.

5. **Torchvision**:  
   - Website: [https://pytorch.org/vision/stable/index.html](https://pytorch.org/vision/stable/index.html)  
   - Used for dataset handling and image transformations.

6. **TQDM**:  
   - GitHub Repository: [https://github.com/tqdm/tqdm](https://github.com/tqdm/tqdm)  
   - Used for displaying progress bars during feature extraction.

### Dataset
1. **LFW (Labeled Faces in the Wild)**:  
   - Website: [http://vis-www.cs.umass.edu/lfw/](http://vis-www.cs.umass.edu/lfw/)  
   - Used as the dataset for face recognition. The images are stored in the `lfw-deepfunneled` folder.

### Papers
1. **FaceNet: A Unified Embedding for Face Recognition and Clustering**  
   - Authors: Florian Schroff, Dmitry Kalenichenko, James Philbin  
   - Paper: [https://arxiv.org/abs/1503.03832](https://arxiv.org/abs/1503.03832)  
   - Description: The foundational paper for the FaceNet model, which inspired the use of embeddings for face recognition.

2. **VGGFace2: A Dataset for Recognising Faces Across Pose and Age**  
   - Authors: Qiong Cao, Li Shen, Weidi Xie, Omkar M. Parkhi, Andrew Zisserman  
   - Paper: [https://arxiv.org/abs/1710.08092](https://arxiv.org/abs/1710.08092)  
   - Description: The dataset used to pre-train the `InceptionResnetV1` model.

How to Run
----------

1. Install the required libraries:
   ```bash
   pip install torch torchvision facenet-pytorch scikit-learn matplotlib tqdm
2. Place the dataset in the 'lfw-deepfunneled' folder.
3. Run the script:
   '''bash
   python main.py

Notes
-----

- Ensure that the dataset is properly structured and accessible.
- The 'max_images' parameter in 'get_image_paths' can be adjusted to limit the number of images for testing.