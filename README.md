# Introduction
Self supervised representational learning is an effective paradigm for multiple downstream tasks.  
Contrastive learning between images and random augmentations is a strong method for such representation learning over visual data.  
The key challenge in contrastive learning is preventing collapse to the trivial solution of an identical encoding for all samples during training.  
VICReg accomplishes this by introducing both variance and covariance losses, as measured over the mini-batch (see paper for details).  
This is an educational project to reproduce VICReg image representation learning paper and evaluate on CIFAR10 dataset.  
Paper: Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning.
arXiv preprint arXiv:2105.04906, 2021  


# Overview of tasks

- Train a VICReg image encoder on the CIFAR10 dataset (without labels)
- Evaluate classification accuracy with linear probing (by training a linear classifier on top of the frozen encoder)
- Evaluate image retrieval using KNN on the encodings
- Evaluate clustering over the encodings

# Technical details

- Image augmentations include: cropping, horizontal flip, color-jitter, gray-scaling and blurring.
- Encoder uses resnet18 architecture for backbone
- Hyperparameters follow the paper (coefficients for invariance, variance and covariance losses: lambda=25, mu=25, nu=1)
- KNN accomplished using FAISS for good efficiency
- Clustering accomplished with vanilla sklearn K-Means

# Results - linear probing

Accuracy of linear probe over trained VICReg encoder: 0.6033  
Baselines:
- Linear probe over an ablated VICReg encoder trained without the variance loss. Accuracy: 0.1437.
- Random baseline is 0.1 (10 classes).
  
Conclusion: VICReg encoding dramatically improves classification accuracy, even though it is trained in self-supervised fashion without labels.

# Results - image retrieval

Data is encoded with the trained VICReg model, a random image is selected from each class, and then 5 nearest neighbors are retrieved (in l2 distance).

![image](https://github.com/ReserveJudgement/VICReg-Reproduction/assets/150562945/2fb2535f-86ed-4eb1-bdfb-c18cf918ffe3)

Conclusion: VICReg encoding manages to embed similar images close by in the embedding space

# Results - clustering

Again data is encoded with trained VICReg model, images are clustered into 10 groups, then dimensionality is reduced to 2D using TSNE and visualized.  
In graph on left, colors represent different clusters. On right, colors represent original class labels. Black points represent cluster centroids in both.

![image](https://github.com/ReserveJudgement/VICReg-Reproduction/assets/150562945/cd0fb962-37bd-4995-bce3-15cccd85a7cd)

Conclusion: VICReg manages to roughly separate out classes without any labels during training.
