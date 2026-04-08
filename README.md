# renati-master-thesis-march-2026

This repository includes the experiments and results of Renati-san (Master's student, ReaLearn) expected for graduation in March 2026



\# Self-Explainable Recognition Models



This repository contains my research code for self-explainable models in image and video recognition.



The project focuses on extending inherently interpretable architectures to video-level inference.



---



\## Included Frameworks



\### 1. ProtoPNet

Prototype-based neural network for interpretable image classification.



\### 2. PatchSAE

Sparse Autoencoder-based explainable model built on CLIP features.



\- Image-level inference (frame-based)

\- Video-level inference (frame sampling + temporal aggregation)



The video module extends the original PatchSAE implementation to support video classification.



---



\## Directory Structure



ProtoPNet/

PatchSAE/

&nbsp; ├── image/   # image-level inference

&nbsp; ├── video/   # video-level inference



---



\## Notes



\- The core PatchSAE implementation follows the original repository.

\- Video-level inference and temporal aggregation were implemented for this project.

\- This repository is organized for research and thesis purposes.



---



\## Reference



PatchSAE original repository:

(Add original link here)



CLIP model:

OpenAI CLIP ViT-B/16



