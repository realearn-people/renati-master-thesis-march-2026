# Image Model Development
This work is based on **ProtoPNet**
## How to Run

### Training
python main.py

## Files
- main.py → Main training script  
- settings.py → Training settings (dataset paths, batch size, epochs, etc.)  
- train_and_test.py → Training & evaluation functions  
- model.py → ProtoPNet model  
- push.py / prune.py → Prototype operations  
- local_analysis.py → Analyze prototypes for one image  

### Analysis
python local_analysis.py

## My Modifications
### main.py
- Added **validation dataset loader** (`val_dataset`, `val_loader`).
- Implemented **best validation accuracy saving**:  
  - The model is saved only when validation accuracy improves (`xxxbestval.pth`).  
- Added **push snapshot saving**:  
  - During push epochs, model is saved as `xxxpush_bestval.pth` if validation improves.  
  - Always save a snapshot `xxxpush_snapshot.pth` when validation accuracy ≥ threshold.  
- This reduces the number of saved model files while keeping the important checkpoints.

### settings.py
- Added `val_dir` and `val_batch_size` for validation dataset.

### train_and_test.py
- Adjusted training loop and logging format (minor changes).

### local_analysis.py
- Tested with my dataset for prototype visualization.

### ravdess_video_to_frames.ipynb
-Extracts frames from RAVDESS videos and saves them into emotion-based folders.
-It can also be extended with face cropping and background removal to prepare images for training.

### test_loader.py
-Evaluate the trained model on the test set, plot the confusion matrix, and compute precision/recall/F1 scores  