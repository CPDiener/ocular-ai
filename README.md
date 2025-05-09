# ocular-ai

A deep learning model for diagnosing eye diseases using retinal images. This project uses a convolutional neural network (ResNet18) fine-tuned on a labeled dataset of retinal images to classify various common eye conditions.

---

## Development Setup (for reproducing training)

To replicate the training and evaluation process:

1. **Clone the repository**
   ```bash
   git clone https://github.com/CPDiener/ocular-ai.git
   cd ocular-ai
    ````

2. **Create and activate a virtual environment**
   (Python 3.12 recommended)

   ```bash
   python -m venv .venv
   source .venv/bin/activate     # On Windows: .venv\Scripts\activate
   ```

3. **Install PyTorch**
   Visit [PyTorch Get Started](https://pytorch.org/get-started/locally/) to install the appropriate version for your system (e.g., CUDA 11.8 or CPU-only).

4. **Install all project dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset Setup

1. **Download the dataset:**
   [Eye Disease Image Dataset (Augmented)](https://data.mendeley.com/datasets/s9bfhswzjb/1)

2. **Place the data into the following folder structure:**

   ```
   ocular-ai/
   └── data/
       ├── raw/
       └── processed/
   ```

3. **Move extracted image folders into `data/raw/`**

4. **Run the one-time dataset split** (from `main.ipynb` or as a script):

    ```python
    from src.data_prep import prepare_dataset
    prepare_dataset()
    ```

This will split the dataset into training and validation subsets inside `data/processed/`.

---

## Model Usage

To use the pretrained model without retraining:

1. **Load the model in a notebook or script:**

    ```python
    import os
    
    from src.model import build_model
    from src.utils import get_device, load_model
   
    device = get_device()
    num_classes = len(os.listdir("../data/processed/train"))
    model = build_model(num_classes)
    
    model = load_model(model, "../saved_models/eye_disease_cnn.pth", device)
    ```

2. **Evaluate the model on the validation set:**

    ```python
    import torch
    from src.data_prep import get_dataloaders
    from src.train import validate_one_epoch
    
    _, val_loader = get_dataloaders("../data/processed", batch_size=32)
    
    criterion = torch.nn.CrossEntropyLoss()
    val_loss, val_acc = validate_one_epoch(model, val_loader, device, criterion)
    
    print(f"Val Loss: {val_loss:.4f}")
    print(f"Val Accuracy: {val_acc:.2f}%")
    ```

3. **Generate metrics and visualizations:**

    ```python
    from src.utils import plot_confusion_matrix
    
    class_names = sorted(os.listdir("../data/processed/train"))
    plot_confusion_matrix(model, val_loader, device, class_names)
    get_classification_report(model, val_loader, device, class_names)
    ```

---

## Results Summary

The model was trained for 5 epochs using a fine-tuned ResNet18 architecture:

* **Final Training Accuracy:** \~92.86%
* **Final Validation Accuracy:** \~86.69%
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam (learning rate = 1e-4)

Model performance was evaluated using:

* Accuracy and loss tracking
* Confusion matrix visualization
* Classification report (precision, recall, F1-score per class)

---

## Repository Structure

```
ocular-ai/
├── data/
│   ├── raw/                # Downloaded images go here
│   └── processed/          # Auto-generated training/validation split
├── notebooks/
│   ├── main.ipynb          # Training and validation
│   └── evaluation.ipynb    # Model evaluation (post-training)
├── saved_models/
│   └── eye_disease_cnn.pth # Trained model
├── src/
│   ├── data_prep.py
│   ├── model.py
│   ├── train.py
│   └── utils.py
├── scripts/
│   └── project_setup.py    # Optional one-time setup script
├── requirements.txt
└── README.md
```

---

## Contributors

* Lead Developer: [Christian Diener](https://github.com/CPDiener)
* Teammates: [Ted Nyberg](https://github.com/TeddyNyberg), [Hennok Tilahun](https://github.com/Nyptes)

---