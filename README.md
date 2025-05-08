# ocular-ai

## Setup Instructions

1. Clone the repo:
```git clone https://github.com/CPDiener/ocular-ai.git```
2. Create a virtual environment (Python 3.12 recommended):
```.venv\Scripts\activate```
3. Download PyTorch from [PyTorch Get Started.](https://pytorch.org/get-started/locally/)
4. Install remaining dependencies:
```pip install -r requirements.txt```


## Data Import

1. Download "Augmented Dataset.zip" data from [Eye Disease Image Dataset.](https://data.mendeley.com/datasets/s9bfhswzjb/1)
2. Create a `data/` directory in the project root, then create `raw/` and `processed/` subdirectories within it as so:
```
ocular-ai/
├── data/
│   └── raw/
│   └── processed/
├── ...
```
3. Extract and move the contents of the download into the `data/raw` directory.
4. Run the first cell of `main.ipynb` which is as follows:
```
from src.data_prep import prepare_dataset

# One-time setup (idempotent)
prepare_dataset()
```
5. Once this cell is run, the data in `data/raw` will be split into training and validation sets within `data/processed`.