
import kagglehub
from pathlib import Path

# Folder where the dataset will be stored
DATASET_DIR = Path(r"V:\Datasets\Liver")

# Create the folder if needed
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# Download the actual dataset files
path = kagglehub.dataset_download(
    "andrewmvd/lits-png",
    output_dir=str(DATASET_DIR)
)

print("Dataset downloaded to:", path)

# Display downloaded files
for file in DATASET_DIR.rglob("*"):
    if file.is_file():
        print(file.relative_to(DATASET_DIR))
