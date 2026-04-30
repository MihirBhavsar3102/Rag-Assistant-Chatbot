import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Project files and folders to be created
list_of_files = [
    ".env",  # for storing GROQ_API_KEY
    "requirements_free.txt",
    "data/.gitkeep",  # keeps the folder in Git even if empty
    "faiss_index/.gitkeep",  # placeholder to ensure index folder is created
    "src/__init__.py",
    "src/ingest.py",
    "src/chat_local.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
