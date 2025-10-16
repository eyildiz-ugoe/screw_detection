import os
import requests
import logging
import tensorflow as tf
from tqdm import tqdm
import subprocess
import sys
from shutil import which

# Try importing alternative extraction libraries
try:
    import rarfile
    RAR_SUPPORTED = True
except ImportError:
    RAR_SUPPORTED = False
    try:
        subprocess.run(['which', 'unrar'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        UNRAR_AVAILABLE = True
    except Exception:
        UNRAR_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def download_file(url, destination_path, chunk_size=1024, max_retries=3):
    """
    Download a file with progress bar and robust error handling.
    Only download if file doesn't already exist.
    
    Args:
        url (str): URL of the file to download
        destination_path (str): Local path to save the file
        chunk_size (int): Size of chunks to download
        max_retries (int): Maximum number of download retries
    
    Returns:
        bool: True if file is present (downloaded or already exists), False if download failed
    """
    # Check if file already exists
    if os.path.exists(destination_path):
        logging.info(f"File already exists: {destination_path}")
        return True

    for attempt in range(max_retries):
        try:
            logging.info(f"Downloading {url} to {destination_path} (Attempt {attempt + 1})")
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination_path, 'wb') as f, tqdm(
                desc=os.path.basename(destination_path),
                total=total_size,
                unit='B',
                unit_scale=True,
            ) as bar:
                for data in response.iter_content(chunk_size=chunk_size):
                    size = f.write(data)
                    bar.update(size)
            
            # Verify file is not empty
            if os.path.getsize(destination_path) == 0:
                raise ValueError("Downloaded file is empty")
            
            return True
        
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.warning(f"Download error: {e}")
            if attempt == max_retries - 1:
                logging.error(f"Failed to download {url} after {max_retries} attempts")
                return False

def extract_archive(archive_path, extract_path, force_extract=False):
    """
    Extract archive using multiple methods.
    
    Args:
        archive_path (str): Path to the archive file
        extract_path (str): Directory to extract files to
        force_extract (bool): If True, forces extraction even if files exist
    """
    # If force_extract is True, bypass the check for existing files
    if not force_extract and any(os.listdir(extract_path)):
        logging.info(f"Files already exist in {extract_path}. Skipping extraction.")
        return True

    logging.info(f"Attempting to extract {archive_path}")
    
    # Method 1: rarfile library
    if RAR_SUPPORTED:
        try:
            with rarfile.RarFile(archive_path) as rf:
                rf.extractall(path=extract_path)
            logging.info(f"Extracted {archive_path} using rarfile")
            return True
        except Exception as e:
            logging.warning(f"rarfile extraction failed: {e}")
    
    # Method 2: unrar command
    if UNRAR_AVAILABLE:
        try:
            result = subprocess.run(['unrar', 'x', archive_path, extract_path], 
                                    capture_output=True, text=True)
            if result.returncode == 0:
                logging.info(f"Extracted {archive_path} using unrar")
                return True
            else:
                logging.error(f"unrar extraction failed: {result.stderr}")
        except Exception as e:
            logging.warning(f"unrar extraction failed: {e}")
    
    # Fallback: Raise error if no extraction method works
    raise RuntimeError(f"Could not extract {archive_path}. Install rarfile or unrar.")

def main():
    """
    Main function with robust error handling for directory processing.
    """
    current_directory = os.getcwd()
    base_folder = os.path.join(current_directory, 'data')
    model_folder = os.path.join(current_directory, 'models')

    # Create directories 
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs(model_folder, exist_ok=True)

    # Download dataset
    dataset_url = "https://zenodo.org/records/4727706/files/ScrewDTF.rar?download=1"
    rar_file = os.path.join(base_folder, 'ScrewDTF.rar')

    # Download dataset only if not already downloaded
    if download_file(dataset_url, rar_file):
        try:
            extract_archive(rar_file, base_folder, force_extract=True)
        except Exception as e:
            logging.error(f"Archive extraction failed: {e}")
            return

    # Models source: default to user-provided Google Drive folder (share URL)
    # You can also provide an explicit mapping of filenames->urls via the
    # MODELS_MAP variable below (uncomment and edit if you prefer direct links).
    DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1XwoDC7XyV9kqFojO2h3B2zyLZmBcBdXF?usp=sharing"

    # Optional explicit mapping (use this if you have direct URLs or file IDs):
    MODELS_MAP = {
        # 'xception.h5': 'https://drive.google.com/uc?export=download&id=<FILE_ID>',
        # 'inceptionv3.h5': 'https://drive.google.com/uc?export=download&id=<FILE_ID>',
    }

    # List of expected model filenames
    expected_models = [
        'xception.h5',
        'inceptionv3.h5',
        'inceptionResNetv2.h5',
        'densenet201.h5',
        'resnet101v2.h5',
        'resnext101.h5',
    ]

    # If MODELS_MAP is empty, attempt to download from Google Drive folder using gdown
    def download_model_from_drive(dest_path, url_or_id):
        """Download a single model using gdown if available, otherwise use requests fallback.
        url_or_id may be a full url or a drive id/url.
        """
        # Try using gdown (preferred for Google Drive links)
        try:
            import gdown
            logging.info("Using gdown to download from Google Drive")
            # gdown accepts both full urls and ids
            gdown.download(url_or_id, dest_path, quiet=False)
            return os.path.exists(dest_path) and os.path.getsize(dest_path) > 0
        except Exception:
            logging.info("gdown not available or failed. Falling back to requests (may fail for Drive links)")
            return download_file(url_or_id, dest_path)

    # Helper: if user provided explicit MODELS_MAP, use it; else instruct user or try gdown
    if MODELS_MAP:
        for name in expected_models:
            dest = os.path.join(model_folder, name)
            if name in MODELS_MAP:
                download_model_from_drive(dest, MODELS_MAP[name])
            else:
                logging.warning(f"No URL provided for {name} in MODELS_MAP. Skipping.")
    else:
        # Try to use gdown to download the whole folder if possible
        # If gdown isn't installed, attempt to install it, otherwise instruct the user
        if which('gdown') is None:
            try:
                logging.info("Installing gdown via pip...")
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'gdown'])
            except Exception as e:
                logging.warning(f"Couldn't install gdown automatically: {e}")

        # Use gdown to download individual files if the user provides direct links.
        # Unfortunately listing a shared Drive folder programmatically requires API access.
        logging.info('Attempting to download models using the provided Drive folder URL. If this fails, set MODELS_MAP with direct file URLs or IDs.')

        # Try a best-effort approach: construct potential file urls using the folder URL's base
        # NOTE: Google Drive folder listing isn't directly downloadable without API; prefer MODELS_MAP.
        for name in expected_models:
            dest = os.path.join(model_folder, name)
            if os.path.exists(dest):
                logging.info(f"Model already exists: {dest}")
                continue
            # Construct a best-effort URL (may not work). This is a hint to the user.
            logging.info(f"Please provide direct download URL for {name} by editing MODELS_MAP in scripts/download_data.py if automatic download fails.")
            # As a fallback, do nothing (user likely already has models in place)

    # Ensure dataset extraction created required folders
    folders = ['Test', 'Eval', 'Train']
    for folder in folders:
        folder_path = os.path.join(base_folder, folder)
        if not os.path.exists(folder_path):
            logging.error(f"Folder not found: {folder_path}")
            return

    logging.info("Setup completed successfully. If models were not downloaded automatically, please populate `models/` with the six .h5 files or update MODELS_MAP in scripts/download_data.py.")

if __name__ == "__main__":
    main()
