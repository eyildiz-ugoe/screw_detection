import os
import requests
import logging
import tensorflow as tf
from tqdm import tqdm
import subprocess

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

    # Download dataset and model
    dataset_url = "https://zenodo.org/records/4727706/files/ScrewDTF.rar?download=1"
    model_url = "https://zenodo.org/records/10474868/files/xception.h5?download=1"

    rar_file = os.path.join(base_folder, 'ScrewDTF.rar')
    model_file = os.path.join(model_folder, 'xception.h5')
    
    # Download dataset only if not already downloaded
    if download_file(dataset_url, rar_file):
        try:
            extract_archive(rar_file, base_folder, force_extract=True)
        except Exception as e:
            logging.error(f"Archive extraction failed: {e}")
            return

    # Download model only if not already downloaded
    download_file(model_url, model_file)

    # Ensure dataset extraction created required folders
    folders = ['Test', 'Eval', 'Train']
    for folder in folders:
        folder_path = os.path.join(base_folder, 'ScrewDTF', folder)
        if not os.path.exists(folder_path):
            logging.error(f"Folder not found: {folder_path}")
            return

    logging.info("Setup completed successfully.")

if __name__ == "__main__":
    main()
