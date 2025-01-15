import requests
import tarfile
import os
from tqdm import tqdm
import spacy
from pathlib import Path


def download_file(url, dest_folder):
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    local_filename = url.split('/')[-1]
    full_local_path = os.path.join(dest_folder, local_filename)

    # Check if the file already exists
    if os.path.exists(full_local_path):
        print(f"File {local_filename} already exists, skipping download.")
        return full_local_path
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_length = int(r.headers.get('content-length'))
            with open(full_local_path, 'wb') as f, tqdm(
                desc=local_filename,
                total=total_length,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in r.iter_content(chunk_size=1024):
                    size = f.write(chunk)
                    bar.update(size)
        return full_local_path
    except requests.RequestException as e:
            print(f"Failed to download {local_filename}: {e}")
            return None
    
def extract_tarfile(file_path, dest_folder):
    if not file_path.endswith((".tar.gz", ".tar")):
        print(f"File {file_path} is not a tar archive.")
        return

    # The name of the folder we expect to see after extraction
    expected_extracted_folder = os.path.join(dest_folder, 'LibriSpeech', os.path.basename(file_path).replace(".tar.gz", ""))

    if os.path.exists(expected_extracted_folder):
        print(f"Archive {file_path} already extracted in {expected_extracted_folder}.")
        return
    
    with tarfile.open(file_path, "r:*") as tar:
        tar.extractall(path=dest_folder)
        print(f"Extracted {file_path} to {dest_folder}")


def download_librispeech_subset(subset, base_url, dest_folder):
    subset_url = base_url + f"{subset}.tar.gz"
    tar_file = download_file(subset_url, dest_folder)
    extract_tarfile(tar_file, dest_folder)
    print(f"Downloaded and extracted {subset}")

def download_entire_librispeech():
    subsets = [
        'train-clean-100', 'train-clean-360', 'train-other-500',
        'dev-clean', 'dev-other', 'test-clean', 'test-other'
    ]
    base_url = "http://www.openslr.org/resources/12/"
    dest_folder = 'librispeech_dataset'

    for subset in subsets:
        download_librispeech_subset(subset, base_url, dest_folder)

def download_spacy_model(model_name):
    try:
        # Attempt to load the model first to check if it's installed
        spacy.load(model_name)
        print(f"Model {model_name} is already installed.")
    except IOError:
        # Model could not be loaded, which likely means it's not installed
        print(f"Model {model_name} not found, attempting to download...")
        try:
            spacy.cli.download(model_name)
            print(f"Successfully downloaded {model_name}")
        except Exception as e:
            print(f"Failed to download {model_name}: {str(e)}")
    except Exception as e:
        print(f"Unexpected error when checking model {model_name}: {str(e)}")

def download_common_voice(dataset_type: str) -> None:
    """
    Download Common Voice dataset based on the specified type.
    
    Args:
        dataset_type (str): Type of dataset to download - "test" or "whole"
    
    Raises:
        ValueError: If dataset_type is not "test" or "whole"
        requests.RequestException: If there's an error during download
    """
    
    # Validate input
    if dataset_type not in ["test", "whole"]:
        raise ValueError('dataset_type must be either "test" or "whole"')
    
    # Configuration values
    dataset = "Common-Voice-Corpus-19.0"
    vps_ip_port = "74.50.66.114:8000"
    
    endpoint = "testsets" if dataset_type == "test" else "wholesets"
    url = f"http://{vps_ip_port}/{dataset}/{endpoint}"
    
    # Create datasets directory if it doesn't exist
    datasets_dir = Path("datasets")
    datasets_dir.mkdir(exist_ok=True)
    
    # Define output filename
    output_filename = f"{dataset}_{endpoint}.tar.gz"
    output_path = datasets_dir / output_filename
    
    # Create extraction directory name (without .tar.gz extension)
    extraction_dir_name = output_filename.rsplit('.tar.gz', 1)[0]
    extraction_path = datasets_dir / extraction_dir_name
    
    print(f"Downloading dataset...")
    print(f"Saving to: {output_path}")
    print(f"Will extract to: {extraction_path}")
    
    try:
        # Make the request
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))
        
        # Download with tqdm progress bar
        with open(output_path, 'wb') as f, tqdm(
            desc=output_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    progress_bar.update(size)
        
        print("\nDownload completed!")
        print(f"File saved to: {output_path}")
        
        # Extract the tar.gz file to its own directory
        if output_path.exists() and str(output_path).endswith('.tar.gz'):
            print(f"Extracting {output_path} to {extraction_path}...")
            
            # Create the extraction directory
            extraction_path.mkdir(exist_ok=True)
            
            # Extract to the specific directory
            with tarfile.open(output_path, 'r:gz') as tar:
                tar.extractall(path=extraction_path)
            print(f"Extraction completed to: {extraction_path}")
            
    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        if output_path.exists():
            output_path.unlink()  # Remove partial download
        raise
    except tarfile.TarError as e:
        print(f"Error extracting file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if output_path.exists():
            output_path.unlink()
        raise

# Download pretrained MOSNet
def download_pretrained_model():
    """
    Downloads the pretrained model and saves it in a 'pretrained' folder.
    """
    model_url = "https://github.com/lochenchou/MOSNet/raw/master/pre_trained/cnn_blstm.h5"
    save_directory = "pretrained"
    save_path = Path(save_directory) / "cnn_blstm.h5"

    # Download the pretrained model
    download_file(model_url, save_directory)

def download_deepfake_model() -> None:
    """
    Download a deepfake detection model and extract it into the `pretrained/model_safetensors` directory.

    Raises:
        requests.RequestException: If there's an error during download
        tarfile.TarError: If there's an error during extraction
    """
    # Configuration values
    vps_ip_port = "74.50.66.114:8000"
    endpoint = "deepfake-model"
    url = f"http://{vps_ip_port}/{endpoint}"

    # Directories and filenames
    pretrained_dir = Path("pretrained")
    target_dir = pretrained_dir / "model_safetensors"
    pretrained_dir.mkdir(exist_ok=True)  # Create `pretrained` directory if it doesn't exist
    target_dir.mkdir(exist_ok=True)      # Create `model_safetensors` directory if it doesn't exist

    output_filename = f"deepfake_model.tar.gz"
    temp_output_path = Path(output_filename)  # Temporary file for download

    print(f"Downloading deepfake detection model...")
    print(f"Saving to: {temp_output_path}")
    print(f"Will extract to: {target_dir}")

    try:
        # Make the request
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Get the total file size
        total_size = int(response.headers.get('content-length', 0))

        # Download with tqdm progress bar
        with open(temp_output_path, 'wb') as f, tqdm(
            desc=output_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as progress_bar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    size = f.write(chunk)
                    progress_bar.update(size)

        print("\nDownload completed!")
        print(f"File saved to: {temp_output_path}")

        # Extract the tar.gz file to the target directory
        if temp_output_path.exists() and str(temp_output_path).endswith('.tar.gz'):
            print(f"Extracting {temp_output_path} to {target_dir}...")

            # Extract to the specific directory
            with tarfile.open(temp_output_path, 'r:gz') as tar:
                for member in tar.getmembers():
                    # Adjust extraction path to place files directly in `model_safetensors`
                    member_path = Path(member.name)
                    if member_path.parts[0] == "model_dir":
                        member.name = str(Path(*member_path.parts[1:]))
                    tar.extract(member, path=target_dir)
            print(f"Extraction completed to: {target_dir}")

        # Remove the downloaded tar.gz file
        temp_output_path.unlink()

    except requests.RequestException as e:
        print(f"Error downloading file: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()  # Remove partial download
        raise
    except tarfile.TarError as e:
        print(f"Error extracting file: {e}")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if temp_output_path.exists():
            temp_output_path.unlink()
        raise


models_to_download = [
    'en_core_web_lg', 'de_core_news_lg', 'fr_core_news_lg',
    'es_core_news_lg', 'pt_core_news_lg', 'it_core_news_lg',
    'nl_core_news_lg', 'el_core_news_md', 'nb_core_news_md',
    'da_core_news_md', 'ja_core_news_md', 'zh_core_web_md',
    'ru_core_news_md', 'pl_core_news_md', 'ca_core_news_md',
    'uk_core_news_md'
]

if __name__ == "__main__":
    # download_entire_librispeech()
    download_common_voice('test')
    download_pretrained_model()
    download_deepfake_model()
    for model in models_to_download:
        download_spacy_model(model)