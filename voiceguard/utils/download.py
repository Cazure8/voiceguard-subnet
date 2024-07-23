import requests
import tarfile
import os
from tqdm import tqdm
import spacy
import subprocess

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

    for model in models_to_download:
        download_spacy_model(model)