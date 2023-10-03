from kaggle.api.kaggle_api_extended import KaggleApi
from kaggle.rest import ApiException
import os
import subprocess
import sys
import ast
import argparse
import zipfile

api = KaggleApi()
api.authenticate()

parser = argparse.ArgumentParser(description='Download a notebook and its datasets')
parser.add_argument('--nb_url', type=str, help='The url of the notebook. It should have the form: https://www.kaggle.com/code/user/slug.', required=True)
parser.add_argument('--skip_private', action='store_true', help="If there are private datasets, don't fail and skip them.")
args = parser.parse_args()

print("THIS ONLY WORKS FOR DATASETS NOT COMPETITIONS.")

def get_user_and_slug(url):
  kernel_slug = url.split('/')[-1]
  username = url.split('/')[-2]
  return username, kernel_slug

def download_nb(download_prefix):
  username, kernel_slug = get_user_and_slug(args.nb_url)

  print(f"--- https://www.kaggle.com/code/{username}/{kernel_slug} ---")

  notebook_root = f"{download_prefix}/{username}/{kernel_slug}"
  # Move the .ipynb here.
  os.makedirs(f"{notebook_root}/src/", exist_ok=True)
  src_dir = f"{notebook_root}/src"
  api.kernels_pull(f"{username}/{kernel_slug}", path=src_dir)
  _ = subprocess.run(["mv", f"{src_dir}/{kernel_slug}.ipynb", f"{src_dir}/bench.ipynb"])

# TODO: There are zips that have other zips inside and you have to unzip
# them. For example: https://www.kaggle.com/code/levitoh/question-similarity/data
# Funnily, I got an error even when I tried to run it on Kaggle.
def unzip_file(zip_filepath, unzip_to):
  zipf = zipfile.ZipFile(zip_filepath, 'r')
  assert zipf is not None
  try:
    zipf.extractall(path=unzip_to)
  except Exception as e:
    zip_file = zip_filepath.split('/')[-1]
    print(f"Unzipping {zip_file} failed.", f"*** EXCEPTION ***\n {e} \n *** END OF EXCEPTION ***")
    return False

  os.remove(zip_filepath)
  return True

def download_dataset(download_prefix):
  username, kernel_slug = get_user_and_slug(args.nb_url)
  kwargs = {"_request_timeout": (1.0, 2.0)}
  data = api.kernel_pull(user_name=username, kernel_slug=kernel_slug, **kwargs)
  metadata = data['metadata']
  dataset_sources = metadata['datasetDataSources']
  has_private_dt = "" in dataset_sources
  if has_private_dt and not args.skip_private:
    print("FAILED: The notebook uses a private dataset. Use --skip_private if it's ok to skip it.")
    sys.exit(1)
  dataset_sources = [s for s in dataset_sources if s != ""]
  
  download_to = f"{download_prefix}/{username}/{kernel_slug}/input"

  for source in dataset_sources:
    splits = source.split('/')
    assert len(splits) == 2
    dataset_user = splits[0]
    dataset_name = splits[1]
    assert dataset_user is not None
    assert dataset_name is not None

    download_name = f"{dataset_user}/{dataset_name}"
    try:
      api.dataset_download_files(download_name, path=download_to, force=True)
    except ApiException as e:
      body_dict = ast.literal_eval(e.body.decode("utf-8"))
      print("--- FAILED ---")
      print(body_dict['message'])
      sys.exit(1)
  # END OF LOOP

  zip_name = dataset_name
  succ = unzip_file(f"{download_to}/{zip_name}.zip", f"{download_to}/{zip_name}")
  if not succ:
    sys.exit(1)

download_prefix = "/home/stef/cs598ms-nbs/notebooks"
download_nb(download_prefix)
download_dataset(download_prefix)

print('SUCCESS')