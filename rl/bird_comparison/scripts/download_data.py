import os
import argparse
import owncloud
import tarfile


def download_and_untargz(oc: owncloud.Client, filename: str, target_dir: str):

    os.makedirs(target_dir, exist_ok=True)

    # download
    print(f'downloading {filename}...')
    target_file = os.path.join(target_dir, filename)
    oc.get_file(filename, local_file=target_file)

    # untar.gz
    print(f'untargzing {filename}...')
    file = tarfile.open(target_file)
    file.extractall(target_dir)
    file.close()

    # delete tar.gz
    print(f'deleting {filename}...')
    os.remove(target_file)


parser = argparse.ArgumentParser(
    description='This tool downloads data for bird comparison')
parser.add_argument('password',
                    help='Password for the NextCloud shared folder')

args = parser.parse_args()
password = args.password

# create the client
url = 'https://nc.elte.hu/s/KrCoC4GAQGkdzo9'
oc = owncloud.Client.from_public_link(url, folder_password=password)

# download
target_base_dir = 'data/bird_comparison'

download_and_untargz(oc=oc,
                     filename='input_stork_data.tar.gz',
                     target_dir=os.path.join(target_base_dir,
                                             'input_stork_data'))

download_and_untargz(oc=oc,
                     filename='wing_loadings_20250424.tar.gz',
                     target_dir=os.path.join(target_base_dir,
                                             'wing_loading_stork'))

download_and_untargz(oc=oc,
                     filename='decomposition_data_20251220.tar.gz',
                     target_dir=os.path.join(target_base_dir,
                                             'decomposed_extrapolated_data'))

download_and_untargz(oc=oc,
                     filename='stork_meta.tar.gz',
                     target_dir=os.path.join(target_base_dir, 'stork_meta'))
