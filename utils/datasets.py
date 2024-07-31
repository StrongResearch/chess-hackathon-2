import torch
from torch.utils.data import Dataset
import numpy as np
import requests
import tarfile
import os
from itertools import accumulate
from h5py import File as h5pyFile
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup

LCZERO_TEST60_URL = 'https://storage.lczero.org/files/training_pgns/test60/'
PGN_CHARS = " #+-./0123456789:=BKLNOQRabcdefghx{}"

def scrape_tar_bz2_links(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to load page: {url}")
    soup = BeautifulSoup(response.content, 'html.parser')
    a_tags = soup.find_all('a')
    tar_bz2_links = [a['href'] for a in a_tags if 'href' in a.attrs and a['href'].endswith('.tar.bz2')]
    return tar_bz2_links

def download_tar_files(urls, dest_dir):

    # Path validation
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest_dir_contents = os.listdir(dest_dir)
    missing_file_urls = [u for u in urls if u.split('/')[-1] not in dest_dir_contents]

    for url in tqdm(missing_file_urls, desc=f"Downloading {len(missing_file_urls)} PGN files"):
        filename = url.split('/')[-1]
        response = requests.get(url)
        with open(os.path.join(dest_dir, filename), 'wb') as file:
            file.write(response.content)

def save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir):
    pgns_array = np.array(pgn_batch, dtype='S')
    hdf_name = f'pgnHDF{hdf_count}.h5'
    hdf_path = os.path.join(dest_dir, hdf_name)
    hf = h5pyFile(hdf_path, 'w')
    hf.create_dataset("pgn_movetext", data=pgns_array, compression='gzip', compression_opts=9)
    hf.close()
    return hdf_name

def compile_tars_to_hdfs(source_dir, dest_dir, batch_size=1_000_000):

    # Path validation
    assert os.path.exists(source_dir), "ERROR: source_dir not found."
    assert not os.path.exists(dest_dir), "ERROR: dest_dir present, please delete first."
    Path(dest_dir).mkdir(parents=True, exist_ok=False)
    tar_files = [f for f in os.listdir(source_dir) if f.endswith('.tar.bz2')]
    
    # Variables
    pgn_batch = []
    hdf_count = 0
    hdf_sizes = []
    hdf_names = []

    for tfile in tqdm(tar_files, desc = "Processing tars into HDFs"):
        with tarfile.open(os.path.join(source_dir, tfile), "r:bz2") as tar:
            pgn_files = [file.name for file in tar.getmembers() if file.name.endswith(".pgn")]

            for pgnfile in pgn_files:
                pgn = tar.extractfile(tar.getmember(pgnfile)).read()
                try:
                    pgn = pgn.decode().strip()
                    assert set(pgn).issubset(set(PGN_CHARS))
                    pgn_batch.append(pgn)

                    if len(pgn_batch) == batch_size:
                        hdf_name = save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir)
                        hdf_sizes.append(batch_size)
                        hdf_names.append(hdf_name)
                        hdf_count += 1
                        pgn_batch = []
                except:
                    print(f"FAILED: {pgn}")
                    continue

    # Store leftover pgns in new hdf
    hdf_name = save_pgn_batch_to_hdf(pgn_batch, hdf_count, dest_dir)
    hdf_sizes.append(len(pgn_batch))
    hdf_names.append(hdf_name)
    hdf_count += 1
    pgn_batch = []

    inventory_path = os.path.join(dest_dir, "inventory.txt")
    with open(inventory_path, "w") as file:
        file.write(f"Total pgns: {sum(hdf_sizes):,}\n")
        for size, name in zip(hdf_sizes, hdf_names):
            file.write(f"{size} {name}\n")

class PGN_HDF_Dataset(Dataset):
    def __init__(self, source_dir=None):
        self.source_dir = source_dir
        self.pgn_chars = PGN_CHARS

        if self.source_dir:
            with open(os.path.join(self.source_dir, "inventory.txt"), "r") as file:
                self.inventory = file.readlines()

            sizes, self.filenames = zip(*[i.split() for i in self.inventory[1:]])
            self.sizes = [int(s) for s in sizes]
            self.len = sum(self.sizes)
            self.breaks = np.array(list(accumulate(self.sizes)))

    def encode(self, pgn):
        return [self.pgn_chars.index(c) for c in pgn]
    
    def decode(self, tokens):
        return [self.pgn_chars[t] for t in tokens]

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        hdf_idx = (self.breaks > idx).argmax().item()
        pgn_idx = idx - sum(self.sizes[:hdf_idx])
        hdf_path = os.path.join(self.source_dir, self.filenames[hdf_idx])
        with h5pyFile(hdf_path, 'r') as hf:
            pgn = hf["pgn_movetext"][pgn_idx].decode('utf-8')
        return pgn