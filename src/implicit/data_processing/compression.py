from fire import Fire
import pandas as pd
import os
import glob

def compress(file : str, delete : bool = False):
    if file.endswith('.jsonl'):
        new_file = file + '.gz'
        frame = pd.read_json(file, lines=True, orient='records')
        frame.to_json(new_file, orient='records', lines=True, compression='gzip')
        if delete:
            os.remove(file)
    else:
        pass

def decompress(file : str, delete : bool = False):
    if file.endswith('.jsonl.gz'):
        new_file = file[:-3]
        frame = pd.read_json(file, lines=True, orient='records', compression='gzip')
        frame.to_json(new_file, orient='records', lines=True)
        if delete:
            os.remove(file)
    else:
        pass

def main(directory : str, mode : str = 'decompress', delete : bool = False, file_type : str = 'jsonl', file : str = None):
    if mode == 'compress':
        files = glob.glob(directory + f'/*.{file_type}') if file is None else [directory + '/' + file]
        for file in files:
            compress(file, delete)
    elif mode == 'decompress':
        files = glob.glob(directory + f'/*.{file_type}.gz') if file is None else [directory + '/' + file]
        for file in files:
            decompress(file, delete)
    else:
        raise ValueError("Mode must be either 'compress' or 'decompress'")

if __name__ == '__main__':
    Fire(main)