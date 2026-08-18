import os
from dotenv import load_dotenv

load_dotenv()

from kaggle.api.kaggle_api_extended import KaggleApi

def download_bbc_dataset(output_dir: str = "./data"):
    api = KaggleApi()
    api.authenticate() 