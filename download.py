import os
from requests import get  # to make GET request
import wget


# def download(url, file_name):
#     # open in binary mode
#     with open(file_name, "wb") as file:
#         # get request
#         response = get(url)
#         # write to file
#         file.write(response.content)


url = "https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/download/train.7z"

# request

folder = 'new_data'
if not os.path.exists(folder):
    os.makedirs(folder)

# download(url, os.path.join(foleder, 'train.7z'))
wget.download(url, os.path.join(folder, 'train.7z'))
