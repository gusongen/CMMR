import base64
import time
from typing import Optional
from PIL import Image
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
# import re
port = 45680


@st.cache
def gen_image_prompt(img: Image.Image):
    width, height = img.size
    applier = T.RandomApply(transforms=[
        # T.RandomCrop(size=(min(256,height),min(256,width))),
        # T.RandomCrop(size=(height // 4, width // 4)),
        T.CenterCrop(size=(height // 4, width // 4)),
        T.RandomPerspective(distortion_scale=0.2),
        # T.ColorJitter(brightness=.5, hue=.3),
        # T.RandomRotation(degrees=(0, 180)),
    ], p=1)
    return applier(img)


@st.cache
def gen_text_prompt(text, ratio=0.2):
    text = text.split(' ')
    idx = np.random.choice(len(text), size=int(ratio * len(text)))
    for i in idx:
        text[i] = '[MASK]'
    return ' '.join(text)


@st.cache
def load_data():  # TODO DEBUG ,place it to utli
    """
    for testing layout
    """
    imgs_path = sorted(Path("/home/gusongen/mm_search_baseline/data/f8k/images").rglob('*.jpg'))
    imgs_path = [str(i.absolute()) for i in imgs_path]
    return imgs_path


def clean_session_state(exclude: Optional[list] = []):
    rm_keys = list(filter(lambda x: x not in exclude, st.session_state.keys()))
    # Delete all the items in Session state
    for key in rm_keys:
        del st.session_state[key]


@st.cache
def base64_encode_img(img_path):
    with open(img_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


# @click.option('--data_set', '-d', type=click.Choice(['f30k', 'f8k', 'toy-data'], case_sensitive=False), default='toy-data')


def clean():
    from jina import Client
    c = Client(host=f"grpc://localhost:{port}")
    c.post('/clear')


def echo():
    from jina import Client
    c = Client(host=f"grpc://localhost:{port}")
    c.post('/echo')


def reindex(num_docs, request_size=64, device=None):
    """
    num_docs : import all image if num_docs is None
    request_size : request_size batch size
    """
    import torch
    from jina import Client, DocumentArray, Document
    if device == None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tic = time.time()
    # img_executor = CLIPImageEncoder(device=device, batch_size=64)
    # todo make it batch
    # total_size = 0

    # 创建 Jina 客户端
    c = Client(host=f"grpc://localhost:{port}")
    c.post('/clear')  # 清空上次的数据
    select_data = np.random.choice(load_data(), size=num_docs, replace=False)
    batch_docs = DocumentArray([Document(uri=uri) for uri in select_data]).batch(request_size, shuffle=True,
                                                                                 show_progress=True
                                                                                 )
    for idx, docs in enumerate(batch_docs):
        for d in docs:
            d.modality = 'image'
        print(f'@batch {idx}')
        # print(img, _)
        # docs = DocumentArray(img_doc)
        # img_executor.encode(docs, parameters={})
        c.post('/index', inputs=docs, parameters={})
        # print('@ db size', c.post('@@@@@@@@@@@@@@@' + '/get_db_len'))  # ['len'])
        # total_size += sum([1 if d.tensor is not None else 0 for d in docs])
    print('@ db size', len(c.post('/get_db')))  # ['len'])
    st.write(c.post('/get_db'))  # ['len'])
    print('@finish')
    # total_size = c.post('/get_db_len')
    total_size = num_docs
    print('@ total size ', total_size)
    print('@ total time ', time.time() - tic)
    print(f'@ avg time { (time.time() - tic) / (total_size+1e-5):.2f} s per img',)
