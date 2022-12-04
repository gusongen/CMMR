# 导入需要的包
import streamlit as st
from jina import Client, DocumentArray, Document
import json
import os
import time
import uuid
import pandas as pd
from pathlib import Path
import base64
import numpy as np
from PIL import Image


# GRPC 监听的端口
port = 45680
# 创建 Jina 客户端
c = Client(host=f"grpc://localhost:{port}")

# 设置标签栏
st.set_page_config(page_title="CMMR", page_icon="🔍")
# 设置标题
st.title('Welcome to CMMR!')
with st.expander("Brief Introduction"):
    st.markdown("""
        **C**Lip **M**ulti **M**odal **R**etrievel is a neural network retrieval system based on the [CLIP](https://github.com/openai/CLIP).
    """)
st.markdown(" <style> .img-wrapper { column-count: 3;  column-gap: 10px; counter-reset: count; width: 100% margin: 0 auto; } .img-wrapper>li { position: relative; margin-bottom: 10px; list-style-type: none; padding:0; margin-left:0; } .img-wrapper>li>img { width: 100%; height: auto; vertical-align: middle; } .img-wrapper>li::after { counter-increment: count; content: counter(count); width: 2em; height: 2em; background-color: rgba(0, 0, 0, 0.9); color: #ffffff; line-height: 2em; text-align: center; position: absolute; font-size: 1em; z-index: 2; left: 0; top: 0; } </style>", unsafe_allow_html=True)


uid = uuid.uuid1()

# 模态选择
modal_select = st.radio(
    "Select the modal of your query.",
    ('Text', 'Image'), horizontal=True)

if modal_select == 'Text':
    # 文本输入框
    query = st.text_input(
        "Query text", placeholder="please input the query description", help='The description of your query')
else:
    query = st.file_uploader("Query image", help='The image of your query', type=['png', 'jpg', 'jpeg'])


# top k 输入框
topn_value = st.text_input(
    "Top N", placeholder="please input an integer", help='The number of results. By default, n equals 1')


# 与后端交互部分
# @st.cache
def search_clip(uid, query, modal, topn_value: int):
    if modal == 'Text':
        query = DocumentArray([Document(text=query, modality='text')])
        resp = c.post('/search', inputs=query, parameters={'relative_score': True, "maxCount": topn_value})
    elif modal == 'Image':
        query = DocumentArray([Document(tensor=np.array(Image.open(query)), modality='image')])
        resp = c.post('/search', inputs=query, parameters={'add_dummy_unknow_prompt': True, 'relative_score': False,
                      'thod': 0., "maxCount": topn_value})  # TODO write it into indexer, judge by modality pair
    else:
        raise NotImplementedError
    # print(resp)
    # st.write(resp[0].matches.to_dict())
    resp = [i .to_dict()['tags']["uri"] for i in resp[0].matches]  # only one text promt
    return resp


@st.cache
def load_data():
    """
    for testing layout
    """
    imgs_path = sorted(Path("/home/gusongen/mm_search_baseline/data/f8k/images").rglob('*.jpg'))
    imgs_path = [str(i.absolute()) for i in imgs_path]
    return imgs_path


def base64_encode_img(img_path):
    with open(img_path, "rb") as image:
        encoded = base64.b64encode(image.read()).decode()
    return f"data:image/jpeg;base64,{encoded}"


imgs_path = load_data()

search_button = st.button("Search")

# search
if search_button:  # 判断是否点击搜索按钮
    if query is None or modal_select == 'Text' and query.strip() == "":   # 判断是否输入查询文本
        st.warning('Please input the query first!')
    else:
        if topn_value == None or topn_value == "" or not topn_value.isnumeric():  # 如果没有输入 top k 则默认设置为1
            topn_value = 1
        topn_value = int(topn_value)
        success_placeholder = st.empty()
        with st.spinner("Processing..."):
            image_urls = search_clip(uid, query, modal_select, topn_value)
            if len(image_urls) == 0:
                st.warning('Nothing similar found!', icon="⚠️")
            else:
                child_html = ['<li><img src="{}" ></img></li>\n'.format(base64_encode_img(url)) for url in image_urls]
                child_html = ''.join(child_html)
                html = f"""<ul class="img-wrapper">{child_html}</ul>"""
                st.markdown(html, unsafe_allow_html=True)
                success_placeholder.success("Done!")


# TODO fix image encoder and text encoder shape mismatch and tesnor/np mismatch
# TODO 干净版本的commit
# TODO middle size dataset
