# 导入需要的包
import streamlit as st
from jina import Client, DocumentArray, Document
import uuid
import pandas as pd
from pathlib import Path
import base64
import numpy as np
from PIL import Image
from loguru import logger
import sys
sys.path.append('..')
sys.path.append('.')
from util import *

TOTAL_EXP = 10  # TODO EXP 10
DEFAULT_TOPK = 5  # (feedback_rate:0.5 * DEFAULT_TOPK ) images per captiong
EXP_ID = 'exp1'
# GRPC 监听的端口
port = 45680
# 创建 Jina 客户端
c = Client(host=f"grpc://localhost:{port}")


def random_choice_text():
    if 'caption_list' not in st.session_state:
        return
    df = st.session_state['caption_list']
    choosen_caption = np.random.choice(df['caption'])
    return choosen_caption


def load_next_exp():
    logger.info(f'@{EXP_ID}current caption feedback done')
    st.session_state['pos_sample'] = 0
    st.session_state['neg_sample'] = 0
    if st.session_state.get(f'{EXP_ID}_cnt', 0) >= TOTAL_EXP:
        if st.session_state.get(f'{EXP_ID}_cnt', 0) == TOTAL_EXP:
            st.session_state[f'{EXP_ID}_cnt'] = st.session_state.get(f'{EXP_ID}_cnt', 0) + 1
            # st.session_state['current_caption'] = ''
            clean_session_state(exclude=[f'{EXP_ID}_cnt', 'log_id'])
        return
    st.session_state['current_caption'] = random_choice_text()
    st.session_state[f'{EXP_ID}_cnt'] = st.session_state.get(f'{EXP_ID}_cnt', 0) + 1


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


def user_feedback(img_url, tag: bool):
    if 'current_caption' not in st.session_state:
        return
    caption = st.session_state['current_caption']
    img_id = img_url.split('/')[-1]
    sample = f'{img_id},{caption},{int(tag)}'
    # print(sampel)
    logger.info(f'@{EXP_ID}@add_user_sample::{sample}')
    if tag:
        st.session_state['pos_sample'] = st.session_state.get('pos_sample', 0) + 1
    else:
        st.session_state['neg_sample'] = st.session_state.get('neg_sample', 0) + 1


def detail_img(img_url):
    logger.info(f'@{EXP_ID}@detail_img')
    print('insert')
    print(img_url)
    with st.sidebar:
        st.image(img_url)
        st.button('POS', type="secondary", on_click=lambda img_url=img_url: user_feedback(img_url, 1), key=f'btn_detail_1')
        st.button('NEG', type="primary", on_click=lambda img_url=img_url: user_feedback(img_url, 0), key=f'btn_detail_0')


def show_img(image_urls):
    cols = st.columns(3)
    for idx, img_url in enumerate(image_urls):
        img = Image.open(img_url)
        cols[idx % 3].image(img, use_column_width=True,
                            # caption='1111',# TODO caption
                            )
        # cols[idx % 3].button('POS', type="secondary", on_click=lambda: print(1), key=f'{idx}_0')
        # cols[idx % 3].button('NEG', type="primary", on_click=lambda: print(2), key=f'{idx}_1')
        cols[idx % 3].button('DETAIL', type="secondary", on_click=lambda img_url=img_url: detail_img(img_url), key=f'btn_{idx}')


def start_exp():
    logger.info(f'@{EXP_ID}@start_exp1')
    # # if 'caption_list' not in st.session_state:
    df = pd.read_csv('/home/gusongen/mm_search_baseline/data/f8k/captions.txt', index_col=None)
    st.session_state['caption_list'] = df
    st.session_state[f'{EXP_ID}_cnt'] = 0
    load_next_exp()

###########
# MAIN PAGE
###########


# 设置标签栏
st.set_page_config(page_title="CMMR", page_icon="🔍", layout='wide', initial_sidebar_state="auto")
if 'log_id' not in st.session_state:
    logger.remove()
    logger.add(sys.stdout)
    st.warning('No log file found, please register user first!')
logger.info(f'@{EXP_ID}@swicth_tab::{EXP_ID}')

# Initialization
# TODO place it in a init function

# sidebar

if f'{EXP_ID}_cnt' not in st.session_state:
    start_btn = st.sidebar.empty()
    if start_btn.button('Start exp'):
        start_exp()
        start_btn.empty()
if f'{EXP_ID}_cnt' in st.session_state:
    with st.sidebar:

        if st.session_state.get(f'{EXP_ID}_cnt', 0) == TOTAL_EXP + 1:
            st.success("Experiment finished!")
            logger.info(f"@{EXP_ID}@Experiment finished!")
        else:
            st.markdown("## Target caption")
            st.markdown(f""">*{ st.session_state['current_caption']}*""")
            # st.code(f""">*{ st.session_state['current_caption']}*""")
            st.markdown(
                f"""
            ---------
            ### Experiment State
            Done / Total : {min(st.session_state.get(f'{EXP_ID}_cnt',0),TOTAL_EXP)} / {TOTAL_EXP}
            ### User Feedback
            Pos / Neg : {st.session_state.get('pos_sample',0)} / {st.session_state.get('neg_sample',0)}
            """
            )
            cols = st.columns(2)
            cols[0].button('Next', on_click=load_next_exp)
            # cols[1].button('Next', on_click=load_next_exp)


# main container
# 设置标题
st.title('Welcome to CMMR!')
with st.expander("Brief Introduction"):
    st.markdown("""
        **C**Lip **M**ulti **M**odal **R**etrievel is a neural network retrieval system based on the [CLIP](https://github.com/openai/CLIP).
    """)
expander = st.expander("实验须知")
with expander:
    st.markdown("""
        * 请对右侧的**Target caption**进行进行进行检索(直接复制)
        * 对返回的图片判断和句子的**是否匹配**(部分语义匹配即可),并点击**DETAIL**进行反馈操作
        * 如果认为是匹配的则点击POS,否则点击NEG
        * 不需要对所有的图像进行反馈,只挑选您认为与caption最匹配/最不匹配的图像
        * 完成后点击右侧**Next**
    """)

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

search_button = st.button("Search", disabled=f'{EXP_ID}_cnt' not in st.session_state or st.session_state[f'{EXP_ID}_cnt'] == TOTAL_EXP + 1)


# search
if search_button:  # 判断是否点击搜索按钮
    if query is None or modal_select == 'Text' and query.strip() == "":   # 判断是否输入查询文本
        st.warning('Please input the query first!')
    else:
        logger.info(f'@{EXP_ID}@search_button_press')
        logger.info(f'@{EXP_ID}@modal_selection::{modal_select}')
        if topn_value == None or topn_value == "" or not topn_value.isnumeric():  # 如果没有输入 top k 则默认设置为1
            topn_value = DEFAULT_TOPK
        topn_value = int(topn_value)
        logger.info(f'@{EXP_ID}@query::{query}')
        logger.info(f'@{EXP_ID}@topn_value::{topn_value}')
        success_placeholder = st.empty()
        with st.spinner("Processing..."):
            uid = uuid.uuid1()
            image_urls = search_clip(uid, query, modal_select, topn_value)
            # image_urls = load_data()[:50]
            logger.info(f'@{EXP_ID}@search_response')
            if len(image_urls) == 0:
                st.warning('Nothing similar found!', icon="⚠️")
            else:
                st.session_state['candidate_images'] = image_urls
                success_placeholder.success("Done!")


# always place it at last
if 'candidate_images' in st.session_state:
    show_img(st.session_state['candidate_images'])
# logger.info('@st.session_state')
# logger.info(st.session_state)
# TODO visualization
# docs = DocumentArray(
#     storage='sqlite',
#     config={
#         'connection': '/home/gusongen/mm_search_baseline/workspace/SimpleIndexer/index_toy-data.db',
#         'table_name': 'simple_indexer_table2',
#     },
# )

# st.write(docs.plot_embeddings())
# TODO fix image encoder and text encoder shape mismatch and tesnor/np mismatch
# TODO 干净版本的commit
# TODO middle size dataset
# TODO 一些log的值能不能提前计算写到log,delta time and trial time
