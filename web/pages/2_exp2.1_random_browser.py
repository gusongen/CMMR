# 导入需要的包
import os
import sys
from typing import Optional
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from loguru import logger
import streamlit.components.v1 as components

sys.path.append('..')
sys.path.append('.')
from util import *


CANDIDATE_SZIE = [10, 50, 100, 150, 200]
REPEAR_EACH = 1  # 每个候选数量的重复次数 # # TODO EXP 5
CANDIDATE_SZIE *= REPEAR_EACH
np.random.shuffle(CANDIDATE_SZIE)
TOTAL_EXP = len(CANDIDATE_SZIE)
DEFAULT_TOPK = 100
IMAGE_DIR = '/home/gusongen/mm_search_baseline/data/f8k/images'
EXP_ID = 'exp2.1'
# TODO 初始状态手动开始实验


def start_exp():
    logger.info(f'@{EXP_ID}@start_exp2.1')
    # if 'caption_list' not in st.session_state:
    df = pd.read_csv('/home/gusongen/mm_search_baseline/data/f8k/captions.txt', index_col=None)
    st.session_state['caption_list'] = df
    st.session_state[f'{EXP_ID}_cnt'] = 0
    load_next_exp()


def random_choice_images(candidate_size):
    """_summary_

    Args:
        candidate_size (int ): candiate image size

    Returns:
        _type_: all candiate image and target image
    """
    if 'caption_list' not in st.session_state:
        return
    df = st.session_state['caption_list']
    candidate_images = np.random.choice(df['image'], size=candidate_size, replace=False)
    candidate_images = [os.path.join(IMAGE_DIR, i)for i in candidate_images]
    target_image = np.random.choice(candidate_images)
    return candidate_images, target_image


def found_result(found: Optional[str] = None, idx=None):
    if st.session_state.get(f'{EXP_ID}_cnt', 0) <= TOTAL_EXP:
        if found is None:
            logger.info(f'@{EXP_ID}@found_result::none')
        else:
            if found.split('/')[-1] == st.session_state.get('target_image', '').split('/')[-1]:
                logger.info(f'@{EXP_ID}@found_result::true')
                logger.info(f'@{EXP_ID}@found_pos::{idx+1}/{CANDIDATE_SZIE[st.session_state.get(f"{EXP_ID}_cnt", 0)]}')
            else:
                logger.info(f'@{EXP_ID}@found_result::false')
        st.session_state['disable_next'] = False
    # load_next_exp()
    # TODO idx and total index


def load_next_exp():
    logger.info(f'@{EXP_ID}current retrieval done')
    if st.session_state.get(f'{EXP_ID}_cnt', 0) >= TOTAL_EXP:
        if st.session_state.get(f'{EXP_ID}_cnt', 0) == TOTAL_EXP:
            st.session_state[f'{EXP_ID}_cnt'] = st.session_state.get(f'{EXP_ID}_cnt', 0) + 1
            # st.session_state.pop('candidate_images')
            clean_session_state(exclude=[f'{EXP_ID}_cnt', 'log_id'])
        return
    st.session_state['disable_next'] = True
    candidate_size = CANDIDATE_SZIE[st.session_state.get(f'{EXP_ID}_cnt', 0)]
    candidate_images, target_image = random_choice_images(candidate_size)
    st.session_state['candidate_images'] = candidate_images
    st.session_state['target_image'] = target_image
    st.session_state[f'{EXP_ID}_cnt'] = st.session_state.get(f'{EXP_ID}_cnt', 0) + 1
    # show_img(st.session_state['candidate_images'])  # bug,不能渲染,只能通过更改sss,渲染放在最后


def show_img(image_urls):
    cols = st.columns(3)
    # cols = img_container.columns(3)
    for idx, img_url in enumerate(image_urls):
        img = Image.open(img_url)
        cols[idx % 3].image(img, use_column_width=True,
                            # caption='1111',# TODO caption
                            )
        # cols[idx % 3].button('POS', type="secondary", on_click=lambda: print(1), key=f'{idx}_0')
        # cols[idx % 3].button('NEG', type="primary", on_click=lambda: print(2), key=f'{idx}_1')
        cols[idx % 3].button('SELECT', key=f'btn_{idx}', on_click=lambda img_url=img_url, idx=idx: found_result(img_url, idx))


# 设置标签栏
st.set_page_config(page_title="CMMR", page_icon="🔍", layout='wide', initial_sidebar_state="auto")
# components.html(
#     "<script>window.parent.document.querySelector('section.main').scrollTo(0, 0);</script>",
#     height=0,
# )
if 'log_id' not in st.session_state:
    logger.remove()
    logger.add(sys.stdout)
    st.warning('No log file found, please register user first!')
logger.info(f'@{EXP_ID}@swicth_tab::{EXP_ID}')
# 设置标题
# st.title('Welcome to CMMR!')
# with st.expander("Brief Introduction"):
#     st.markdown("""
#         **C**Lip **M**ulti **M**odal **R**etrievel is a neural network retrieval system based on the [CLIP](https://github.com/openai/CLIP).
#     """)
st.title('实验2.1')
# st.write(st.session_state)
expander = st.expander("实验须知")
with expander:
    st.markdown("""
        * 请在下列图片中寻找右侧的**Target image**
        * 找到后点击图片下方**SELECT**完成当前实验
        * 不保证目标一定存在
        * 如果认为目标图片不存在或者检索困难可以点击右侧**Not Found**结束当前实验
        * 完成后点击右侧**Next**
    """)

# Using "with" notation


# img_container = st.container()


if f'{EXP_ID}_cnt' not in st.session_state:
    start_btn = st.sidebar.empty()
    if start_btn.button('Start exp'):
        start_exp()
        start_btn.empty()

if f'{EXP_ID}_cnt' in st.session_state:
    with st.sidebar:
        if st.session_state.get(f'{EXP_ID}_cnt', 0) > TOTAL_EXP:
            st.success("Experiment finished!")
            logger.info(f"@{EXP_ID}@Experiment finished!")
        else:
            st.markdown("## Target image")
            st.image(Image.open(st.session_state['target_image']))
            st.markdown(
                f"""
            ---------
            ### Experiment State
            Done / Total : {min(st.session_state.get(f'{EXP_ID}_cnt',0),TOTAL_EXP)} / {TOTAL_EXP}

            """
            )
            cols = st.columns(2)
            cols[0].button('Not Found', type='primary', on_click=found_result)
            cols[1].button('Next', on_click=load_next_exp, disabled=st.session_state.get('disable_next', True))
if 'candidate_images' in st.session_state:
    show_img(st.session_state['candidate_images'])
# logger.info('@st.session_state')
# logger.info(st.session_state)
