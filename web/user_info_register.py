from loguru import logger
import os
from datetime import datetime
# import httplib2
import pandas as pd
import streamlit as st
import uuid
import sys
from xpinyin import Pinyin

st.set_page_config(page_title="user exp", page_icon="🐞", layout="centered")
if 'log_id' not in st.session_state:
    logger.remove()
    logger.add(sys.stdout)
    st.warning('No log file found, please register user first!')
logger.info(f'@swicth_tab::user_info_register')


def logger_init(log_file_name='monitor',
                log_dir='../logs/',
                only_file=False):
    # 指定路径
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_file_name + "_{time}.log")
    log_id = logger.add(log_path)
    # log_path = os.path.join(log_dir, log_file_name + '_' + str(datetime.now())[:10] + '.txt')
    # # open(log_path,'w')
    # fmt = '[%(asctime)s] - %(levelname)s: %(message)s'
    # formatter = logging.Formatter(fmt=fmt, datefmt='%Y-%d-%m %H:%M:%S')
    # logger = logging.getLogger()
    # logger.setLevel(log_level)
    # # stream
    # sh = logging.StreamHandler(sys.stdout)
    # sh.setLevel(log_level)
    # sh.setFormatter(formatter)
    # logger.addHandler(sh)
    # # file
    # fh = logging.FileHandler(log_path)
    # fh.setLevel(log_level)
    # fh.setFormatter(formatter)
    # logger.addHandler(fh)

    logger.info(f"@logger initialized")
    return log_id


# def init_exp(user_name, age, gender, exp_type, comment):
def init_exp(user_name, age, gender, comment):
    """
    TODO 创建一个log file
    TODO 创建feedback example ,句柄
    TODO 跳转下一个page
    TODO 保存到state
    """
    uid = uuid.uuid1()
    user_name = user_name if user_name.strip() else 'default'
    age = int(age) if age.isdigit() else 0
    # log_fie_name = f'user_exp_{exp_type}_{Pinyin().get_pinyin(user_name)}_{uid}'
    log_fie_name = f'user_exp_{Pinyin().get_pinyin(user_name)}_{uid}'
    log_id = logger_init(log_fie_name)
    st.session_state['log_id'] = log_id
    # logger.info(f"@user_data::exp:{exp_type}, user_name:{user_name}, age:{age}, gender:{gender}")
    logger.info(f"@user_data::user_name:{user_name}, age:{age}, gender:{gender}")
    logger.info(f"@comment:{comment}")


st.title("🐞User experiment register !")

form = st.form(key="annotation")

with form:
    cols = st.columns((1, 1))
    user_name = cols[0].text_input("姓名:", placeholder='default')
    age = cols[1].text_input("年龄:", placeholder='0')
    gender = cols[0].selectbox(
        "性别:", ["male", "female"], index=0
    )
    # exp_type = cols[1].selectbox(
    #     "实验类型:", ["exp1.feedback"], index=0
    # )
    comment = st.text_area("备注:")
    # cols = st.columns(2)
    # date = cols[0].date_input("Bug date occurrence:")
    # bug_severity = cols[1].slider("Bug severity:", 1, 5, 2)
    submitted = st.form_submit_button(label="提交")


if submitted:
    # init_exp(user_name, age, gender, exp_type, comment)
    init_exp(user_name, age, gender, comment)
    st.success("Thanks! Your information was recorded.")
    st.balloons()

expander = st.expander("see more detail")
with expander:
    st.write(f"todo")
