import streamlit as st
import sys
sys.path.append('..')
sys.path.append('.')
from util import *


st.write(st.session_state)  # TODO debug


st.button('refresh')

st.button('clean session state', on_click=lambda: clean_session_state(exclude=['log_id']))
max_bd_size = st.number_input('max_bd_size', step=1, min_value=0, max_value=len(load_data()))
if st.button('reindex') and int(max_bd_size):  # on_click=lambda: reindex(max_bd_size))
    reindex(int(max_bd_size))


st.button('clean bd', on_click=clean)
st.button('echo', on_click=echo)
