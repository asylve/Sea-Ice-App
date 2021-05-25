# -*- coding: utf-8 -*-
"""
Created on Mon May 24 10:50:15 2021

@author: DELL
"""

import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd

st.title('My first app')

st.map()

if st.checkbox('Show dataframe'):
    chart_data = pd.DataFrame(
       np.random.randn(20, 3),
       columns=['a', 'b', 'c'])

    st.write(chart_data)