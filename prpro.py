import numpy as np
import matplotlib.pyplot as plt

# For preprocessing
from sklearn.preprocessing import StandardScaler


import torch # For building the networks
import torchtuples as tt # Some useful functions

from pycox.datasets import metabric
from pycox.models import PMF
from pycox.evaluation import EvalSurv
import pandas as pd
import streamlit as st
from torch import device
import os

##########

model = torch.load('PMF_5_model.pt',weights_only = False)


#######

st.title('PMF 模型')

p1 = st.number_input(label = '载脂蛋白A1(g/L)',
                     step=float(0.01))

p2 = st.number_input(label = '高密度脂蛋白胆固醇(mmol/L)',
                     step=float(0.01))
p3 = st.number_input(label = '总胆固醇(mmol/L)',
                     step=float(0.01))
p4 = st.number_input(label = '乳酸脱氢酶(U/L)',
                     step=float(0.01))
p5 = st.number_input(label = '磷(mmol/L)',
                     step=float(0.01))

p6 = st.number_input(label = '时间（天）',
                     min_value=0,
                     max_value=1000,
                     value=0)


ipcom = pd.DataFrame([p1,p2,p3,p4,p5]).T.values.astype("float32")

pred = model.interpolate(10).predict_surv_df(ipcom)

pred_ev = EvalSurv(pred,
                   np.array([30]),
                   np.array([1]),
                   censor_surv='km')

pred_reu = 1 - pred_ev.surv_at_times(p6)
pred_reu1 = round(float(pred_reu*100),2)

if st.button("预测"):
    st.write(f'{p6}天复发率: {pred_reu1}%')




