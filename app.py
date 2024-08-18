#streamlit run app.py

import streamlit as st
import pandas as pd
import torch
from torch.utils.data import DataLoader
import tkinter as tk
from tkinter import filedialog
import os
import load_data
import matplotlib.pyplot as plt
import shap
from PIL import Image

st.set_page_config(layout='wide')
#-----------------------侧边栏------------------------

#introduction
introduction_click=st.sidebar.title("Introduction")
st.sidebar.write("This is a app which can help user quickly predict FOP ")

#data specification
data_specification=st.sidebar.button("Data Specification")
if(data_specification):
    sp = pd.read_excel("./data.xlsx", 'Sheet4')
    st.sidebar.write(sp)

#-------------------------读取文件-----------------------------

st.title("AMI_LSTM")

model = torch.load('./save/lstm_model_100.pth')

#data_selection
file=st.file_uploader("choose a data file",type={'xlsx'})

if file is not None:
    # 读取上传的文件
    df = pd.read_excel(file)
    # 对文件进行处理或展示
    train_loader, test_loader, train_dataset, test_dataset,one_hot,scaler=load_data.load_data()

    X_ONE = df[
        ['性别', '婚姻状况', '胸闷', '腹痛', '背痛', '心悸', '恶心', '大汗', '乏力', '呼吸困难', '是否合并高血压',
         '是否合并糖尿病', '是否合并高脂血症', '是否PCI', '疾病类型', 'PTCA', '溶栓', '药物保守治疗', '医疗付费形式',
         '工作状态']]
    X_ONE = one_hot.transform(X_ONE)
    x1 = X_ONE.toarray()

    X_value = df[['疾病不确定感', '总支持', '心理']]
    x2 = scaler.transform(X_value)

    x3 = df.drop(
        columns=['编号', '编码', 'FOP', '性别', '婚姻状况', '胸闷', '腹痛', '背痛', '心悸', '恶心', '大汗', '乏力',
                 '呼吸困难', '是否合并高血压', '是否合并糖尿病', '是否合并高脂血症', '是否PCI', '疾病类型', 'PTCA',
                 '溶栓', '药物保守治疗', '医疗付费形式', '工作状态', '疾病不确定感', '总支持', '心理'])

    x4 = df[['FOP']]

    X = []

    for i in range(X_ONE.shape[0]):
        a = list(x1[i])
        b = list(x2[i])
        c = list(x3.iloc[i])
        d = list(x4.iloc[i])
        X.append(a + b + c + d)

    n = pd.DataFrame(X)

    n.columns=['Gender-male','Gender-female','Marital status-married','Marital status-Unmarried','Without chest pain','With chest pain',
               'Without abdominal pain','With abdominal pain','Without back pain','With back pain',
               'Without palpitations','With palpitations','Without nausea','With nausea','Without profuse sweating','With profuse sweating',
               'Without fatigue','With fatigue','Without shortness of breath','With shortness of breath','Without hypertension','With hypertension',
               'Without diabetes','With diabetes','Without hyperlipidemia','With hyperlipidemia','No PCI treatment','PCI treatment','ST-Elevation Myocardial Infarction','Non-ST-Elevation Myocardial Infarction',
               'Non-PTCA treatment','PTCA treatment','Non-thrombolytic treatment','Thrombolytic treatment ','Non-pharmacological conservative treatment','Pharmacological conservative treatment',
               'Health insurance','New Rural Cooperative Medical Scheme','Self-pay','Public funding',
               'Employment status-Employed','Employment status-Retired','Employment status-Unemployed',
               'Uncertainty about the disease', 'Social support', 'Psychological flexibility','Age groups ', 'Educational level', 'Monthly household incom', 'BMI classification', 'Smoking status',
               'Alcohol consumption', 'Exercise habits', 'Killip classification','Number of vessel lesions','FOP']

    st.title("Data after data processing ")

    st.write(n)

    if(st.button("Predict")):

        st.title("Result")

        data= load_data.CustomDataset(n)

        test_loader= DataLoader(data, batch_size=64, shuffle=False)

        model.eval()

        thresold = 0.463

        temp = []

        for batch in test_loader:
            X_batch = batch['X'].unsqueeze(1).to('cuda')  # 增加序列维度并传递给cuda计算
            temp.append(model(X_batch))

        prediction_pro = []  # 存储预测值概率

        prediction = []  # 存储预测值

        for i in range(len(temp)):
            for j in temp[i]:
                prediction_pro.append(j.item())

        for i in range(len(prediction_pro)):
            if (prediction_pro[i] > thresold):
                prediction.append(1)
            else:
                prediction.append(0)

        n.insert(loc=len(n.columns),column='Prediction',value=pd.Series(prediction))

        tab1,tabs2=st.tabs(['Prediction','Chart'])

        end_prediction=n.drop(columns=['FOP'])

        end=n.drop(columns=['FOP','Prediction'])

        #shap
        model.to('cpu')
        model.train()
        data = []
        shap_value = []
        for batch in test_loader:
            X_batch = batch['X'].unsqueeze(1)
            data.append(X_batch)
        i=1
        raw=data[0]
        if(len(data)>1):
            while(i<len(data)):
                raw=torch.cat((raw,data[i]),0)
                i=i+1
        explainer = shap.GradientExplainer(model=model, data=raw)
        shap_value = explainer.shap_values(raw)
        shap_value = shap_value.reshape(end.shape[0], 55)

        features_names = ['Gender-male', 'Gender-female', 'Marital status-married', 'Marital status-Unmarried',
                          'Without chest pain', 'With chest pain',
                          'Without abdominal pain', 'With abdominal pain', 'Without back pain', 'With back pain',
                          'Without palpitations', 'With palpitations', 'Without nausea', 'With nausea',
                          'Without profuse sweating', 'With profuse sweating',
                          'Without fatigue', 'With fatigue', 'Without shortness of breath', 'With shortness of breath',
                          'Without hypertension', 'With hypertension',
                          'Without diabetes', 'With diabetes', 'Without hyperlipidemia', 'With hyperlipidemia',
                          'No PCI treatment', 'PCI treatment', 'ST-Elevation Myocardial Infarction',
                          'Non-ST-Elevation Myocardial Infarction',
                          'Non-PTCA treatment', 'PTCA treatment', 'Non-thrombolytic treatment',
                          'Thrombolytic treatment ', 'Non-pharmacological conservative treatment',
                          'Pharmacological conservative treatment',
                          'Health insurance', 'New Rural Cooperative Medical Scheme', 'Self-pay', 'Public funding',
                          'Employment status-Employed', 'Employment status-Retired', 'Employment status-Unemployed',
                          'Uncertainty about the disease', 'Social support', 'Psychological flexibility', 'Age groups ',
                          'Educational level', 'Monthly household incom', 'BMI classification', 'Smoking status',
                          'Alcohol consumption', 'Exercise habits', 'Killip classification', 'Number of vessel lesions']



        with tab1:
            st.write(end_prediction)
        with tabs2:
            shap.summary_plot(shap_value, end, feature_names=features_names, max_display=20,show=False)
            fig1=plt.gcf()
            fig1.savefig('./fig/fig.png')
            image=Image.open('./fig/fig.png')
            st.image(image,width=600)


