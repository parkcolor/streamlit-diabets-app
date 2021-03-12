import streamlit as st
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg') #서버에서 화면에 표시하기 위해서 필요
import seaborn as sns
import joblib


def main():
    st.title('당뇨병 판정시스템')
    st.subheader('이 머신러닝은 당신의 건강데이터를 바탕으로 당뇨병 예측을 합니다')

    st.write('아래 데이터는 머신러닝에 사용될 데이터입니다')
    df = pd.read_csv('diabetes.csv.txt')
    st.dataframe(df)

    st.subheader('아래 항목을 입력하세요')
    preganacies = st.number_input('임신횟수를 입력하세요',0.0)
    glucose = st.number_input('혈당 수치를 입력하세요',0.0)
    bloodpressure = st.number_input('혈압을 입력하세요',0.0)
    skin = st.number_input('피하지방 측정값을 입력하세요', 0.0)
    insulin = st.number_input('인슐린 수치를 입력하세요',0.0)
    bmi = st.number_input('BMI 측정값을 입력하세요',0.0)
    dpf = st.number_input('당뇨 가족력을 입력하세요',0.0)
    age = st.number_input('나이를 입력하세요',0.0)

    #2. 예측한다
    #2-1 모델불러오기
    model = joblib.load('best_model.joblib')
    #2-2 스케일링
    new_data = np.array([preganacies,glucose,bloodpressure,skin,insulin,bmi,dpf,age])
    new_data = new_data.reshape(1,-1)
    
    if st.button('예측하기'):
        y_pred = model.predict(new_data)
 
        if y_pred == 1:
            st.error('당뇨에 걸릴 확률이 높습니다')
        else:
            st.success('당뇨에 걸릴 확률이 낮습니다')
    



if __name__ == '__main__':
    main()