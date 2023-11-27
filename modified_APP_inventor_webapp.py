# -*- coding: utf-8 -*-

# Import necessary libraries
import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import ai_wonder as wonder

# Arrange radio buttons horizontally
st.write('<style> div.row-widget.stRadio > div { flex-direction: row; } </style>',
         unsafe_allow_html=True)

# The driver
if __name__ == "__main__":
    # Streamlit interface
    st.subheader(f"건축 허가 예측 AI")
    st.markdown("기말고사 대체과제 :blue[**앱개발(app_inventor)**]")

    # User inputs
    시도 = st.selectbox("시도", ['경기도', '전라북도', '경상남도', '대구광역시', '제주특별자치도', '충청남도', '서울특별시', '강원도', '충청북도',
                      '경상북도', '부산광역시', '전라남도', '제주특별시', '인천광역시', '울산광역시', '광주광역시', '대전광역시', '세종특별자치시'], index=14)
    건축면적 = st.text_input("건축면적(_)", value="19.5")
    연면적 = st.text_input("연면적(_)", value="284.963")
    용적률산전용면적 = st.text_input("용적률산전용면적(_)", value="1,239.32")
    용적률 = st.text_input("용적률(%)", value="11.1182")
    지붕구조동 = st.selectbox("지붕구조(동)", [
                         '기타지붕', '(철근)콘크리트', '슬레이트', '기와', '철근콘크리트', '목구조', '철근콘크리트구조'], index=5)

    # Make datapoint from user input
    point = pd.DataFrame({
        '시도': [시도],
        '건축면적(_)': [건축면적],
        '연면적(_)': [연면적],
        '용적률산전용면적(_)': [용적률산전용면적],
        '용적률(%)': [용적률],
        '지붕구조(동)': [지붕구조동],
    })

    # Predict and Explain
    if st.button('예측하기'):
        state = wonder.load_saved_states(f'modified_APP_inventor_state.pkl')

        model = wonder.input_piped_model(state)
        prediction = str(model.predict(point)[0])
        st.success(f"예측결과는 **{prediction}** 입니다.")

        st.info("예측에 영향을 미친 요소")
        importances = pd.DataFrame(wonder.local_explanations(
            state, point), columns=["항목", "요소", "영향도"])
        st.dataframe(importances, hide_index=True)

        st.info("반사실 요소: 판단을 뒤집기 위한 방법")
        tests = wonder.inverse_transform(state, state.X_test)
        st.dataframe(wonder.what_if_instances(
            state, point, tests).iloc[:20], hide_index=True)
###
