# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import ai_wonder as wonder

# Input with default values
def user_input(prompt, default):
    response = input(f"{prompt} (default: {default}): ")
    return response if response else default

# The driver
if __name__ == "__main__":
    print(f"Modified App Inventor '허가_비허가' Predictor")
    print("Powered by AI Wonder\n")
    
    # User inputs
    시도 = user_input("시도", "'부산광역시'")
    건축면적 = user_input("건축면적(_)", "'206.63'")
    연면적 = user_input("연면적(_)", "'81.06'")
    용적률산전용면적 = user_input("용적률산전용면적(_)", "'79.38'")
    용적률 = user_input("용적률(%)", "'4.2165'")
    지붕구조동 = user_input("지붕구조(동)", "'목구조'")

    # Make datapoint from user input
    point = pd.DataFrame({
        '시도': [시도],
        '건축면적(_)': [건축면적],
        '연면적(_)': [연면적],
        '용적률산전용면적(_)': [용적률산전용면적],
        '용적률(%)': [용적률],
        '지붕구조(동)': [지붕구조동],
    })

    # Predict
    model = wonder.load_saved_model(f'modified_APP_inventor_model.pkl')
    prediction = str(model.predict(point)[0])
    print(f"\nPrediction of '허가_비허가' is '{prediction}'.")
###
