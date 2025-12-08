##install streamlit
#pip list-->to get all pip elemnts
#pip install streamlit

import streamlit as st
import pickle
from PIL import Image

#create a function
def main():
    #to add titile
    st.title(':violet[IRIS SPECIES PREDICTOR]')
    #to read image
    image=Image.open(r'inbox_19517213_157356294f32ae95956a495960b7ea39_1_ZK9_HrpP_lhSzTq9xVJUQw.png')
    st.image(image,width=600)

    #identitfy the features

    #input features
    Sepal_Length= st.text_input('sepal Length', 'Type here')
    Sepal_Width = st.text_input('Sepal_Width', 'Type here')
    Petal_Length = st.text_input('Petal_Length', 'Type here')
    Petal_Width = st.text_input('Petal_Width', 'Type here')
    

    #store all feartures in a vraiable

    f = [Sepal_Length, Sepal_Width, Petal_Length, Petal_Width]

    # load model and scaler
    model1 = pickle.load(open('mdlknn', 'rb'))
    scaler1 = pickle.load(open('slrmodel', 'rb'))

    # predict button
    pred = st.button("PREDICT")

    if pred:
        # make prediction
        prediction = model1.predict(scaler1.transform([f]))[0]

        # display result
        if prediction == 0:
            prediction = "virginica"
            st.success(f"The predicted Iris species is: {prediction} 🌸")
            st.write("🎉 Congratulations! Prediction successful!")
        
        
        else:
            prediction = "Setosa"
            st.success(f"The predicted Iris species is: {prediction} 🌸")
            st.balloons()
            st.write("🎉 Congratulations! Prediction successful!")


main()