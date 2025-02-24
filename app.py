import streamlit as st
from fastai.vision.all import *
import plotly.express as px




st.write("## BMI")
st.write(
    ":hospital: Malumot dastur haqida"
)
st.sidebar.write("## Rasmini Yuklash :gear:")

col1, col2,col3 = st.columns(3)
file=st.sidebar.file_uploader("Rasm yuklash",type=['png','jpeg','svg','jpg'])


if file:
    col1.write("Yuklangan Rasm :camera:")
    with col1:
        st.image(file)

    img =PILImage.create(file)

    model = load_learner("cat_dog_model.pkl")

    pred, pred_id, probs =model.predict(img)

    col2.write("Natijalar 🧬 🧠")
    with col2:
        st.success(f"Bashorat : {pred}")
        st.info(f"Ehtimollik : {probs[pred_id]*100:.1f}%")
        
    col3.write("Bashorat aniqlik Grafigi 📈")
    with col3:
        fig = px.bar(y=probs*100,x=model.dls.vocab,width=400, height=350)
        st.plotly_chart(fig)
else:
    col1.write("Yuklangan Rasm :camera:")
    with col1:
        st.image("x.jpg")
    col2.write("Natijalar 🧬 🧠")
    with col2:
        st.success(f"Bashorat Qiymati")
        st.info(f"Bashorat aniqligi (%)")
    col3.write("Bashorat aniqlik Grafigi 📈")
    with col3:
        st.image("grafik.jpg")
      

with st.expander("Foydalanish Qo'llanmasi"):
    st.markdown("""
                ##### Foydalanish:
                * Ekraning chap tomonidagi "Rasm yuklash" qismi orqali rasmingizni yuklang
                * Natijalar : Matn,Foiz va Grafik ko'rinishida chiqadi

                ##### Ogohlantirish :
                * Ishlov berilgan rasmni yuklash xato natija chiqishiga olib keladi !
                """)




