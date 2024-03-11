
import streamlit as st
import datetime
from pathlib import Path

import os

import matplotlib.pyplot as plt
import cloudpickle

import mpld3
import streamlit.components.v1 as components

from sdk_model.create_lm_from_2points import create_save_model


from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_squared_error
import numpy as np

st.header('Creazione Modello Lineare partendo da due punti')

with st.form('my_form'):
    st.subheader('INSERISCI nomi (modello, variabili):')
    visibility_form = 'visible' # 'visible', 'hidden' or 'collapsed'
    nome_modello = st.text_input('Nome modello: ', label_visibility=visibility_form)
    # st.subheader('Nome Variabile Target:')
    nome_var_target = st.text_input('Nome Variabile Target: ', label_visibility=visibility_form)
    # st.subheader('Nome Variabile Esplicativa:')
    nome_var_x = st.text_input('Nome Variabile Esplicativa:', label_visibility=visibility_form)

    st.subheader('INSERISCI coordinate punti:')
    col1, col2 = st.columns(2)
    with col1:
        # st.write('Primo punto')
        x1 = st.number_input('Punto I asse X (esplicativa): ', value=0)
        y1 = st.number_input('Punto I asse Y (target): ', value=0)
    with col2:
        # st.write('Secondo punto')
        x2 = st.number_input('Punto II asse X (esplicativa): ', value=0)
        y2 = st.number_input('Punto II asse Y (target): ', value=0)

    # Every form must have a submit button
    submitted = st.form_submit_button('Create model and save in Downloads')


if submitted:
    # Check data consistancy for model createion
    consistant_data = True
    if consistant_data:
        st.subheader('Risultato:')

        dir_save_model = str(Path.home() / "Downloads")
        model_dt = f"{datetime.datetime.now()}".split('.')[0][:-3]
        model_dt = model_dt.replace(" ", "_").replace(":", "").replace("-", "")
        file_name = f'{nome_modello}_{model_dt}.pkl'

        X, Y = create_save_model(x1, y1, x2, y2, nome_modello, nome_var_x, nome_var_target, dir_save_model, file_name)

        st.write(f"Saved Model path: {os.path.join(dir_save_model, file_name)}")

        fig = plt.figure()
        plt.plot(X, Y)
        plt.xlabel(nome_var_x)
        plt.ylabel(nome_var_target)
        plt.title(nome_modello)
        # plt.legend()
        # st.pyplot(fig)

        fig_html = mpld3.fig_to_html(fig)
        components.html(fig_html, height=600)

    else:
        st.write('☝️ I dati inseriti non sono consistenty. Esempio:\n * Punto I uguale con Punto II ...')

else:
    st.write('☝️ Inserisci valori desiderati!')
