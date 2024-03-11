import datetime

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components
from sdk_model.create_forecast_model import model_params, create_features, create_model
from pathlib import Path

@st.cache_data
def upload_csv_file(uploaded_file, sep):
    dataframe = pd.read_csv(uploaded_file, sep=sep)
    return dataframe

# if "file_path" not in st.session_state:
#     st.session_state.file_path = None
dataframe = None
with st.expander("**1 - Upload data**"):
    csv_file = st.file_uploader("Choose a CSV file:") #, key="file_path")

    if csv_file is not None:
        st.selectbox("Select separator:", (";", ","), key='csv_sep')
        # check if it is realy csv file
        dataframe = upload_csv_file(csv_file, sep=st.session_state.csv_sep)
        st.write(dataframe)
        st.write('CSV file was uploaded succesfully')


@st.cache_data
def create_time_index(df, time_field, format):
    try:
        df_wt = df.copy()
        if time_field == '<index>':
            df_wt.index = pd.to_datetime(df_wt.index, format=format)
        else:
            df_wt.index = pd.to_datetime(df_wt[time_field], format=format)
            df_wt.drop(columns = time_field, inplace=True)

        df_wt.sort_index(inplace=True)
        return True, df_wt

    except Exception as e:
        return False, e


if "dataframe_wt" not in st.session_state:
    st.session_state.dataframe_wt = None
if "time_format" not in st.session_state:
    st.session_state.time_format = None

with st.expander("**2 - Set time**", ):
    if dataframe is not None:
        st.write('First 4 rows from uploaded file:')
        st.write(dataframe.head(4))
        st.selectbox("Select the column (index) with time", ['<index>'] + list(dataframe.columns), key='time_column')

        col1, col2, col3 = st.columns([3, 1, 3])
        with col1:
            # st.write('Primo punto')
            time_format_select = st.selectbox("Select the time format", ['%Y-%m-%d %H:%M:%S.%f', '%Y%m%d %H:%M:%S.%f',
                                                    '%d-%m-%Y %H:%M:%S.%f', '%d%m%Y %H:%M:%S.%f'])
            if st.button(f'Use selected format:\n{time_format_select}'):
                st.session_state.time_format = time_format_select

        with col2:
            st.write('OR')

        with col3:
            # st.write('Secondo punto')
            time_format_input = st.text_input('Insert/Change time format: ', value='%Y-%m-%d %H:%M:%S.%f',
                               key='time_format_input')
            if st.button(f'Use inserted format:\n{time_format_input}'):
                st.session_state.time_format = time_format_input

        if st.session_state.time_format is not None:
            st.write(f"**CHECK before apply**: time column: {st.session_state.time_column}; time format: {st.session_state.time_format}")

            if st.button('Apply column and format'):
                st.session_state.success_time_format, st.session_state.dataframe_wt = \
                    create_time_index(dataframe, st.session_state.time_column, st.session_state.time_format)

            if st.session_state.dataframe_wt is None:
                st.write('NO RESULT: Apply the settings')
            else:
                st.write('**FINAL RESULT:**')
                if (type(st.session_state.dataframe_wt) == pd.DataFrame) and st.session_state.success_time_format:
                    st.write(st.session_state.dataframe_wt)
                    st.write('Time was created successfully')
                else:
                    st.write('Error when creating time:')
                    st.write(st.session_state.dataframe_wt)
    else:
        st.write('No uploaded CSV')


if "var_4_time_series_plot" not in st.session_state:
    st.session_state.var_4_time_series_plot = None
if "var_x_4_plot" not in st.session_state:
    st.session_state.var_x_4_plot = None
if "var_y_4_plot" not in st.session_state:
    st.session_state.var_y_4_plot = None
if "var_4_nulls_plot" not in st.session_state:
    st.session_state.var_4_nulls_plot = None

@st.cache_data
def get_df_stats(df):
    df1 = df.describe(include='all')

    df1.loc['dtype'] = df.dtypes
    df1.loc['size'] = len(df)
    df1.loc['nulls'] = df.isnull().mean()
    return df1

with st.expander("**3 - EDA**", ):
    if st.session_state.dataframe_wt is None:
        st.write('Please do previous steps')
    else:
        st.write('Statistics')
        st.write(get_df_stats(st.session_state.dataframe_wt))

        st.selectbox("Choose a plot", ['-', 'time series', 'scatter plot', 'Nulls plot'], key='choosen_plot')

        if st.session_state.choosen_plot == 'time series':
            col1, col2 = st.columns(2)
            with col2:
                st.selectbox("Choose the Variable", ['-']+list(st.session_state.dataframe_wt.columns), key='var_4_time_series_plot')

            if (st.session_state.var_4_time_series_plot is not None) and (st.session_state.var_4_time_series_plot != '-'):
                fig = plt.figure()
                plt.plot(st.session_state.dataframe_wt[st.session_state.var_4_time_series_plot])
                plt.xlabel('Time')
                plt.ylabel(st.session_state.var_4_time_series_plot)
                plt.title(f"Time series for: {st.session_state.var_4_time_series_plot}")

                fig_html = mpld3.fig_to_html(fig)
                components.html(fig_html, height=600)

        elif st.session_state.choosen_plot == 'scatter plot':
            col1, col2 = st.columns(2)
            with col2:
                st.selectbox("Choose the Variable on X", ['-']+list(st.session_state.dataframe_wt.columns), key='var_x_4_plot')
                st.selectbox("Choose the Variable on Y", ['-']+list(st.session_state.dataframe_wt.columns), key='var_y_4_plot')

            if (st.session_state.var_x_4_plot is not None) and (st.session_state.var_y_4_plot is not None) and\
                (st.session_state.var_x_4_plot != '-') and (st.session_state.var_y_4_plot != '-'):

                fig = plt.figure()
                plt.scatter(st.session_state.dataframe_wt[st.session_state.var_x_4_plot], \
                            st.session_state.dataframe_wt[st.session_state.var_y_4_plot])
                plt.xlabel(st.session_state.var_x_4_plot)
                plt.ylabel(st.session_state.var_y_4_plot)
                plt.title(f"Scatter plot")

                fig_html = mpld3.fig_to_html(fig)
                components.html(fig_html, height=600)

        elif st.session_state.choosen_plot == 'Nulls plot':
            col1, col2 = st.columns(2)
            with col2:
                st.selectbox("Choose the Variable", ['-']+list(st.session_state.dataframe_wt.columns), key='var_4_nulls_plot')

            if (st.session_state.var_4_nulls_plot is not None) and (st.session_state.var_4_nulls_plot != '-'):
                mask_nulls = st.session_state.dataframe_wt[st.session_state.var_4_nulls_plot].isnull()
                fig = plt.figure()
                plt.scatter(st.session_state.dataframe_wt.index[mask_nulls==False],
                            st.session_state.dataframe_wt.loc[mask_nulls==False, st.session_state.var_4_nulls_plot],
                            label=st.session_state.var_4_nulls_plot)
                plt.scatter(st.session_state.dataframe_wt.index[mask_nulls],
                            st.session_state.dataframe_wt.loc[mask_nulls, st.session_state.var_4_nulls_plot],
                            label=f"Nulls")

                plt.xlabel('Time')
                plt.ylabel(st.session_state.var_4_nulls_plot)
                plt.legend()
                plt.title(f"Nulls inspection {st.session_state.var_4_nulls_plot}")

                fig_html = mpld3.fig_to_html(fig)
                components.html(fig_html, height=600)


if 'fe_lags' not in st.session_state:
    st.session_state.fe_lags = None

with st.expander("**4 - SELECT COLUMNS OF INTEREST**", ):
    if st.session_state.dataframe_wt is not None:
        st.multiselect(
            'Select / Deselect desired columns',
            list(st.session_state.dataframe_wt.columns),
            list(st.session_state.dataframe_wt.columns),
            key='selected_columns'
        )

        st.write('You selected:', '; '.join(st.session_state.selected_columns))
        st.session_state.fe_lags = {el: 0 for el in st.session_state.selected_columns}
        st.session_state.fe_time = {
                "giorno_settimana": False,
                "ora": False,
                "mese": False,
                "fine_settimana": False,
                "numero_settimana": False,
                "giorno_festivo": False
            }
    else:
        st.write('Please do previous steps')



@st.cache_data
def filter_create_new_features(df, selected_columns, fe_dict, fe_time_dict):
    df_result = df[selected_columns].copy()
    return create_features(df_result, {'add_lags': fe_dict}, fe_time_dict)

if "dataframe_fe" not in st.session_state:
    st.session_state.dataframe_fe = None

with st.expander("**5 - CREATE NEW FEATURES**", ):
    if st.session_state.fe_lags is None:
        st.write('Please do previous steps')
    else:
        with st.form('fe_form'):
            st.write('Create Features from LAGS')

            for k, v in st.session_state.fe_lags.items():
                st.session_state.fe_lags[k] = st.number_input(f'Nbr Lags for "{k}": ', value=v)

            st.write('Create Features from Time')
            for k, v in st.session_state.fe_time.items():
                st.session_state.fe_time[k] = st.selectbox(k, [False, True])

            # Every form must have a submit button
            submitted = st.form_submit_button('Create LAGS and Time features')


        if submitted:
            st.write('Created lags:')
            col1, col2 = st.columns([1, 10])
            with col2:
                for k, v in st.session_state.fe_lags.items():
                    if v != 0:
                        st.write(f'for column "{k}": {v} {"lag" if v == 1 else "lags"}')

            st.write('Created Time features:')
            col1, col2 = st.columns([1, 10])
            with col2:
                for k, v in st.session_state.fe_time.items():
                    if v:
                        st.write(f'{k}')

            st.session_state.dataframe_fe = filter_create_new_features(st.session_state.dataframe_wt,
                                                                       st.session_state.selected_columns,
                                                                       st.session_state.fe_lags,
                                                                       st.session_state.fe_time)
        if st.session_state.dataframe_fe is not None:
            st.write('**RESULT:**')
            st.write(st.session_state.dataframe_fe)



def get_values_from_dict(dict_in):
    dict_out = {}

    for k, v in dict_in.items():
        if type(v) == list:
            dict_out[k] = v[:1]
        else:
            dict_out[k] = [v]

    return dict_out


st.session_state.model_training = {}
st.session_state.model_training['model_params'] = {k: get_values_from_dict(v) for k,v in model_params.items()}
st.session_state.model_training['model_params_view'] = {k: list(v.keys())[:3] for k,v in model_params.items()}

if 'model_training' not in st.session_state:
    st.session_state.model_training = {}

if 'model_params' not in st.session_state.model_training.keys():
    st.session_state.model_training['model_params'] = {k: get_values_from_dict(v) for k,v in model_params.items()}
    st.session_state.model_training['model_params_view'] = {k: list(v.keys())[:3] for k,v in model_params.items()}

if 'model' not in st.session_state:
    st.session_state.model = None

if 'mlpregressor_params_view_to_set' not in st.session_state:
    st.session_state.mlpregressor_params_view_to_set = []

@st.cache_data
def create_save_model(df_filtered, dict_fe, fe_time, model_training, dir_save_model, file_name):
    return create_model(df_filtered, dict_fe, fe_time, model_training, dir_save_model, file_name)


with st.expander("**6 - SET MODEL TRAINING**", ):

    if (st.session_state.fe_lags is None) or (st.session_state.dataframe_fe is None):
        st.write('Please do previous steps')
    else:
        st.session_state.model_training['model_name'] = st.text_input('Insert model name:', value='model')
        st.session_state.model_training['target_column'] = st.selectbox("Select Target", ['-'] + list(st.session_state.dataframe_fe.columns))
        st.session_state.model_training['perc_split'] = st.selectbox("Perc Train/Validation split", [60, 65, 70, 75, 80, 85, 90, 95], index=4)
        st.session_state.model_training['model_type'] = st.selectbox("Model", ['-', 'mlpregressor']) # 'XGBoost', 'mlpregressor', 'LSTM'

        if st.session_state.model_training['model_type'] == 'mlpregressor':
            st.write('Model params:')
            mlp_params = st.session_state.model_training['model_params']['mlpregressor']

            st.session_state.mlpregressor_params_view_to_set = st.multiselect(
                'Select / Deselect desired params to be setted',
                list(mlp_params.keys()),
                list(st.session_state.model_training['model_params_view']['mlpregressor'])
            )

            col1, col2 = st.columns([1, 10])
            with col2:
                for k in st.session_state.mlpregressor_params_view_to_set:
                    initial_v = model_params['mlpregressor'][k]
                    v = mlp_params[k]
                    mlp_params[k] = eval(st.text_input(f'{k}: example: {initial_v}', value=f"{v}"))


        if st.button(f'Train Model:'):
            dir_save_model = str(Path.home() / "Downloads")
            model_dt = f"{datetime.datetime.now()}".split('.')[0][:-3]
            model_dt = model_dt.replace(" ", "_").replace(":", "").replace("-", "")
            file_name = f'{st.session_state.model_training["model_name"]}_{model_dt}.pkl'

            st.write(dir_save_model, file_name)

            df_filtered = st.session_state.dataframe_wt[st.session_state.selected_columns].copy()
            y_train, y_hat_train, y_test, y_hat_test = create_save_model(df_filtered,
                                                                    {'add_lags': st.session_state.fe_lags},
                                                                    st.session_state.fe_time,
                                                                    st.session_state.model_training,
                                                                  dir_save_model,
                                                                  file_name)

            st.write('TRAINED Model Result')

            fig = plt.figure()
            plt.scatter(y_train, y_hat_train)
            plt.xlabel('Real')
            plt.ylabel('Predicted')
            plt.title(f"Real VS Predicted (Test)")

            fig_html = mpld3.fig_to_html(fig)
            components.html(fig_html, height=600)

        else:
            st.write('Please Train the model')

