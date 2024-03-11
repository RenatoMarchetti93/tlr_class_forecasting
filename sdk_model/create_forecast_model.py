import pandas as pd
import numpy as np
import pickle as pk
import holidays
import datetime as dt
import sklearn
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
import cloudpickle
import os

import joblib
import dill

model_params = {'mlpregressor': {'activation': ['relu', 'identity', 'logistic', 'tanh'],
                                 'alpha': 0.0001,
                                 # 'batch_size': ['auto'],
                                 'beta_1': 0.9,
                                 'beta_2': 0.999,
                                 'early_stopping': [False, True],
                                 'epsilon': 1e-08,
                                 'hidden_layer_sizes': (100,),
                                 'learning_rate': ['constant', 'invscaling', 'adaptive'],
                                 'learning_rate_init': 0.001,
                                 'max_fun': 15000,
                                 'max_iter': 200,
                                 'momentum': 0.9,
                                 'n_iter_no_change': 10,
                                 'nesterovs_momentum': [True, False],
                                 'power_t': 0.5,
                                 # 'random_state': "None",
                                 'shuffle': [True, False],
                                 'solver': ['adam', 'lbfgs', 'sgd'],
                                 'tol': 0.0001,
                                 'validation_fraction': 0.1
                                 # 'verbose': [False, True],
                                 # 'warm_start': [False, True]
                                 }
                }


class RebeccaCustomModel:
    """
    Inizializza la classe con un DataFrame e un dizionario di features.
    :param df: DataFrame da manipolare ( index datetime con Frequenza oraria )
    :param dict_features: Dizionario con le specifiche delle feature da calcolare.

    # example
        dict_fengin = {
            "datetime_info": {
                "giorno_settimana": False,
                "ora": False,
                "mese": False,
                "fine_settimana": False,
                "numero_settimana": False,
                "giorno_festivo": False
            },

            "add_lags": {
                "var_x1": 1,
                "var_x2": 0
            },

            "add_seasonal_lags": {
                "var_x1": 1,
                "var_x2": 0
            },

            "add_stats": {
                "var_x1": 1,
                "var_x2": 0
            }


        }

    """

    def __init__(self
                 # , model,
                 # params: dict = None,
                 # # scale_function = None,
                 # target_variable: str = None,
                 # input_variables: list[str] = None,
                 # dict_fengin_from_time: dict = {},
                 # dict_fengin_from_inputs: dict = {},
                 # dict_fengin_from_target: dict = {}
                 ):

        # self.input_variables = input_variables #['Var 1 name', 'Var 2 name', 'Var 3 name', 'Var 4 name']
        # self.target_variable = target_variable # 'Target name'
        # self.dict_fengin_from_time = dict_fengin_from_time
        # self.dict_fengin_from_inputs = dict_fengin_from_inputs
        # self.dict_fengin_from_target = dict_fengin_from_target
        # self.model = model
        # self.params = params

        self.input_variables = ['Var 1 name', 'Var 2 name', 'Var 3 name', 'Var 4 name']
        self.target_variable = 'Target name'
        self.dict_fengin_from_time = {}
        self.dict_fengin_from_inputs = {}
        self.dict_fengin_from_target = {}
        self.model = MLPRegressor()
        self.params = {}

        self.inputs_scaler = MinMaxScaler()
        self.target_scaler = MinMaxScaler()

    @staticmethod
    def _is_holiday(date):
        """
        Verifica se una data è un giorno festivo in Italia.
        :param date: Data da verificare.
        :return: True se la data è un giorno festivo, altrimenti False.
        """

        return date in holidays.Italy()

    @staticmethod
    def add_datetime_info(df_in: pd.DataFrame, add_info: dict) -> pd.DataFrame:
        """
        Aggiunge informazioni temporali al DataFrame, come giorno della settimana, ora, mese, ecc.
        :return: DataFrame arricchito con informazioni temporali.

        """

        df_out = df_in.copy()
        if add_info.get("giorno_settimana", 'False'):
            df_out['giorno_settimana'] = df_out.index.dayofweek

        if add_info.get("ora", 'False'):
            df_out['ora'] = df_out.index.hour

        if add_info.get("mese", 'False'):
            df_out['mese'] = df_out.index.month

        if add_info.get("fine_settimana", 'False'):
            df_out['fine_settimana'] = (df_out.index.dayofweek >= 5).astype(int)

        if add_info.get("numero_settimana", 'False'):
            df_out['numero_settimana'] = df_out.index.isocalendar().week

        if add_info.get("giorno_festivo", 'False'):
            df_out['giorno_festivo'] = df_out.index.map(RebeccaCustomModel._is_holiday).astype(int)

        return df_out

    @staticmethod
    def add_lagged_features(df_in: pd.DataFrame, add_lags: dict) -> pd.DataFrame:
        """
        Aggiunge al DataFrame le feature lagged in base al dizionario di features.
        :param df: DataFrame a cui aggiungere le feature.
        :return: DataFrame arricchito con le feature lagged.
        """

        df_out = df_in.copy()
        for one_col in df_out.columns:
            n_lags = add_lags.get(one_col, 0)
            if type(n_lags) == int and n_lags > 0:
                for lag in range(1, n_lags + 1):
                    df_out[f'{one_col}_lag{lag}'] = df_out[one_col].shift(lag)

        return df_out

    @staticmethod
    def add_rolling_stats(df_in: pd.DataFrame, add_stats: dict) -> pd.DataFrame:
        """
        Aggiunge statistiche rolling al DataFrame, come mediana, deviazione standard, ecc.
        :param df: DataFrame a cui aggiungere le statistiche.
        :return: DataFrame arricchito con statistiche rolling.
        """

        df_out = df_in.copy()
        for one_col in df_out.columns:
            col_details = add_stats.get(one_col, {})
            h_window = col_details.get('window', 0)

            if type(h_window) == int and h_window > 1:
                rolling_window = df_out[one_col].rolling(f'{h_window}H', min_periods=h_window)

                if col_details.get('median_rolling', False):
                    df_out[f'{one_col}_median_rolling_w{h_window}'] = rolling_window.median()

                if col_details.get('ha', False):
                    df_out[f'{one_col}_dev_std_rolling_w{h_window}'] = rolling_window.std()

                if col_details.get('mean_rolling', False):
                    df_out[f'{one_col}_mean_rolling_w{h_window}'] = rolling_window.mean()

                if col_details.get('max_rolling', False):
                    df_out[f'{one_col}_max_rolling_w{h_window}'] = rolling_window.max()

                if col_details.get('min_rolling', False):
                    df_out[f'{one_col}_min_rolling_w{h_window}'] = rolling_window.min()

                if col_details.get('perc_25_rolling', False):
                    df_out[f'{one_col}_perc_25_rolling_w{h_window}'] = rolling_window.quantile(0.25)

                if col_details.get('perc_75_rolling', False):
                    df_out[f'{one_col}_perc_75_rolling_w{h_window}'] = rolling_window.quantile(0.75)

        return df_out

    def get_time_shifts(self):
        """
            Restituisce la lista degli shift temporali per ogni variabile.
            :return: lista degli shift temporali.
            """
        return {'dict_fengin_from_inputs': self.dict_fengin_from_inputs,
                'dict_fengin_from_target': self.dict_fengin_from_target}

    def preprocess(self, df_in: pd.DataFrame, drop_nulls=True, arange_vars=True) -> pd.DataFrame:
        """
        Esegue l'intero processo di preprocessing sul DataFrame.
        :return: DataFrame preprocessato.
        """

        if arange_vars:
            df_out = df_in[self.input_variables + [self.target_variable]].copy()
        else:
            df_out = df_in.copy()

        df_out.sort_index(inplace=True)

        df_out = self.add_lagged_features(df_out, self.dict_fengin_from_inputs.get('add_lags', {}))
        # df_out = self.add_rolling_stats(df_out, self.dict_fengin_from_inputs['add_stats'])

        df_out = self.add_lagged_features(df_out, self.dict_fengin_from_target.get('add_lags', {}))
        # df_out = self.add_rolling_stats(df_out, self.dict_fengin_from_inputs['add_stats'])

        df_out = self.add_datetime_info(df_out, self.dict_fengin_from_time)

        if drop_nulls:
            df_out = df_out.dropna()

        return df_out

    def update_target_preprocess(self, df_in: pd.DataFrame, drop_nulls=True) -> pd.DataFrame:
        """
        Esegue l'intero processo di preprocessing sul DataFrame.
        :return: DataFrame preprocessato.
        """

        df_out = self.add_lagged_features(df_in, self.dict_fengin_from_target['add_lags'])
        # df_out = self.add_rolling_stats(df_out, self.dict_fengin_from_inputs['add_stats'])
        # df_out = self.add_seasonal_lagged_features(df_out, self.dict_features['add_seasonal_lags'])
        # df_out = self.add_seasonal_stats(df_out, self.dict_features['add_seasonal_lags'])

        if drop_nulls:
            df_out = df_out.dropna()

        return df_out

    @staticmethod
    def split_data_temporally(df: pd.DataFrame, target_column, split_date: str = None, train_ratio: float = 0.8,
                              entire_df: bool = True):
        """
        Divide il DataFrame in set di allenamento e test basandosi su una divisione temporale.

        :param df: DataFrame completo.
        :param target_column: Nome della colonna target.
        :param split_date: Data specifica per dividere i dati (formato 'YYYY-MM-DD').
        :param train_ratio: Percentuale di dati da usare per l'allenamento (es. 0.8 per 80%).
        :return: Tuple (X_train, y_train, X_test, y_test).
        """
        # if random_flag:
        #     print('Random')
        #     df = df.sample(frac=1).reset_index(drop=True)
        #     split_idx = int(len(df) * train_ratio)
        #     train_df = df.iloc[:split_idx].copy()
        #     test_df = df.iloc[split_idx:].copy()
        df.sort_index(inplace=True)
        if split_date:
            train_df = df[df.index < split_date].copy()
            test_df = df[df.index >= split_date].copy()

        else:
            split_idx = int(len(df) * train_ratio)
            train_df = df.iloc[:split_idx].copy()
            test_df = df.iloc[split_idx:].copy()

        if entire_df:
            return train_df, test_df

        # else:
        #     raise ValueError("È necessario fornire una split_date o un train_ratio.")

        X_train = train_df.drop(target_column, axis=1)
        y_train = train_df[target_column]
        X_test = test_df.drop(target_column, axis=1)
        y_test = test_df[target_column]

        return X_train, y_train, X_test, y_test

    def fit_scale_inputs(self, df_in):
        if self.inputs_scaler is None:
            pass
        else:
            self.inputs_scaler.fit(df_in)

    def fit_scale_target(self, df_in):
        if self.target_scaler is None:
            pass
        else:
            self.target_scaler.fit(df_in)

    def scale_inputs(self, df_in):
        if self.inputs_scaler is None:
            return df_in
        else:
            return self.inputs_scaler.transform(df_in)

    def scale_target(self, df_in):
        if self.target_scaler is None:
            return df_in
        else:
            return self.target_scaler.transform(df_in)

    def inverse_scale_target(self, df_in):
        if self.target_scaler is None:
            return df_in
        else:
            return self.target_scaler.inverse_transform(df_in)

    def fit(self, df_in: pd.DataFrame, preprocess: bool = True, cv=5):
        df_train = self.preprocess(df_in)
        X_train = df_train.drop(columns=self.target_variable).values
        y_train = df_train[self.target_variable].values.reshape(-1, 1)

        self.fit_scale_inputs(X_train)
        X_train_normalized = self.scale_inputs(X_train)
        self.fit_scale_target(y_train)
        y_train_normalized = self.scale_target(y_train)

        grid_search = GridSearchCV(self.model, self.params, cv=cv, n_jobs=-1, verbose=2)
        grid_search.fit(X_train_normalized, y_train_normalized)

        self.model = MLPRegressor(**grid_search.best_params_)
        self.model.fit(X_train_normalized, y_train_normalized)
        return self.model

    def inner_predict(self, df_in: pd.DataFrame) -> pd.Series:

        """
        Un passo della predizione ricorsiva ( "Predizione ad un passo")
        predico il valore target al tempo t in funzione di features relative istanti precedenti fino al tempo t

        :param : DataFrame contenente le feature necessarie al modello.
        """
        df_predict = self.preprocess(df_in)

        X_predict = df_predict.drop(columns=self.target_variable)

        X_predict_normalized = self.scale_inputs(X_predict)

        y_hat_nomalized = self.model.predict(X_predict_normalized)
        y_hat = self.inverse_scale_target(y_hat_nomalized.reshape(-1, 1))
        X_predict['y_predicted'] = y_hat

        df_out = df_in.join(X_predict[['y_predicted']])
        return df_out

    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Valuta il modello sui dati di test.

        :param X_test: DataFrame contenente le feature di test.
        :param y_test: Serie contenente il target di test.
        :return: Dizionario con le metriche di valutazione.
        """
        predictions = self.model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    def predict(self, df_in: pd.DataFrame, flag_all=False) -> pd.Series:
        df_predict = self.inner_predict(df_in)

        if flag_all:
            return df_predict

        #### predict the desired horizont
        #
        y_pred = []

        return y_pred


def create_features(df, fe_dict, fe_time_dict):
    forecast_model = RebeccaCustomModel()
    forecast_model.dict_fengin_from_time = fe_time_dict
    forecast_model.dict_fengin_from_inputs = fe_dict

    df_result = forecast_model.preprocess(df, drop_nulls=False, arange_vars=False)

    return df_result


def create_model(df, fe_dict, fe_time_dict, model_training, dir_save_model, file_name):
    # df.to_csv('tmp.csv', index=True, sep=';')
    # print(f"{fe_dict=}")
    # print(f"{fe_time_dict=}")
    # print(f"{model_training=}")
    col_target = model_training['target_column']

    fe_inputs_dict = {k_fe: {k: v for k, v in dict_fe.items() if k != col_target} for k_fe, dict_fe in fe_dict.items()}
    fe_target_dict = {k_fe: {k: v for k, v in dict_fe.items() if k == col_target} for k_fe, dict_fe in fe_dict.items()}

    if model_training['model_type'] == 'mlpregressor':
        forecast_model = RebeccaCustomModel()
        forecast_model.params = model_training['model_params']['mlpregressor']
    else:
        forecast_model = RebeccaCustomModel()
        forecast_model.params = model_training['model_params']['mlpregressor']

    forecast_model.target_variable = col_target
    forecast_model.input_variables = [el for el in df.columns if el != col_target]
    forecast_model.dict_fengin_from_time = fe_time_dict
    forecast_model.dict_fengin_from_inputs = fe_inputs_dict
    forecast_model.dict_fengin_from_target = fe_target_dict

    df_train, df_test = forecast_model.split_data_temporally(df, col_target, train_ratio=model_training['perc_split'] / 100)
    print(df_train.shape)
    print(df_test.shape)
    forecast_model.fit(df_train)
    df_train_with_prediction = forecast_model.predict(df_train, flag_all=True)
    df_test_with_prediction = forecast_model.predict(df_test, flag_all=True)

    with open(os.path.join(dir_save_model, file_name), 'wb') as handle:
        cloudpickle.dump(forecast_model, handle)

    df_train_with_prediction = df_train_with_prediction[df_train_with_prediction['y_predicted'].isnull()==False]
    df_test_with_prediction = df_test_with_prediction[df_test_with_prediction['y_predicted'].isnull()==False]
    return df_train_with_prediction[col_target], df_train_with_prediction['y_predicted'], \
        df_test_with_prediction[col_target], df_test_with_prediction['y_predicted']



if __name__ == '__main__':
    N_obs = 10

    forecast_model = RebeccaCustomModel()

    # forecast_model.model =
    forecast_model.input_variables = ['x1', 'x2']
    forecast_model.target_variable = 'target'
    forecast_model.dict_fengin_from_time = {
        "giorno_settimana": True,
        "ora": True,
        "mese": True,
        "fine_settimana": True,
        "numero_settimana": True,
        "giorno_festivo": True
    }
    forecast_model.dict_fengin_from_inputs = {"add_lags": {
        "x1": 1,
        "x2": 2
    }}
    forecast_model.dict_fengin_from_target = {"add_lags": 2}
    # forecast_model.model = MLPRegressor()
    # forecast_model.params = {}

    df = pd.DataFrame(np.arange(N_obs * 3).reshape((N_obs, 3)), columns=['x1', 'x2', 'target'])
    df.index = pd.to_datetime(
        dict(year=np.ones(N_obs) * 2024, month=np.ones(N_obs), day=np.ones(N_obs), hour=np.arange(N_obs)))

    forecast_model.fit(df)
