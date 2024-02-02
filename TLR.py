import pandas as pd 
import numpy as np 
import pickle as pk 
import holidays 
import datetime as dt
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import joblib

import dill 

class Preprocessing():

    """
    Inizializza la classe con un DataFrame e un dizionario di features.
    :param df: DataFrame da manipolare ( index datetime con Frequenza oraria )
    :param dict_features: Dizionario con le specifiche delle feature da calcolare.
    """

    def __init__(self,df: pd.DataFrame, dict_features: dict):
        self.df = df
        self.dict_features = dict_features
        self.time_shifts = [] # numero di ore indietro

    @staticmethod
    def _is_holiday(date):

        """
        Verifica se una data è un giorno festivo in Italia.
        :param date: Data da verificare.
        :return: True se la data è un giorno festivo, altrimenti False.
        """

        return date in holidays.Italy()
    
    def add_datetime_info(self) -> pd.DataFrame:
        """
        Aggiunge informazioni temporali al DataFrame, come giorno della settimana, ora, mese, ecc.
        :return: DataFrame arricchito con informazioni temporali.

        """

        df = self.df.copy()
        df['giorno_settimana'] = df.index.dayofweek
        df['ora'] = df.index.hour
        df['mese'] = df.index.month
        df['fine_settimana'] = (df['giorno_settimana'] >= 5).astype(int)
        df['numero_settimana'] = df.index.isocalendar().week
        df['giorno_festivo'] = df.index.map(self._is_holiday).astype(int)
        return df

    def add_lagged_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggiunge al DataFrame le feature lagged in base al dizionario di features.
        :param df: DataFrame a cui aggiungere le feature.
        :return: DataFrame arricchito con le feature lagged.
        """
        for var, (n_lag, _, _) in self.dict_features.items():

            self.time_shifts.append(n_lag)

            for lag in range(1, n_lag + 1):
                df[f'{var}_lag_{lag}'] = df[var].shift(lag)

        return df
    
    def add_rolling_stats(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Aggiunge statistiche rolling al DataFrame, come mediana, deviazione standard, ecc.
        :param df: DataFrame a cui aggiungere le statistiche.
        :return: DataFrame arricchito con statistiche rolling.
        """
        for var, (_, h_window, _) in self.dict_features.items():

            # h_window = numero di ore dell'intervallo da considerare per costruire le statistiche.
            # l'intervallo ha il tempo t come estremo destro (compreso) 

            self.time_shifts.append(h_window - 1)

            if var != target:

                rolling_window = df[var].rolling(f'{h_window}H', min_periods=h_window)
                df[f'median_rolling_{var}'] = rolling_window.median()
                df[f'dev_std_rolling_{var}'] = rolling_window.std()
                df[f'mean_rolling_{var}'] = rolling_window.mean()
                df[f'max_rolling_{var}'] = rolling_window.max()
                df[f'min_rolling_{var}'] = rolling_window.min()
                df[f'perc_25_rolling_{var}'] = rolling_window.quantile(0.25)
                df[f'perc_75_rolling_{var}'] = rolling_window.quantile(0.75)
            
            else:
                var = target + "_lag_1"
                rolling_window = df[var].rolling(f'{h_window-1}H', min_periods=h_window-1)
                df[f'median_rolling_{var}'] = rolling_window.median()
                df[f'dev_std_rolling_{var}'] = rolling_window.std()
                df[f'mean_rolling_{var}'] = rolling_window.mean()
                df[f'max_rolling_{var}'] = rolling_window.max()
                df[f'min_rolling_{var}'] = rolling_window.min()
                df[f'perc_25_rolling_{var}'] = rolling_window.quantile(0.25)
                df[f'perc_75_rolling_{var}'] = rolling_window.quantile(0.75)
        
        self.target = target
        return df
    
    def add_seasonal_stats(self, df: pd.DataFrame, target: str) -> pd.DataFrame:
        """
        Aggiunge statistiche stagionali al DataFrame.
        :param df: DataFrame a cui aggiungere le statistiche stagionali. ( al tempo t, t-1*24, ...,  t-M_period*24)
        :return: DataFrame arricchito con statistiche stagionali.
        """

        for var, (_, _, M_period) in self.dict_features.items():
            self.time_shifts.append(M_period*24)

            if var != target:
                seasonal_window = df[var].rolling(window = M_period, min_periods=M_period)
                df[f'min_seasonal_{var}'] = seasonal_window.min()
                df[f'max_seasonal_{var}'] = seasonal_window.max()
                df[f'mean_seasonal_{var}'] = seasonal_window.mean()

            else:
                var = target+ "_lag_1"
                seasonal_window = df[var].rolling(window = M_period, min_periods = M_period)
                df[f'min_seasonal_{var}'] = seasonal_window.min()
                df[f'max_seasonal_{var}'] = seasonal_window.max()
                df[f'mean_seasonal_{var}'] = seasonal_window.mean()

        self.target = target
        return df
    
    def get_time_shifts(self):
            """
            Restituisce la lista degli shift temporali per ogni variabile.
            :return: lista degli shift temporali.
            """
            return self.time_shifts

    def preprocess(self, target: str) -> pd.DataFrame:

        """
        Esegue l'intero processo di preprocessing sul DataFrame.
        :return: DataFrame preprocessato.
        """

        df = self.add_datetime_info()
        df = self.add_lagged_features(df)
        df = self.add_rolling_stats(df, target)
        df = self.add_seasonal_stats(df, target)

        df = df.dropna()
        self.df = df
        return df

    def split_data_temporally(self, df: pd.DataFrame, target_column: str, split_date: str = None, train_ratio: float = None):
        ################################# QUI #############################
        """
        Divide il DataFrame in set di allenamento e test basandosi su una divisione temporale.

        :param df: DataFrame completo.
        :param target_column: Nome della colonna target.
        :param split_date: Data specifica per dividere i dati (formato 'YYYY-MM-DD').
        :param train_ratio: Percentuale di dati da usare per l'allenamento (es. 0.8 per 80%).
        :return: Tuple (X_train, y_train, X_test, y_test).
        """
        if split_date:
            train_df = df[df.index < split_date]
            test_df = df[df.index >= split_date]

        elif train_ratio:
            split_idx = int(len(df) * train_ratio)
            train_df = df.iloc[:split_idx]
            test_df = df.iloc[split_idx:]

        else:
            raise ValueError("È necessario fornire una split_date o un train_ratio.")

        X_train = train_df.drop(target_column, axis=1)
        y_train = train_df[target_column]
        X_test = test_df.drop(target_column, axis=1)
        y_test = test_df[target_column]

        return X_train, y_train, X_test, y_test

    def fit_grid(self, models_params: list, X_train: pd.DataFrame, y_train: pd.Series, cv=5):
        
        # creare attributo model params in modo da non passarlo nel fit_grid

        """
        Esegue Grid Search su più modelli per trovare il miglior set di iperparametri.

        :param models_params: Lista di tuple, ciascuna contenente un modello e la sua griglia di parametri.
        :param X_train: DataFrame contenente le feature di allenamento.
        :param y_train: Serie contenente il target di allenamento.
        :param cv: Numero di fold per la cross-validation.
        :return: Il miglior modello e i suoi parametri.
        """
        best_score = float('-inf')
        for model, params in models_params:

            grid_search = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=2)
            grid_search.fit(X_train, y_train)

            if grid_search.best_score_ > best_score:

                best_score = grid_search.best_score_
                self.best_model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_

        self.model = self.best_model

        return self.model, self.best_params

    def inner_predict(self, X_test: pd.DataFrame) -> pd.Series:

        """
        Un passo della predizione ricorsiva ( "Predizione ad un passo")
        predico il valore target al tempo t in funzione di features relative istanti precedenti fino al tempo t

        :param X_test: DataFrame contenente le feature necessarie al modello.
        """
        return self.model.predict(X_test)

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
        r2  = r2_score(y_test, predictions)

        return {'MSE': mse, 'MAE': mae, 'R2': r2}

    # def pre_predict()
    def predict(self, df: pd.DataFrame, start:str, horizon_window: int) -> pd.Series:
        
        y_pred = []
        max_hour_back = self.time_shifts
        ind = df.loc[pd.to_datetime(start)-pd.Timedelta(hours = max_hour_back):pd.to_datetime(start) + pd.Timedelta(hours = horizon_window)].index
        # da tradurre in indici numerici e non temporali
        for i in ind:

            for var, (n_lag, h_window, M_period) in self.dict_features.items():

                for lag in range(1, n_lag + 1):
                    df.loc[i,f'{var}_lag_{lag}'] = df.loc[i - pd.Timedelta(hours=lag), var]

                df.loc[i, f'median_rolling_{self.target}_lag_1'] = df.loc[(i - pd.Timedelta(hours=h_window-1)):i, self.target + '_lag_1'].median()
                df.loc[i, f'dev_std_rolling_{self.target}_lag_1'] = df.loc[i - pd.Timedelta(hours=h_window-1):i, self.target + '_lag_1'].std()
                df.loc[i, f'media_rolling_{self.target}_lag_1'] = df.loc[i - pd.Timedelta(hours=h_window-1):i, self.target + '_lag_1'].mean()

                
                if var != self.target:

                    rolling_window = df[var].rolling(f'{h_window}H', min_periods=h_window)
                    df.loc[i,f'median_rolling_{var}'] = rolling_window.median()
                    df.loc[i,f'dev_std_rolling_{var}'] = rolling_window.std()
                    df.loc[i,f'mean_rolling_{var}'] = rolling_window.mean()
                    df.loc[i,f'max_rolling_{var}'] = rolling_window.max()
                    df.loc[i,f'min_rolling_{var}'] = rolling_window.min()
                    df.loc[i,f'perc_25_rolling_{var}'] = rolling_window.quantile(0.25)
                    df.loc[i,f'perc_75_rolling_{var}'] = rolling_window.quantile(0.75)
                    
                    seasonal_window = df[var].rolling(window = M_period, min_periods=M_period)
                    df.loc[i,f'min_seasonal_{var}'] = seasonal_window.min()
                    df.loc[i,f'max_seasonal_{var}'] = seasonal_window.max()
                    df.loc[i,f'mean_seasonal_{var}'] = seasonal_window.mean()

                else:
                    var = self.target + "_lag_1"

                    rolling_window = df[var].rolling(f'{h_window-1}H', min_periods=h_window-1)
                    df.loc[f'median_rolling_{var}'] = rolling_window.median()
                    df.loc[f'dev_std_rolling_{var}'] = rolling_window.std()
                    df.loc[f'mean_rolling_{var}'] = rolling_window.mean()
                    df.loc[f'max_rolling_{var}'] = rolling_window.max()
                    df.loc[f'min_rolling_{var}'] = rolling_window.min()
                    df.loc[f'perc_25_rolling_{var}'] = rolling_window.quantile(0.25)
                    df.loc[f'perc_75_rolling_{var}'] = rolling_window.quantile(0.75)
                    
                    seasonal_window = df[var].rolling(window = M_period, min_periods = M_period)
                    df.loc[i,f'min_seasonal_{var}'] = seasonal_window.min()
                    df.loc[i,f'max_seasonal_{var}'] = seasonal_window.max()
                    df.loc[i,f'mean_seasonal_{var}'] = seasonal_window.mean()

            y_pred = y_pred.append(self.model.predict(df.loc[i,:].values.reshape(1,-1)))

        y_pred = pd.Series(y_pred, name = self.target)

        return y_pred


    def save_model(self, path: str):
        """
        Salva il modello in un file.

        :param path: Percorso del file in cui salvare il modello.
        """
        joblib.dump(self.model, path)

    def load_model(self, path: str):
        """
        Carica un modello da un file.

        :param path: Percorso del file da cui caricare il modello.
        """
        self.model = joblib.load(path)




# class ForecastingModelTrainer:
#     """
#     Classe per gestire l'allenamento di modelli di forecasting, supportando MLP e XGBoost,
#     e per eseguire Grid Search per trovare il miglior set di iperparametri.
#     """

#     def split_data_temporally(self, df: pd.DataFrame, target_column: str, split_date: str = None, train_ratio: float = None):
#         ################################# QUI #############################
#         """
#         Divide il DataFrame in set di allenamento e test basandosi su una divisione temporale.

#         :param df: DataFrame completo.
#         :param target_column: Nome della colonna target.
#         :param split_date: Data specifica per dividere i dati (formato 'YYYY-MM-DD').
#         :param train_ratio: Percentuale di dati da usare per l'allenamento (es. 0.8 per 80%).
#         :return: Tuple (X_train, y_train, X_test, y_test).
#         """
#         if split_date:
#             train_df = df[df.index < split_date]
#             test_df = df[df.index >= split_date]

#         elif train_ratio:
#             split_idx = int(len(df) * train_ratio)
#             train_df = df.iloc[:split_idx]
#             test_df = df.iloc[split_idx:]

#         else:
#             raise ValueError("È necessario fornire una split_date o un train_ratio.")

#         X_train = train_df.drop(target_column, axis=1)
#         y_train = train_df[target_column]
#         X_test = test_df.drop(target_column, axis=1)
#         y_test = test_df[target_column]

#         return X_train, y_train, X_test, y_test

#     # def _initialize_model(self):
#     #     """
#     #     Inizializza il modello in base al tipo specificato.

#     #     :return: Il modello inizializzato.
#     #     """
#     #     if self.model_type == 'mlp':
#     #         return MLPRegressor()
#     #     elif self.model_type == 'xgboost':
#     #         return XGBRegressor()
#     #     else:
#     #         raise ValueError("Tipo di modello non supportato. Scegliere tra 'mlp' o 'xgboost'.")

#     # def fit(self, X_train: pd.DataFrame, y_train: pd.Series):

#     #     """
#     #     Allena il modello sui dati forniti.

#     #     :param X_train: DataFrame contenente le feature di allenamento.
#     #     :param y_train: Serie contenente il target di allenamento.
#     #     """
#     #     self.model.fit(X_train, y_train)

#     def fit_grid(self, models_params: list, X_train: pd.DataFrame, y_train: pd.Series, cv=5):

#         """
#         Esegue Grid Search su più modelli per trovare il miglior set di iperparametri.

#         :param models_params: Lista di tuple, ciascuna contenente un modello e la sua griglia di parametri.
#         :param X_train: DataFrame contenente le feature di allenamento.
#         :param y_train: Serie contenente il target di allenamento.
#         :param cv: Numero di fold per la cross-validation.
#         :return: Il miglior modello e i suoi parametri.
#         """
#         best_score = float('-inf')
#         for model, params in models_params:

#             grid_search = GridSearchCV(model, params, cv=cv, n_jobs=-1, verbose=2)
#             grid_search.fit(X_train, y_train)

#             if grid_search.best_score_ > best_score:

#                 best_score = grid_search.best_score_
#                 self.best_model = grid_search.best_estimator_
#                 self.best_params = grid_search.best_params_

#         self.model = self.best_model

#         return self.model, self.best_params

#     def inner_predict(self, X_test: pd.DataFrame) -> pd.Series:

#         """
#         Un passo della predizione ricorsiva ( "Predizione ad un passo")
#         predico il valore target al tempo t in funzione di features relative istanti precedenti fino al tempo t

#         :param X_test: DataFrame contenente le feature necessarie al modello.
#         """
#         return self.model.predict(X_test)
    
#     def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:

#         """
#         Valuta il modello sui dati di test.

#         :param X_test: DataFrame contenente le feature di test.
#         :param y_test: Serie contenente il target di test.
#         :return: Dizionario con le metriche di valutazione.
#         """
#         predictions = self.model.predict(X_test)
#         mse = mean_squared_error(y_test, predictions)
#         mae = mean_absolute_error(y_test, predictions)
#         r2  = r2_score(y_test, predictions)

#         return {'MSE': mse, 'MAE': mae, 'R2': r2}

#     # def evaluate_forecasting_model(self, X: pd.DataFrame, y: pd.Series, start_window: str , horizon_window: int):

#     #     for var in target_stats:
#     #         df.loc[i, f'median_rolling_{var}_lag_1'] = df.loc[(i - pd.Timedelta(hours=23)):i, var + '_lag_1'].median()
#     #         df.loc[i, f'dev_std_rolling_{var}_lag_1'] = df.loc[i - pd.Timedelta(hours=23):i, var + '_lag_1'].std()
#     #         df.loc[i, f'media_rolling_{var}_lag_1'] = df.loc[i - pd.Timedelta(hours=23):i, var + '_lag_1'].mean()


        


#     def save_model(self, path: str):
#         """
#         Salva il modello in un file.

#         :param path: Percorso del file in cui salvare il modello.
#         """
#         joblib.dump(self.model, path)

#     def load_model(self, path: str):
#         """
#         Carica un modello da un file.

#         :param path: Percorso del file da cui caricare il modello.
#         """
#         self.model = joblib.load(path)



## MAIN ##

dict_var_lag = {'T2M': (5,24,4), 'RH2M': (4,24,4), 'PRECTOTCORR': (4,24,4), 'PS': (4,24,4)}


path = r'Meteo_1Gen_Dic2023.csv'
df = pd.read_csv(path)
df.Datetime = pd.to_datetime(df.Datetime)
df.set_index('Datetime', inplace = True)
df.drop(columns = 'Unnamed: 0', inplace = True)
prep = Preprocessing(df,dict_var_lag)
df_prep = prep.preprocess('T2M')
# time_shifts = prep.get_time_shifts()
# # max(time_shifts)
# # df_prep = df
# # # df_prep.to_excel('prova_classe_Preprocessing.xlsx')
model_params = [(MLPRegressor(), {
                                'hidden_layer_sizes': [(100,)],
                                'activation': ['tanh'],
                                'solver': ['adam'],
                                'alpha': [ 0.05],
                                'learning_rate': ['constant'],
                                'batch_size': [32, ]
                                }
)]
x = 1
(0.53*0.00369)*0.914*(x) + 0.935*(1-x)
X_train, y_train, X_test, y_test = prep.split_data_temporally(df_prep, prep.target, '2023-10-30')
model, best_params = prep.fit_grid(model_params, X_train, y_train)

# FINO a qui funziona, testare il resto!