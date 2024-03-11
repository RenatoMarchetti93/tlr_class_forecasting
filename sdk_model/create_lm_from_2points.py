import cloudpickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import os


class RebeccaCustomModel(object):
    def __init__(self):
        '''
        Fill in the model specification as specified in the base_sdk_doc file
        '''

        self.name = 'Regressor Example'
        self.input_variables = ['Var_X1_name']
        self.output_variables = ['Target name']
        self.task = 'regression'
        self.metrics = {'rmse': None}
        self.model = LinearRegression()
        # standard deviation of the target variable
        self.sigma = {'Target': None}

    def predict(self, x):
        '''
        Called when the model is used to make predictions.
        This method should return a prediction for each row of x.
        '''
        return self.model.predict(x)

    def fit(self, x, y):
        '''
        Called when the model is retrained.
        Return self.
        '''
        self.model.fit(x, y)
        return self


def create_data_from_2_points(point_1, point_2, n=100):
    x1, y1 = point_1
    x2, y2 = point_2

    delta_x = (x2 - x1)/n
    delta_y = (y2 - y1)/n

    x_result = [[x1+delta_x*i] for i in range(n+1)]
    y_result = [y1+delta_y*i for i in range(n+1)]

    return np.array(x_result), np.array(y_result)


def get_fit_metrics(y_true, y_pred):
    msqe = mean_squared_error(y_true, y_pred)
    sigma_target = np.std(y_true-y_pred)

    return msqe, sigma_target


def create_save_model(x1, y1, x2, y2, nome_modello, nome_var_x, nome_var_target, dir_save_model, file_name):
        # Instantiate model
        rebecca_model = RebeccaCustomModel()
        rebecca_model.name = nome_modello
        rebecca_model.input_variables = [nome_var_x]
        rebecca_model.output_variables = [nome_var_target]

        # Create data to fit
        X, Y = create_data_from_2_points((x1, y1), (x2, y2))
        rebecca_model.fit(X, Y)

        # metrics
        # msqe, sigma_target = get_fit_metrics(Y, rebecca_model.predict(X))

        # model_dt = f"{datetime.datetime.now()}".split('.')[0][:-3]
        # model_dt = model_dt.replace(" ", "_").replace(":", "").replace("-", "")
        #
        # # dir_save_model = str(Path.home() / "Downloads")
        # # if not os.path.exists(dir_save_model):
        # #     os.mkdir(dir_save_model)
        #
        # file_name = f'{nome_modello}_{model_dt}.pkl'
        with open(os.path.join(dir_save_model, file_name), 'wb') as handle:
            cloudpickle.dump(rebecca_model, handle)

        return X, Y


# This is the function that will be called when running this file.
if __name__ == '__main__':
    import cloudpickle as cp
    from pathlib import Path
    import os
    import datetime

    nome_modello = 'nome_modello'

    # create_data_from_2_points((0,10), (10, 20), 10)
    # create_data_from_2_points((10,0), (20, 10), 10)

    rebecca_model = RebeccaCustomModel()
    # Create data to fit
    X, Y = create_data_from_2_points((0, 10), (10, 20), 10)
    rebecca_model.fit(X, Y)
    y_pred = rebecca_model.predict(X)

    dir_save_model = str(Path.home() / "Downloads")

    model_dt = f"{datetime.datetime.now()}".split('.')[0][:-3]
    model_dt = model_dt.replace(" ", "_").replace(":", "").replace("-", "")

    with open(os.path.join(dir_save_model, f'{nome_modello} {model_dt}.pkl'), 'wb') as handle:
        cp.dump(rebecca_model, handle)  # cloudpickle

    print('END')

