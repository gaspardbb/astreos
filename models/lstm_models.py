import itertools
import os
import shutil

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

from models.sk_model import *
from utils import utils
from utils.utils import load_obj

DEFAULT_BATCH_SIZE = 200
DEFAULT_HIDDEN_DIM = 16
DEFAULT_NUM_LAYERS = 2
DEFAULT_LOSS = nn.MSELoss()
DEFAULT_NUM_EPOCHS = 40
DEFAULT_LR = 5e-3
ROOT = utils.get_project_root()
LSTM_MODEL_PATH = os.path.join(ROOT, 'models/saved_models/lstm_models')


def CAPE_loss(y_pred, y_true):
    loss = 100 * (y_pred - y_true).abs().sum() / y_true.sum()
    return loss


def preprocess_lstm(features_df, shifts=3):
    """ Take features DataFrame and build array with each line containing past information + current one
        Parameters
        ----------
        features_df: pd.DataFrame
            features table for one WF (preferably should not contain diff column)
        shifts: int
            number of past elements that will be taken into account for current prediction

        Returns
        -------
        train_X: np.array
            array of shape (nb of entries, shift + 1 (past + current), nb of features
    """
    aux_df = features_df.copy()
    train_df = aux_df.to_numpy()
    for i in range(1, shifts + 1):
        sh_df = aux_df.shift(i, fill_value=0).to_numpy()
        train_df = np.hstack((train_df, sh_df))
    train_df = train_df.reshape((len(features_df.index), shifts + 1, -1))
    train_df = train_df[:, ::-1]  # put in chronological order
    return train_df


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=3, dropout=0.5, batch_first=True):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.drop_prob = dropout
        self.batch_first = batch_first

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, dropout=self.drop_prob,
                            batch_first=self.batch_first)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self, X=None):
        # This is what we'll initialise our hidden state as
        batch_size = self.batch_size if X is None else X.shape[0 if self.batch_first else -1]
        return (torch.zeros(self.num_layers, batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, batch_size, self.hidden_dim))

    def forward(self, x):
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(x)

        # Only take the output from the final timestep
        y_pred = torch.relu(self.linear(lstm_out[:, -1]))
        return y_pred.view(-1)


class LstmRegressor(object):
    """ Apply sklearn regressor to each windfarm separately """

    def __init__(self, lstm_configs, shift, id='mean', train_valid_ratio=0.2, valid_test_ratio=0.5, verbose=1):
        """ Initialize model, notably specifying model parameters

            Parameters
            ----------
            lstm_configs: dict[str: List]
                dict containing parameters to test for the lstm model
                possible keys: "hidden_dim" (16), "num_layers" (2), "num_epochs" (40), "dropout" (0.5),
                               "loss" (torch.nn.MSELoss()), "lr" (5e-3), "batch_size" (200)
            shift: int
                Number of past elements to take into account in the lstm model
            id: str
                identify saved models (could account for the features selected)
            train_valid_ratio: float
                ratio applied for fit
            valid_test_ratio: float
                ratio applied for fit
        """

        self.train_valid_ratio = train_valid_ratio
        self.valid_test_ration = valid_test_ratio
        self.lstm_configs = lstm_configs
        self.shift = shift
        self.id = id

        self.scalers = {}
        self.best_models = []
        self.best_scores = []

        self.dir_path = os.path.join(LSTM_MODEL_PATH, 'LSTM_' + id)  # path of the lstm regressors
        self.verbose = verbose

        if not os.path.exists(self.dir_path):
            os.mkdir(self.dir_path)

    def fit(self, X, Y=None):
        """ Require X to be the full dataFrame with all WF and column production """

        full_df_wfs = list(map(lambda features_df: features_df.drop('Production', axis=1), split_data_wf(X)))
        prod_df_wfs = list(map(lambda features_df: features_df['Production'], split_data_wf(X)))

        for i, full_df_wf, prod_df_wf in zip(*(np.arange(1, len(full_df_wfs) + 1), full_df_wfs, prod_df_wfs)):
            self.best_scores.append(np.inf)
            self.best_models.append(None)

            # ----------  data conditioning  ---------- #
            # fill na
            full_df_wf = handle_nan(full_df_wf)

            # split_dataset data between train / valid / test
            train_X_df, valid_X_df, test_X_df, train_Y, valid_Y, test_Y = split_dataset(full_df_wf, prod_df_wf,
                                                                                        self.train_valid_ratio,
                                                                                        self.valid_test_ration)
            # scale
            scaler = StandardScaler()
            train_X_df[:] = scaler.fit_transform(train_X_df)
            valid_X_df[:], test_X_df[:] = scaler.transform(valid_X_df), scaler.transform(test_X_df)
            self.scalers[i] = scaler

            # shift
            train_X = preprocess_lstm(train_X_df, shifts=self.shift)
            valid_X, test_X = preprocess_lstm(valid_X_df, shifts=self.shift), preprocess_lstm(test_X_df,
                                                                                              shifts=self.shift)

            # remove lines containing nan in predictions
            train_X, train_Y = handle_nan_Y(train_X, train_Y, self.verbose)
            valid_X, valid_Y = handle_nan_Y(valid_X, valid_Y, self.verbose)
            test_X, test_Y = handle_nan_Y(test_X, test_Y, self.verbose)

            # torchize
            train_data = TensorDataset(torch.from_numpy(train_X.copy()).float(), torch.from_numpy(train_Y.copy()))
            val_data = TensorDataset(torch.from_numpy(valid_X.copy()).float(), torch.from_numpy(valid_Y.copy()))
            test_data = TensorDataset(torch.from_numpy(test_X.copy()).float(), torch.from_numpy(test_Y.copy()))

            model_path = os.path.join(self.dir_path, str(i))
            try:
                os.mkdir(model_path)
            except FileExistsError as e:
                logger.info(f"File {model_path} already exists, replace it with new model")
                shutil.rmtree(model_path)
                os.mkdir(model_path)
            tmp_model_path = os.path.join(model_path, 'tmp')
            os.mkdir(tmp_model_path)  # save temporary trained models here, will be removed

            # iterate on model configurations
            keys, values = zip(*self.lstm_configs.items())
            for v in itertools.product(*values):
                experiment_param = dict(zip(keys, v))
                if "batch_size" not in experiment_param:
                    experiment_param["batch_size"] = DEFAULT_BATCH_SIZE
                if "hidden_dim" not in experiment_param:
                    experiment_param["hidden_dim"] = DEFAULT_HIDDEN_DIM
                if "num_layers" not in experiment_param:
                    experiment_param["num_layers"] = DEFAULT_NUM_LAYERS
                batch_size = experiment_param["batch_size"]
                if self.verbose > 0:
                    logger.info(f"{i} - {experiment_param} - current best score: {self.best_scores[-1]:2f}")
                # loss, lr, num_epochs should not be fed into the model
                loss = DEFAULT_LOSS if 'loss' not in experiment_param else experiment_param.pop('loss')
                lr = DEFAULT_LR if 'lr' not in experiment_param else experiment_param.pop('lr')
                num_epochs = DEFAULT_NUM_EPOCHS if 'num_epochs' not in experiment_param else experiment_param.pop(
                    'num_epochs')

                train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

                lstm_input_size = train_X.shape[-1]
                model_wf = LSTM(lstm_input_size, output_dim=1, batch_first=True, **experiment_param)

                train_model(model_wf, train_loader, val_data, loss, tmp_model_path, num_epochs=num_epochs, lr=lr,
                            verbose=self.verbose)

                # Loading the best model to evaluate on test
                model_wf.load_state_dict(torch.load(os.path.join(tmp_model_path, 'state_dict.pt')))
                model_wf.eval()
                test_score = CAPE_loss(model_wf(test_data.tensors[0]).squeeze(), test_data.tensors[1]).item()
                if self.verbose > 1:
                    logger.info("Test loss: {:.3f}".format(test_score))

                if test_score < self.best_scores[-1]:
                    if self.verbose > 1:
                        logger.info(f'Save new model for WF {i}...')
                    torch.save(model_wf.state_dict(), os.path.join(model_path, 'state_dict.pt'))
                    model_config = {'shift': self.shift, 'scaler': self.scalers[i]}
                    model_config.update(experiment_param)
                    utils.save_obj(model_path, model_config, 'config')
                    self.best_models[-1] = model_wf
                    self.best_scores[-1] = test_score
            # remove tmp file
            shutil.rmtree(tmp_model_path)

    def predict(self, X):
        """ X should be full feature DataFrame for all WF """

        full_df_wfs = split_data_wf(X)
        if 'Production' in full_df_wfs[0].columns:
            full_df_wfs = list(map(lambda features_df: features_df.drop('Production', axis=1), full_df_wfs))

        predictions = []

        for i, full_df_wf in enumerate(full_df_wfs):
            i += 1
            full_df_wf = full_df_wf.copy()

            model_path = os.path.join(self.dir_path, str(i))
            model_config = load_obj(model_path, 'config')
            shift = model_config.pop('shift')
            assert shift == self.shift

            # preprocess full_df_wf
            full_df_wf = handle_nan(full_df_wf)  # fill na
            scaler = model_config.pop('scaler')
            full_df_wf[:] = scaler.transform(full_df_wf)  # scale
            X_wf = preprocess_lstm(full_df_wf, shifts=self.shift)  # shift
            X_wf = torch.from_numpy(X_wf.copy()).float()  # torchize
            lstm_input_size = X_wf.shape[-1]

            # load best model, model forward to get predictions
            if len(self.best_models) > 0:
                model_wf = self.best_models[i - 1]
            else:
                model_wf = LSTM(lstm_input_size, output_dim=1, batch_first=True, **model_config)
            model_wf.load_state_dict(torch.load(os.path.join(model_path, 'state_dict.pt')))
            model_wf.eval()
            prediction_wf = model_wf(X_wf).detach().numpy()

            predictions.append(prediction_wf)

        return predictions


def handle_nan_Y(X, Y, verbose=0):
    if np.any(np.isnan(Y)):
        if verbose > 0:
            logger.info(f'{np.isnan(Y).sum()} nan found in dataset, removing lines')
        X = X[~np.isnan(Y)]
        Y = Y[~np.isnan(Y)]
    return X, Y


def train_model(model, train_loader, val_data, loss, save_path, num_epochs, lr, print_every=100,
                verbose=1):
    """ Train LSTM model on training batches from train_loader and validate on val_loader """
    model.train()
    loss_fn = loss
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    valid_loss_min = np.Inf

    counter = 0

    for t in range(num_epochs):
        # Initialise hidden state
        model.hidden = model.init_hidden()

        for x_train, y_train in train_loader:
            counter += 1
            model.zero_grad()

            # Forward pass
            y_pred = model(x_train).double()
            loss = loss_fn(y_pred, y_train)
            # Zero out gradient, else they will accumulate between epochs
            optimiser.zero_grad()

            # Backward pass
            loss.backward()

            # Update parameters
            # nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimiser.step()

            if counter % print_every == 0:
                val_h = model.init_hidden()
                model.eval()
                val_loss = CAPE_loss(model(val_data.tensors[0]).squeeze(), val_data.tensors[1])

                model.train()
                if verbose > 0:
                    logger.info("Epoch: {}/{}...".format(t + 1, num_epochs))
                    logger.info("Step: {}...".format(counter))
                    logger.info("Loss: {:.6f}...".format(loss.item()))
                    logger.info("Val Loss: {:.6f}".format(val_loss))
                if val_loss < valid_loss_min:
                    torch.save(model.state_dict(), os.path.join(save_path, 'state_dict.pt'))
                    if verbose > 0:
                        logger.info(
                            'Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,
                                                                                                      val_loss))
                    valid_loss_min = val_loss

    return


def split_dataset(full_X_df, full_Y_df, train_valid_ratio=0.8, valid_test_ratio=0.5):
    train_X_df, test_X_df, train_Y_df, test_Y_df = train_test_split(full_X_df, full_Y_df, train_size=train_valid_ratio,
                                                                    shuffle=False)
    valid_X_df, test_X_df, valid_Y_df, test_Y_df = train_test_split(test_X_df, test_Y_df, train_size=valid_test_ratio,
                                                                    shuffle=False)
    train_X_df, valid_X_df, test_X_df = train_X_df.copy(), valid_X_df.copy(), test_X_df.copy()
    train_Y, valid_Y, test_Y = train_Y_df.values, valid_Y_df.values, test_Y_df.values

    return train_X_df, valid_X_df, test_X_df, train_Y, valid_Y, test_Y


if __name__ == '__main__':
    df, target = load_data()
    features = all_features(df, get_diff=[], test_set=True)

    full_df = pd.concat([features, target], axis=1)

    lstm_configs = {'loss': [nn.MSELoss(), CAPE_loss, nn.L1Loss()], 'lr': [5e-3, 1e-3, 5e-4],
                    'batch_size': [64, 128, 256]}

    lstm_regressor = LstmRegressor(lstm_configs, shift=12, id="mean_12")
    lstm_regressor.fit(full_df)

    logger.info(f"Done... got scores {lstm_regressor.best_scores} (mean: {np.mean(lstm_regressor.best_scores)}")
