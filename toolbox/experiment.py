import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from functools import partial
from pathlib import Path
import time
import numpy as np
import pandas as pd
import netCDF4
from keras import backend as K
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.optimizers import adam
from toolbox.data import load_data_pair
from toolbox.data import load_data_single
from toolbox.metrics import psnr
from toolbox.models import bicubic
from toolbox.paths import data_dir



class Experiment(object):


    def __init__(self, scale=3, load_set=None, build_model=None, optimizer='adam', save_dir='.'):
        self.scale = scale
        self.load_set = partial(load_set, scale=scale)
        self.build_model = partial(build_model, scale=scale)
        self.optimizer = optimizer
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.save_dir / 'config.yaml'
        self.model_file = self.save_dir / 'model.hdf5'
        self.train_dir = self.save_dir / 'train'
        self.train_dir.mkdir(exist_ok=True)
        self.history_file = self.train_dir / 'history.csv'
        self.weights_dir = self.train_dir / 'weights'
        self.weights_dir.mkdir(exist_ok=True)
        self.test_dir = self.save_dir / 'test'
        self.test_dir.mkdir(exist_ok=True)
        self.apply_dir = self.save_dir / 'apply'
        self.apply_dir.mkdir(exist_ok=True)


    def weights_file(self, epoch=None):
        if epoch is None:
            return self.weights_dir / 'ep{epoch:04d}.hdf5'
        else:
            return self.weights_dir / f'ep{epoch:04d}.hdf5'


    @property
    def latest_epoch(self):
        try:
            return pd.read_csv(str(self.history_file))['epoch'].iloc[-1]
        except (FileNotFoundError, pd.io.common.EmptyDataError):
            pass
        return -1


    def _ensure_dimension(self, array, dim):
        while len(array.shape) < dim:
            array = array[np.newaxis, ...]
        return array


    def _ensure_channel(self, array, c):
        return array[..., c:c+1]


    def pre_process(self, array):
        array = self._ensure_dimension(array, 4)
        array = self._ensure_channel(array, 0)
        return array


    def post_process(self, array, auxiliary_array):
        array = np.concatenate([array, auxiliary_array[..., 1:]], axis=-1)
        return array


    def inverse_post_process(self, array):
        array = self._ensure_dimension(array, 4)
        array = self._ensure_channel(array, 0)
        return array


    def compile(self, model):
        """Compile model with default settings."""
        model.compile(optimizer=self.optimizer, loss='mse', metrics=[psnr])
        return model


    def train(self, train_set, dev_set, epochs=1, resume=True):

        # Load and process data
        x_train, y_train = self.load_set(train_set)
        x_dev, y_dev = self.load_set(dev_set)
        x_train, x_dev = [self.pre_process(x)
                          for x in [x_train, x_dev]]
        y_train, y_dev = [self.inverse_post_process(y)
                          for y in [y_train, y_dev]]

        # Compile model
        model = self.compile(self.build_model(x_train))
        model.summary()

        # Save model architecture
        # Currently in Keras 2 it's not possible to load a model with custom
        # layers. So we just save it without checking consistency.
        self.config_file.write_text(model.to_yaml())

        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
        # Inherit weights
        if resume:
            latest_epoch = self.latest_epoch
            if latest_epoch > -1:
                weights_file = self.weights_file(epoch=latest_epoch)
            initial_epoch = latest_epoch + 1
        else:
            initial_epoch = 0

        # Set up callbacks
        callbacks = []
        callbacks += [ModelCheckpoint(str(self.model_file))]
        callbacks += [ModelCheckpoint(str(self.weights_file()),
                                      save_weights_only=True)]
        callbacks += [CSVLogger(str(self.history_file), append=resume)]

        # Train
        model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks,
                  validation_data=(x_dev, y_dev), initial_epoch=initial_epoch)

        # Plot metrics history
        prefix = str(self.history_file).rsplit('.', maxsplit=1)[0]
        df = pd.read_csv(str(self.history_file))
        epoch = df['epoch']
        for metric in ['Loss', 'PSNR']:
            train = df[metric.lower()]
            dev = df['val_' + metric.lower()]
            plt.figure()
            plt.plot(epoch, train, label='train')
            plt.plot(epoch, dev, label='dev')
            plt.legend(loc='best')
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.savefig('.'.join([prefix, metric.lower(), 'png']))
            plt.close()


    def test(self, test_set, metrics=[psnr]):
        print('Test on', test_set)
        datatest_dir = self.test_dir / test_set
        datatest_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each data
        rows = []
        for data_path in (data_dir / test_set).glob('*'):
            rows += [self.test_on_data(str(data_path),
                                        str(datatest_dir / data_path.stem),
                                        metrics=metrics)]
        df = pd.DataFrame(rows)

        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)

        df.to_csv(str(self.test_dir / f'{test_set}/metrics.csv'))


    def test_on_data(self, path, prefix, suffix='nc', metrics=[psnr]):
        # Load data
        lr_data, hr_data = load_data_pair(path, scale=self.scale)
        lr_data.resize(1,lr_data.shape[0],lr_data.shape[1],1)
        hr_data = np.array(hr_data)
        hr_data.resize(hr_data.shape[0],hr_data.shape[1],1)

        # Generate bicubic dataset
        bicubic_model = bicubic(lr_data, scale=self.scale)
        bicubic_array = bicubic_model.predict_on_batch(lr_data)

        # Generate output data and measure run time
        x = self.pre_process(lr_data)
        model = self.compile(self.build_model(x))
        if self.model_file.exists():
            model.load_weights(str(self.model_file))
        start = time.perf_counter()
        y_pred = model.predict_on_batch(x)
        end = time.perf_counter()
        output_array = self.post_process(y_pred[0], bicubic_array[0])

        # Record metrics
        row = pd.Series()
        row['name'] = Path(path).stem
        row['time'] = end - start
        y_true = self.inverse_post_process(hr_data)
        for metric in metrics:
            row[metric.__name__] = K.eval(metric(y_true, y_pred))

        # Save data
        data_to_save = []
        data_to_save += [(hr_data, 'original')]
        data_to_save += [(output_array, 'output')]
        data_to_save += [(lr_data, 'input')]
        for subdata, label in data_to_save:
            subdata = np.squeeze(subdata)
            [Nlat, Nlon] = subdata.shape
            ncout = netCDF4.Dataset('.'.join([prefix, label, suffix]),'w')
            ncout.createDimension("lat", Nlat)
            ncout.createDimension("lon", Nlon)
            dSSH = ncout.createVariable("SSH", "f4", ("lat","lon"))
            dSSH.long_name = "Sea Surface Height Above Geoid"
            dSSH.units = "m"
            dSSH[:] = subdata
            ncout.close()

        return row


    def apply(self, apply_set):
        print('Apply on', apply_set)
        dataapply_dir = self.apply_dir / apply_set
        dataapply_dir.mkdir(exist_ok=True)

        # Evaluate metrics on each data
        rows = []
        for data_path in (data_dir / apply_set).glob('*'):
            rows += [self.apply_on_data(str(data_path),str(dataapply_dir / data_path.stem))]
        df = pd.DataFrame(rows)

        # Compute average metrics
        row = pd.Series()
        row['name'] = 'average'
        for col in df:
            if col != 'name':
                row[col] = df[col].mean()
        df = df.append(row, ignore_index=True)

        df.to_csv(str(self.apply_dir / f'{apply_set}/metrics.csv'))


    def apply_on_data(self, path, prefix, suffix='nc'):
        # Load data
        lr_data = load_data_single(path, scale=self.scale)
        lr_data.resize(1,lr_data.shape[0],lr_data.shape[1],1)

        # Generate bicubic dataset
        bicubic_model = bicubic(lr_data, scale=self.scale)
        bicubic_array = bicubic_model.predict_on_batch(lr_data)

        # Generate output data and measure run time
        x = self.pre_process(lr_data)
        model = self.compile(self.build_model(x))
        if self.model_file.exists():
            model.load_weights(str(self.model_file))
        start = time.perf_counter()
        y_pred = model.predict_on_batch(x)
        end = time.perf_counter()
        output_array = self.post_process(y_pred[0], bicubic_array[0])

        # Record metrics
        row = pd.Series()
        row['name'] = Path(path).stem
        row['time'] = end - start

        # Save data
        data_to_save = []
        data_to_save += [(output_array, 'output')]
        data_to_save += [(lr_data, 'input')]
        for subdata, label in data_to_save:
            subdata = np.squeeze(subdata)
            [Nlat, Nlon] = subdata.shape
            ncout = netCDF4.Dataset('.'.join([prefix, label, suffix]),'w')
            ncout.createDimension("lat", Nlat)
            ncout.createDimension("lon", Nlon)
            dSSH = ncout.createVariable("SSH", "f4", ("lat","lon"))
            dSSH.long_name = "Sea Surface Height Above Geoid"
            dSSH.units = "m"
            dSSH[:] = subdata
            ncout.close()

        return row

