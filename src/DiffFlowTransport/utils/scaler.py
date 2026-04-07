"""A couple of utility functions and classes."""

# Author: Peishi Jiang 
# Email: shixijps@gmail.com

import jax
import equinox as eqx
import jax.numpy as jnp
import numpy as np

from jaxtyping import Array

from sklearn.preprocessing import StandardScaler, MinMaxScaler


# TODO: Scale Q using log after any adopted scaler
def scale_df(df, scaler_type='', logQ=False, Q_sym='Q'):
    """Scale the dataframe
    Args:
        df (pandas dataframe): the pandas dataframe with shape (Ns, Nx)
        scaler_type (str): the type of xdata scaler, either 'minmax', 'normalize', 'standard', 'log', or ''
    """
    scaler, df_norm = {}, df.copy()
    varns = df.columns

    # Take the logarithmic transform of Q before doing anything else
    if logQ is True:
        assert Q_sym in df.columns
        scaler_type_q = f'log-{scaler_type}'
        scaler_v = get_scaler_by_type(scaler_type_q)
        df_norm[Q_sym] = scaler_v.fit_transform(df[[Q_sym]].values)
        scaler[Q_sym] = scaler_v
        varns = varns.drop(Q_sym)

    # Get the scaler of the rest of the variables
    for varn in df.columns:
        scaler_v = get_scaler_by_type(scaler_type)
        df_norm[varn] = scaler_v.fit_transform(df[[varn]].values)
        scaler[varn] = scaler_v
    
    # Return
    return scaler, df_norm


def get_scaler_by_type(scaler_type=''):
    scaler_type_l = scaler_type.lower()
    if scaler_type_l == 'minmax':
        scaler = MinMaxScaler()

    elif scaler_type_l == 'standard':
        scaler = StandardScaler()
    
    elif scaler_type_l == 'log':
        scaler = LogScaler()
    
    elif scaler_type_l in {'log_minmax', 'log-minmax', 'log+minmax', 'logminmax'}:
        scaler = LogTransformScaler(transform_type='minmax')

    elif scaler_type_l in {'log_standard', 'log-standard', 'log+standard', 'logstandard'}:
        scaler = LogTransformScaler(transform_type='standard')

    elif scaler_type == '':
        scaler = IdentifyScaler()
    
    else:
        raise Exception("Unknown scaler type: %s" % scaler_type)
    
    return scaler


def get_scaler(data=None, scaler_type=''):
    """Get the data scaler.

    Args:
        data (array-like or None): the data array with shape (Ns, Nx)
        scaler_type (str): the type of xdata scaler, either 'minmax', 'normalize', 'standard', 'log', or ''
    """
    scaler_type_l = scaler_type.lower()
    if scaler_type_l == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler
        # scaler = MinMaxScaler(data)
        # return scaler

    elif scaler_type_l == 'standard':
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler
        # scaler = StandardScaler(data)
        # return scaler
    
    elif scaler_type_l == 'log':
        return LogScaler()
    
    elif scaler_type_l in {'log_minmax', 'log-minmax', 'log+minmax', 'logminmax'}:
        scaler = LogTransformScaler(transform_type='minmax')
        scaler.fit(data)
        return scaler

    elif scaler_type_l in {'log_standard', 'log-standard', 'log+standard', 'logstandard'}:
        scaler = LogTransformScaler(transform_type='standard')
        scaler.fit(data)
        return scaler

    elif scaler_type == '':
        return IdentifyScaler()
    
    else:
        raise Exception("Unknown scaler type: %s" % scaler_type)


class BaseScaler(eqx.Module):
    def fit(self, data):
        return self

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        pass

    def inverse_transform(self, data):
        pass


# class MinMaxScaler(BaseScaler):
#     """The scaler performs nothing."""
#     xmin: Array
#     xmax: Array

#     def __init__(self, data):
#         if data is None:
#             self.xmin, self.xmax = jnp.array(0.), jnp.array(1.)
#         else:
#             data = jnp.array(data)
#             self.xmin = jnp.min(data, axis=0) # (Nx,)
#             self.xmax = jnp.max(data, axis=0) # (Nx,)

#     def transform(self, data):
#         # data: (Ns, Nx)
#         xmin, xmax = self.xmin, self.xmax
#         scaled_data = jax.vmap(
#             lambda x,xmi,xma: (x-xmi) / (xma-xmi), in_axes=(0, None, None)
#         )(data, xmin, xmax)
#         return scaled_data

#     def inverse_transform(self, scaled_data):
#         # scaled_data: (Ns, Nx)
#         xmin, xmax = self.xmin, self.xmax
#         data = jax.vmap(
#             lambda x,xmi,xma: x * (xma-xmi) + xmi, in_axes=(0, None, None)
#         )(scaled_data, xmin, xmax)
#         return data


# class StandardScaler(BaseScaler):
#     """The scaler performs nothing."""
#     xmean: Array
#     xstd: Array

#     def __init__(self, data):
#         if data is None:
#             self.xmean, self.xstd = jnp.array(0.), jnp.array(1.)
#         else:
#             data = jnp.array(data)
#             self.xmean = jnp.min(data, axis=0) # (Nx,)
#             self.xstd = jnp.max(data, axis=0) # (Nx,)

#     def transform(self, data):
#         # data: (Ns, Nx)
#         xmean, xstd = self.xmean, self.xstd
#         scaled_data = jax.vmap(
#             lambda x,xmu,xst: (x-xmu) / xst, in_axes=(0, None, None)
#         )(data, xmean, xstd)
#         return scaled_data

#     def inverse_transform(self, scaled_data):
#         # scaled_data: (Ns, Nx)
#         xmean, xstd = self.xmean, self.xstd
#         data = jax.vmap(
#             lambda x,xmu,xst: x * xst + xmu, in_axes=(0, None, None)
#         )(scaled_data, xmean, xstd)
#         return data


class IdentifyScaler(BaseScaler):
    """The scaler performs nothing."""

    def transform(self, data):
        return data

    def inverse_transform(self, scaled_data):
        data = scaled_data
        return data


class LogScaler(BaseScaler):
    """The scaler perform the logrithmic transformation."""
    base: Array

    def __init__(self, data=None, base=10.):
        self.base = jnp.array(base)

    def transform(self, data):
        return jnp.log(data) / jnp.log(self.base)

    def inverse_transform(self, scaled_data):
        return jnp.power(self.base, scaled_data)


class LogTransformScaler(BaseScaler):
    """Apply logarithmic transform first, then the selected scaler."""
    base: Array
    transform_type: str
    scaler: object

    def __init__(self, data=None, transform_type='minmax', base=10.):
        self.base = jnp.array(base)
        self.transform_type = transform_type.lower()
        if self.transform_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif self.transform_type == 'standard':
            self.scaler = StandardScaler()
        elif self.transform_type == '':
            self.scaler = IdentifyScaler()
        else:
            raise Exception("Unknown transform type after log: %s" % transform_type)

        if data is not None:
            self.fit(data)

    def _log_transform(self, data):
        data = jnp.array(data)
        return jnp.log(data) / jnp.log(self.base)

    def fit(self, data):
        log_data = np.asarray(self._log_transform(data))
        if hasattr(self.scaler, 'fit'):
            self.scaler.fit(log_data)
        return self

    def transform(self, data):
        log_data = np.asarray(self._log_transform(data))
        if hasattr(self.scaler, 'transform'):
            out = self.scaler.transform(log_data)
        else:
            out = log_data
        return jnp.array(out)

    def inverse_transform(self, scaled_data):
        scaled_data = np.asarray(scaled_data)
        if hasattr(self.scaler, 'inverse_transform'):
            log_data = self.scaler.inverse_transform(scaled_data)
        else:
            log_data = scaled_data
        return jnp.power(self.base, jnp.array(log_data))
