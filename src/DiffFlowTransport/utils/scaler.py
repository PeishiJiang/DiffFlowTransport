"""A couple of utility functions and classes."""

# Author: Peishi Jiang 
# Email: shixijps@gmail.com

import jax
import equinox as eqx
import jax.numpy as jnp

from jaxtyping import Array

from sklearn.preprocessing import StandardScaler, MinMaxScaler


def scale_df(df, scaler_type=''):
    """Scale the dataframe
    Args:
        df (pandas dataframe): the pandas dataframe with shape (Ns, Nx)
        scaler_type (str): the type of xdata scaler, either 'minmax', 'normalize', 'standard', 'log', or ''
    """
    scaler, df_norm = {}, df.copy()
    for varn in df.columns:
        scaler_v = get_scaler_by_type(scaler_type)
        df_norm[varn] = scaler_v.fit_transform(df[[varn]].values)
        scaler[varn] = scaler_v
    return scaler, df_norm


def get_scaler_by_type(scaler_type=''):
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()

    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
    
    elif scaler_type.lower() == 'log':
        scaler = LogScaler()

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
    if scaler_type.lower() == 'minmax':
        scaler = MinMaxScaler()
        scaler.fit(data)
        return scaler
        # scaler = MinMaxScaler(data)
        # return scaler

    elif scaler_type.lower() == 'standard':
        scaler = StandardScaler()
        scaler.fit(data)
        return scaler
        # scaler = StandardScaler(data)
        # return scaler
    
    elif scaler_type.lower() == 'log':
        return LogScaler()

    elif scaler_type == '':
        return IdentifyScaler()
    
    else:
        raise Exception("Unknown scaler type: %s" % scaler_type)


class BaseScaler(eqx.Module):
    # def __init__(self, data=None):
    #     self.data = data

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
        super().__init__(data)
        self.base = jnp.array(base)

    def transform(self, data):
        return jnp.log(data) / jnp.log(self.base)

    def inverse_transform(self, scaled_data):
        return jnp.power(self.base, scaled_data)