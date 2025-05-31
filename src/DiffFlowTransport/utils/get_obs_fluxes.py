"""Functions for getting observations from pandas dataframe."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import pandas as pd
import jax.numpy as jnp


def read_obs_csv(f_data, watershed=''):
    if watershed.lower() == 'plynlimon':
        df = pd.read_csv(f_data, index_col=1)
        df.index = pd.to_datetime(df.index, format='%m/%d/%y')
    
    elif watershed.lower() == 'oakcreek':
        df = pd.read_csv(f_data, index_col=0)
        df.index = pd.to_datetime(df.index)
    
    else:
        df = pd.read_csv(f_data, index_col=0)
        df.index = pd.to_datetime(df.index)
    
    return df


def get_transport_obs_fluxes(df, watershed=''):
    if watershed.lower() == 'plynlimon':
        return get_fluxes_from_df_plynlimon(df)
    
    elif watershed.lower() == 'oakcreek':
        return get_fluxes_from_df_oakcreek(df)
    
    else:
        raise Exception(
            f'Unknown watershed {watershed}. Please create a function for getting its observed transport data!'
        )


def get_fluxes_from_df_plynlimon(df, nτ=None):
    J = jnp.array(df['J'].values)
    Q = jnp.array(df['Q'].values)
    ET = jnp.array(df['ET'].values)
    C_J = jnp.array(df[['Cl mg/l']].values)
    C_Q_J = jnp.array(df['Q Cl mg/l'].values)
    time = df.index

    return J, Q, ET, C_J, C_Q_J, time


def get_fluxes_from_df_oakcreek(df):
    # J = jnp.array(df['P'].values)
    # C_J = jnp.array(df[['C_P']].values)
    J = jnp.array(df['P2'].values)
    # C_J = jnp.array(df[['C_P2']].values)
    C_J = jnp.array(df[['C_P2b']].values)
    Q = jnp.array(df['Q'].values)
    ET = jnp.array(df['ET'].values)
    C_Q_J = jnp.array(df['C_Q'].values)
    time = df.index

    return J, Q, ET, C_J, C_Q_J, time