"""The SAS functions."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import jax
import jax.numpy as jnp
import equinox as eqx
from equinox.nn import MLP, Linear

from jaxtyping import Array
from typing import Dict


# The base class
class SASBase(eqx.Module):
    loc: Array
    scale: Array
    
    def __init__(self, loc=0.0, scale=1.0):
        self.loc = jnp.array(loc)
        self.scale = jnp.array(scale)
    
    def pdf(self, Si, x):
        # Return PDF
        raise Exception('Not implemented')
    
    def __call__(self, Si, x):
        # Return CDF:
        raise Exception('Not implemented')


# The null class
class SAS_null(SASBase):
    loc: Array
    scale: Array
    
    def pdf(self, Si, x):
        # Return PDF
        return 0.
    
    def __call__(self, Si, x):
        # Return CDF:
        return 0.


class SAS_Uniform(SASBase):
    
    def __init__(self, scale):
        super().__init__(0.0, scale)

    def pdf(self, Si, x=None):
        loc, scale = self.loc, self.scale
        return jax.scipy.stats.uniform.pdf(Si, loc=loc, scale=scale)
    
    def __call__(self, Si, x=None):
        loc, scale = self.loc, self.scale
        return jax.scipy.stats.uniform.cdf(Si, loc=loc, scale=scale)


class SAS_Uniform_VaryingScale(SASBase):
    
    def __init__(self, scale):
        super().__init__(0.0, scale)

    def pdf(self, Si, x=1.0):
        loc, scale = self.loc, jnp.array(x)
        return jax.scipy.stats.uniform.pdf(Si, loc=loc, scale=scale)
    
    def __call__(self, Si, x=1.0):
        loc, scale = self.loc, jnp.array(x)
        return jax.scipy.stats.uniform.cdf(Si, loc=loc, scale=scale)


# Gamme distribution
class SAS_Gamma(SASBase):
    a: Array
    
    def __init__(self, a, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.a = jnp.array(a)
    
    def pdf(self, Si, x=None):
        a, loc, scale = self.a, self.loc, self.scale
        y = (Si - loc) / scale
        # TODO: the minimum scaled y is needed to avoid
        # the FloatingPointError in jit operations!
        y = jax.lax.max(1e-20, y)
        return jax.scipy.stats.gamma.pdf(y, a, loc=0., scale=1.)
    
    def __call__(self, Si, x=None):
        a, loc, scale = self.a, self.loc, self.scale
        y = (Si - loc) / scale
        y = jax.lax.max(1e-20, y)
        return jax.scipy.stats.gamma.cdf(y, a, loc=0., scale=1.)


class SAS_Gamma_VaryingScale(SASBase):
    a: Array
    
    def __init__(self, a, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.a = jnp.array(a)
    
    def pdf(self, Si, x=1.0):
        a, loc, scale = self.a, self.loc, jnp.array(x)
        y = (Si - loc) / scale
        # TODO: the minimum scaled y is needed to avoid
        # the FloatingPointError in jit operations!
        y = jax.lax.max(1e-20, y)
        return jax.scipy.stats.gamma.pdf(y, a, loc=0., scale=1.)
    
    def __call__(self, Si, x=1.0):
        a, loc, scale = self.a, self.loc, jnp.array(x)
        y = (Si - loc) / scale
        y = jax.lax.max(1e-20, y)
        # print(scale.shape, x.shape, x.flatten().shape, Si.shape, y.shape)
        return jax.scipy.stats.gamma.cdf(y, a, loc=0., scale=1.)


# Beta distribution
class SAS_Beta(SASBase):
    a: Array
    b: Array
    
    def __init__(self, a, b, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.a, self.b = jnp.array(a), jnp.array(b)
    
    def pdf(self, Si, x=None):
        a, b, loc, scale = self.a, self.b, self.loc, self.scale
        return jax.scipy.stats.beta.pdf(Si, a, b, loc, scale)
    
    def __call__(self, Si, x=None):
        a, b, loc, scale = self.a, self.b, self.loc, self.scale
        return jax.scipy.stats.beta.cdf(Si, a, b, loc, scale)


# Kumaraswamy distribution 
class SAS_Kumaraswamy(SASBase):
    a: Array
    b: Array
    
    def __init__(self, a, b, loc=0.0, scale=1.0):
        super().__init__(loc, scale)
        self.a, self.b = jnp.array(a), jnp.array(b)
    
    def pdf(self, Si, x=None):
        a, b, loc, scale = self.a, self.b, self.loc, self.scale
        y = (Si - loc) / scale
        y = jax.lax.min(jax.lax.max(0., y), 1.)
        return a * b * y ** (a-1) * (1-y**a) ** (b-1)
    
    def __call__(self, Si, x=None):
        # Return CDF:
        a, b, loc, scale = self.a, self.b, self.loc, self.scale
        y = (Si - loc) / scale
        y = jax.lax.min(jax.lax.max(0., y), 1.)
        return 1. - (1. - y**a) ** b


# Mixture density network
# The code is modified from the following post: 
# https://github.com/hardmaru/mdn_jax_tutorial/blob/master/mixture_density_networks_jax.ipynb
class SAS_NormalMDN(SASBase):
    m: eqx.Module
    m_α: eqx.Module
    m_μ: eqx.Module
    m_σ: eqx.Module
    m_scale: eqx.Module
    # pdf_func: Callable
    # cdf_func: Callable
    
    def __init__(self, n_input, n_hidden, n_mixture, key, loc=0., scale=1., **mlp_kwargs):
        super().__init__(loc, scale)
        key = jax.random.key(key)
        key1, key2, key3, key4, key5 = jax.random.split(key, 5)
        
        # MLP model for predicting the hidden states
        # n_mlp_output = (n_output+2) * n_mixture
        self.m = MLP(in_size=n_input, out_size=n_hidden, key=key1, **mlp_kwargs)
        
        # MLP model for predicting α, μ, and σ
        self.m_α = Linear(in_features=n_hidden, out_features=n_mixture, key=key2)
        self.m_μ = Linear(in_features=n_hidden, out_features=n_mixture, key=key3)
        self.m_σ = Linear(in_features=n_hidden, out_features=n_mixture, key=key4)
        self.m_scale = Linear(in_features=n_hidden, out_features=1, key=key5)
        
        # # PDF function of mixture distributions 
        # self.pdf_func = jax.scipy.stats.norm.pdf
        
        # # CDF function of mixture distributions 
        # self.cdf_func = jax.scipy.stats.norm.cdf
    
    def get_param(self, x):
        # Calculate the weights, mean, and standard deviation of the mixture distributions
        z = self.m(x)  # shape: (n_hidden,)
        z_α, z_μ, z_σ = self.m_α(z), self.m_μ(z), self.m_σ(z) # shape: (n_mixture,) (n_mixture,) (n_mixture)
        
        # Calculate the scale
        scale_α = self.m_scale(z) # shape: (1,)
        scale_α = jax.nn.softplus(scale_α)
        scale = scale_α * self.scale

        # The weights (n_mixture,)
        α = jax.nn.softmax(z_α)
        
        # The mean (n_mixture,)
        μ = z_μ
        
        # The standard deviation (n_mixture,)
        σ = jnp.exp(z_σ)
        
        return α, μ, σ, scale
    
    def pdf(self, Si, x):
        α, μ, σ, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        pdfs = jax.vmap(jax.scipy.stats.norm.pdf, in_axes=(None,0,0))(y, μ, σ)
        return jnp.sum(jnp.dot(α, pdfs))
    
    def __call__(self, Si, x):
        α, μ, σ, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        cdfs = jax.vmap(jax.scipy.stats.norm.cdf, in_axes=(None,0,0))(y, μ, σ)
        return jnp.sum(jnp.dot(α, cdfs))


class SAS_GammaMDN(SASBase):
    m: eqx.Module
    m_α: eqx.Module
    m_a: eqx.Module
    m_scale: eqx.Module
    # pdf_func: Callable
    # cdf_func: Callable
    
    def __init__(self, n_input, n_hidden, n_mixture, key, loc=0., scale=1., **mlp_kwargs):
        super().__init__(loc, scale)
        key = jax.random.key(key)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # MLP model for predicting the hidden states
        # n_mlp_output = (n_output+2) * n_mixture
        self.m = MLP(in_size=n_input, out_size=n_hidden, key=key1, **mlp_kwargs)
        
        # MLP model for predicting α, coefficients, and scales
        self.m_α = Linear(in_features=n_hidden, out_features=n_mixture, key=key2)
        self.m_a = Linear(in_features=n_hidden, out_features=n_mixture, key=key3)
        self.m_scale = Linear(in_features=n_hidden, out_features=1, key=key4)
        
        # # PDF function of mixture distributions 
        # self.pdf_func = jax.scipy.stats.norm.pdf
        
        # # CDF function of mixture distributions 
        # self.cdf_func = jax.scipy.stats.norm.cdf
    
    def get_param(self, x):
        z = self.m(x)  # shape: (n_hidden,)

        # Calculate The weights (n_mixture,)
        z_α = self.m_α(z)
        α = jax.nn.softmax(z_α)

        # Calculate the shape parameter of the Gamma distribution
        z_a = self.m_a(z) # shape: (n_mixture,)
        a = jax.nn.sigmoid(z_a)
        
        # Calculate the scale
        scale_α = self.m_scale(z) # shape: (1,)
        scale_α = jax.nn.softplus(scale_α)
        scale = self.scale * scale_α
        
        return α, a, scale
    
    def pdf(self, Si, x):
        α, a, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        y = jax.lax.max(1e-20, y)
        pdfs = jax.vmap(jax.scipy.stats.gamma.pdf, in_axes=(None,0))(y, a)
        return jnp.sum(jnp.dot(α, pdfs))
    
    def __call__(self, Si, x):
        α, a, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        y = jax.lax.max(1e-20, y)
        cdfs = jax.vmap(jax.scipy.stats.gamma.cdf, in_axes=(None,0))(y, a)
        return jnp.sum(jnp.dot(α, cdfs))


class SAS_MDN(SASBase):
    """A general MDN consisting of a Uniform, a Gaussian, and a Gamma distributions"""
    m: eqx.Module
    m_α: eqx.Module
    m_a: eqx.Module
    m_scale: eqx.Module
    
    def __init__(self, n_input, n_hidden, key, loc=0., scale=1., **mlp_kwargs):
        super().__init__(loc, scale)
        key = jax.random.key(key)
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # MLP model for predicting the hidden states
        # n_mlp_output = (n_output+2) * n_mixture
        self.m = MLP(in_size=n_input, out_size=n_hidden, key=key1, **mlp_kwargs)
        
        # MLP model for predicting α, coefficients a, and the scale
        self.m_α = Linear(in_features=n_hidden, out_features=3, key=key2)
        self.m_a = Linear(in_features=n_hidden, out_features=3, key=key3)
        self.m_scale = Linear(in_features=n_hidden, out_features=1, key=key4)

    def get_param(self, x):
        z = self.m(x)  # shape: (n_hidden,)

        # Calculate The weights (n_mixture,)
        z_α = self.m_α(z)
        α = jax.nn.softmax(z_α)

        # Calculate the parameters of distributions
        z_a = self.m_a(z) # shape: (n_mixture,)
        Γa, μ, σ = z_a
        # Βa = jax.nn.softplus(Βa)  # parameter a of beta distribution
        # Βb = jax.nn.softplus(Βb)  # parameter b of beta distribution
        Γa = jax.nn.sigmoid(Γa)  # shape parameter of gamma distribution
        μ = μ  # mean of normal distribution
        σ = jnp.exp(σ)  # std of normal distribution
        
        # Calculate the scale
        scale_α = self.m_scale(z) # shape: (1,)
        scale_α = jax.nn.softplus(scale_α)
        scale = self.scale * scale_α
        
        return α, Γa, μ, σ, scale
    
    def pdf(self, Si, x):
        α, Γa, μ, σ, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        y = jax.lax.max(1e-20, y)
        pdf_norm = jax.scipy.stats.norm.cdf(y, μ, σ)
        pdf_uniform = jax.scipy.stats.uniform.cdf(y, loc=0., scale=1.)
        pdf_gamma = jax.scipy.stats.gamma.cdf(y, Γa, loc=0., scale=1.)
        # pdf_beta = jax.scipy.stats.beta.cdf(y, Βa, Βb, loc=0., scale=1.)
        pdfs = jnp.array([pdf_norm, pdf_uniform, pdf_gamma])
        return jnp.sum(jnp.dot(α, pdfs))
    
    def __call__(self, Si, x):
        α, Γa, μ, σ, scale = self.get_param(x)
        y = (Si - self.loc) / scale
        y = jax.lax.max(1e-20, y)
        cdf_norm = jax.scipy.stats.norm.cdf(y, μ, σ)
        cdf_uniform = jax.scipy.stats.uniform.cdf(y, loc=0., scale=1.)
        cdf_gamma = jax.scipy.stats.gamma.cdf(y, Γa, loc=0., scale=1.)
        # cdf_beta = jax.scipy.stats.beta.cdf(y, Βa, Βb, loc=0., scale=1.)
        cdfs = jnp.array([cdf_norm, cdf_uniform, cdf_gamma])
        return jnp.sum(jnp.dot(α, cdfs))


def initialize_sas_model(sas_params: Dict):
    model_type = sas_params['func']
    model_params = sas_params['args']

    if model_type.lower() == 'uniform':
        model = SAS_Uniform

    elif model_type.lower() == 'uniform_varyingscale':
        model = SAS_Uniform_VaryingScale

    elif model_type.lower() == 'gamma':
        model = SAS_Gamma

    elif model_type.lower() == 'gamma_varyingscale':
        model = SAS_Gamma_VaryingScale

    elif model_type.lower() == 'beta':
        model = SAS_Beta

    elif model_type.lower() == 'kumaraswamy':
        model = SAS_Kumaraswamy

    elif model_type.lower() == 'normalmdn':
        model = SAS_NormalMDN

    elif model_type.lower() == 'gammamdn':
        model = SAS_GammaMDN

    elif model_type.lower() == 'mdn':
        model = SAS_MDN

    return model(**model_params)