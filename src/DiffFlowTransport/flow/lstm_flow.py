"""The RNN model for rainfall-runoff modeling."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray

import equinox as eqx
from equinox._module import field
from equinox._misc import default_floating_dtype

import math
from typing import Optional


class LSTMCell(eqx.Module, strict=True):
    weight_ih: Array
    weight_hh: Array
    biasi: Optional[Array]
    biash: Optional[Array]
    input_size: int = field(static=True)
    hidden_size: int = field(static=True)
    use_bias: bool = field(static=True)

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        use_bias: bool = True,
        dtype=None,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `input_size`: The dimensionality of the input vector at each time step.
        - `hidden_size`: The dimensionality of the hidden state passed along between
            time steps.
        - `use_bias`: Whether to add on a bias after each update.
        - `dtype`: The dtype to use for all weights and biases in this LSTM cell.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending on
            whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)
        """
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bkey = jrandom.split(key, 3)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (4 * hidden_size, input_size)
        self.weight_ih = jrandom.uniform(ihkey, ihshape, dtype, minval=-lim, maxval=lim)
        hhshape = (4 * hidden_size, hidden_size)
        self.weight_hh = jrandom.uniform(hhkey, hhshape, dtype, minval=-lim, maxval=lim)
        bshape = (4 * hidden_size,)
        self.biasi = jrandom.uniform(bkey, bshape, dtype, minval=-lim, maxval=lim) if use_bias else None
        self.biash = jrandom.uniform(bkey, bshape, dtype, minval=-lim, maxval=lim) if use_bias else None

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias

    def __call__(self, input, hidden, *, key=None):
        """**Arguments:**

        - `input`: The input, which should be a JAX array of shape `(input_size,)`.
        - `hidden`: The hidden state, which should be a 2-tuple of JAX arrays, each of
            shape `(hidden_size,)`.
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        **Returns:**

        The updated hidden state, which is a 2-tuple of JAX arrays, each of shape
        `(hidden_size,)`.
        """
        h, c = hidden
        lin = self.weight_ih @ input + self.weight_hh @ h
        if self.use_bias:
            lin = lin + self.biasi + self.biash
        i, f, g, o = jnp.split(lin, 4)
        i = jnn.sigmoid(i)
        f = jnn.sigmoid(f)
        g = jnn.tanh(g)
        o = jnn.sigmoid(o)
        c = f * c + i * g
        h = o * jnn.tanh(c)
        return (h, c)


class LSTM(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(self, in_size, out_size, hidden_size, *, key):
        ckey, lkey = jrandom.split(key)
        self.hidden_size = hidden_size
        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, use_bias=False, key=ckey)
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)

    def __call__(self, input):
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))

        def f(carry, inp):
            h,c = self.cell(inp, carry)
            return (h,c), (h,c)
        states_t, states_all = jax.lax.scan(f, init_state, input)
        
        ht, ct = states_t
        return jax.nn.relu(self.linear(ht))
    
    def output_all(self, input):
        init_state = (jnp.zeros(self.cell.hidden_size),
                      jnp.zeros(self.cell.hidden_size))

        def f(carry, inp):
            h,c = self.cell(inp, carry)
            return (h,c), (h,c)
        states_t, states_all = jax.lax.scan(f, init_state, input)
        
        h_all, c_all = states_all
        vmap = jax.vmap(lambda h: self.linear(h), in_axes=0)
        h_trans_all = vmap(h_all)
        c_trans_all = vmap(c_all)
        # return h_trans_all, c_trans_all
        ht, ct = states_t
        outh, outc = self.linear(ht), self.linear(ct)
        return outh, outc, h_trans_all, c_trans_all