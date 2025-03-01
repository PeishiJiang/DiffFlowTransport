"""Key functions for SAS-based transport process."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

import jax
import jax.numpy as jnp
import equinox as eqx

from jaxtyping import Array
from typing import Callable

from .solver import solve_step_rk4
from ..sas import SASBase


class SASTransport(eqx.Module):
    sas_Q: SASBase
    sas_ET: SASBase
    solver: Callable
    dt: float
    α_Q: float
    α_ET: float
    nm: int
    k1: Array
    C_eq: Array
    C_Q_old: Array

    def __init__(
        self, sas_Q, sas_ET, dt, α_Q, α_ET, k1, C_eq, C_Q_old, 
        solver=solve_step_rk4
    ):
        """Initialization function

        Args:
            sas_Q (SASBase): The SAS function model for Q.
            sas_ET (SASBase): The SAS function model for ET.
            dt (float): The time step.
            α_Q (float): The fractionation in Q.
            α_ET (float): The fractionation in ET.
            k1 (Array): The reaction rate (nm,).
            C_eq (Array): The equilibrium reaction concentration (nm,).
            C_Q_old (Array): The old water concentration (nm,).
            solver (Callable): The numerical solver.
        """
        self.sas_Q = sas_Q
        self.sas_ET = sas_ET
        self.dt = jnp.array(dt)
        self.α_Q = jnp.array(α_Q)
        self.α_ET = jnp.array(α_ET)
        self.k1 = jnp.array(k1)
        self.C_eq = jnp.array(C_eq)
        self.C_Q_old = jnp.array(C_Q_old)
        self.solver = solver

        # Number of solutes
        self.nm = self.C_eq.size

    @eqx.filter_jit
    def __call__(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        """Solve the SAS-based transport model.

        Args:
            J (Array): The input flux (nt,).
            C_J (Array): The input concentration (nt, nm).
            Q (Array): The output flow (nt,).
            ET (Array): The output evapotranspiration (nt,).
            sTmT_init (Array): The initial age-ranked storage and mass (nτ, 1+nm).
            sas_Q_args (Array): The arguments to the SAS function for Q (nt, narg).
            sas_ET_args (Array): The arguments to the SAS function for ET (nt, narg).

        Returns:
            _type_: _description_
        """
        nm = self.nm
        sas_Q = self.sas_Q
        sas_ET = self.sas_ET
        dt = self.dt
        α_Q = self.α_Q
        α_ET = self.α_ET
        k1 = self.k1
        C_eq = self.C_eq
        C_Q_old = self.C_Q_old
        solver = self.solver

        nτ, _ = sTmT_init.shape
        nt = J.size
        J_full = jnp.concat([J[None,:], jnp.zeros([nτ-1, nt])])

        # Initialize other initial conditions
        sTmT_0 = jnp.zeros([nt, nm+1])
        ST_top_0 = jnp.zeros(nt+1)
        ST_bot_0 = jnp.zeros(nt+1)
        P_Q_old = jnp.ones(nt)

        # Solve the TTDs' ODE and return
        # sTmTs: (nτ, nt, nm+1)
        # mQETs: (nτ, nt, nm, 2)
        # pQETs: (nτ, nt, 2)
        # mR: (nτ, nt, nm)
        sTmTs, mQETs, pQETs, mRs = solve_ttd_sTmT(
            fvec, solver,
            J_full, C_J, Q, ET, dt,
            sTmT_0, ST_top_0, ST_bot_0, sTmT_init,  
            sas_Q, sas_ET, sas_Q_args, sas_ET_args,
            α_Q, α_ET, k1, C_eq, 
            # solver=solver, f=f
        )
    
        sT = sTmTs[...,0]  # (nτ, nt)
        mT = sTmTs[...,1:]  # (nτ, nt, nm)
        
        sT_init, mT_init = sTmT_init[:,0], sTmT_init[:,1:]
        sT = jnp.concat([sT_init[:,None], sT], axis=1)  # (nτ, nt+1)
        mT = jnp.concat([mT_init[:,None,:], mT], axis=1)  # (nτ, nt+1, nm)

        # Calculate C_Q
        def divide_nan(mq, q):
            qy = jnp.where(q==0.0, 1.0, q)
            return jnp.where(q==0.0, 0.0, mq / qy)
        mQs = mQETs[...,0]  # (nτ, nt, nm)
        C_Q1 = jax.vmap(divide_nan, in_axes=(0,0))(
            mQs.sum(axis=0) * dt, Q
        )  # (nt, nm)

        # Contribution from old water
        pQs = pQETs[...,0]  # (nτ, nt)
        P_Q_old = P_Q_old - pQs.sum(axis=0) * dt  # (nt,)
        # P_Q_old = jax.nn.relu(P_Q_old)
        C_Q2 = jnp.vectorize(lambda a,b: a*b)(
            jnp.stack([C_Q_old]*nt), jnp.stack([P_Q_old]*nm).T
        )  # (nt,nm)
        # The old water concentration should be zero if Q is zero
        def convert_zeroQ_to_zeroC(q, c):
            return jnp.where(q==0.0, 0.0, c)
        C_Q2 = jax.vmap(convert_zeroQ_to_zeroC, in_axes=(0,0))(
            Q, C_Q2
        )

        C_Q = C_Q1 + C_Q2  # (nt, nm)
        # print(C_Q1.shape, C_Q2.shape, C_Q_old.shape, P_Q_old.shape, nm, nt, pQs.shape)
        
        return sT, mT, mQETs, pQETs, mRs, C_Q

    def get_sT(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return sT

    def get_ST(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return jnp.cumsum(sT * self.dt, axis=0)  # (nτ, nt+1)

    def get_pQ(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return pQETs[...,0]  # (nτ, nt)

    def get_pET(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return pQETs[...,1]  # (nτ, nt)

    def get_mT(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return mT

    def get_mR(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        return mRs

    # def get_CT(
    #     self, J, C_J, Q, ET, sTmT_init, sTmT_0, ST_top_0, ST_bot_0, P_Q_old
    # ):
    #     pass

    def get_CQ(self, J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args, mask=None):
        sT, mT, mQETs, pQETs, mRs, C_Q = self(
            J, C_J, Q, ET, sTmT_init, sas_Q_args, sas_ET_args
        )
        # Note: mask is used to mask out the C_Q where observations are not given.
        mask = jnp.ones(C_Q.shape) if mask is None else mask
        return C_Q[mask]  # (nt, nm)


# Compute mQ -- mass loss with flow
def compute_mQ(sT, mT, α, Q_t, pQ):
    # sT, mT, α, Q_t, pQ: (1,), (1,), (1,), (1,), (1,)
    # return mT / sT * (α * Q_t * pQ)
    # return jnp.where(sT == 0.0, 0.0, mT / sT * (α * Q_t * pQ))
    part_a = mT * (α * Q_t * pQ)
    y = jnp.where(sT == 0.0, 1.0, sT)
    y = jnp.where(sT == 0.0, 0.0, 1./ y)
    return part_a * y


# Compute mR -- Reaction rate
def compute_mR(sT, mT, k1, C_eq):
    # sT, mT, k1, C_eq: (1,), (1,), (1,), (1,)
    return k1 * (C_eq*sT - mT)

compute_mR_vec = jax.vmap(compute_mR, in_axes=(None,0,0,0))


# SAS-based transport forward function
def f(sTmT, ST, dt, 
      J_τ_t, C_J_τ_t, Q_t, ET_t, 
      sas_Q, sas_ET, sas_arg_Q, sas_arg_ET, 
      α_Q, α_ET, k1, C_eq
):
    # sTmT: (1+nm,)
    # ST: (1,)
    # dt: (1,)
    # J_τ_t: (1,)
    # C_J_τ_t: (nm,)
    # Q_t: (1,)
    # ET_t: (1,)
    # α_Q: (1,)
    # α_ET: (1,)
    # sas_Q: (1,)
    # sas_arg_Q: (1,)
    # sas_ET: (1,)
    # sas_arg_ET: (1,)
    # k1: (nm,)
    # C_eq: (nm,)
    
    sT = sTmT[0] # (1,)
    mT = sTmT[1:] # (nm,)
    
    # jax.debug.print("sas_arg_Q shape: {x}", x=sas_arg_Q.shape)
    # print(sas_arg_Q.shape, ST.shape)
    # Calculate the pQ, mQ, and mR
    PQ1, PQ2 = sas_Q(ST+sT*dt, sas_arg_Q), sas_Q(ST, sas_arg_Q)
    pQ = (PQ1 - PQ2) / dt
    PET1, PET2 = sas_ET(ST+sT*dt, sas_arg_ET), sas_ET(ST, sas_arg_ET)
    pET = (PET1 - PET2) / dt
    mQ = compute_mQ(sT, mT, α_Q, Q_t, pQ) # (nm,)
    mET = compute_mQ(sT, mT, α_ET, ET_t, pET) # (nm,)

    # print(pQ.shape, pET.shape)
    pQET = jnp.array([pQ, pET])  # (2,)
    mQET = jnp.array([mQ, mET]).T  # (nm, 2)
    
    # Calculate mR
    mR = compute_mR_vec(sT, mT, k1, C_eq)  # (nm,)
    
    # Update the change of sT and mT
    δs = (J_τ_t - Q_t * pQ * dt - ET_t * pET * dt) / dt  # (1,)

    # Update the change of sT and mT
    δm = (J_τ_t * C_J_τ_t - mQ * dt - mET * dt + mR * dt) / dt  # (nm,)

    return jnp.concat([jnp.array([δs]), δm]), pQET, mQET, mR

fvec = jax.vmap(
    f, in_axes=(0,0,None,0,0,0,0,None,None,0,0,
                None,None,None,None)
)


# SAS-based transport function
def solve_ttd_sTmT(
    f, solver,
    J_full, C_J, Q, ET, dt,
    sTmT_0, ST_top_0, ST_bot_0, sTmT_init,  
    sas_Q, sas_ET,
    sas_Q_args, sas_ET_args,
    α_Q, α_ET, k1, C_eq, 
):
    # J_full: (nτ, nt)
    # C_J: (nt, nm)
    # Q: (nt,)
    # ET: (nt,)
    # dt: (1,)
    # sTmT_0: (nt, 1+nm)
    # ST_top_0: (nt,)
    # ST_bot_0: (nt,)
    # sTmT_init: (nτ, 1+nm)
    # sas_Q, sas_ET: (1,)
    # sas_Q_args, sas_ET_args: (nt, 1)
    # α_Q, α_ET: (1,)
    # k1, C_eq: (nm,)
    
    sas_funcs = [sas_Q, sas_ET]
    other_args = [α_Q, α_ET, k1, C_eq]
    
    def step_τ(states, x):
        # sTmT_init_τ : (1+nm,)
        sTmT_init_τ, J_τ = x
        sT_init_τ = sTmT_init_τ[0]
    
        # sTmT : (nt, 1+nm)
        # ST_top, ST_bot : (nt+1,)
        sTmT, ST_top, ST_bot = states
        
        # Get fluxes and sas arguments
        fluxes = [J_τ, C_J, Q, ET]
        sas_args = [sas_Q_args, sas_ET_args]
        # sas_args = [Q[:,None], ET[:,None]]  # TODO: this should be the function input!
    
        # Solve the ODE system
        # sTmT_new: (nt, 1+nm)
        # pQET_new: (nt, 2)
        # mQET_new: (nt, nm, 2)
        # mR_new: (nt, nm)
        sTmT_new, pQET_new, mQET_new, mR_new = solver(
            sTmT, ST_top[:-1], ST_bot[1:], dt, 
            *fluxes, *sas_funcs, *sas_args, *other_args,
            f=f
        )
        sT_new = sTmT_new[...,0]

        # Update ST
        ST_top_new = ST_bot
        ST_bot_new = jnp.concat([sT_init_τ[None] * dt, ST_bot[1:] + sT_new * dt])
        # jax.debug.print('sT_new: {x}; ST_bot: {y}; args: {z}', x=sT_new, y=ST_bot, z=args)
        
        # Variables as the initial condition to the next step
        sTmT_new_rotate = jnp.concat([sTmT_init_τ[None,...], sTmT_new[:-1]], axis=0)
        
        # Variables to save
        state_to_save = [sTmT_new, mQET_new, pQET_new, mR_new]
        # jax.debug.print("sTmT_new: {x}", x=sTmT_new)
        
        return (sTmT_new_rotate, ST_top_new, ST_bot_new), state_to_save
    
    _, state_to_save = jax.lax.scan(step_τ, (sTmT_0, ST_top_0, ST_bot_0), (sTmT_init,J_full))

    sTmTs, mQETs, pQETs, mRs = state_to_save
    # jax.debug.print("sTmTs: {x}", x=sTmTs)

    # Return
    # sTmTs: (nτ, nt, 1+nm)
    # pQETs: (nτ, nt, 2)
    # mQETs: (nτ, nt, nm, 2)
    # mR: (nτ, nt, nm)
    return sTmTs, mQETs, pQETs, mRs
