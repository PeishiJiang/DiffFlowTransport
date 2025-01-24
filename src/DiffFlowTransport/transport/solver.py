"""Multiple solvers."""

# Author: Peishi Jiang
# Email: shixijps@gmail.com

# TODO: Note that the usage of nn.relu would affect the accuracy of implicit differentiation

import jax


def solve_step_euler(states, ST_top, ST_bot, dt, *args, f):
    """Explicit Euler solver.

    Args:
        states (Array): The model states (nt, ns)
        ST_top (Array): The age-ranked storage at time j and age i (nt,)
        ST_bot (Array): The age-ranked storage at time j+1 and age i (nt,)
        dt (Float): The time step size
        f (Callable): The forward function
        args (List): The other arguments of the forward functions

    Returns:
        _type_: _description_
    """
    # Note: we use relu to make sure the updated states are non-zero
    ST = ST_top
    r, pQET, mQET, mR = f(states, ST, dt, *args)

    # states_new = states + r * dt
    states_new = jax.nn.relu(states + r * dt)

    return states_new, pQET, mQET, mR


def solve_step_rk2(states, ST_top, ST_bot, dt, *args, f):
    """The second-order Runge-Kutta method.

    Args:
        states (Array): The model states (nt, ns)
        ST_top (Array): The age-ranked storage at time j and age i (nt,)
        ST_bot (Array): The age-ranked storage at time j+1 and age i (nt,)
        dt (Float): The time step size
        f (Callable): The forward function
        args (List): The other arguments of the forward functions

    Returns:
        _type_: _description_
    """
    # Note: we use relu to make sure the updated states are non-zero
    ST1 = ST_top
    states1 = states
    r1, pQET1, mQET1, mR1 = f(states1, ST1, dt, *args)
    
    ST2 = ST_bot
    states2 = jax.nn.relu(states+1.*r1*dt)
    # states2 = states+1.*r1*dt
    r2, pQET2, mQET2, mR2 = f(states2, ST2, dt, *args)

    pQET = 1./2 * (pQET1 + pQET2)
    mQET = 1./2 * (mQET1 + mQET2)
    mR = 1./2 * (mR1 + mR2)
    states_new = jax.nn.relu(states + (r1 + r2)/2. * dt)
    # states_new = states + (r1 + r2)/2. * dt

    return states_new, pQET, mQET, mR


def solve_step_rk4(states, ST_top, ST_bot, dt, *args, f):
    """The fourth-order Runge-Kutta method.

    Args:
        states (Array): The model states (nt, ns)
        ST_top (Array): The age-ranked storage at time j and age i (nt,)
        ST_bot (Array): The age-ranked storage at time j+1 and age i (nt,)
        dt (Float): The time step size
        f (Callable): The forward function
        args (List): The other arguments of the forward functions

    Returns:
        _type_: _description_
    """
    # Note: we use relu to make sure the updated states are non-zero
    ST1 = ST_top
    states1 = states
    r1, pQET1, mQET1, mR1 = f(states1, ST1, dt, *args)
    # jax.debug.print('r1: {x}', x=r1)
    
    ST2 = ST_top/2 + ST_bot/2
    states2 = jax.nn.relu(states+0.5*r1*dt)
    # states2 = states+0.5*r1*dt
    r2, pQET2, mQET2, mR2 = f(states2, ST2, dt, *args)
    # jax.debug.print('r2: {x}', x=r2)
    
    ST3 = ST_top/2 + ST_bot/2
    states3 = jax.nn.relu(states+0.5*r2*dt)
    # states3 = states+0.5*r2*dt
    r3, pQET3, mQET3, mR3 = f(states3, ST3, dt, *args)
    # jax.debug.print('r3: {x}', x=r3)
    
    ST4 = ST_bot
    states4 = jax.nn.relu(states+1.*r3*dt)
    # states4 = states+1.*r3*dt
    r4, pQET4, mQET4, mR4 = f(states4, ST4, dt, *args)
    # jax.debug.print('r4: {x}', x=r4)

    # states_new = states + 1./6 * (r1 + 2 * r2 + 2 * r3 + r4) * dt
    states_new = jax.nn.relu(states + 1./6 * (r1 + 2 * r2 + 2 * r3 + r4) * dt)
    pQET = 1./6 * (pQET1 + 2 * pQET2 + 2 * pQET3 + pQET4)
    mQET = 1./6 * (mQET1 + 2 * mQET2 + 2 * mQET3 + mQET4)
    mR = 1./6 * (mR1 + 2 * mR2 + 2 * mR3 + mR4)

    # jax.debug.print('states_new: {x}', x=states_new)

    return states_new, pQET, mQET, mR
