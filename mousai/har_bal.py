"""Harmonic balance solvers and other related tools."""
import warnings
import numpy as np
import scipy as sp
import scipy.fftpack as fftp
import scipy.linalg as la
import logging
import ipdb
from scipy.optimize import newton_krylov, anderson, broyden1, broyden2, \
    excitingmixing, linearmixing, diagbroyden, fsolve
import scipy.optimize as op


def hb_time(sdfunc, x0=None, omega=1, method='newton_krylov', num_harmonics=1,
            num_variables=None, eqform='second_order', params={}, realify=True,
            **kwargs):
    r"""Harmonic balance solver for first and second order ODEs.

    Obtains the solution of a first-order and second-order differential
    equation under the presumption that the solution is harmonic using an
    algebraic time method.

    Returns `t` (time), `x` (displacement), `v` (velocity), and `a`
    (acceleration) response of a first- or second- order linear ordinary
    differential equation defined by
    :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)` or
    :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`.

    For the state space form, the function `sdfunc` should have the form::

        def duff_osc_ss(x, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            xdot = np.array([[x[1]],[-x[0]-.1*x[0]**3-.1*x[1]+1*sin(omega*t)]])
            return xdot

    In a state space form solution, the function must take the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    For the second order form the function `sdfunc` should have the form::

        def duff_osc(x, v, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            return np.array([[-x-.1*x**3-.2*v+sin(omega*t)]])

    In a second-order form solution the function must take the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    Parameters
    ----------
    sdfunc : function
        For `eqform='first_order'`, name of function that returns **column
        vector** first derivative given `x`, `omega` and \*\*kwargs. This is
        *NOT* a string.

        :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`

        For `eqform='second_order'`, name of function that returns **column
        vector** second derivative given `x`, `v`, `omega` and \*\*kwargs. This
        is *NOT* a string.

        :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)`
    x0 : array_like, somewhat optional
        n x m array where n is the number of equations and m is the number of
        values representing the repeating solution.
        It is required that :math:`m = 1 + 2 num_{harmonics}`. (we will
        generalize allowable default values later.)
    omega : float
        assumed fundamental response frequency in radians per second.
    method : str, optional
        Name of optimization method to be used.
    num_harmonics : int, optional
        Number of harmonics to presume. The omega = 0 constant term is always
        presumed to exist. Minimum (and default) is 1. If num_harmonics*2+1
        exceeds the number of columns of `x0` then `x0` will be expanded, using
        Fourier analaysis, to include additional harmonics with the starting
        presumption of zero values.
    num_variables : int, somewhat optional
        Number of states for a state space model, or number of generalized
        dispacements for a second order form.
        If `x0` is defined, num_variables is inferred. An error will result if
        both `x0` and num_variables are left out of the function call.
        `num_variables` must be defined if `x0` is not.
    eqform : str, optional
        `second_order` or `first_order`. (second order is default)
    params : dict, optional
        Dictionary of parameters needed by sdfunc.
    realify : boolean, optional
        Force the returned results to be real.
    other : any
        Other keyword arguments available to nonlinear solvers in
        `scipy.optimize.nonlin
        <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_.
        See `Notes`.

    Returns
    -------
    t, x, e, amps, phases : array_like
        time, displacement history (time steps along columns), errors,
    amps : float array
        amplitudes of displacement (primary harmonic) in column vector format.
    phases : float array
        amplitudes of displacement (primary harmonic) in column vector format.

    Examples
    --------
    >>> import mousai as ms
    >>> t, x, e, amps, phases = ms.hb_time(ms.duff_osc,
    ...                                    np.array([[0,1,-1]]),
    ...                                    omega = 0.7)

    Notes
    -----
    .. seealso::

       ``hb_freq``

    This method is not reliable for a low number of harmonics.

    Calls a linear algebra function from
    `scipy.optimize.nonlin
    <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_ with
    `newton_krylov` as the default.

    Evaluates the differential equation/s at evenly spaced points in time. Each
    point in time yields a single equation. One harmonic plus the constant term
    results in 3 points in time over the cycle.

    Solver should gently "walk" solution up to get to nonlinearities for hard
    nonlinearities.

    Algorithm:
        1. calls `hb_err` with `x` as the variable to solve for.
        2. `hb_err` uses a Fourier representation of `x` to obtain
           velocities (after an inverse FFT) then calls `sdfunc` to determine
           accelerations.
        3. Accelerations are also obtained using a Fourier representation of x
        4. Error in the accelerations (or state derivatives) are the functional
           error used by the nonlinear algebraic solver
           (default `newton_krylov`) to be minimized by the solver.

    Options to the nonlinear solvers can be passed in by \*\*kwargs (keyward
    arguments) identical to those available to the nonlinear solver.

    """
    # Initial conditions exist?
    if x0 is None:
        if num_variables is not None:
            x0 = np.zeros((num_variables, 1 + num_harmonics * 2))
        else:
            print('Error: Must either define number of variables or initial\
                  guess for x.')
            return
    elif num_harmonics is None:
        num_harmonics = int((x0.shape[1] - 1) / 2)
    elif 1 + 2 * num_harmonics > x0.shape[1]:
        x_freq = fftp.fft(x0)
        x_zeros = np.zeros((x0.shape[0], 1 + num_harmonics * 2 - x0.shape[1]))
        x_freq = np.insert(x_freq, [x0.shape[1] - x0.shape[1] // 2], x_zeros,
                           axis=1)

        x0 = fftp.ifft(x_freq) * (1 + num_harmonics * 2) / x0.shape[1]
        x0 = np.real(x0)
    if isinstance(sdfunc, str):
        sdfunc = globals()[sdfunc]
        print("`sdfunc` is expected to be a function name, not a string")
    params['function'] = sdfunc  # function that returns SO derivative
    time = np.linspace(0, 2 * np.pi / omega, num=x0.shape[1], endpoint=False)
    params['time'] = time
    params['omega'] = omega
    params['n_har'] = num_harmonics

    def hb_err(x):
        r"""Array (vector) of hamonic balance second order algebraic errors.

        Given a set of second order equations
        :math:`\ddot{x} = f(x, \dot{x}, \omega, t)`
        calculate the error :math:`E = \ddot{x} - f(x, \dot{x}, \omega, t)`
        presuming that :math:`x` can be represented as a Fourier series, and
        thus :math:`\dot{x}` and :math:`\ddot{x}` can be obtained from the
        Fourier series representation of :math:`x`.

        Parameters
        ----------
        x : array_like
            x is an :math:`n \\times m` by 1 array of presumed displacements.
            It must be a "list" array (not a linear algebra vector). Here
            :math:`n` is the number of displacements and :math:`m` is the
            number of times per cycle at which the displacement is guessed
            (minimum of 3)

        **kwargs : string, float, variable
            **kwargs is a packed set of keyword arguments with 3 required
            arguments.
                1. `function`: a string name of the function which returned
                the numerically calculated acceleration.

                2. `omega`: which is the defined fundamental harmonic
                at which the is desired.

                3. `n_har`: an integer representing the number of harmonics.
                Note that `m` above is equal to 1 + 2 * `n_har`.

        Returns
        -------
        e : array_like
            2d array of numerical error of presumed solution(s) `x`.

        Notes
        -----
        `function` and `omega` are not separately defined arguments so as to
        enable algebraic solver functions to call `hb_time_err` cleanly.

        The algorithm is as follows:
            1. The velocity and accelerations are calculated in the same shape
               as `x` as `vel` and `accel`.
            3. Each column of `x` and `v` are sent with `t`, `omega`, and other
               `**kwargs** to `function` one at a time with the results
               agregated into the columns of `accel_num`.
            4. The difference between `accel_num` and `accel` is reshaped to be
               :math:`n \\times m` by 1 and returned as the vector error used
               by the numerical algebraic equation solver.

        """
        nonlocal params  # Will stay out of global/conflicts
        n_har = params['n_har']
        omega = params['omega']
        time = params['time']
        m = 1 + 2 * n_har
        vel = harmonic_deriv(omega, x)
        if eqform == 'second_order':
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                # might be able to be commented out, left as example
                t = time[i]
                params['cur_time'] = time[i]  # loops
                # Note that everything in params can be accessed within
                # `function`.
                accel_from_deriv[:, i] = params['function'](x[:, i], vel[:, i],
                                                            params)[:, 0]
            e = accel_from_deriv - accel
        elif eqform == 'first_order':

            vel_from_deriv = np.zeros_like(vel)
            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                t = time[i]
                params['cur_time'] = time[i]
                # Note that everything in params can be accessed within
                # `function`.
                vel_from_deriv[:, i] =\
                    params['function'](x[:, i], params)[:, 0]

            e = vel_from_deriv - vel
        else:
            print('eqform cannot have a value of ', eqform)
            return 0, 0, 0, 0, 0
        return e

    try:
        x = globals()[method](hb_err, x0, **kwargs)
    except:
        x = x0  # np.full([x0.shape[0],x0.shape[1]],np.nan)
        amps = np.full([x0.shape[0], ], np.nan)
        phases = np.full([x0.shape[0], ], np.nan)
        e = hb_err(x)  # np.full([x0.shape[0],x0.shape[1]],np.nan)
        raise
    else:
        xhar = fftp.fft(x) * 2 / len(time)
        amps = np.absolute(xhar[:, 1])
        phases = np.angle(xhar[:, 1])
        e = hb_err(x)

    if realify is True:
        x = np.real(x)
    else:
        print('x was real')
    return time, x, e, amps, phases


def hb_freq(sdfunc, x0=None, omega=1, method='newton_krylov', num_harmonics=1,
            num_variables=None, mask_constant=True, eqform='second_order',
            params={}, realify=True, num_time_steps=51, **kwargs):
    r"""Harmonic balance solver for first and second order ODEs.

    Obtains the solution of a first-order and second-order differential
    equation under the presumption that the solution is harmonic using an
    algebraic time method.

    Returns `t` (time), `x` (displacement), `v` (velocity), and `a`
    (acceleration) response of a first or second order linear ordinary
    differential equation defined by
    :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)` or
    :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`.

    For the state space form, the function `sdfunc` should have the form::

        def duff_osc_ss(x, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            x_dot = np.array([[x[1]],
                              [-x[0]-.1*x[0]**3-.1*x[1]+1*sin(omega*t)]])
            return xdot

    In a state space form solution, the function must take the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    For the second order form the function `sdfunc` should have the form::

        def duff_osc(x, v, params):  # params is a dictionary of parameters
            omega = params['omega']  # `omega` will be put into the dictionary
                                     # for you
            t = params['cur_time']   # The time value is available as
                                     # `cur_time` in the dictionary
            return np.array([[-x-.1*x**3-.2*v+sin(omega*t)]])

    In a second-order form solution the function must take the states and the
    `params` dictionary. This dictionary should be used to obtain the
    prescribed response frequency and the current time. These plus any other
    parameters are used to calculate the state derivatives which are returned
    by the function.

    Parameters
    ----------
    sdfunc : function
        For `eqform='first_order'`, name of function that returns **column
        vector** first derivative given `x`, `omega` and \*\*kwargs. This is
        *NOT* a string.

        :math:`\dot{\mathbf{x}}=f(\mathbf{x},\omega)`

        For `eqform='second_order'`, name of function that returns **column
        vector** second derivative given `x`, `v`, `omega` and \*\*kwargs. This
        is *NOT* a string.

        :math:`\ddot{\mathbf{x}}=f(\mathbf{x},\mathbf{v},\omega)`
    x0 : array_like, somewhat optional
        n x m array where n is the number of equations and m is the number of
        values representing the repeating solution.
        It is required that :math:`m = 1 + 2 num_{harmonics}`. (we will
        generalize allowable default values later.)
    omega : float
        assumed fundamental response frequency in radians per second.
    method : str, optional
        Name of optimization method to be used.
    num_harmonics : int, optional
        Number of harmonics to presume. The `omega` = 0 constant term is always
        presumed to exist. Minimum (and default) is 1. If num_harmonics*2+1
        exceeds the number of columns of `x0` then `x0` will be expanded, using
        Fourier analaysis, to include additional harmonics with the starting
        presumption of zero values.
    num_variables : int, somewhat optional
        Number of states for a state space model, or number of generalized
        dispacements for a second order form.
        If `x0` is defined, num_variables is inferred. An error will result if
        both `x0` and num_variables are left out of the function call.
        `num_variables` must be defined if `x0` is not.
    eqform : str, optional
        `second_order` or `first_order`. (`second order` is default)
    params : dict, optional
        Dictionary of parameters needed by sdfunc.
    realify : boolean, optional
        Force the returned results to be real.
    mask_constant : boolean, optional
        Force the constant term of the series representation to be zero.
    num_time_steps : int, default = 51
        number of time steps to use in time histories for derivative
        calculations.
    other : any
        Other keyword arguments available to nonlinear solvers in
        `scipy.optimize.nonlin
        <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_.
        See Notes.

    Returns
    -------
    t, x, e, amps, phases : array_like
        time, displacement history (time steps along columns), errors,
    amps : float array
        amplitudes of displacement (primary harmonic) in column vector format.
    phases : float array
        amplitudes of displacement (primary harmonic) in column vector format.

    Examples
    --------
    >>> import mousai as ms
    >>> t, x, e, amps, phases = ms.hb_freq(ms.duff_osc,
    ...                                    np.array([[0,1,-1]]),
    ...                                    omega = 0.7)

    Notes
    -----
    .. seealso::

       `hb_time`

    Calls a linear algebra function from
    `scipy.optimize.nonlin
    <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_ with
    `newton_krylov` as the default.

    Evaluates the differential equation/s at evenly spaced points in time
    defined by the user (default 51). Uses error in FFT of derivative
    (acceeration or state equations) calculated based on:

    1. governing equations
    2. derivative of `x` (second derivative for state method)

    Solver should gently "walk" solution up to get to nonlinearities for hard
    nonlinearities.

    Algorithm:
        1. calls `hb_time_err` with x as the variable to solve for.
        2. `hb_time_err` uses a Fourier representation of x to obtain
           velocities (after an inverse FFT) then calls `sdfunc` to determine
           accelerations.
        3. Accelerations are also obtained using a Fourier representation of x
        4. Error in the accelerations (or state derivatives) are the functional
           error used by the nonlinear algebraic solver
           (default `newton_krylov`) to be minimized by the solver.

    Options to the nonlinear solvers can be passed in by \*\*kwargs.

    """
    # Initial conditions exist?
    if x0 is None:
        if num_variables is not None:
            x0 = np.zeros((num_variables, 1 + num_harmonics * 2))
        else:
            print('Error: Must either define number of variables or initial\
                  guess for x.')
            return
    elif num_harmonics is None:
        num_harmonics = int((x0.shape[1] - 1) / 2)
    elif 1 + 2 * num_harmonics > x0.shape[1]:
        x_freq = fftp.fft(x0)
        x_zeros = np.zeros((x0.shape[0], 1 + num_harmonics * 2 - x0.shape[1]))
        x_freq = np.insert(x_freq, [x0.shape[1] - x0.shape[1] // 2], x_zeros,
                           axis=1)

        x0 = fftp.ifft(x_freq) * (1 + num_harmonics * 2) / x0.shape[1]
        x0 = np.real(x0)
    if isinstance(sdfunc, str):
        sdfunc = globals()[sdfunc]
        print("`sdfunc` is expected to be a function name, not a string")
    params['function'] = sdfunc  # function that returns SO derivative
    time = np.linspace(0, 2 * np.pi / omega, num=x0.shape[1], endpoint=False)
    params['time'] = time
    params['omega'] = omega
    params['n_har'] = num_harmonics

    X0 = fftp.rfft(x0)
    if mask_constant is True:
        X0 = X0[:, 1:]

    params['mask_constant'] = mask_constant

    def hb_err(X):
        """Return errors in equation eval versus derivative calculation."""
        # r"""Array (vector) of hamonic balance second order algebraic errors.
        #
        # Given a set of second order equations
        # :math:`\ddot{x} = f(x, \dot{x}, \omega, t)`
        # calculate the error :math:`E = \mathcal{F}(\ddot{x}
        # - \mathcal{F}\left(f(x, \dot{x}, \omega, t)\right)`
        # presuming that :math:`x` can be represented as a Fourier series, and
        # thus :math:`\dot{x}` and :math:`\ddot{x}` can be obtained from the
        # Fourier series representation of :math:`x` and :math:`\mathcal{F}(x)`
        # represents the Fourier series of :math:`x(t)`
        #
        # Parameters
        # ----------
        # X : float array
        #     X is an :math:`n \\times m` by 1 array of sp.fft.rfft
        #     fft coefficients lacking the constant (first) element.
        #     Here :math:`n` is the number of displacements and :math:`m` 2
        #     times the number of harmonics to be solved for.
        #
        # **kwargs : string, float, variable
        #     **kwargs is a packed set of keyword arguments with 3 required
        #     arguments.
        #         1. `function`: a string name of the function which returned
        #         the numerically calculated acceleration.
        #
        #         2. `omega`: which is the defined fundamental harmonic
        #         at which the is desired.
        #
        #         3. `n_har`: an integer representing the number of harmonics.
        #         Note that `m` above is equal to 2 * `n_har`.
        #
        # Returns
        # -------
        # e : float array
        #     2d array of numerical errors of presumed solution(s) `X`. Error
        #     between first (or second) derivative via Fourier analysis and via
        #     solution of the governing equation.
        #
        # Notes
        # -----
        # `function` and `omega` are not separately defined arguments so as to
        # enable algebraic solver functions to call `hb_err` cleanly.
        #
        # The algorithm is as follows:
        #     1. X is prepended with a zero vector (to represent the constant
        #        value)
        #     2. `x` is calculated via an inverse `numpy.fft.rfft`
        #     1. The velocity and accelerations are calculated in the same
        #        shape as `x` as `vel` and `accel`.
        #     3. Each column of `x` and `v` are sent with `t`, `omega`, and
        #        other `**kwargs** to `function` one at a time with the results
        #        agregated into the columns of `accel_num`.
        #     4. The rfft is taken of `accel_num` and `accel`.
        #     5. The first column is stripped out of both `accel_num_freq and
        #        `accel_freq`.

        # """
        nonlocal params  # Will stay out of global/conflicts
        omega = params['omega']
        time = params['time']
        mask_constant = params['mask_constant']
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))

        x = fftp.irfft(X)
        time_e, x = time_history(time, x, num_time_points=num_time_steps)
#        m = 2 * n_har

        vel = harmonic_deriv(omega, x)
        # print('vel = ', vel)
        m = num_time_steps

        if eqform == 'second_order':
            accel = harmonic_deriv(omega, vel)
            accel_from_deriv = np.zeros_like(accel)

            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                # might be able to be commented out, left as example
                t = time_e[i]
                params['cur_time'] = time_e[i]  # loops
                # Note that everything in params can be accessed within
                # `function`.
                accel_from_deriv[:, i] = params['function'](x[:, i], vel[:, i],
                                                            params)[:, 0]
            e = (accel_from_deriv - accel)/np.max(np.abs(accel))
            #print(accel)
            # print('accel from derive = ', accel_from_deriv)
            # print('accel = ', accel)
            # print(e)
        elif eqform == 'first_order':

            vel_from_deriv = np.zeros_like(vel)
            # Should subtract in place below to save memory for large problems
            for i in np.arange(m):
                # This should enable t to be used for current time in loops
                t = time_e[i]
                params['cur_time'] = time_e[i]
                # Note that everything in params can be accessed within
                # `function`.
                vel_from_deriv[:, i] =\
                    params['function'](x[:, i], params)[:, 0]

            e = (vel_from_deriv - vel)/np.max(np.abs(vel))
        else:
            print('eqform cannot have a value of ', eqform)
            return 0, 0, 0, 0, 0
        e_fft = fftp.fft(e)
        # print(e_fft)
        e_fft_condensed = condense_fft(e_fft, num_harmonics)
        # print(e_fft_condensed)
        e = fft_to_rfft(e_fft_condensed)
        if mask_constant is True:
            e = e[:, 1:]
        # print('e ', e, ' X ', X)
        # print('1 eval')
        return e


    try:
        X = globals()[method](hb_err, X0, **kwargs)
        # print('tried')
    except:  # Catches and raises errors- needs actual error listed.
        print(
            'Excepted- search failed for omega = {:6.4f} rad/s.'.format(omega))
        print("""What ever error this is, please put into har_bal
               after the excepts (2 of them)""")
        X = X0
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))
        amps = np.full([X0.shape[0], ], np.nan)
        phases = np.full([X0.shape[0], ], np.nan)
        e = hb_err(X)  # np.full([x0.shape[0],X0.shape[1]],np.nan)
        raise
    else:  # Runs if there are no errors
        if mask_constant is True:
            X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))
        xhar = rfft_to_fft(X) * 2 / len(time)
        amps = np.absolute(xhar[:, 1])
        phases = np.angle(xhar[:, 1])
        e = hb_err(X)

    # if mask_constant is True:
    #    X = np.hstack((np.zeros_like(X[:, 0]).reshape(-1, 1), X))

    # print('\n\n\n\n', X)
    # print(e)
    # amps = np.sqrt(X[:, 1]**2 + X[:, 2]**2)
    # phases = np.arctan2(X[:, 2], X[:, 1])
    # e = hb_err(X)

    x = fftp.irfft(X)

    if realify is True:
        x = np.real(x)
    else:
        print('x was real')
    return time, x, e, amps, phases


def hb_so(sdfunc, **kwargs):
    """Deprecated function name. Use hb_time."""
    message = 'hb_so is deprecated. Please use hb_time or an alternative.'
    warnings.warn(message, DeprecationWarning)
    return hb_time(sdfunc, kwargs)


def harmonic_deriv(omega, r):
    r"""Returns derivative of a harmonic function using frequency methods.

    Parameters
    ----------
    omega: float
        Fundamendal frequency, in rad/sec, of repeating signal
    r: array_like
        | Array of rows of time histories to take the derivative of.
        | The 1 axis (each row) corresponds to a time history.
        | The length of the time histories *must be an odd integer*.

    Returns
    -------
    s: array_like
        Function derivatives.
        The 1 axis (each row) corresponds to a time history.

    Examples
    --------
    >>> import matplotlib.pyplot as plt
    >>> from mousai import *
    >>> import scipy as sp
    >>> from scipy import pi, sin, cos
    >>> f = 2
    >>> omega = 2.*pi * f
    >>> numsteps = 11
    >>> t = sp.arange(0,1/omega*2*pi,1/omega*2*pi/numsteps)
    >>> x = sp.array([sin(omega*t)])
    >>> v = sp.array([omega*cos(omega*t)])
    >>> states = sp.append(x,v,axis = 0)
    >>> state_derives = harmonic_deriv(omega,states)
    >>> plt.plot(t,states.T,t,state_derives.T,'x')
    [<matplotlib.line...]

    """
    s = np.zeros_like(r)
    for i in np.arange(r.shape[0]):
        s[i, :] = fftp.diff(r[i, :]) * omega
    return np.real(s)


def solmf(x, v, M, C, K, F):
    r"""Acceleration of second order linear matrix system.

    Parameters
    ----------
    x, v, F : array_like
        :math:`n\times 1` arrays of current displacement, velocity, and Force.
    M, C, K : array_like
        Mass, damping, and stiffness matrices.

    Returns
    -------
    a : array_like
        :math:`n\\times 1` acceleration vector

    Examples
    --------
    >>> import numpy as np
    >>> M = np.array([[2,0],[0,1]])
    >>> K = np.array([[2,-1],[-1,3]])
    >>> C = 0.01 * M + 0.01 * K
    >>> x = np.array([[1],[0]])
    >>> v = np.array([[0],[10]])
    >>> F = v * 0.1
    >>> a = solmf(x, v, M, C, K, F)
    >>> print(a)
        [[-0.95]
         [ 1.6 ]]

    """
    return -la.solve(M, C @ v + K @ x - F)


def duff_osc(x, v, params):
    """Duffing oscillator acceleration."""
    omega = params['omega']
    t = params['cur_time']
    acceleration = np.array([[-x - .1 * x**3. - 0.2 * v + np.sin(omega * t)]])
    return acceleration


def time_history(t, x, realify=True, num_time_points=200):
    r"""Generate refined time history from harmonic balance solution.

    Harmonic balance solutions presume a limited number of harmonics in the
    solution. The result is that the time history is usually a very limited
    number of values. Plotting these results implies that the solution isn't
    actually a continuous one. This function fills in the gaps using the
    harmonics obtained in the solution.

    Parameters
    ----------
    t: array_like
        1 x m array where m is the number of
        values representing the repeating solution.
    x: array_like
        n x m array where m is the number of equations and m is the number of
        values representing the repeating solution.
    realify: boolean
        Force the returned results to be real.
    num_time_points: int
        number of points desired in the "smooth" time history.

    Returns
    -------
    t: array_like
        1 x num_time_points array.
    x: array_like
        n x num_time_points array.

    Examples
    --------
    >>> import numpy as np
    >>> import mousai as ms
    >>> x = np.array([[-0.34996499,  1.36053998, -1.11828552]])
    >>> t = np.array([0.        , 2.991993  , 5.98398601])
    >>> t_full, x_full = ms.time_history(t, x, num_time_points=300)

    Notes
    -----
    The implication of this function is that the higher harmonics that
    were not determined in the solution are zero. This is indeed the assumption
    made when setting up the harmonic balance solution. Whether this is a valid
    assumption is something that the user must judge when obtaining the
    solution.

    """
    dt = t[1]
    t_length = t.size
    t = sp.linspace(0, t_length * dt, num_time_points, endpoint=False)
    x_freq = fftp.fft(x)
    x_zeros = sp.zeros((x.shape[0], t.size - x.shape[1]))
    x_freq = sp.insert(x_freq, [t_length - t_length // 2], x_zeros, axis=1)
    x = fftp.ifft(x_freq) * num_time_points / t_length
    if realify is True:
        x = sp.real(x)
    else:
        print('x was real')

    return t, x


def condense_fft(X_full, num_harmonics):
    """Create equivalent amplitude reduced-size FFT from longer FFT."""
    X_red = np.hstack((X_full[:, 0:(num_harmonics + 1)],
                       X_full[:, -1:-(num_harmonics + 1):-1]))\
        * (2 * num_harmonics + 1) / X_full[0, :].size
    return X_red


def rfft_to_fft(X_real):
    """Switch from SciPy real fft form to complex fft form."""
    X = fftp.fft(fftp.irfft(X_real))
    return X


def fft_to_rfft(X_real):
    """Switch from complex form fft form to SciPy rfft form."""
    X_complex = fftp.rfft(np.real(fftp.ifft(X_real)))
    return X_complex


def hb_freq_cont(Fnlfunc, X0=None, Ws=0.01, We=1.00, ds=0.01, maxNp=1000, deriv=False, fnform='Time',
                 method='newton_krylov', num_harmonics=1, mask_constant=True,
                 eqform='second_order', params={}, num_time_points=128, solep=1e-6,
                 ITMAX=100, dsmax=None, dsmin=None, scf=None, Nop=None, angop=None,
                 zt=None, **kwargs):
    r"""Harmonic balance solver with continuation for first and second order ODEs.
    
    Obtains and continues the solution of a first-order and second-order differential 
    equation under the presumption that the solution is harmonic using an algebraic time method.

    Returns 'X' (Solution with harmonics and continuation parameter), 
    'dX' (Unit tangents along solution curve), 'R' (residuals)

    Parameters
    ----------
    Fnlfunc : function
         Should return the nonlinear force in Time domain if `fnform` is 'Time' 
         and in Frequency domain in `fnform` is 'Freq'

         If `deriv` is True, the function should also return the Jacobian in the 
         following format:
         For `fnform='Time'`, function must return 
         :math: `F_{nl}(t), \frac{dF_{nl}}{dX}(t), \frac{dF_{nl}}{d\dot{X}}(t), \frac{dF_{nl}}{d\omega}(t)`
         eg: `Fnl, dFnldX, dFnldXd, dFnldw = Fnlfunc(xt,params)`
         When `xt` is n x m, `Fnl` is n x m, `dFnldX` is n x m, `dFnldXd` is n x m, `dFnldw` is n x m
         For `fnform='Frequency'`, function must return
         :math: '\tilde{F_{nl}}, \frac{d\tilde{F}_{nl}}{d\tilde{X}}, \frac{d\tilde{F}_{nl}}{d\omega}`
         eg: `Fnl, dFnldX, dFnldw = Fnlfunc(x,params)`
         Here, `x` is always n(2Nh+1) x 1, (Nh is number of harmonics) and, 
         `Fnl` is n(2Nh+1) x 1, `dFnldX` is n(2Nh+1) x n(2Nh+1), `dFnldw` is n(2Nh+1) x 1
    X0 : array_like, somewhat optional
         n(2Nh+1) x 1 giving the initial guess. All zeros assumed if not provided.
    Ws : float
         Starting frequency for continuation
    We : float
         Ending frequency for continuation
    method : str, optional
        Name of optimization method to be used.    
    num_harmonics : int, optional
        Number of harmonics to presume. The `omega` = 0 constant term is always
        presumed to exist. Minimum (and default) is 1.
    eqform : str, optional
        `second_order` or `first_order`. (`second order` is default)
    params : dict
        Dictionary of parameters required by Fnlfunc
    num_time_points : int, default = 128
        number of time steps for AFT
    other : any
        Other keyword arguments available to nonlinear solvers in
        `scipy.optimize.nonlin
        <https://docs.scipy.org/doc/scipy/reference/optimize.nonlin.html>`_.
        See Notes.

    Returns
    -------
    X, dX, R : array_like
         Solution, Unit tangent along arc, Residual
    X : float matrix
        (n(2Nh+1)+1) x Npoints. Format of each column:
        a_0^1  ---
        a_0^2  Zero harmonics
        ...    of all DOF's
        a_0^n  ---
        a_1^1  ---
        a_1^2  First cosine harmonics 
        ...    of all DOF's
        a_1^n ---
        b_1^1  ---
        b_1^2  First Sine harmonics
        ...    of all DOF's
        b_1^n ---
        .
        .
        .
        a_Nh^1  ---
        a_Nh^2  Last cosine harmonics 
        ...     of all DOF's
        a_Nh^n ---
        b_Nh^1  ---
        b_Nh^2  Last Sine harmonics
        ...     of all DOF's
        b_Nh^n ---  
        W      Fundamental frequency
    dX : float matrix
         (n(2Nh+1)+1) x Npoints. Format same as above. Stores the normalizedd tangent 
         vectors at each solution point.
    R  : float matrix
         (n(2Nh+1)+1) x Npoints. Format same as above. Stores the residual at each point.
    
    Examples
    --------
    <TO BE MADE>
    """
    # Check for matrices
    if eqform=='first_order':
        if not set(['C','K']).issubset(list(params.keys())):
            print('Error: Please provide C & K matrices in the params dictionary')
            return
        else:
            params['M'] = np.zeros_like(params['K'])
    elif eqform=='second_order':
        if not set(['M','C','K']).issubset(list(params.keys())):
            print('Error: Please provide M,C & K matrices in the params dictionary')
            return
    else:
        print('Error: Unknown string in eqform')
        return
    nd = np.shape(params['K'])[0]    
    if  not set(['fh','Fc','Fs']).issubset(list(params.keys())):
        print('Error: Please specify forcing details')
        return
    elif np.shape(params['Fc'])[0]!=nd or np.shape(params['Fs'])[0]!=nd:
        print('Error: Dimension mismatch in Fc &/or Fs')
        return
    
    if num_harmonics is None:
        print('Error: Invalid number of harmonics')
        return
    Nh = num_harmonics
    Nt = num_time_points
    # Initial conditions exist?
    if X0 is None:
        X0 = np.zeros((nd*(2*Nh+1),1))
    elif X0.size!=nd*(2*Nh+1):
        print('Error: Invalid initial guess')
        return -1
    else:
        X0 = np.reshape(X0,(X0.size,1))
        
    # Correct Initial Solution
    print('Initial Correction')
    X0c = op.root(lambda X,Ws,params,nd,Nh,Nt,Fnlfunc,fnform,deriv: hb_res(np.block([[np.reshape(X,(X.size,1))],[Ws]]), params, nd, Nh, Nt, Fnlfunc, fnform, deriv), X0, args=(Ws,params,nd,Nh,Nt,Fnlfunc,fnform,deriv), method='hybr', jac=True, tol=1e-16);
    if not X0c.success and X0c.status<1:
        print('Initial Correction failed with message: ',X0c.message,': Quitting.')
        return X0c
    print('Initial Correction succeeded after ',X0c.nfev,' iterations with a final norm of',la.norm(X0c.fun))
    X0 = np.reshape(X0c.x,(X0.size,1))
    Xi = np.block([[X0],[Ws]])
    R,dRdX,dRdW = hb_res(Xi, params, nd, Nh, Nt, Fnlfunc, fnform, deriv)
    
    # Initiate continuation
    X = np.zeros((nd*(2*Nh+1)+1,maxNp))
    X[:,0:1] = Xi
    dir = np.sign(We-Ws)

    if zt is None:
        zt = 1.0/(len(Xi)-1)
    # Tangent
    z = -np.sqrt(zt)*la.solve(dRdX,dRdW)
    alpha = dir/np.sqrt(1.0 + np.dot(z.T,z))
    alphap = alpha
    w = Ws
    npt = 0
    if dsmax is None:
        dsmax = 5*ds
    if dsmin is None:
        dsmin = ds/5
    Dscale = np.ones_like(Xi[:,0])

    
    def resfn(X, params, nd, Nh, Nt, Fnlfunc, fnform, deriv, DXds, X0, ds):
        X = np.reshape(X,(X.size,1))
        R,dRdX,dRdW = hb_res(X, params, nd, Nh, Nt, Fnlfunc, fnform, deriv)
        RR = np.block([R,np.asscalar(np.dot(DXds.T,X-X0)-ds)])
        JAC = np.block([[dRdX,dRdW],[DXds.T]])
        return np.reshape(RR,(RR.size,)),JAC
    while dir*w<dir*We:
        # Predictor
        dWds = alpha
        dXds = alpha*z
        DXds = np.block([[dXds],[dWds]])
        X0 = Xi
        Xp = X0 + DXds*ds
        
        zp = z
        alphap = alpha
        # Corrector
        Xi = X0
        X0c = op.root(resfn, Xi, args=(params,nd,Nh,Nt,Fnlfunc,fnform,deriv,DXds,X0,ds), method='hybr', jac=True, tol=1e-10);
        if not X0c.success and X0c.status<1:
            print('Correction failed with message: ',X0c.message,': Quitting.')
            X = X[:,0:npt]
            return X
        print('Correction succeeded after ',X0c.nfev,' iterations with a final norm of',la.norm(X0c.fun))
        itern = X0c.nfev
        Xi = np.reshape(X0c.x,(X0c.x.size,1))

        R,dRdX,dRdW = hb_res(Xi, params, nd, Nh, Nt, Fnlfunc, fnform, deriv)
        R = np.reshape(R,(R.size,1))
        res = la.norm(R)
        # Tangent
        z = -np.sqrt(zt)*la.solve(dRdX,dRdW)
        alpha = 1.0/np.sqrt(1.0+np.dot(z.T,z))
        angbw = np.arccos(alpha*alphap*(1.0+np.dot(zp.T,z)))
        alpha = alpha*np.sign(np.cos(angbw))

        npt = npt+1
        X[:,npt:npt+1] = Xi
        w = np.asscalar(Xi[-1,0])
        # Adapt step size
        if Nop is None:
            scf = 1.0
        else:
            scf = Nop/itern;
        if angop is not None:
            angvar = min(np.rad2deg(angbw),np.abs(180.0-np.rad2deg(angbw)))
            scf = scf*angop/angvar
        if scf>2.0:
            scf = 2.0
        elif scf<0.5:
            scf = 0.5
        ds = max(min(dsmax,ds*scf),dsmin)
        print('N=',npt,'; W=', Xi[-1,0],'; ds=',ds,'; It=',itern,'; R=',res,'; Ang=',np.asscalar(np.rad2deg(angbw)))
        if npt==maxNp:
            print('Max points exceeded')
            break                 
    X = X[:,0:npt]
    return X

def time2freq(X, Nh):
    Nt,n = np.shape(X)
    if Nt%2==0:
        Nht = int(Nt/2-1)
    else:
        Nht = int((Nt-1)/2)
    Xf = fftp.rfft(X,axis=0)*2/Nt
    Xf[0,:] = Xf[0,:]/2
    Xf = Xf[0:(2*Nh+1),:]
    Xf[2::2,:] = -Xf[2::2,:]
    Xfo = np.zeros((2*Nh+1,n))
    if Nh>Nht:
        Xfo[0:(2*Nht+1),:] = Xf[0:(2*Nht+1),:]
    else:
        Xfo = Xf[0:(2*Nh+1),:]
    return Xfo
    

def freq2time(X, Nt):
    Nh = int((np.shape(X)[0]-1)/2)
    if Nt%2==0:
        Nht = int(Nt/2-1)
    else:
        Nht = int((Nt-1)/2)    
    n = np.shape(X)[1]
    Xf = np.zeros((Nt,n))
    if Nh>Nht:
        Xf[0:(2*Nht+1),:] = X[0:(2*Nht+1),:]
    else:
        Xf[0:(2*Nh+1),:] = X[0:(2*Nh+1),:]
    Xf[0,:] = Xf[0,:]*Nt
    Xf[1::2,:] = Xf[1::2,:]*Nt/2
    Xf[2::2,:] = -Xf[2::2,:]*Nt/2
    Xt = fftp.irfft(Xf,axis=0)
    return Xt


def hb_res(X, params, nd, Nh, Nt, Fnlfn, fnform, deriv):
    E = np.zeros((nd*(2*Nh+1),nd*(2*Nh+1)))
    dEdw = E.copy()
    M = params['M']
    C = params['C']
    K = params['K']
    E[0:nd,0:nd] = K
    w = np.asscalar(X[-1])
    for k in range(1,Nh+1):
        cst = nd + (k-1)*2*nd
        cen = nd + (k-1)*2*nd + nd
        sst = nd + (k-1)*2*nd + nd
        sen = nd + (k-1)*2*nd + 2*nd
        E[cst:cen,cst:cen] = E[sst:sen,sst:sen] = K - (k*w)**2*M
        E[cst:cen,sst:sen] = (k*w)*C
        E[sst:sen,cst:cen] = -E[cst:cen,sst:sen]

        dEdw[cst:cen,cst:cen] = dEdw[sst:sen,sst:sen] = -2.*w*k**2*M
        dEdw[cst:cen,sst:sen] = k*C
        dEdw[sst:sen,cst:cen] = -dEdw[cst:cen,sst:sen]

    # Nonlinear forcing & Jacobian calculation
    if fnform=='Time': ## INCOMPLETE: Got to add provision for velocity dependent nonlinearities
        t = np.linspace(0,2*np.pi,Nt,endpoint=0)
        t = np.reshape(t,(Nt,1))
        Xd = np.reshape(X[0:-1,0],(2*Nh+1,nd))
        Xt = freq2time(Xd, Nt)
        if deriv==True:
            Fnlt, dFnldxt, dFnldxdt, dFnldWt = Fnlfn(Xt, params)
        else:
            Fnlt = Fnlfn(Xt, params)
            _h_ = 1e-3
            dFnldx = np.zeros((Nt,nd*nd))
            for i in range(0,nd):
                Xp = Xt.copy()
                Xp[:,i] = Xt[:,i] + _h_
                dFnldx[:,i::nd] = (Fnlfn(Xp, params)-Fnlt)/_h_
                dFnldxdt = np.zeros_like(dFnldx)
                dFnldWt = np.zeros_like(Fnlt)

        Fnl = np.reshape(time2freq(Fnlt, Nh),(nd*(2*Nh+1),1))
        dFnldW = np.reshape(time2freq(dFnldWt, Nh),(nd*(2*Nh+1),1))

        Jnl = np.zeros((nd*(2*Nh+1),nd*(2*Nh+1)))
        tmp = time2freq(dFnldxt, Nh)
        for ii in range(0,nd):
            Jnl[ii::nd,0:nd] = tmp[:,ii*nd:(ii+1)*nd]
        for k in range(1,Nh+1):
            cst = nd + (k-1)*2*nd
            cen = nd + (k-1)*2*nd + nd
            sst = nd + (k-1)*2*nd + nd
            sen = nd + (k-1)*2*nd + 2*nd                
            dFnldA = time2freq(dFnldxt*np.cos(k*t)-k*dFnldxdt*np.sin(k*t), Nh)
            dFnldB = time2freq(dFnldxt*np.sin(k*t)+k*dFnldxdt*np.cos(k*t), Nh)
            for ii in range(0,nd):
                Jnl[ii::nd,cst:cen] = dFnldA[:,ii*nd:(ii+1)*nd]
                Jnl[ii::nd,sst:sen] = dFnldB[:,ii*nd:(ii+1)*nd]
    elif fnform=='Freq':
        if deriv==True:
            Fnl, Jnl, dFnldW = Fnlfn(X, params)
        else:
            Fnl = Fnlfn(X, params)
            Jnl = np.zeros((Fnl.size,Fnl.size))
            Xp = X.copy()
            _h_ = 1e-3
            for k in range(0,Fnl.size):
                Xp[k,:] = Xp[k,:]+_h_
                Jnl[:,k] = (Fnlfn(Xp, params)-Fnl)/_h_
                Xp[k,:] = Xp[k,:]-_h_
                Xp[-1,:] = Xp[-1,:]+_h_
                dFnldW = (Fnlfn(Xp,params)-Fnl)/_h_
    else:
        raise ValueError('Unknown fnform')

    Fl = np.zeros((nd*(2*Nh+1),1))
    fh = params['fh']
    Fl[(nd+(fh-1)*2*nd):(2*nd+(fh-1)*2*nd),:] = params['Fc']
    Fl[(2*nd+(fh-1)*2*nd):(3*nd+(fh-1)*2*nd),:] = params['Fs']
    R = np.dot(E,X[:-1,:]) + Fnl - Fl
    dRdX = E+Jnl
    dRdW = np.dot(dEdw,X[:-1,:]) + dFnldW

    R = np.reshape(R,(R.size,))
    return R,dRdX,dRdW
