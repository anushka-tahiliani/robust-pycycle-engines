"""Complex-step safe math helpers and the mass-flow root solve shared by the initial guesses."""

import numpy as np
import openmdao.api as om


def abs_cs(x):
    """
    Complex-step safe version of numpy.abs.

    Parameters
    ----------
    x : float or ndarray
        Value to take the absolute value of.

    Returns
    -------
    float or ndarray
        Absolute value, preserving the imaginary perturbation.
    """
    if isinstance(x, np.ndarray):
        return np.where(x.real < 0.0, -x, x)
    if x.real < 0.0:
        return -x
    return x


def cs_clip(x, lo, hi):
    """
    Complex-step safe version of numpy.clip.

    Parameters
    ----------
    x : float or ndarray
        Value to clip.
    lo : float
        Lower bound.
    hi : float
        Upper bound.

    Returns
    -------
    ndarray
        Clipped value, preserving the imaginary perturbation.
    """
    x = np.asarray(x)
    x = np.where(x.real < lo, lo + 1j * x.imag, x)
    return np.where(x.real > hi, hi + 1j * x.imag, x)


def cs_max(x, lo=0.0):
    """
    Complex-step safe lower bound.

    Parameters
    ----------
    x : float or ndarray
        Value to bound.
    lo : float
        Lower bound.

    Returns
    -------
    ndarray
        Bounded value, preserving the imaginary perturbation.
    """
    x = np.asarray(x)
    return np.where(x.real < lo, lo + 1j * x.imag, x)


class _MassFlowBalance(om.ImplicitComponent):
    """Solve for the mass flow W at which the cycle output equals the target."""

    def initialize(self):
        self.options.declare('compute_cycle', types=object)
        self.options.declare('output', types=str)

    def setup(self):
        self.add_input('target', val=1000.0)
        self.add_output('W', val=50.0)
        self.declare_partials(of='W', wrt='*', method='cs')

    def apply_nonlinear(self, inputs, outputs, residuals):
        out = self.options['compute_cycle'](outputs['W'][0])
        residuals['W'] = out[self.options['output']] - inputs['target'][0]

    def solve_nonlinear(self, inputs, outputs):
        pass


def _newton_mass_flow(compute_cycle, target, W0, output):
    """Run a Newton solve on mass flow starting from W0 and return the converged value."""
    prob = om.Problem(reports=False)
    prob.model.add_subsystem(
        'balance', _MassFlowBalance(compute_cycle=compute_cycle, output=output), promotes=['*']
    )
    prob.model.nonlinear_solver = om.NewtonSolver(solve_subsystems=False)
    prob.model.nonlinear_solver.options['maxiter'] = 20
    prob.model.nonlinear_solver.options['iprint'] = 0
    prob.model.nonlinear_solver.options['err_on_non_converge'] = True
    prob.model.linear_solver = om.DirectSolver()
    prob.setup()
    prob.set_val('target', target)
    prob.set_val('W', W0)
    prob.run_model()
    return prob.get_val('W').item()


def _drop_zero_imag(value):
    """Return a real float if the value has no imaginary part, and the value itself otherwise."""
    if np.ndim(value) == 0 and np.imag(value) == 0.0:
        return float(np.real(value))
    return value


def solve_mass_flow(compute_cycle, target, W_array, output, flow):
    """
    Find the cycle whose output matches the target by sweeping and then refining mass flow.

    Parameters
    ----------
    compute_cycle : callable
        Closes the cycle for a given mass flow and returns a dictionary of cycle quantities.
    target : float
        Target value of the cycle output (thrust [lbf] or shaft power [hp]).
    W_array : ndarray
        Mass flows to sweep [kg/s].
    output : str
        Key of the cycle output in the compute_cycle result.
    flow : str
        Key of the swept mass flow in the compute_cycle result.

    Returns
    -------
    dict
        Cycle state at the solved mass flow. If several brackets exist, the one at the highest
        mass flow is used.

    Raises
    ------
    ValueError
        If no swept mass flow interval brackets the target.
    """
    sweep = [compute_cycle(W) for W in W_array]

    W_start = None
    for low, high in zip(sweep[:-1], sweep[1:]):
        out_low, out_high = np.real(low[output]), np.real(high[output])
        if out_low < target < out_high or out_high < target < out_low:
            W_start = 0.5 * (np.real(low[flow]) + np.real(high[flow]))
    if W_start is None:
        raise ValueError(
            f'No mass flow between {W_array[0]:.1f} and {W_array[-1]:.1f} kg/s brackets the '
            f'target {output} of {target:.1f}.'
        )

    cycle = compute_cycle(_newton_mass_flow(compute_cycle, target, W_start, output))
    return {key: _drop_zero_imag(value) for key, value in cycle.items()}


def compressor_exit(Tt_in, pi, eta_p, gamma=1.4):
    """
    Compute the exit total temperature and isentropic efficiency of a compressor.

    Parameters
    ----------
    Tt_in : float
        Inlet total temperature [K].
    pi : float
        Total pressure ratio. A stage with a pressure ratio of 1 is absent and does no work.
    eta_p : float
        Polytropic efficiency.
    gamma : float
        Ratio of specific heats.

    Returns
    -------
    Tt_out : float
        Exit total temperature [K].
    eta : float
        Isentropic efficiency, 1 for an absent stage.
    """
    if np.real(pi) == 1.0:
        return Tt_in, 1.0
    tau_ideal = pi ** ((gamma - 1) / gamma) - 1
    eta = tau_ideal / (pi ** ((gamma - 1) / (eta_p * gamma)) - 1)
    return Tt_in * (1 + tau_ideal / eta), eta


def split_pressure_ratio(opr, *stage_prs):
    """
    Give the axial HPC the part of the overall pressure ratio the other stages do not supply.

    Parameters
    ----------
    opr : float
        Overall pressure ratio.
    *stage_prs : float
        Pressure ratios of the other compression stages (fan, LPC, centrifugal compressor).

    Returns
    -------
    float
        Axial HPC pressure ratio, exactly 1 when the other stages supply the whole OPR.

    Raises
    ------
    ValueError
        If the other stages supply more than the overall pressure ratio.
    """
    pi_hpc = opr / np.prod(stage_prs)
    if abs(np.real(pi_hpc) - 1.0) < 1e-9:
        return 1.0
    if np.real(pi_hpc) < 1.0:
        raise ValueError(
            f'OPR {np.real(opr):.4g} is below the product of the other stage pressure ratios '
            f'{np.real(np.prod(stage_prs)):.4g}, which would need an axial HPC pressure ratio below 1.'
        )
    return pi_hpc
