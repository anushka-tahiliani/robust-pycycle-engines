"""
OpenMDAO components the cycle models add to pyCycle's elements.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import openmdao.api as om
import pycycle.api as pyc
from pycycle.elements.US1976 import P_interp, T_interp

from robust_pycycle_engines.cycles.cooling import TurbineCooling
from robust_pycycle_engines.initial_guess.balances import K_TO_R, PA_TO_PSI


def guess_off_design_statics(inputs, outputs, cycle, stations):
    """
    Seed an off-design point's freestream totals and station statics from its flight condition.

    Off-design, each station's statics are solved for its fixed design area. Started from stale
    values, those solves can stall at their bounds or return NaN while the balances are fine.
    Station totals are taken from the design state of the initial guess, with temperatures
    scaled by the freestream total temperature and pressures by the freestream total pressure,
    as at a fixed corrected operating point.

    Parameters
    ----------
    inputs : openmdao.vectors.Vector
        Inputs of the cycle, including those of its `fc` element.
    outputs : openmdao.vectors.Vector
        Outputs of the cycle to seed.
    cycle : dict
        Initial guess, with the design freestream totals `Tt0` [K] and `Pt0` [Pa].
    stations : dict
        (Tt [K], Pt [Pa], MN) of each element's exit at design, keyed by element name.

    Returns
    -------
    tuple of float
        Ratios of the off-design to design freestream total temperature and total pressure.
    """
    alt = inputs['fc.ambient.readAtmTable.alt']  # [ft]
    ram = 1.0 + 0.2 * inputs['fc.conv.fs.exit_static.ps_resid.MN'] ** 2
    Tt0 = T_interp(alt) * ram  # [degR]
    Pt0 = P_interp(alt) * ram**3.5  # [psi]
    outputs['fc.balance.Tt'] = Tt0
    outputs['fc.balance.Pt'] = Pt0
    T_ratio = Tt0 / (cycle['Tt0'] * K_TO_R)
    P_ratio = Pt0 / (cycle['Pt0'] * PA_TO_PSI)
    for element, (Tt, Pt, MN) in stations.items():
        stat = 1.0 + 0.2 * MN**2
        outputs[f'{element}.out_stat.balance.T'] = Tt * T_ratio / stat  # [degK]
        outputs[f'{element}.out_stat.ps_resid.Ps'] = Pt * P_ratio / stat**3.5 / 1e5  # [bar]
    return float(T_ratio[0]), float(P_ratio[0])


def limit_inlet_to_subsonic(model, point, MN, alt):
    """
    Bound an off-design point's inlet static state so that the inlet exit stays subsonic.

    The inlet static state is solved for a fixed area and mass flow, which has a subsonic and a
    supersonic solution. Only the subsonic one is physical, so the sonic static temperature and
    pressure at the flight condition are the lower bounds of the state. Call from the model's
    `configure`.

    Parameters
    ----------
    model : openmdao.api.Group
        Multipoint model holding the off-design point.
    point : str
        Name of the off-design point.
    MN : float
        Mach number of the flight condition.
    alt : float
        Altitude of the flight condition [ft].
    """
    ram = 1.0 + 0.2 * MN**2
    Tt = float(T_interp(alt)) * ram / K_TO_R  # [degK]
    Pt = float(P_interp(alt)) * ram**3.5 / PA_TO_PSI / 1e5  # [bar]
    stat = f'{point}.inlet.out_stat'
    model.set_output_solver_options(f'{stat}.balance.T', lower=Tt / 1.2)
    model.set_output_solver_options(f'{stat}.ps_resid.Ps', lower=Pt / 1.2**3.5)


def isentropic_efficiency(eta_p_name, eta_p=0.9, constant=False):
    """
    Build a component converting a compressor's polytropic efficiency to isentropic.

    Parameters
    ----------
    eta_p_name : str
        Name of the polytropic efficiency input.
    eta_p : float
        Polytropic efficiency.
    constant : bool
        If True, the polytropic efficiency is fixed rather than an input.

    Returns
    -------
    openmdao.api.ExecComp
        Component with inputs `PR` and `eta_p_name`, and output `eff`.
    """
    return om.ExecComp(
        f'eff = ((PR)**((gamma - 1)/gamma) - 1) / ((PR)**((gamma - 1)/({eta_p_name} * gamma)) - 1)',
        PR={'val': 2.0},
        gamma={'val': 1.4, 'constant': True},
        eff={'val': 0.9},
        **{eta_p_name: {'val': eta_p, 'constant': constant}},
    )


def add_compressor_efficiencies(mp, compressors):
    """
    Set design compressor efficiencies from polytropic efficiencies and pressure ratios.

    Parameters
    ----------
    mp : pycycle.api.MPCycle
        Multipoint cycle during its setup, with pressure ratio outputs `DV.<comp>_PR` and a
        design point `DESIGN`.
    compressors : dict
        (pressure ratio, polytropic efficiency) keyed by compressor name. A compressor with a
        pressure ratio of 1 does no work and gets an efficiency of 1.
    """
    for comp, (PR, eta_p) in compressors.items():
        if PR == 1.0:
            mp.set_input_defaults(f'DESIGN.{comp}.eff', 1.0)
            continue
        name = f'eta_p{comp}'
        mp.add_subsystem(f'{comp}_em', isentropic_efficiency(name, eta_p=eta_p, constant=True))
        mp.connect(f'DV.{comp}_PR', f'{comp}_em.PR')
        mp.connect(f'{comp}_em.eff', f'DESIGN.{comp}.eff')


class BleedMargin(om.ExplicitComponent):
    """
    LPC R-line schedule for the off-design bleed, linear in throttle.

    The R-line falls from `rline_start` at throttle `pc_start` to `rline_end` at `pc_end`, keeping
    the LPC away from stall as the engine throttles back.
    """

    def initialize(self):
        self.options.declare('pc_start', default=1.0, desc='Throttle where the line starts.')
        self.options.declare('rline_start', default=1.6, desc='R-line at `pc_start`.')
        self.options.declare('pc_end', default=0.625, desc='Throttle where the line ends.')
        self.options.declare('rline_end', default=1.1, desc='R-line at `pc_end`.')

    def setup(self):
        self.add_input('PC', val=1.0, desc='Throttle, as a fraction of maximum T4.')
        self.add_output('LPC_RlineMap', val=1.0, desc='Target LPC R-line.')
        self.declare_partials('LPC_RlineMap', 'PC', val=self._slope())

    def _slope(self):
        opts = self.options
        return (opts['rline_end'] - opts['rline_start']) / (opts['pc_end'] - opts['pc_start'])

    def compute(self, inputs, outputs):
        opts = self.options
        outputs['LPC_RlineMap'] = (
            self._slope() * (inputs['PC'] - opts['pc_start']) + opts['rline_start']
        )


class OneSidedBleed(om.ImplicitComponent):
    """
    LPC bleed that acts only when the LPC would otherwise fall below its R-line schedule.

    Either the bleed is zero with the R-line at or above the schedule, or the bleed is positive
    with the R-line on the schedule. The Fischer-Burmeister function writes that condition as one
    smooth residual, so Newton needs no switch between the two cases.
    """

    def initialize(self):
        self.options.declare('smoothing', default=1e-4, desc='Smoothing of the residual.')

    def setup(self):
        self.add_input('RlineMap', val=1.6, desc='LPC R-line.')
        self.add_input('RlineTarget', val=1.6, desc='LPC R-line schedule.')
        self.add_output('bleed_frac', val=0.0, lower=0.0, upper=0.2, desc='LPC bleed fraction.')
        self.declare_partials('bleed_frac', '*')

    def apply_nonlinear(self, inputs, outputs, residuals):
        bleed = outputs['bleed_frac']
        margin = inputs['RlineMap'] - inputs['RlineTarget']
        norm = (bleed**2 + margin**2 + self.options['smoothing'] ** 2) ** 0.5
        residuals['bleed_frac'] = bleed + margin - norm

    def linearize(self, inputs, outputs, partials):
        bleed = outputs['bleed_frac']
        margin = inputs['RlineMap'] - inputs['RlineTarget']
        norm = (bleed**2 + margin**2 + self.options['smoothing'] ** 2) ** 0.5
        partials['bleed_frac', 'bleed_frac'] = 1.0 - bleed / norm
        partials['bleed_frac', 'RlineMap'] = 1.0 - margin / norm
        partials['bleed_frac', 'RlineTarget'] = margin / norm - 1.0


def hp_speed():
    """
    Build the stress-limited HP shaft speed of a shroudless nickel turbine (AN**2 limit).

    Returns
    -------
    openmdao.api.ExecComp
        Component with input `A_hpt_exit` and output `N_hpt`, with a 1.4 factor of safety.
    """
    return om.ExecComp(
        'N_hpt = ((((3/2)*0.25*UTS_Ni)*3600/(2*pi*rho_Ni*A_hpt_exit))**0.5)/1.4',
        rho_Ni={'val': 8500.0, 'units': 'kg/m**3'},
        UTS_Ni={'val': 1.2e9, 'units': 'Pa'},
        A_hpt_exit={'val': 0.15, 'units': 'm**2'},
        N_hpt={'val': 15000.0, 'units': 'rpm'},
    )


def ip_speed():
    """
    Build the stress-limited IP shaft speed of a steel LPT (AN**2 limit).

    Returns
    -------
    openmdao.api.ExecComp
        Component with input `A_lpt_exit` and output `N_lpt`, with a 1.4 factor of safety.
    """
    return om.ExecComp(
        'N_lpt = ((sigma_lpt * 3600.0) / (2.0 * pi * rho_Steel * A_lpt_exit)) ** 0.5 / 1.4',
        sigma_lpt={'val': (3 / 2) * 0.25 * 1.05e9, 'units': 'Pa'},
        rho_Steel={'val': 7900.0, 'units': 'kg/m**3'},
        A_lpt_exit={'val': 0.15, 'units': 'm**2'},
        N_lpt={'val': 15000.0, 'units': 'rpm'},
    )


def add_hpt_cooling(cycle, balance):
    """
    Size the design HPT cooling flows with the two-stage turbine cooling model of `cooling`.

    The NGV cooling leaves `bld3` through port `bld_inlet` and the rotor and second-stage cooling
    through port `bld_exit`; balances on `balance` set both bleed fractions.

    Parameters
    ----------
    cycle : pycycle.api.Cycle
        Design point with elements `bld3`, `burner` and `hpt`, during its setup.
    balance : openmdao.api.BalanceComp
        Balance of the design point.
    """
    cycle.add_subsystem(
        'hpt_cooling',
        TurbineCooling(n_stages=2, thermo_data=pyc.species_data.janaf, T_metal=2460.0),
    )
    cycle.add_subsystem('hpt_post_NGV', pyc.CombineCooling(n_ins=3))
    cycle.pyc_connect_flow('bld3.bld_inlet', 'hpt_cooling.Fl_cool', connect_stat=False)
    cycle.pyc_connect_flow('burner.Fl_O', 'hpt_cooling.Fl_turb_I')
    cycle.pyc_connect_flow('hpt.Fl_O', 'hpt_cooling.Fl_turb_O')
    for row in (1, 2, 3):
        cycle.connect(f'hpt_cooling.row_{row}.W_cool', f'hpt_post_NGV.W_{row}')
    cycle.connect('hpt.power', 'hpt_cooling.turb_pwr')

    balance.add_balance('cool_NGV_frac_W', lower=0.0, upper=0.3, eq_units='lbm/s')
    cycle.connect('balance.cool_NGV_frac_W', 'bld3.bld_inlet:frac_W')
    cycle.connect('bld3.bld_inlet:stat:W', 'balance.lhs:cool_NGV_frac_W')
    cycle.connect('hpt_cooling.row_0.W_cool', 'balance.rhs:cool_NGV_frac_W')

    balance.add_balance('cool_turb_frac_W', lower=0.0, upper=0.3, eq_units='lbm/s')
    cycle.connect('balance.cool_turb_frac_W', 'bld3.bld_exit:frac_W')
    cycle.connect('bld3.bld_exit:stat:W', 'balance.lhs:cool_turb_frac_W')
    cycle.connect('hpt_post_NGV.W_cool', 'balance.rhs:cool_turb_frac_W')


def fan_radius(hub_tip=0.3125):
    """
    Build the fan tip radius from the fan face area.

    Parameters
    ----------
    hub_tip : float
        Fan hub-to-tip radius ratio.

    Returns
    -------
    openmdao.api.ExecComp
        Component with input `area` and output `FanRad`.
    """
    return om.ExecComp(
        'FanRad = (area/(pi*(1.0 - hub_tip**2.0)))**0.5',
        area={'val': 7000.0, 'units': 'm**2'},
        hub_tip={'val': hub_tip, 'units': None},
        FanRad={'val': 50.0, 'units': 'm'},
    )


def fan_speed(M_tip, M_2):
    """
    Build the fan speed that gives a relative tip Mach number at the fan face.

    Parameters
    ----------
    M_tip : float
        Fan relative tip Mach number.
    M_2 : float
        Axial Mach number at the fan face.

    Returns
    -------
    openmdao.api.ExecComp
        Component with inputs `r_tip` and `T_0` and output `N_fan`.
    """
    return om.ExecComp(
        'N_fan = (60/(2*pi))*((M_tip**2 - M_2**2)*gamma*R*T_0/(1 + 0.5*(gamma - 1)*M_2**2))**0.5'
        ' / r_tip',
        M_tip={'val': M_tip, 'units': None},
        M_2={'val': M_2, 'units': None},
        gamma={'val': 1.4, 'units': None},
        R={'val': 287.0, 'units': 'J/(kg*K)'},
        T_0={'val': 288.0, 'units': 'K'},
        r_tip={'val': 1.0, 'units': 'm'},
        N_fan={'val': 6000.0, 'units': 'rpm'},
    )


def hpc_tip_clearance():
    """
    Build the HPC last-stage blade height and its smooth tip-clearance efficiency penalty.

    The penalty applies only when the blade is shorter than `height_ref`, using a smooth ramp so
    Newton sees continuous derivatives.

    Returns
    -------
    openmdao.api.Group
        Group with input `A3` and output `eta_phpc_axi_tl`.
    """
    group = om.Group()
    group.add_subsystem(
        'blade_h',
        om.ExecComp(
            'height = (1 - hub_tip_ratio) * power(A3 / (pi * (1 - hub_tip_ratio**2)), 0.5) * 1000',
            A3={'val': 0.05, 'units': 'm**2'},
            hub_tip_ratio={'val': 0.93, 'constant': True},
            height={'val': 10.0, 'units': 'mm'},
        ),
        promotes_inputs=['A3'],
    )
    group.add_subsystem(
        'loss',
        om.ExecComp(
            'eta_phpc_axi_tl = eta_phpc_axi - 0.06 * ( '
            ' (1.0 - power(height/height_ref, 0.6)) '
            '+ power( (1.0 - power(height/height_ref, 0.6))**2 + 1e-12, 0.5 ) '
            ')',
            height={'val': 15.0, 'units': 'mm'},
            height_ref={'val': 13.0, 'units': 'mm'},
            eta_phpc_axi={'val': 0.9},
            eta_phpc_axi_tl={'val': 0.85},
        ),
        promotes_outputs=['eta_phpc_axi_tl'],
    )
    group.connect('blade_h.height', 'loss.height')
    return group
