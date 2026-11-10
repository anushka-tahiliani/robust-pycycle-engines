"""
pyCycle model of a direct-drive, two-spool, separate-flow turbofan.

Flow path: inlet, fan, splitter; the core runs LPC, axial HPC, centrifugal stage, combustor, cooled
HPT, LPT and core nozzle, and the bypass runs a bleed and the bypass nozzle. The LPC is left out
when `pi_lpc` is 1 and the centrifugal stage, on the LPC map as pyCycle has no centrifugal map,
when `pi_centrif` is 1. The design point is sized from the initial guess of
`robust_pycycle_engines.initial_guess.turbofan`; HP speed comes from the HPT AN**2 limit and LP
speed from the fan tip Mach number. Off-design throttles by a fraction of the maximum T4, and LPC
bleed can keep the LPC off its stall line, at or above an R-line schedule (`BleedMargin`).

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import openmdao.api as om
import pycycle.api as pyc

from robust_pycycle_engines.cycles import components
from robust_pycycle_engines.cycles.maps import FAN_MAP, HPC_MAP, LPC_MAP, maps_at_design_points
from robust_pycycle_engines.initial_guess._gas_path import COOL_IG_FRAC_WORK
from robust_pycycle_engines.initial_guess.balances import KG_TO_LBM, turbofan_design_guess
from robust_pycycle_engines.initial_guess.turbofan import DUCT_LOSSES

# Settings shared by every turbofan, overridden per engine by `engines.TURBOFAN_MODELS`
DEFAULT_SETTINGS = {
    'fan_tip_mach': 1.1,  # fan relative tip Mach number, sets LP speed
    'fan_face_mach': 0.751,  # axial Mach number at the fan face in the fan speed model
    # [lbm/s] upper bounds of the design and off-design mass flow; None scales the guessed design
    # mass flow by `W_MAX_FACTOR`
    'W_max_design': None,
    'W_max_off_design': None,
    'BPR_max': 20.0,  # upper bound of the off-design bypass ratio
    'N_min': 100.0,  # [rpm] lower bound of the off-design shaft speeds
    'maxiter': 15,  # Newton iterations coupling the design and off-design points
    'bleed_pc_start': 1.0,  # throttle where the LPC R-line schedule starts
    'design_alt': 35000.0,  # [ft] design altitude
    'map_points': {},  # design point of a compressor on its map, see `maps_at_design_points`
}

# Default map of each compressor
COMPRESSOR_MAPS = {'fan': FAN_MAP, 'lpc': LPC_MAP, 'hpc_axi': HPC_MAP, 'hpc_centri': LPC_MAP}

W_MAX_FACTOR = {'W_max_design': 3.0, 'W_max_off_design': 5.0}

# Design Mach numbers at each station
STATION_MN = {
    'inlet': 0.751,
    'fan': 0.4578,
    'splitter.MN1': 0.3104,
    'splitter.MN2': 0.4518,
    'duct4': 0.4,
    'lpc': 0.3059,
    'duct6': 0.3563,
    'hpc_axi': 0.25,
    'hpc_centri': 0.2,
    'bld3': 0.2,
    'burner': 0.15,
    'hpt': 0.3,
    'duct11': 0.3063,
    'lpt': 0.4,
    'duct13': 0.4,
    'byp_bld': 0.4489,
    'duct15': 0.4589,
}

# Losses and fixed fractions shared by the design and off-design points
CYCLE_PARAMS = {
    'inlet.ram_recovery': 0.999,
    'duct4.dPqP': 0.0,
    'duct6.dPqP': DUCT_LOSSES['duct6'],
    'burner.dPqP': 0.054,
    'duct11.dPqP': DUCT_LOSSES['duct11'],
    'duct13.dPqP': DUCT_LOSSES['duct13'],
    'duct15.dPqP': 0.0149,
    'core_nozz.Cv': 0.9933,
    'byp_bld.bypBld:frac_W': 0.0,
    'byp_nozz.Cv': 0.9939,
    'lpc.cool_IG:frac_work': COOL_IG_FRAC_WORK,
    'lpc.bleed_out:frac_work': 1.0,
}

# Map scalars and areas fixed at design and carried to off-design
_TURBOMACHINES = {
    'fan': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'lpc': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpc_axi': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpc_centri': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpt': ('s_PR', 's_Wp', 's_eff', 's_Np'),
    'lpt': ('s_PR', 's_Wp', 's_eff', 's_Np'),
}
_AREAS = [
    ('inlet.Fl_O:stat:area', 'inlet.area'),
    ('fan.Fl_O:stat:area', 'fan.area'),
    ('splitter.Fl_O1:stat:area', 'splitter.area1'),
    ('splitter.Fl_O2:stat:area', 'splitter.area2'),
    ('duct4.Fl_O:stat:area', 'duct4.area'),
    ('lpc.Fl_O:stat:area', 'lpc.area'),
    ('duct6.Fl_O:stat:area', 'duct6.area'),
    ('hpc_axi.Fl_O:stat:area', 'hpc_axi.area'),
    ('hpc_centri.Fl_O:stat:area', 'hpc_centri.area'),
    ('bld3.Fl_O:stat:area', 'bld3.area'),
    ('burner.Fl_O:stat:area', 'burner.area'),
    ('hpt.Fl_O:stat:area', 'hpt.area'),
    ('duct11.Fl_O:stat:area', 'duct11.area'),
    ('lpt.Fl_O:stat:area', 'lpt.area'),
    ('duct13.Fl_O:stat:area', 'duct13.area'),
    ('byp_bld.Fl_O:stat:area', 'byp_bld.area'),
    ('duct15.Fl_O:stat:area', 'duct15.area'),
    ('core_nozz.Throat:stat:area', 'balance.rhs:W'),
    ('byp_nozz.Throat:stat:area', 'balance.rhs:BPR'),
]


def _absent_elements(cycle):
    """Return the optional core elements left out of pyCycle: a stage is modeled only above 1."""
    absent = set()
    if cycle['pi_lpc'] <= 1.0:
        absent |= {'lpc', 'duct6'}
    if cycle['pi_centrif'] <= 1.0:
        absent.add('hpc_centri')
    return absent


def _present(name, absent):
    """Return whether a promoted name belongs to an element that is in the model."""
    return name.split('.')[0] not in absent


class Turbofan(pyc.Cycle):
    """
    One operating point of the turbofan.

    The design point balances mass flow to thrust, FAR to T4, turbine pressure ratios to shaft
    power, and HPT cooling to the cooling model. Off-design balances mass flow and BPR to the
    design nozzle throats, FAR to `PC` times `T4_max`, and shaft speeds to shaft power.
    """

    def initialize(self):
        self.options.declare('cycle', types=dict, desc='Design cycle state from the initial guess.')
        self.options.declare('settings', types=dict, desc='Engine settings, see DEFAULT_SETTINGS.')
        self.options.declare(
            'lpc_bleed',
            default=True,
            types=bool,
            desc='If True, LPC bleed keeps the LPC at or above its R-line schedule (`OneSidedBleed`).',
        )
        self.options.declare(
            'flow_scale', default=1.0, desc='Factor on the off-design mass flow guess.'
        )
        super().initialize()

    def setup(self):
        design = self.options['design']
        settings = dict(self.options['settings'])
        W_guess = self.options['cycle']['W_total'] * KG_TO_LBM
        for bound, factor in W_MAX_FACTOR.items():
            if settings[bound] is None:
                settings[bound] = factor * W_guess
        absent = _absent_elements(self.options['cycle'])
        maps = maps_at_design_points(COMPRESSOR_MAPS, settings['map_points'])
        self._apply_guess = True
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem(
            'fan',
            pyc.Compressor(map_data=maps['fan'], bleed_names=[], map_extrap=True),
            promotes_inputs=[('Nmech', 'LP_Nmech')],
        )
        self.add_subsystem('splitter', pyc.Splitter())
        self.add_subsystem('duct4', pyc.Duct())
        if 'lpc' not in absent:
            self.add_subsystem(
                'lpc',
                pyc.Compressor(
                    map_data=maps['lpc'], map_extrap=True, bleed_names=['cool_IG', 'bleed_out']
                ),
                promotes_inputs=[('Nmech', 'LP_Nmech')],
            )
            self.add_subsystem('duct6', pyc.Duct())
        self.add_subsystem(
            'hpc_axi',
            pyc.Compressor(map_data=maps['hpc_axi'], map_extrap=True),
            promotes_inputs=[('Nmech', 'HP_Nmech')],
        )
        if 'hpc_centri' not in absent:
            self.add_subsystem(
                'hpc_centri',
                pyc.Compressor(map_data=maps['hpc_centri'], map_extrap=True),
                promotes_inputs=[('Nmech', 'HP_Nmech')],
            )
        self.add_subsystem('bld3', pyc.BleedOut(bleed_names=['bld_inlet', 'bld_exit']))
        self.add_subsystem('burner', pyc.Combustor(fuel_type='FAR'))
        self.add_subsystem(
            'hpt',
            pyc.Turbine(
                map_data=pyc.HPTMap, bleed_names=['bld_inlet', 'bld_exit'], map_extrap=True
            ),
            promotes_inputs=[('Nmech', 'HP_Nmech')],
        )
        self.add_subsystem('duct11', pyc.Duct())
        self.add_subsystem(
            'lpt',
            pyc.Turbine(map_data=pyc.LPTMap, map_extrap=True),
            promotes_inputs=[('Nmech', 'LP_Nmech')],
        )
        self.add_subsystem('duct13', pyc.Duct())
        self.add_subsystem('core_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))
        self.add_subsystem('byp_bld', pyc.BleedOut(bleed_names=['bypBld']))
        self.add_subsystem('duct15', pyc.Duct())
        self.add_subsystem('byp_nozz', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        lp_spool = [e for e in ('fan', 'lpc', 'lpt') if e not in absent]
        hp_spool = [e for e in ('hpc_axi', 'hpc_centri', 'hpt') if e not in absent]
        self.add_subsystem(
            'lp_shaft', pyc.Shaft(num_ports=len(lp_spool)), promotes_inputs=[('Nmech', 'LP_Nmech')]
        )
        self.add_subsystem(
            'hp_shaft', pyc.Shaft(num_ports=len(hp_spool)), promotes_inputs=[('Nmech', 'HP_Nmech')]
        )
        self.add_subsystem('perf', pyc.Performance(num_nozzles=2, num_burners=1))

        self.connect('inlet.Fl_O:tot:P', 'perf.Pt2')
        self.connect(f'{hp_spool[-2]}.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('core_nozz.Fg', 'perf.Fg_0')
        self.connect('byp_nozz.Fg', 'perf.Fg_1')
        for shaft, spool in (('lp_shaft', lp_spool), ('hp_shaft', hp_spool)):
            for port, element in enumerate(spool):
                self.connect(f'{element}.trq', f'{shaft}.trq_{port}')
        # Nozzles expand to ambient pressure
        self.connect('fc.Fl_O:stat:P', 'core_nozz.Ps_exhaust')
        self.connect('fc.Fl_O:stat:P', 'byp_nozz.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            self._setup_design(balance, settings)
        else:
            self._setup_off_design(balance, settings)

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I')
        self.pyc_connect_flow('inlet.Fl_O', 'fan.Fl_I')
        self.pyc_connect_flow('fan.Fl_O', 'splitter.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O1', 'duct4.Fl_I')
        core = [
            e for e in ('duct4', 'lpc', 'duct6', 'hpc_axi', 'hpc_centri', 'bld3') if e not in absent
        ]
        for upstream, downstream in zip(core[:-1], core[1:]):
            self.pyc_connect_flow(f'{upstream}.Fl_O', f'{downstream}.Fl_I')
        self.pyc_connect_flow('bld3.Fl_O', 'burner.Fl_I')
        self.pyc_connect_flow('burner.Fl_O', 'hpt.Fl_I')
        self.pyc_connect_flow('hpt.Fl_O', 'duct11.Fl_I')
        self.pyc_connect_flow('duct11.Fl_O', 'lpt.Fl_I')
        self.pyc_connect_flow('lpt.Fl_O', 'duct13.Fl_I')
        self.pyc_connect_flow('duct13.Fl_O', 'core_nozz.Fl_I')
        self.pyc_connect_flow('splitter.Fl_O2', 'byp_bld.Fl_I')
        self.pyc_connect_flow('byp_bld.Fl_O', 'duct15.Fl_I')
        self.pyc_connect_flow('duct15.Fl_O', 'byp_nozz.Fl_I')
        # HPT cooling flows, injected at the NGV inlet and after the first rotor
        self.pyc_connect_flow('bld3.bld_inlet', 'hpt.bld_inlet', connect_stat=False)
        self.pyc_connect_flow('bld3.bld_exit', 'hpt.bld_exit', connect_stat=False)

        newton = self.nonlinear_solver = om.NewtonSolver()
        newton.options['atol'] = 1e-6
        newton.options['rtol'] = 1e-6
        newton.options['iprint'] = 2
        newton.options['maxiter'] = 50
        newton.options['solve_subsystems'] = True
        newton.options['max_sub_solves'] = 50
        newton.options['err_on_non_converge'] = True
        newton.options['reraise_child_analysiserror'] = False
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        self.linear_solver = om.DirectSolver()

        super().setup()

    def _setup_design(self, balance, settings):
        balance.add_balance(
            'W', lower=10.0, upper=settings['W_max_design'], units='lbm/s', eq_units='lbf'
        )
        self.connect('balance.W', 'fc.W')
        self.connect('perf.Fn', 'balance.lhs:W')
        self.promotes('balance', inputs=[('rhs:W', 'Fn_DES')])

        balance.add_balance('FAR', eq_units='degR', lower=1e-4)
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

        # Turbine pressure ratios zero the net shaft power
        balance.add_balance('lpt_PR', lower=1.001, eq_units='hp', use_mult=True, mult_val=-1)
        self.connect('balance.lpt_PR', 'lpt.PR')
        self.connect('lp_shaft.pwr_in_real', 'balance.lhs:lpt_PR')
        self.connect('lp_shaft.pwr_out_real', 'balance.rhs:lpt_PR')

        balance.add_balance('hpt_PR', lower=1.001, eq_units='hp', use_mult=True, mult_val=-1)
        self.connect('balance.hpt_PR', 'hpt.PR')
        self.connect('hp_shaft.pwr_in_real', 'balance.lhs:hpt_PR')
        self.connect('hp_shaft.pwr_out_real', 'balance.rhs:hpt_PR')

        # HPC efficiency, reduced when the last-stage blades are short
        self.add_subsystem('hpc_tip_clearance', components.hpc_tip_clearance())
        self.connect('hpc_axi.Fl_O:stat:area', 'hpc_tip_clearance.A3')
        self.add_subsystem('hpc_axi_em', components.isentropic_efficiency('eta_phpc_axi'))
        self.connect('hpc_tip_clearance.eta_phpc_axi_tl', 'hpc_axi_em.eta_phpc_axi')
        self.connect('hpc_axi_em.eff', 'hpc_axi.eff')

        # Shaft speeds: HP from the HPT stress limit, LP from the fan tip Mach number
        self.add_subsystem('hp_speed', components.hp_speed())
        self.connect('hpt.Fl_O:stat:area', 'hp_speed.A_hpt_exit')
        self.connect('hp_speed.N_hpt', 'HP_Nmech')
        self.add_subsystem('fan_radius', components.fan_radius())
        self.connect('inlet.Fl_O:stat:area', 'fan_radius.area')
        self.add_subsystem(
            'fan_speed', components.fan_speed(settings['fan_tip_mach'], settings['fan_face_mach'])
        )
        self.connect('fan_radius.FanRad', 'fan_speed.r_tip')
        self.connect('fc.Fl_O:stat:T', 'fan_speed.T_0')
        self.connect('fan_speed.N_fan', 'LP_Nmech')

        components.add_hpt_cooling(self, balance)

    def _setup_off_design(self, balance, settings):
        # Throttle: T4 is the fraction PC of T4_max
        balance.add_balance('FAR', val=0.017, lower=1e-4, eq_units='degR', use_mult=True)
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.rhs:FAR')
        self.promotes('balance', inputs=[('mult:FAR', 'PC'), ('lhs:FAR', 'T4_max')])

        balance.add_balance(
            'W', units='lbm/s', lower=10.0, upper=settings['W_max_off_design'], eq_units='inch**2'
        )
        self.connect('balance.W', 'fc.W')
        self.connect('core_nozz.Throat:stat:area', 'balance.lhs:W')

        balance.add_balance(
            'BPR', lower=1.00001, upper=settings['BPR_max'], eq_units='inch**2', use_mult=True
        )
        self.connect('balance.BPR', 'splitter.BPR')
        self.connect('byp_nozz.Throat:stat:area', 'balance.lhs:BPR')

        # Shaft speeds zero the net shaft power
        for shaft, spool in (('lp_shaft', 'LP_Nmech'), ('hp_shaft', 'HP_Nmech')):
            balance.add_balance(
                spool,
                val=1.5,
                units='rpm',
                lower=settings['N_min'],
                eq_units='hp',
                use_mult=True,
                mult_val=-1,
            )
            self.connect(f'balance.{spool}', spool)
            self.connect(f'{shaft}.pwr_in_real', f'balance.lhs:{spool}')
            self.connect(f'{shaft}.pwr_out_real', f'balance.rhs:{spool}')

        if self.options['lpc_bleed']:
            self.add_subsystem(
                'bleed_margin', components.BleedMargin(pc_start=settings['bleed_pc_start'])
            )
            self.promotes('bleed_margin', inputs=['PC'])
            self.add_subsystem('bleed', components.OneSidedBleed())
            self.connect('bleed.bleed_frac', 'lpc.bleed_out:frac_W')
            self.connect('lpc.map.RlineMap', 'bleed.RlineMap')
            self.connect('bleed_margin.LPC_RlineMap', 'bleed.RlineTarget')

        # Design fan speed, used to seed the LP speed
        self.add_subsystem(
            'lp_seed',
            om.ExecComp(
                'LP_Nmech_seed = N_fan',
                N_fan={'val': 6000.0, 'units': 'rpm'},
                LP_Nmech_seed={'val': 6000.0, 'units': 'rpm'},
            ),
            promotes=['*'],
        )

    def guess_nonlinear(self, inputs, outputs, residuals):
        """
        Seed the balances from the initial guess.

        The design point is reseeded on every solve. An off-design point is seeded with the
        design state once, until `reset_guess` is called; between throttles the deck generator
        seeds it with the last converged point instead.
        """
        cycle = self.options['cycle']
        if self.options['design']:
            for name, value in turbofan_design_guess(cycle).items():
                outputs[name] = value
            return
        if not self._apply_guess:
            return
        inlet = (cycle['Tt0'], cycle['Pt0'] * CYCLE_PARAMS['inlet.ram_recovery'])
        T_ratio, P_ratio = components.guess_off_design_statics(
            inputs, outputs, cycle, {'inlet': (*inlet, STATION_MN['inlet'])}
        )
        # Design corrected flow at the off-design inlet conditions: the design mass flow itself
        # can exceed the choked flow of the fixed inlet area
        W_guess = cycle['W_total'] * KG_TO_LBM * P_ratio / T_ratio**0.5
        outputs['balance.W'] = W_guess * self.options['flow_scale']
        outputs['balance.HP_Nmech'] = cycle['HP_Nmech_max']
        outputs['balance.LP_Nmech'] = inputs['N_fan']
        outputs['balance.FAR'] = cycle['FAR']
        outputs['balance.BPR'] = cycle['BPR']
        self._apply_guess = False

    def reset_guess(self):
        """Seed the off-design point from the initial guess again on its next solve."""
        self._apply_guess = True


class MPTurbofan(pyc.MPCycle):
    """
    Design point `DESIGN` and one off-design point `OD` of a turbofan.

    The off-design HPT cooling flows match the design cooling mass flows, and the design map
    scalars and flow areas fix the off-design geometry.
    """

    def initialize(self):
        self.options.declare('engine', types=dict, desc='Entry of `engines.TURBOFANS`.')
        self.options.declare('cycle', types=dict, desc='Design cycle state from the initial guess.')
        self.options.declare('settings', types=dict, desc='Engine settings, see DEFAULT_SETTINGS.')
        self.options.declare(
            'lpc_bleed', default=True, types=bool, desc='Off-design LPC bleed, see `Turbofan`.'
        )
        self.options.declare(
            'flow_scale', default=1.0, desc='Factor on the off-design mass flow guess.'
        )
        self.options.declare(
            'flight_condition',
            default=None,
            desc='(Mach number, altitude [ft]) of the off-design points, to keep their fan face '
            'or inlet subsonic.',
        )
        super().initialize()

    def setup(self):
        engine = self.options['engine']
        cycle = self.options['cycle']
        settings = self.options['settings']
        absent = _absent_elements(cycle)
        lpc_bleed = self.options['lpc_bleed'] and 'lpc' not in absent

        if cycle['pi_hpc'] <= 1.0:
            raise ValueError('The turbofan cycle model needs an axial HPC; raise opr.')

        # Pressure ratios as split by the initial guess, so the guess and the model agree
        dv = self.add_subsystem('DV', om.IndepVarComp())
        dv.add_output('fan_PR', val=engine['pi_fan'])
        dv.add_output('hpc_axi_PR', val=cycle['pi_hpc'])
        if 'lpc' not in absent:
            dv.add_output('lpc_PR', val=cycle['pi_lpc'])
        if 'hpc_centri' not in absent:
            dv.add_output('hpc_centri_PR', val=cycle['pi_centrif'])

        # Compressor efficiencies from the polytropic efficiencies of the initial guess
        efficiencies = {
            'fan': (engine['pi_fan'], engine.get('eta_pfan', 0.9)),
            'lpc': (cycle['pi_lpc'], engine.get('eta_plpc', 0.9)),
            'hpc_centri': (cycle['pi_centrif'], engine.get('eta_pcentrif', 0.85)),
        }
        components.add_compressor_efficiencies(
            self, {comp: eta for comp, eta in efficiencies.items() if comp not in absent}
        )
        self.connect('DV.hpc_axi_PR', 'DESIGN.hpc_axi_em.PR')

        self.pyc_add_pnt(
            'DESIGN', Turbofan(design=True, thermo_method='TABULAR', cycle=cycle, settings=settings)
        )
        for comp in ('fan', 'lpc', 'hpc_axi', 'hpc_centri'):
            if comp not in absent:
                self.connect(f'DV.{comp}_PR', f'DESIGN.{comp}.PR')

        self.set_input_defaults('DESIGN.splitter.BPR', engine['BPR'])
        for station, MN in STATION_MN.items():
            if _present(station, absent):
                name = station if '.' in station else f'{station}.MN'
                self.set_input_defaults(f'DESIGN.{name}', MN)

        for name, value in CYCLE_PARAMS.items():
            if _present(name, absent):
                self.pyc_add_cycle_param(name, value)
        self.pyc_add_cycle_param('hp_shaft.HPX', engine['pwr_offtk_hpt'], units='hp')
        if 'lpc' not in absent:
            self.pyc_add_cycle_param('lpc.cool_IG:frac_W', engine.get('coolIG_frac_W', 0.0))
            if not lpc_bleed:
                self.pyc_add_cycle_param('lpc.bleed_out:frac_W', 0.0)

        self.pyc_add_pnt(
            'OD',
            Turbofan(
                design=False,
                thermo_method='TABULAR',
                cycle=cycle,
                settings=settings,
                lpc_bleed=lpc_bleed,
                flow_scale=self.options['flow_scale'],
            ),
        )
        self.set_input_defaults('OD.fc.MN', engine['M'])
        self.set_input_defaults('OD.fc.alt', settings['design_alt'], units='ft')
        self.connect('DESIGN.fan_speed.N_fan', 'OD.N_fan')

        # Off-design cooling fractions deliver the design cooling mass flows
        bal = self.add_subsystem('OD_bleedW_bal', om.BalanceComp())
        for port, name in (('bld_inlet', 'bld_inlet_fracW'), ('bld_exit', 'bld_exit_fracW')):
            bal.add_balance(name, val=0.02, lower=0.0, upper=1, eq_units='lbm/s')
            self.connect(f'OD.bld3.{port}:stat:W', f'OD_bleedW_bal.lhs:{name}')
            self.connect(f'DESIGN.bld3.{port}:stat:W', f'OD_bleedW_bal.rhs:{name}')
            self.connect(f'OD_bleedW_bal.{name}', f'OD.bld3.{port}:frac_W')

        for comp, scalars in _TURBOMACHINES.items():
            if comp not in absent:
                for scalar in scalars:
                    self.pyc_connect_des_od(f'{comp}.{scalar}', f'{comp}.{scalar}')
        for des, od in _AREAS:
            if _present(des, absent):
                self.pyc_connect_des_od(des, od)

        newton = self.nonlinear_solver = om.NewtonSolver(solve_subsystems=True)
        newton.options['atol'] = 1e-4
        # Absolute test only: after a large first residual, a relative test passes too early
        newton.options['rtol'] = 0.0
        newton.options['maxiter'] = settings['maxiter']
        newton.options['iprint'] = 2
        newton.options['err_on_non_converge'] = True
        newton.linesearch = om.BoundsEnforceLS()
        newton.linesearch.options['bound_enforcement'] = 'scalar'
        newton.linesearch.options['iprint'] = -1
        self.linear_solver = om.DirectSolver()

        super().setup()

    def configure(self):
        """Keep the off-design inlet subsonic at the flight condition, if one is given."""
        super().configure()
        if self.options['flight_condition'] is not None:
            components.limit_inlet_to_subsonic(self, 'OD', *self.options['flight_condition'])
