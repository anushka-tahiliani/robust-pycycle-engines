"""
pyCycle model of a three-spool turboshaft with a free power turbine.

Flow path: inlet, LPC, axial HPC, centrifugal stage, combustor, cooled HPT, LPT and power turbine,
then the exhaust nozzle. A compressor is modeled only with a pressure ratio above 1, so the LPC
(and with it the LPT and IP shaft), the axial HPC or the centrifugal stage can be left out. The LPC
and centrifugal stage use the LPC map, as pyCycle has no centrifugal map. The design point is sized
from the initial guess of `robust_pycycle_engines.initial_guess.turboshaft`; HP and IP speeds come
from turbine AN**2 limits. Off-design throttles by a fraction `PC` of the maximum T4, and LPC bleed
can keep the LPC off its stall line at part throttle.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import openmdao.api as om
import pycycle.api as pyc

from robust_pycycle_engines.cycles import components
from robust_pycycle_engines.cycles.maps import HPC_MAP, LPC_MAP, maps_at_design_points
from robust_pycycle_engines.initial_guess._gas_path import COOL_IG_FRAC_WORK
from robust_pycycle_engines.initial_guess.balances import turboshaft_design_guess

# Settings shared by every turboshaft, overridden per engine by `engines.TURBOSHAFT_MODELS`
DEFAULT_SETTINGS = {
    'maxiter': 15,  # Newton iterations coupling the design and off-design points
    'design_alt': 0.0,  # [ft] design altitude
    'cycle_params': {},  # overrides of `CYCLE_PARAMS`, such as duct losses
    'lpc_bleed': False,  # off-design LPC bleed keeps the LPC at or above its R-line schedule
    'bleed_pc_start': 1.0,  # throttle where the LPC R-line schedule starts
    'map_points': {},  # design point of a compressor on its map, see `maps_at_design_points`
}

# Default map of each compressor
COMPRESSOR_MAPS = {'lpc': LPC_MAP, 'hpc_axi': HPC_MAP, 'hpc_centri': LPC_MAP}

# Design Mach numbers at each station
STATION_MN = {
    'inlet': 0.4,
    'duct225': 0.4,
    'lpc': 0.3,
    'icduct': 0.3,
    'bld25': 0.25,
    'hpc_axi': 0.25,
    'hpc_centri': 0.2,
    'bld3': 0.2,
    'burner': 0.15,
    'hpt': 0.3,
    'lpt': 0.4,
    'itduct': 0.4,
    'pt': 0.4,
}


def _exit_totals(cycle):
    """Return the design (Tt [K], Pt [Pa]) at each element's exit from the initial guess."""
    inlet = (cycle['Tt2'], cycle['Pt2'])
    lpc = (cycle['Tt24'], cycle['Pt24'])
    axial = (cycle['Tt3'], cycle['Pt3'])
    centrif = (cycle['Tt3_centrif'], cycle['Pt3_centrif'])
    lpt = (cycle['Tt45_lpt'], cycle['Pt45_lpt'])
    return {
        'inlet': inlet,
        'duct225': inlet,
        'lpc': lpc,
        'icduct': lpc,
        'hpc_axi': axial,
        'bld25': axial,
        'hpc_centri': centrif,
        'bld3': centrif,
        'burner': (cycle['Tt4'], cycle['Pt4'] * (1.0 - CYCLE_PARAMS['burner.dPqP'])),
        'hpt': (cycle['Tt45'], cycle['Pt45']),
        'lpt': lpt,
        'itduct': lpt,
        'pt': (cycle['Tt5'], cycle['Pt5']),
    }


# Losses and fixed fractions shared by the design and off-design points
CYCLE_PARAMS = {
    'inlet.ram_recovery': 1.0,
    'duct225.dPqP': 0.0,
    'icduct.dPqP': 0.0,
    'bld25.cool1:frac_W': 0.0,
    'bld25.cool2:frac_W': 0.0,
    'burner.dPqP': 0.05,
    'itduct.dPqP': 0.0,
    'nozzle.Cv': 0.99,
    'lpc.cool_IG:frac_work': COOL_IG_FRAC_WORK,
    'lpc.bleed_out:frac_work': 1.0,
    'lpt.cool1:frac_P': 1.0,
    'lpt.cool2:frac_P': 0.0,
}

# Map scalars and areas fixed at design and carried to off-design
_TURBOMACHINES = {
    'lpc': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpc_axi': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpc_centri': ('s_PR', 's_Wc', 's_eff', 's_Nc'),
    'hpt': ('s_PR', 's_Wp', 's_eff', 's_Np'),
    'lpt': ('s_PR', 's_Wp', 's_eff', 's_Np'),
    'pt': ('s_PR', 's_Wp', 's_eff', 's_Np'),
}
_AREAS = [
    ('inlet.Fl_O:stat:area', 'inlet.area'),
    ('duct225.Fl_O:stat:area', 'duct225.area'),
    ('lpc.Fl_O:stat:area', 'lpc.area'),
    ('icduct.Fl_O:stat:area', 'icduct.area'),
    ('bld25.Fl_O:stat:area', 'bld25.area'),
    ('hpc_axi.Fl_O:stat:area', 'hpc_axi.area'),
    ('hpc_centri.Fl_O:stat:area', 'hpc_centri.area'),
    ('bld3.Fl_O:stat:area', 'bld3.area'),
    ('burner.Fl_O:stat:area', 'burner.area'),
    ('hpt.Fl_O:stat:area', 'hpt.area'),
    ('lpt.Fl_O:stat:area', 'lpt.area'),
    ('itduct.Fl_O:stat:area', 'itduct.area'),
    ('pt.Fl_O:stat:area', 'pt.area'),
    ('nozzle.Throat:stat:area', 'balance.rhs:W'),
]


def _absent_elements(cycle):
    """Return the optional elements left out of pyCycle: a stage is modeled only above 1."""
    absent = set()
    if cycle['pi_lpc'] <= 1.0:
        absent |= {'lpc', 'bld25', 'lpt', 'ip_shaft'}
    if cycle['pi_hpc'] <= 1.0:
        absent.add('hpc_axi')
    if cycle['pi_centrif'] <= 1.0:
        absent.add('hpc_centri')
    return absent


class Turboshaft(pyc.Cycle):
    """
    One operating point of the turboshaft.

    The design point balances mass flow to the nozzle pressure ratio, FAR to T4, turbine
    pressure ratios to shaft power, and HPT cooling to the cooling model. Off-design balances
    mass flow to the design nozzle throat, HP and IP speeds to shaft power, and FAR to `PC` times
    `T4_max`. With `lpc_bleed`, LPC bleed keeps the LPC at or above its R-line schedule
    (`BleedMargin`, `OneSidedBleed`).
    """

    def initialize(self):
        self.options.declare('cycle', types=dict, desc='Design cycle state from the initial guess.')
        self.options.declare(
            'flow_scale', default=1.0, desc='Factor on the off-design mass flow guess.'
        )
        self.options.declare(
            'lpc_bleed', default=False, types=bool, desc='Off-design LPC bleed, see `Turboshaft`.'
        )
        self.options.declare(
            'bleed_pc_start', default=1.0, desc='Throttle where the LPC R-line schedule starts.'
        )
        self.options.declare('map_points', default={}, types=dict, desc='See `DEFAULT_SETTINGS`.')
        super().initialize()

    def setup(self):
        design = self.options['design']
        absent = _absent_elements(self.options['cycle'])
        hp_spool = [e for e in ('hpc_axi', 'hpc_centri', 'hpt') if e not in absent]
        maps = maps_at_design_points(COMPRESSOR_MAPS, self.options['map_points'])
        self._apply_guess = True
        self.options['thermo_method'] = 'TABULAR'
        self.options['thermo_data'] = pyc.AIR_JETA_TAB_SPEC

        self.add_subsystem('fc', pyc.FlightConditions())
        self.add_subsystem('inlet', pyc.Inlet())
        self.add_subsystem('duct225', pyc.Duct())
        if 'lpc' not in absent:
            self.add_subsystem(
                'lpc',
                pyc.Compressor(
                    map_data=maps['lpc'], map_extrap=True, bleed_names=['cool_IG', 'bleed_out']
                ),
                promotes_inputs=[('Nmech', 'IP_Nmech')],
            )
        self.add_subsystem('icduct', pyc.Duct())
        if 'hpc_axi' not in absent:
            self.add_subsystem(
                'hpc_axi',
                pyc.Compressor(map_data=maps['hpc_axi'], map_extrap=True),
                promotes_inputs=[('Nmech', 'HP_Nmech')],
            )
        if 'bld25' not in absent:
            self.add_subsystem('bld25', pyc.BleedOut(bleed_names=['cool1', 'cool2']))
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
        if 'lpt' not in absent:
            self.add_subsystem(
                'lpt',
                pyc.Turbine(map_data=pyc.LPTMap, map_extrap=True, bleed_names=['cool1', 'cool2']),
                promotes_inputs=[('Nmech', 'IP_Nmech')],
            )
        self.add_subsystem('itduct', pyc.Duct())
        self.add_subsystem(
            'pt',
            pyc.Turbine(map_data=pyc.LPTMap, map_extrap=True),
            promotes_inputs=[('Nmech', 'LP_Nmech')],
        )
        self.add_subsystem('nozzle', pyc.Nozzle(nozzType='CV', lossCoef='Cv'))

        self.add_subsystem(
            'lp_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech', 'LP_Nmech')]
        )
        if 'ip_shaft' not in absent:
            self.add_subsystem(
                'ip_shaft', pyc.Shaft(num_ports=2), promotes_inputs=[('Nmech', 'IP_Nmech')]
            )
        self.add_subsystem(
            'hp_shaft', pyc.Shaft(num_ports=3), promotes_inputs=[('Nmech', 'HP_Nmech')]
        )
        self.add_subsystem('perf', pyc.Performance(num_nozzles=1, num_burners=1))

        self.connect('duct225.Fl_O:tot:P', 'perf.Pt2')
        self.connect(f'{hp_spool[-2]}.Fl_O:tot:P', 'perf.Pt3')
        self.connect('burner.Wfuel', 'perf.Wfuel_0')
        self.connect('inlet.F_ram', 'perf.ram_drag')
        self.connect('nozzle.Fg', 'perf.Fg_0')
        self.connect('lp_shaft.pwr_in', 'perf.power')
        self.connect('pt.trq', 'lp_shaft.trq_0')
        if 'ip_shaft' not in absent:
            self.connect('lpc.trq', 'ip_shaft.trq_0')
            self.connect('lpt.trq', 'ip_shaft.trq_1')
        for port, element in enumerate(hp_spool):
            self.connect(f'{element}.trq', f'hp_shaft.trq_{port}')
        self.connect('fc.Fl_O:stat:P', 'nozzle.Ps_exhaust')

        balance = self.add_subsystem('balance', om.BalanceComp())
        if design:
            self._setup_design(balance)
        else:
            self._setup_off_design(balance)

        self.pyc_connect_flow('fc.Fl_O', 'inlet.Fl_I', connect_w=False)
        gas_path = (
            'inlet',
            'duct225',
            'lpc',
            'icduct',
            'hpc_axi',
            'bld25',
            'hpc_centri',
            'bld3',
            'burner',
            'hpt',
            'lpt',
            'itduct',
            'pt',
            'nozzle',
        )
        path = [e for e in gas_path if e not in absent]
        for upstream, downstream in zip(path[:-1], path[1:]):
            self.pyc_connect_flow(f'{upstream}.Fl_O', f'{downstream}.Fl_I')
        # LPT cooling from the inter-compressor bleed, HPT cooling from the HPC exit
        if 'bld25' not in absent:
            self.pyc_connect_flow('bld25.cool1', 'lpt.cool1', connect_stat=False)
            self.pyc_connect_flow('bld25.cool2', 'lpt.cool2', connect_stat=False)
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

    def _setup_design(self, balance):
        absent = _absent_elements(self.options['cycle'])
        balance.add_balance('W', lower=1, units='lbm/s', eq_units=None)
        self.connect('balance.W', 'inlet.Fl_I:stat:W')
        self.connect('nozzle.PR', 'balance.lhs:W')

        balance.add_balance('FAR', eq_units='degR', lower=1e-4)
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.lhs:FAR')

        # Turbine pressure ratios zero the net shaft power
        for turbine, shaft in (('lpt', 'ip_shaft'), ('hpt', 'hp_shaft'), ('pt', 'lp_shaft')):
            if turbine in absent:
                continue
            balance.add_balance(f'{turbine}_PR', lower=1, eq_units='hp', rhs_val=0.0)
            self.connect(f'balance.{turbine}_PR', f'{turbine}.PR')
            self.connect(f'{shaft}.pwr_net', f'balance.lhs:{turbine}_PR')

        # Shaft speeds from the turbine stress limits; the power turbine speed is fixed
        self.add_subsystem('hp_speed', components.hp_speed())
        self.connect('hpt.Fl_O:stat:area', 'hp_speed.A_hpt_exit')
        self.connect('hp_speed.N_hpt', 'HP_Nmech')
        if 'lpt' not in absent:
            self.add_subsystem('ip_speed', components.ip_speed())
            self.connect('lpt.Fl_O:stat:area', 'ip_speed.A_lpt_exit')
            self.connect('ip_speed.N_lpt', 'IP_Nmech')

        # Axial HPC efficiency, reduced when the last-stage blades are short
        if 'hpc_axi' not in absent:
            self.add_subsystem('hpc_tip_clearance', components.hpc_tip_clearance())
            self.connect('hpc_axi.Fl_O:stat:area', 'hpc_tip_clearance.A3')
            self.add_subsystem('hpc_axi_em', components.isentropic_efficiency('eta_phpc_axi'))
            self.connect('hpc_tip_clearance.eta_phpc_axi_tl', 'hpc_axi_em.eta_phpc_axi')
            self.connect('hpc_axi_em.eff', 'hpc_axi.eff')

        components.add_hpt_cooling(self, balance)

    def _setup_off_design(self, balance):
        # Throttle: T4 is the fraction PC of T4_max
        balance.add_balance('FAR', lower=1e-4, upper=0.06, eq_units='degR', use_mult=True)
        self.connect('balance.FAR', 'burner.Fl_I:FAR')
        self.connect('burner.Fl_O:tot:T', 'balance.rhs:FAR')
        self.promotes('balance', inputs=[('mult:FAR', 'PC'), ('lhs:FAR', 'T4_max')])

        balance.add_balance('W', lower=1, units='lbm/s', eq_units='inch**2')
        self.connect('balance.W', 'inlet.Fl_I:stat:W')
        self.connect('nozzle.Throat:stat:area', 'balance.lhs:W')

        # Shaft speeds zero the net shaft power
        absent = _absent_elements(self.options['cycle'])
        for spool, shaft in (('IP_Nmech', 'ip_shaft'), ('HP_Nmech', 'hp_shaft')):
            if shaft in absent:
                continue
            balance.add_balance(spool, units='rpm', lower=1.001, eq_units='hp', rhs_val=0.0)
            self.connect(f'balance.{spool}', spool)
            self.connect(f'{shaft}.pwr_net', f'balance.lhs:{spool}')

        if self.options['lpc_bleed']:
            self.add_subsystem(
                'bleed_margin', components.BleedMargin(pc_start=self.options['bleed_pc_start'])
            )
            self.promotes('bleed_margin', inputs=['PC'])
            self.add_subsystem('bleed', components.OneSidedBleed())
            self.connect('bleed.bleed_frac', 'lpc.bleed_out:frac_W')
            self.connect('lpc.map.RlineMap', 'bleed.RlineMap')
            self.connect('bleed_margin.LPC_RlineMap', 'bleed.RlineTarget')

    def guess_nonlinear(self, inputs, outputs, residuals):
        """
        Seed the balances from the initial guess.

        The design point is reseeded on every solve. An off-design point is seeded with the
        design state once, until `reset_guess` is called.
        """
        cycle = self.options['cycle']
        absent = _absent_elements(cycle)
        if self.options['design']:
            for name, value in turboshaft_design_guess(cycle).items():
                if name != 'balance.lpt_PR' or 'lpt' not in absent:
                    outputs[name] = value
            return
        if not self._apply_guess:
            return
        outputs['balance.HP_Nmech'] = cycle['HP_Nmech_max']
        if 'ip_shaft' not in absent:
            outputs['balance.IP_Nmech'] = cycle['IP_Nmech_max']
        outputs['balance.FAR'] = cycle['FAR']
        T_ratio, P_ratio = components.guess_off_design_statics(
            inputs,
            outputs,
            cycle,
            {
                element: (*totals, STATION_MN[element])
                for element, totals in _exit_totals(cycle).items()
                if element not in absent
            },
        )
        # Design corrected flow at the off-design inlet conditions
        W_guess = cycle['W_core_lbs'] * P_ratio / T_ratio**0.5
        outputs['balance.W'] = W_guess * self.options['flow_scale']
        self._apply_guess = False

    def reset_guess(self):
        """Seed the off-design point from the initial guess again on its next solve."""
        self._apply_guess = True


class MPTurboshaft(pyc.MPCycle):
    """
    Design point `DESIGN` and one off-design point `OD` of a turboshaft.

    The off-design HPT cooling flows match the design cooling mass flows, and the design map
    scalars and flow areas fix the off-design geometry.
    """

    def initialize(self):
        self.options.declare('engine', types=dict, desc='Entry of `engines.TURBOSHAFTS`.')
        self.options.declare('cycle', types=dict, desc='Design cycle state from the initial guess.')
        self.options.declare('settings', types=dict, desc='Engine settings, see DEFAULT_SETTINGS.')
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
        lpc_bleed = settings['lpc_bleed'] and 'lpc' not in absent
        if {'hpc_axi', 'hpc_centri'} <= absent:
            raise ValueError('The turboshaft cycle model needs an HP compressor; raise opr.')

        # Pressure ratios as split by the initial guess, so the guess and the model agree
        dv = self.add_subsystem('DV', om.IndepVarComp())
        if 'lpc' not in absent:
            dv.add_output('lpc_PR', val=cycle['pi_lpc'])
        if 'hpc_axi' not in absent:
            dv.add_output('hpc_axi_PR', val=cycle['pi_hpc'])
        if 'hpc_centri' not in absent:
            dv.add_output('hpc_centri_PR', val=cycle['pi_centrif'])

        # Compressor efficiencies from the polytropic efficiencies of the initial guess
        efficiencies = {
            'lpc': (cycle['pi_lpc'], engine.get('eta_plpc', 0.9)),
            'hpc_centri': (cycle['pi_centrif'], engine.get('eta_pcentrif', 0.85)),
        }
        components.add_compressor_efficiencies(
            self, {comp: eta for comp, eta in efficiencies.items() if comp not in absent}
        )

        self.pyc_add_pnt(
            'DESIGN',
            Turboshaft(
                design=True,
                thermo_method='TABULAR',
                cycle=cycle,
                map_points=settings['map_points'],
            ),
        )
        for comp in ('lpc', 'hpc_axi', 'hpc_centri'):
            if comp not in absent:
                self.connect(f'DV.{comp}_PR', f'DESIGN.{comp}.PR')
        if 'hpc_axi' not in absent:
            self.connect('DV.hpc_axi_PR', 'DESIGN.hpc_axi_em.PR')
            self.set_input_defaults(
                'DESIGN.hpc_tip_clearance.loss.eta_phpc_axi', engine.get('eta_phpc_init', 0.9)
            )

        for station, MN in STATION_MN.items():
            if station not in absent:
                self.set_input_defaults(f'DESIGN.{station}.MN', MN)
        self.set_input_defaults('DESIGN.lp_shaft.HPX', engine['target_hp'], units='hp')

        for name, value in {**CYCLE_PARAMS, **settings['cycle_params']}.items():
            if name.split('.')[0] not in absent:
                self.pyc_add_cycle_param(name, value)
        self.pyc_add_cycle_param('hp_shaft.HPX', engine['pwr_offtk_hpt'], units='hp')
        if 'lpc' not in absent:
            self.pyc_add_cycle_param('lpc.cool_IG:frac_W', engine.get('coolIG_frac_W', 0.0))
            if not lpc_bleed:
                self.pyc_add_cycle_param('lpc.bleed_out:frac_W', 0.0)

        self.pyc_add_pnt(
            'OD',
            Turboshaft(
                design=False,
                thermo_method='TABULAR',
                cycle=cycle,
                flow_scale=self.options['flow_scale'],
                lpc_bleed=lpc_bleed,
                bleed_pc_start=settings['bleed_pc_start'],
                map_points=settings['map_points'],
            ),
        )
        self.set_input_defaults('OD.fc.MN', engine['M'])
        self.set_input_defaults('OD.fc.alt', settings['design_alt'], units='ft')

        # Off-design cooling fractions deliver the design cooling mass flows
        bal = self.add_subsystem('OD_bleedW_bal', om.BalanceComp())
        for port, val in (('bld_inlet', 0.02), ('bld_exit', 0.01)):
            name = f'{port}_fracW'
            bal.add_balance(name, val=val, lower=0.0, upper=1, eq_units='lbm/s')
            self.connect(f'OD.bld3.{port}:stat:W', f'OD_bleedW_bal.lhs:{name}')
            self.connect(f'DESIGN.bld3.{port}:stat:W', f'OD_bleedW_bal.rhs:{name}')
            self.connect(f'OD_bleedW_bal.{name}', f'OD.bld3.{port}:frac_W')

        for comp, scalars in _TURBOMACHINES.items():
            if comp not in absent:
                for scalar in scalars:
                    self.pyc_connect_des_od(f'{comp}.{scalar}', f'{comp}.{scalar}')
        for des, od in _AREAS:
            if des.split('.')[0] not in absent:
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
