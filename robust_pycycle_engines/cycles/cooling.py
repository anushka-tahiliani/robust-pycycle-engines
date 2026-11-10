"""
pyCycle's turbine cooling model with rotor coolant mixed after the rotor does work.

pyCycle's `TurbineCooling` subtracts each rotor's work from the primary flow after mixing in that
rotor's coolant, so the coolant is charged for work it never does, and it places each row's exit
pressure at whole-stage boundaries. Here a stator mixes its coolant at the stage inlet pressure,
and a rotor does its work on the primary flow first and then mixes its coolant at the stage exit
pressure. Everything else is pyCycle's model.

Author: Anushka Tahiliani (MIT Gas Turbine Lab), Contact: anushkat@mit.edu
"""

import openmdao.api as om
from pycycle.elements import cooling


class CoolingCalcs(cooling.CoolingCalcs):
    """Cooling flow, exit enthalpy and exit pressure of one turbine blade row."""

    def _row(self, inputs):
        """Return the stage fraction, gas temperature and its derivative, and work of the row."""
        n_stages = self.options['n_stages']
        i_row = self.options['i_row']
        stage_index = i_row // 2
        if i_row % 2 == 0:  # stator: no work, at the stage inlet pressure
            i_stage = stage_index / n_stages
            T_gas = inputs['Tt_primary'] + self.options['T_safety']
            dTgas_dTprimary = 1.0
            dh = 0.0
            ddh_dturb_pwr = 0.0
        else:  # rotor: does its share of the turbine work, ends at the stage exit pressure
            i_stage = (stage_index + 1) / n_stages
            T_gas = 0.92 * inputs['Tt_primary'] + self.options['T_safety']
            dTgas_dTprimary = 0.92
            dh = inputs['turb_pwr'] / n_stages
            ddh_dturb_pwr = 1.0 / n_stages
        return i_stage, T_gas, dTgas_dTprimary, dh, ddh_dturb_pwr

    def compute(self, inputs, outputs):
        i_row = self.options['i_row']
        T_metal = self.options['T_metal']
        self.i_stage, T_gas, _, dh, _ = self._row(inputs)
        profile_factor = 0.3 if i_row == 0 else 0.13

        W_primary = inputs['W_primary']
        if T_gas < T_metal:
            W_cool = 0.0
        else:
            phi = (T_gas - T_metal) / (T_gas - inputs['Tt_cool'])
            phi_prime = (phi + profile_factor) / (profile_factor + 1.0)
            try:
                W_cool = (
                    0.022
                    * inputs['x_factor']
                    * (4.0 / 3.0)
                    * W_primary
                    * (phi_prime / (1 - phi_prime)) ** 1.25
                )
            except FloatingPointError as err:
                raise om.AnalysisError(
                    f'bad flow values in {self.pathname}; W: {W_primary}'
                ) from err
        outputs['W_cool'] = W_cool

        # Stator dh is zero, so this mixes immediately; a rotor works before mixing
        ht_primary_out = inputs['ht_primary'] - dh / W_primary
        outputs['ht_out'] = (W_primary * ht_primary_out + W_cool * inputs['ht_cool']) / (
            W_primary + W_cool
        )
        Pt_in, Pt_out = inputs['Pt_in'], inputs['Pt_out']
        outputs['Pt_stage'] = Pt_out + (Pt_in - Pt_out) * self.i_stage

    def compute_partials(self, inputs, J):
        i_row = self.options['i_row']
        T_metal = self.options['T_metal']
        self.i_stage, T_gas, dTgas_dTprimary, dh, ddh_dturb_pwr = self._row(inputs)
        profile_factor = 0.3 if i_row == 0 else 0.13

        W_primary = inputs['W_primary']
        if T_gas < T_metal:
            dWc_dx_factor = dWc_dWp = dWc_dTt_primary = dWc_dTt_cool = W_cool = 0.0
        else:
            T_cool = inputs['Tt_cool']
            x_factor = inputs['x_factor']
            phi = (T_gas - T_metal) / (T_gas - T_cool)
            phi_prime = (phi + profile_factor) / (profile_factor + 1.0)
            const = 0.022 * 4.0 / 3.0
            phi_prime_term = (phi_prime / (1.0 - phi_prime)) ** 1.25
            dphi_prime_dphi = 1.0 / (profile_factor + 1.0)
            dphi_dTgas = 1.0 / (T_gas - T_cool) - (T_gas - T_metal) / (T_gas - T_cool) ** 2
            dphi_dTcool = (T_gas - T_metal) / (T_gas - T_cool) ** 2
            dterm_dphi_prime = (1.25 * (phi_prime / (1.0 - phi_prime)) ** 0.25) * (
                1.0 / (1.0 - phi_prime) + phi_prime / (1.0 - phi_prime) ** 2
            )
            dWc_dx_factor = const * phi_prime_term * W_primary
            dWc_dWp = const * phi_prime_term * x_factor
            dWc_dphi = const * W_primary * x_factor * dterm_dphi_prime * dphi_prime_dphi
            dWc_dTt_primary = dWc_dphi * dphi_dTgas * dTgas_dTprimary
            dWc_dTt_cool = dWc_dphi * dphi_dTcool
            W_cool = const * x_factor * W_primary * phi_prime_term

        ht_primary = inputs['ht_primary']
        ht_cool = inputs['ht_cool']
        W_out = W_primary + W_cool
        dht_dWc = (W_primary * (ht_cool - ht_primary) + dh) / W_out**2
        dht_dWp = (W_cool * (ht_primary - ht_cool) + dh) / W_out**2

        J['W_cool', 'W_primary'] = dWc_dWp
        J['W_cool', 'x_factor'] = dWc_dx_factor
        J['W_cool', 'Tt_primary'] = dWc_dTt_primary
        J['W_cool', 'Tt_cool'] = dWc_dTt_cool
        J['ht_out', 'x_factor'] = dht_dWc * dWc_dx_factor
        J['ht_out', 'W_primary'] = dht_dWp + dht_dWc * dWc_dWp
        J['ht_out', 'Tt_primary'] = dht_dWc * dWc_dTt_primary
        J['ht_out', 'Tt_cool'] = dht_dWc * dWc_dTt_cool
        J['ht_out', 'ht_primary'] = W_primary / W_out
        J['ht_out', 'ht_cool'] = W_cool / W_out
        J['ht_out', 'turb_pwr'] = -ddh_dturb_pwr / W_out
        J['Pt_stage', 'Pt_in'] = self.i_stage
        J['Pt_stage', 'Pt_out'] = 1 - self.i_stage


class _Row(cooling.Row):
    def add_subsystem(self, name, subsys, **kwargs):
        if name == 'cooling_calcs':
            opts = ('n_stages', 'i_row', 'T_safety', 'T_metal')
            subsys = CoolingCalcs(**{opt: subsys.options[opt] for opt in opts})
        return super().add_subsystem(name, subsys, **kwargs)


class TurbineCooling(cooling.TurbineCooling):
    """pyCycle's `TurbineCooling`, with each row computed by `CoolingCalcs`."""

    def add_subsystem(self, name, subsys, **kwargs):
        if isinstance(subsys, cooling.Row):
            row_opts = (
                'n_stages',
                'i_row',
                'T_metal',
                'T_safety',
                'thermo_method',
                'thermo_data',
                'main_flow_composition',
                'bld_flow_composition',
                'mix_flow_composition',
            )
            subsys = _Row(**{opt: subsys.options[opt] for opt in row_opts})
        return super().add_subsystem(name, subsys, **kwargs)
