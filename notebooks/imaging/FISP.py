from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np

# TODO: move to library
def float_array(v):
    a = np.array(v, dtype=float)
    a.setflags(write=False)
    return a

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

g_ZERO = float_array((0,0,0))

PARDEF = [
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_pulse', float, 0, min=0, max=1, unit=''),
    ParDef('t_pulse', float, 32e-6, unit='s'),
    ParDef('n_leading_pulses', int, 20, min=0, unit=''),
    ParDef('t_dw', float, 5e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 200, min=2, unit=''),
    ParDef('t_read', float, 1000e-6, unit='s'),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_spoil', float, 100e-6, unit='s'),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_phase', float, 500e-6, unit='s'),
    ParDef('g_phase_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('n_phase_1', int, 1, min=1, unit=''),
    ParDef('g_phase_1', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('n_phase_2', int, 1, min=1, unit=''),
    ParDef('g_phase_2', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1), unit=''),
    ParDef('t_grad_stab', float, 100e-6, unit='s'),
    ParDef('t_end', float, 10e-3, unit='s'),
    ParDef('shim_x', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_y', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_z', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_z2', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_zx', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_zy', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_xy', float, 0, min=-1, max=1, unit=''),
    ParDef('shim_x2y2', float, 0, min=-1, max=1, unit='')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(par: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Scans(
        p.n_scans,
        datalayout.Repetitions(
            p.n_phase_1,
            datalayout.Repetitions(
                p.n_phase_2,
                datalayout.Acquisition(
                    n_samples=p.n_samples,
                    t_dw=p.t_dw))))


def main(par: ParameterSet):
    # gradient duty cycle check TODO: implement duty cycle checks in driver
    t_rep = par.t_pulse + 2*par.t_phase + par.t_grad_stab + par.t_read + par.t_spoil + par.t_end
    grad_total_area = (
        (np.abs(par.g_phase_read)+np.abs(par.g_phase_1) + np.abs(par.g_phase_2))*par.t_phase
        + np.abs(par.g_read)*par.t_read
        + np.abs(par.g_spoil)*par.t_spoil
    )
    grad_duty_cycle = grad_total_area/t_rep
    log.debug(f'gradient duty cycle: {str(grad_duty_cycle)}')
    if np.any(grad_duty_cycle > 0.3):
        raise Exception('Gradient duty cycle too high!')
    
    if par.n_phase_1>1:
#         g_phase_1_step = (par.g_phase_1_end - par.g_phase_1_start)/(par.n_phase_1-1)
        g_phase_1_step = -par.g_phase_1/(par.n_phase_1//2)
    else:
        g_phase_1_step = 0
    log.debug(f'phase 1 step: {str(g_phase_1_step)}')
    if par.n_phase_2>1:
#         g_phase_2_step = (par.g_phase_2_end - par.g_phase_2_start)/(par.n_phase_2-1)
         g_phase_2_step = -par.g_phase_2/(par.n_phase_2//2)
    else:
        g_phase_2_step = 0
    log.debug(f'phase 2 step: {str(g_phase_2_step)}')
    
    if par.t_read < par.n_samples*par.t_dw:
        raise Exception('Read gradient time too short for acquisition!')
    
    t_evo = par.t_pulse/2 + par.t_phase + par.t_grad_stab + par.t_read/2
    log.debug(f"evolution time: {t_evo}")
    
    rf_pulse_even = (
        seq.pulse_start(par.f, 0, par.a_pulse)
        + seq.wait(par.t_pulse)
        + seq.pulse_end()
    )
    
    rf_pulse_odd = (
        seq.pulse_start(par.f, 180, par.a_pulse)
        + seq.wait(par.t_pulse)
        + seq.pulse_end()
    )
    
    readout_even = seq.gradient(*par.g_read) + seq.acquire(par.f, 0, par.t_dw, par.n_samples) + seq.wait(par.t_read)
    readout_odd = seq.gradient(*par.g_read) + seq.acquire(par.f, 180, par.t_dw, par.n_samples) + seq.wait(par.t_read)

    phase = 0
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
    
    # do a few dummy pulses to reach steady state
    for i in range(par.n_leading_pulses):
        phase = (phase + 180) % 360
        yield seq.pulse_start(par.f, phase, par.a_pulse)
        yield seq.wait(par.t_pulse)
        yield seq.pulse_end()
        yield seq.wait(par.t_phase)
        yield seq.wait(par.t_grad_stab)
        yield seq.wait(par.t_read)
        yield seq.wait(par.t_phase)
        yield seq.gradient(*par.g_spoil)
        yield seq.wait(par.t_spoil)
        yield seq.gradient(*g_ZERO)
        yield seq.wait(par.t_end)
    
    i = 0
    for i_scan in range(par.n_scans):
        for i_phase_1 in range(par.n_phase_1):
            g_phase_1_i = par.g_phase_1 + i_phase_1*g_phase_1_step
            for i_phase_2 in range(par.n_phase_2):
                g_phase_2_i = par.g_phase_2 + i_phase_2*g_phase_2_step
                
                even = i%2==0
                
                yield (
                    (rf_pulse_even if even else rf_pulse_odd)
                    
                    + seq.gradient(*(par.g_phase_read+g_phase_1_i+g_phase_2_i))
                    + seq.wait(par.t_phase)
                    
                    # avoid artifacts by allowing the gradients to come to zero before performing read
                    + seq.gradient(*g_ZERO)
                    + seq.wait(par.t_grad_stab)
                    
                    + (readout_even if even else readout_odd)
                    
                    + seq.gradient(*(-g_phase_1_i-g_phase_2_i))
                    + seq.wait(par.t_phase)
                    
                    + seq.gradient(*par.g_spoil)
                    + seq.wait(par.t_spoil)
                    
                    + seq.gradient(*g_ZERO)
                    + seq.wait(par.t_end)
                )
                
                i += 1
