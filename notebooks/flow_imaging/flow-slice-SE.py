from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np
from functools import partial
from matipo.util.pulseshape import calc_soft_pulse

# TODO: move to library
def float_array(v):
    a = np.array(v, dtype=float)
    a.setflags(write=False)
    return a

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

def gen_soft_pulse_cycle(freq, phase_cycle, amp, width, bandwidth):
    N, dt, pts = calc_soft_pulse(width, bandwidth)
    log.debug(f'softpulse N: {N}, dt: {dt}')
    pts *= amp
    softpulse_cycle = []
    for phase in phase_cycle:
        softpulse = seq.pulse_start(freq, phase, pts[0]) + seq.wait(dt)
        for amp in pts[1:]:
            softpulse += seq.pulse_update(freq, phase, amp) + seq.wait(dt)
        softpulse += seq.pulse_end()
        softpulse_cycle.append(softpulse)
    return softpulse_cycle

g_ZERO = float_array((0,0,0))
def gen_grad_ramp(g_start, g_end, t, n):
    g_step = (g_end - g_start)/n
    t_step = t/n
    ret = b''
    for i in range(n):
        ret += seq.gradient(*(g_start+(i+1)*g_step))
        ret += seq.wait(t_step)
    return ret

PARDEF = [
    ParDef('n_scans', int, 1, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_90', float, 0),
    ParDef('t_90', float, 32e-6),
    ParDef('bw_90', float, 0),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 5e-6),
    ParDef('n_samples', int, 128),
    ParDef('t_echo', float, 2000e-6),
    ParDef('g_slice', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_phase', float, 320e-6),
    ParDef('g_phase_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_1', int, 1, min=1),
    ParDef('g_phase_1', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_2', int, 1, min=1),
    ParDef('g_phase_2', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_flow', float, 160e-6),
    ParDef('t_flow_spacing', float, 320e-6),
    ParDef('t_flow_tweak', float, 0),
    ParDef('g_flow', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_spoil', float, 100e-6),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_grad_stab', float, 100e-6),
    ParDef('t_end', float, 1),
    ParDef('t_grad_ramp', float, 100e-6),
    ParDef('n_grad_ramp', int, 10),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1)
]

# TODO: move to library
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(par: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Repetitions(
        p.n_phase_1,
        datalayout.Repetitions(
            p.n_phase_2,
            datalayout.Scans(
                p.n_scans,
                datalayout.Acquisition(
                    n_samples=p.n_samples,
                    t_dw=p.t_dw))))


def main(par: ParameterSet):
    # TODO: use trapezoidal pulses
    t_read = par.n_samples * par.t_dw
    
    t_unslice = (par.t_90 - par.t_grad_ramp)/2
    if t_unslice < 1e-6:
        t_unslice = 1e-6
    
    # for 90 - 180 timings must have:
    # t_90/2 + 3*t_grad_ramp + t_unslice + t_flow + t_flow_unflow + t_flow + t_flow_180 + t_180/2 = t_echo/2
    t_flow_hold = par.t_flow - par.t_grad_ramp
    t_flow_hold_2 = t_flow_hold + par.t_flow_tweak
    
    if t_flow_hold <= 0 or t_flow_hold_2 <= 0:
        raise Exception(f'Flow encode gradient time too short, minimum {par.t_grad_ramp}')
    
    t_flow_unflow = par.t_flow_spacing - (t_flow_hold + 2*par.t_grad_ramp)
    if t_flow_unflow <= 0:
        raise Exception(f'Flow encode gradient spacing too short, minimum {(t_flow_hold + 2*par.t_grad_ramp)}')
    
    t_flow_180 = (par.t_echo - par.t_90 - par.t_180)/2 - (3*par.t_grad_ramp + t_unslice) - t_flow_hold - t_flow_hold_2 - 4*par.t_grad_ramp - t_flow_unflow
    log.debug('t_flow_180: %f', t_flow_180)
    if t_flow_180 <= 0:
        raise Exception(f'Echo time too short, or flow encoding/spacing too long (t_flow_180={t_flow_180}.')
    
    # for 180 - read timings must have:
    # t_180/2 + t_180_phase + t_phase + t_grad_stab + t_read/2 = t_echo/2
    t_180_phase = (par.t_echo - par.t_180 - t_read)/2 - par.t_phase - par.t_grad_stab
    if t_180_phase <= 0:
        raise Exception('Echo time too short, or acquisition time too long.')
    
    if par.n_phase_1>1:
        g_phase_1_step = -par.g_phase_1/(par.n_phase_1//2)
    else:
        g_phase_1_step = 0
    
    if par.n_phase_2>1:
        g_phase_2_step = -par.g_phase_2/(par.n_phase_2//2)
    else:
        g_phase_2_step = 0
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    # pregenerate fixed ramps
    g_ramp_zero_to_slice = gen_grad_ramp(g_ZERO, par.g_slice, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_slice_to_zero = gen_grad_ramp(par.g_slice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_zero_to_unslice = gen_grad_ramp(g_ZERO, -par.g_slice, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_unslice_to_zero = gen_grad_ramp(-par.g_slice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_zero_to_flow = gen_grad_ramp(g_ZERO, par.g_flow, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_flow_to_zero = gen_grad_ramp(par.g_flow, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_zero_to_unflow = gen_grad_ramp(g_ZERO, -par.g_flow, par.t_grad_ramp, par.n_grad_ramp)
    g_ramp_unflow_to_zero = gen_grad_ramp(-par.g_flow, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    
    enable_softpulse = par.bw_90 > 0
    if enable_softpulse:
        log.debug('using softpulse')
        softpulse_cycle = gen_soft_pulse_cycle(par.f, phase_cycle_90, par.a_90, par.t_90, par.bw_90)
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
        
    # TODO: do some dummy pulses to reach equilibrium?
    
    for i_phase_1 in range(par.n_phase_1):
        g_phase_1_i = par.g_phase_1 + i_phase_1*g_phase_1_step
        for i_phase_2 in range(par.n_phase_2):
            g_phase_2_i = par.g_phase_2 + i_phase_2*g_phase_2_step
            for i_scan in range(par.n_scans):
                p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
                p_180 = phase_cycle_180[i_scan % n_phase_cycle]
                
                # slice select 90 pulse
                yield g_ramp_zero_to_slice
                if enable_softpulse:
                    yield softpulse_cycle[i_scan % n_phase_cycle]
                else:
                    yield seq.pulse_start(par.f, phase, par.a_90)
                    yield seq.wait(par.t_90)
                    yield seq.pulse_end()
                yield g_ramp_slice_to_zero
                # phase unwrapping gradient to cancel second half of g_slice
                yield g_ramp_zero_to_unslice
                yield seq.wait(t_unslice)
                yield g_ramp_unslice_to_zero

                # flow encode positive gradient
                yield g_ramp_zero_to_flow
                yield seq.wait(t_flow_hold)
                yield g_ramp_flow_to_zero
                # wait
                yield seq.wait(t_flow_unflow)
                # flow encode negative gradient
                yield g_ramp_zero_to_unflow
                yield seq.wait(t_flow_hold_2)
                yield g_ramp_unflow_to_zero
                # wait to centre 180 pulse at half echo time
                yield seq.wait(t_flow_180)
                # 180 pulse
                yield seq.pulse_start(par.f, p_180, par.a_180)
                yield seq.wait(par.t_180)
                yield seq.pulse_end()
                # wait
                yield seq.wait(t_180_phase)
                # phase gradient
                yield seq.gradient(*(par.g_phase_read+g_phase_1_i+g_phase_2_i))
                yield seq.wait(par.t_phase)
                yield seq.gradient(0, 0, 0)
                yield seq.wait(par.t_grad_stab)
                # read gradient + acquisition
                yield seq.gradient(*par.g_read)
                yield seq.acquire(par.f, p_acq, par.t_dw, par.n_samples)
                yield seq.wait(t_read)
                # balance phase gradient to avoid artifacts at short TR
                yield seq.gradient(*(-g_phase_1_i-g_phase_2_i))
                yield seq.wait(par.t_phase)
                # spoiler
                yield seq.gradient(*par.g_spoil)
                yield seq.wait(par.t_spoil)
                # wait end time
                yield seq.gradient(0, 0, 0)
                yield seq.wait(par.t_end)
