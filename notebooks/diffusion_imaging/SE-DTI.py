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

g_ZERO = float_array((0,0,0))

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

# TODO: move to library
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
    ParDef('g_slice', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 5e-6),
    ParDef('n_samples', int, 128),
#     ParDef('t_echo', float, 2000e-6),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_phase', float, 320e-6),
    ParDef('g_phase_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_1', int, 1, min=1),
    ParDef('g_phase_1', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_2', int, 1, min=1),
    ParDef('g_phase_2', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_diff', float, 1e-3),
    ParDef('g_diff', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_spoil', float, 1000e-6),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_grad_ramp', float, 100e-6),
    ParDef('n_grad_ramp', int, 10),
    ParDef('t_end', float, 1),
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


def get_datalayout(par: ParameterSet):
    return datalayout.Repetitions(
        par.n_phase_1,
        datalayout.Repetitions(
            par.n_phase_2,
            datalayout.Scans(
                par.n_scans,
                datalayout.Acquisition(
                    n_samples=par.n_samples,
                    t_dw=par.t_dw))))


def main(par: ParameterSet):
    # pulse timing sequence:
    # t_90 | t_90_diff | t_grad_ramp | t_diff | t_grad_ramp | t_180 | t_grad_ramp | t_diff | t_grad_ramp | t_phase | t_grad_ramp | t_read | t_phase | t_spoil | t_end | Repeat for n_scans
    t_read = par.n_samples * par.t_dw
    # equations for first and second halves of t_echo:
    # 1: t_echo/2 = t_90/2 + t_90_diff + t_grad_ramp + t_diff + t_grad_ramp + t_180/2
    # 2: t_echo/2 = t_180/2 + t_grad_ramp + t_diff + t_grad_ramp + t_phase + t_grad_ramp + t_read/2
    t_echo = 2*(par.t_180/2 + par.t_grad_ramp + par.t_diff + par.t_grad_ramp + par.t_phase + par.t_grad_ramp + t_read/2)  # from 2
    t_90_diff = t_echo/2 - (par.t_90/2 + par.t_grad_ramp + par.t_diff + par.t_grad_ramp + par.t_180/2)  # from 1
    
    log.info('t_echo: %.2e, t_90_diff: %.2e', t_echo, t_90_diff)
    
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
    
    enable_softpulse = par.bw_90 > 0
    if enable_softpulse:
        log.debug('using softpulse')
        t_unslice = t_90_diff - 3*par.t_grad_ramp
        if t_unslice <= 0:
            raise Exception('Timing Error')
        g_unslice = -par.g_slice*0.5*(par.t_90+par.t_grad_ramp)/(t_unslice+par.t_grad_ramp)
        log.debug(g_unslice)
        softpulse_cycle = gen_soft_pulse_cycle(par.f, phase_cycle_90, par.a_90, par.t_90, par.bw_90)
    
    g_ZERO_diff = gen_grad_ramp(g_ZERO, par.g_diff, par.t_grad_ramp, par.n_grad_ramp)
    g_diff_ZERO = gen_grad_ramp(par.g_diff, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
        
    # TODO: do some dummy pulses to reach steady state for short rep times?
    
    for i_phase_1 in range(par.n_phase_1):
        g_phase_1_i = par.g_phase_1 + i_phase_1*g_phase_1_step
        for i_phase_2 in range(par.n_phase_2):
            g_phase_2_i = par.g_phase_2 + i_phase_2*g_phase_2_step
            for i_scan in range(par.n_scans):
                p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
                p_180 = phase_cycle_180[i_scan % n_phase_cycle]

                # 90 pulse
#                 yield seq.pulse_start(par.f, p_90, par.a_90)
#                 yield seq.wait(par.t_90)
#                 yield seq.pulse_end()
                if enable_softpulse:
                    yield (
                        gen_grad_ramp(g_ZERO, par.g_slice, par.t_grad_ramp, par.n_grad_ramp)
                        + softpulse_cycle[i_scan % n_phase_cycle]
                        + gen_grad_ramp(par.g_slice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
                        + seq.pulse_update(par.f, 0, 0)
                        + gen_grad_ramp(g_ZERO, g_unslice, par.t_grad_ramp, par.n_grad_ramp)
                        + seq.wait(t_unslice)
                        + gen_grad_ramp(g_unslice, g_ZERO, par.t_grad_ramp, par.n_grad_ramp)
                )
                else:
                    yield (
                        seq.pulse_start(par.f, p_90, par.a_90)
                        + seq.wait(par.t_90)
                        + seq.pulse_end()
                        + seq.wait(t_90_diff)
                    )
                
                # diffusion gradient
                yield g_ZERO_diff
                yield seq.wait(par.t_diff)
                yield g_diff_ZERO
                
                # 180 pulse
                yield seq.pulse_start(par.f, p_180, par.a_180)
                yield seq.wait(par.t_180)
                yield seq.pulse_end()
                
                # diffusion gradient
                yield g_ZERO_diff
                yield seq.wait(par.t_diff)
                yield g_diff_ZERO
                
                # phase gradient
                yield seq.gradient(*(par.g_phase_read+g_phase_1_i+g_phase_2_i))
                yield seq.wait(par.t_phase)
                yield seq.gradient(0, 0, 0)
                yield seq.wait(par.t_grad_ramp)
                
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
                yield seq.gradient(0, 0, 0)
                
                # wait end time
                yield seq.wait(par.t_end)
