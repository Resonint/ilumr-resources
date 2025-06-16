# See figure 6 of https://doi.org/10.1016/j.jmr.2006.06.017

from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np
from functools import partial
from matipo.util.pulseshape import calc_soft_pulse
from matipo.util import etl

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
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 5e-6),
    ParDef('n_samples', int, 128),
    # ParDef('t_flow_echo', float, 2000e-6),
    ParDef('g_read', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_phase', float, 320e-6),
    ParDef('n_phase_1', int, 1, min=1),
    ParDef('g_phase_1', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_phase_2', int, 1, min=1),
    ParDef('g_phase_2', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('n_ETL', int, 0),
    ParDef('t_flow', float, 400e-6),
    ParDef('t_flow_spacing', float, 500e-6),
    # ParDef('t_flow_tweak', float, 0),
    ParDef('g_flow', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('g_flow_tweak', float_array, (1,1,1), min=(0,0,0), max=(2, 2, 2)), # scales second flow encode gradient, used to achieve perfect match
    ParDef('t_qfilter', float, 100e-6),
    ParDef('g_qfilter', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_spoil', float, 100e-6),
    ParDef('g_spoil', float_array, (0, 0, 0), min=(-1, -1, -1), max=(1, 1, 1)),
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
    n_runs, n_echos = etl.sequence_format(p.n_ETL, p.n_phase_1, p.n_phase_2)
    # note: if the total number of phase steps does not divide evenly into n_ETL
    # there will be additional data in the last run with no phase encoding
    # which may be ignored or exploited
    return datalayout.Scans(
        p.n_scans,
        datalayout.Repetitions(
            n_runs,
            datalayout.Repetitions(
                n_echos,
                datalayout.Acquisition(
                    n_samples=p.n_samples,
                    t_dw=p.t_dw))))


def main(p: ParameterSet):
    t_read = p.n_samples * p.t_dw
    t_phaseread = t_read/2 - p.t_grad_ramp/2
    t_flow_hold = p.t_flow - p.t_grad_ramp
    # t_flow_hold_2 = t_flow_hold + p.t_flow_tweak
    
    if t_flow_hold <= 0: # or t_flow_hold_2 <= 0:
        raise Exception(f'Flow encode gradient time (t_flow) too short, minimum {p.t_grad_ramp}')
    
    # PGSE echo time constraint:
    # LHS = t_90/2 + (t_grad_ramp+t_flow_hold+t_grad_ramp) + t_flow_180 + t_180/2
    # RHS = t_180/2 + t_180_unflow + (t_grad_ramp+t_flow_hold+t_grad_ramp) + t_90/2
    # LHS and RHS are both adjustable, so calculate minimum possible t_echo_pgse
    t_flow_180 = 1e-6
    t_180_unflow = 1e-6
    t_echo_pgse = 2*max(
        p.t_90/2 + (p.t_grad_ramp+t_flow_hold+p.t_grad_ramp) + t_flow_180 + p.t_180/2,
        p.t_180/2 + t_180_unflow + (p.t_grad_ramp+t_flow_hold+p.t_grad_ramp) + p.t_90/2)
    # and recalculate t_flow_180 and t_180_unflow to achieve this t_echo_pgse
    t_flow_180 = t_echo_pgse/2 - (p.t_90/2 + (p.t_grad_ramp+t_flow_hold+p.t_grad_ramp) + p.t_180/2)
    t_180_unflow = t_echo_pgse/2 - (p.t_180/2 + (p.t_grad_ramp+t_flow_hold+p.t_grad_ramp) + p.t_90/2)
    
    # adjusting t_echo_pgse to achieve t_flow_spacing constraint:
    # LHS = t_flow_hold/2 + t_grad_ramp + t_flow_180 + t_180 + t_180_unflow + t_grad_ramp + t_flow_hold/2
    # RHS = t_flow_spacing
    # calculate flow spacing with minimum possible t_echo_pgse:
    t_flow_spacing_min = t_flow_hold/2 + p.t_grad_ramp + t_flow_180 + p.t_180 + t_180_unflow + p.t_grad_ramp + t_flow_hold/2
    if p.t_flow_spacing < t_flow_spacing_min:
        raise Exception(f'Flow spacing (t_flow_spacing) time too short, minimum {t_flow_spacing_min}')
    # increase t_flow_180 and t_180_unflow equally to achieve t_flow_spacing constraint:
    t_flow_spacing_pad = (p.t_flow_spacing-t_flow_spacing_min)/2
    t_flow_180 += t_flow_spacing_pad
    t_180_unflow += t_flow_spacing_pad
    # recalculate t_echo_pgse
    t_echo_pgse = 2*(p.t_90/2 + (p.t_grad_ramp+t_flow_hold+p.t_grad_ramp) + t_flow_180 + p.t_180/2)
    
    # 1st RARE echo time constraint:
    # LHS = t_90/2 + t_qfilter_phaseread + (t_grad_ramp+t_phaseread+t_grad_ramp) + t_180/2
    # RHS = t_180/2 + (t_phase+t_grad_ramp) + (t_grad_ramp+t_read+t_grad_ramp)/2
    # RHS is fixed, use to calculate t_echo_cpmg
    t_echo_cpmg = 2*(p.t_180/2 + (p.t_phase+p.t_grad_ramp) + (p.t_grad_ramp+t_read+p.t_grad_ramp)/2)
    # and calculate t_qfilter_phaseread from LHS:
    t_qfilter_phaseread = t_echo_cpmg/2 - (p.t_90/2 + (p.t_grad_ramp+t_phaseread+p.t_grad_ramp) + p.t_180/2)
    # check t_qfilter_phaseread is positive:
    if t_qfilter_phaseread <= 0:
        raise Exception('RARE echo time too short, reduce t_qfilter or increase t_phase')
    
    if p.n_phase_1>1:
        g_phase_1_step = -p.g_phase_1/(p.n_phase_1//2)
    else:
        g_phase_1_step = 0
    log.debug(f'phase 1 step: {str(g_phase_1_step)}')
    if p.n_phase_2>1:
         g_phase_2_step = -p.g_phase_2/(p.n_phase_2//2)
    else:
        g_phase_2_step = 0
    log.debug(f'phase 2 step: {str(g_phase_2_step)}')
    
    n_runs, n_echos = etl.sequence_format(p.n_ETL, p.n_phase_1, p.n_phase_2)
    
    log.debug(f"n_runs: {n_runs}, n_echos: {n_echos}")
    
    n_phase_cycle = 4
    # phase_cycle_90 = [0, 0, 180, 180, 0, 0, 180, 180]
    # phase_cycle_pgse = [90, 90, 90, 90, 90, 90, 90, 90]
    # phase_cycle_qfilter = [0, 90, 0, 90, 0, 90, 0, 90]
    # phase_cycle_cpmg = [90, 0, 90, 0, 270, 180, 270, 180] # alternating CPMG/CP sequences
    phase_cycle_90 = [0, 0, 180, 180]
    phase_cycle_pgse = [90, 90, 90, 90]
    phase_cycle_qfilter = [0, 90, 0, 90, 180]
    phase_cycle_cpmg = [90, 0, 90, 0] # alternating CPMG/CP sequences
    
    # pregenerate fixed ramps
    g_ramp_zero_to_flow = gen_grad_ramp(g_ZERO, p.g_flow, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_flow_to_zero = gen_grad_ramp(p.g_flow, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_zero_to_unflow = gen_grad_ramp(g_ZERO, p.g_flow*p.g_flow_tweak, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_unflow_to_zero = gen_grad_ramp(p.g_flow*p.g_flow_tweak, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_zero_to_read = gen_grad_ramp(g_ZERO, p.g_read, p.t_grad_ramp, p.n_grad_ramp)
    g_ramp_read_to_zero = gen_grad_ramp(p.g_read, g_ZERO, p.t_grad_ramp, p.n_grad_ramp)
    
    yield seq.shim(p.shim_x, p.shim_y, p.shim_z, p.shim_z2, p.shim_zx, p.shim_zy, p.shim_xy, p.shim_x2y2)
    yield seq.wait(0.01)
    
    for i_scan in range(p.n_scans):
        p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
        p_pgse = phase_cycle_pgse[i_scan % n_phase_cycle]
        p_qfilter = phase_cycle_qfilter[i_scan % n_phase_cycle]
        p_cpmg = phase_cycle_cpmg[i_scan % n_phase_cycle]
        
        for i_run in range(n_runs):
            # # prewarm gradients for better stability
            # yield g_ramp_zero_to_unflow
            # yield seq.wait(t_flow_hold)
            # yield g_ramp_unflow_to_zero
            # yield seq.wait(t_flow_180+p.t_180+t_180_unflow)
            # yield g_ramp_zero_to_flow
            # yield seq.wait(t_flow_hold)
            # yield g_ramp_flow_to_zero
            # yield seq.wait(t_echo_cpmg)
            
            # 90 pulse
            yield seq.pulse_start(p.f, p_90, p.a_90)
            yield seq.wait(p.t_90)
            yield seq.pulse_end()
            
            # flow encode
            yield g_ramp_zero_to_flow
            yield seq.wait(t_flow_hold)
            yield g_ramp_flow_to_zero
            
            # wait to centre 180 pulse at half echo time
            yield seq.wait(t_flow_180)
            
            # 180 pulse
            yield seq.pulse_start(p.f, p_pgse, p.a_180)
            yield seq.wait(p.t_180)
            yield seq.pulse_end()
            
            yield seq.wait(t_180_unflow)
            
            # unflow encode
            yield g_ramp_zero_to_unflow
            yield seq.wait(t_flow_hold)
            yield g_ramp_unflow_to_zero
            
            # Quadrature phase filter, selects real part for CPMG, imag part for CP
            yield seq.pulse_start(p.f, p_qfilter, p.a_90)
            yield seq.wait(p.t_90)
            yield seq.pulse_end()
            yield seq.gradient(*p.g_qfilter)
            yield seq.wait(p.t_qfilter)
            yield seq.gradient(*g_ZERO)
            yield seq.wait(p.t_grad_ramp)
            yield seq.pulse_start(p.f, p_qfilter, p.a_90)
            yield seq.wait(p.t_90)
            yield seq.pulse_end()
            
            # delay to set CPMG echo time
            yield seq.wait(t_qfilter_phaseread)
            # read prephasing
            yield g_ramp_zero_to_read
            yield seq.wait(t_phaseread)
            yield g_ramp_read_to_zero

            for i_echo in range(n_echos):
                # calculate interlaced phase index
                i_phase = i_echo * n_runs + i_run
                if i_phase < p.n_phase_1*p.n_phase_2:
                    # calculate gradient
                    i_phase_1 = i_phase % p.n_phase_1
                    i_phase_2 = i_phase // p.n_phase_1
                    g_phase = p.g_phase_1 + i_phase_1*g_phase_1_step + p.g_phase_2 + i_phase_2*g_phase_2_step
                else:
                    # special case when p.n_phase_1*p.n_phase_2 does not divide evenly into p.n_ETL
                    # and there are extra echos
                    g_phase = g_ZERO
                
                # 180 pulse
                yield seq.pulse_start(p.f, p_cpmg, p.a_180)
                yield seq.wait(p.t_180)
                yield seq.pulse_end()
                
                # phase gradient
                yield seq.gradient(*g_phase)
                yield seq.wait(p.t_phase)
                yield seq.gradient(*g_ZERO)
                yield seq.wait(p.t_grad_ramp)
                
                # read gradient + acquisition
                yield g_ramp_zero_to_read
                yield seq.acquire(p.f, p_acq, p.t_dw, p.n_samples)
                yield seq.wait(t_read)
                yield g_ramp_read_to_zero
                
                # balance phase gradient
                yield seq.gradient(*(-g_phase))
                yield seq.wait(p.t_phase)
                yield seq.gradient(*g_ZERO)
                yield seq.wait(p.t_grad_ramp)
                
            # spoiler
            yield seq.gradient(*p.g_spoil)
            yield seq.wait(p.t_spoil)
            
            # wait end time
            yield seq.gradient(*g_ZERO)
            yield seq.wait(p.t_end)
