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

PARDEF = [
    ParDef('n_scans', int, 1, min=1),
    ParDef('f', float, 1e6),
    ParDef('a_90', float, 0),
    ParDef('t_90', float, 32e-6),
    ParDef('a_180', float, 0),
    ParDef('t_180', float, 32e-6),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6),
    ParDef('n_samples', int, 1000),
    ParDef('n_echo', int, 1000),
    ParDef('t_echo', float, 200e-6),
    ParDef('t_end', float, 1),
    ParDef('g_diff', float_array, [(0, 0, 0)], min=(-1, -1, -1), max=(1, 1, 1)),
    ParDef('t_diff', float, 1e-3),
    ParDef('t_diff_spacing', float, 2e-3),
    ParDef('t_grad_ramp', float, 200e-6),
    ParDef('shim_x', float, 0, min=-1, max=1),
    ParDef('shim_y', float, 0, min=-1, max=1),
    ParDef('shim_z', float, 0, min=-1, max=1),
    ParDef('shim_z2', float, 0, min=-1, max=1),
    ParDef('shim_zx', float, 0, min=-1, max=1),
    ParDef('shim_zy', float, 0, min=-1, max=1),
    ParDef('shim_xy', float, 0, min=-1, max=1),
    ParDef('shim_x2y2', float, 0, min=-1, max=1),
]

# TODO: move to library
ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(par: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(par: ParameterSet):
    return datalayout.Repetitions(
        par.g_diff.shape[0],
        datalayout.Scans(
            par.n_scans,
            datalayout.Repetitions(
                par.n_echo,
                datalayout.Acquisition(
                    n_samples=par.n_samples,
                    t_dw=par.t_dw))))


def main(par: ParameterSet):
    
    # calculate some timing parameters for the sequence
    t_acq = par.n_samples * par.t_dw # acuisition duration
    
    # constraint equations:
    # t_diffecho/2 = t_90/2 + t_90_diff + t_diff + t_diff_180 + t_180/2
    # t_diffecho/2 = t_180/2 + t_180_diff + t_diff + t_diff_acq + t_acq/2
    # t_diff_spacing = t_diff + t_diff_180 + t_180 + t_180_diff
    # t_diff_180 > t_grad_ramp
    # t_diff_acq > t_grad_ramp
    
    t_diff_180 = t_180_diff = (par.t_diff_spacing - par.t_diff - par.t_180)/2
    if t_diff_180 < par.t_grad_ramp:
        raise Exception('t_diff_spacing too short or t_diff too long!')
        
    # set minimum t_90_diff 
    t_90_diff  = 1e-6
    # set minimum t_diff_acq 
    t_diff_acq = par.t_grad_ramp
    
    t_diffecho_1 = 2*(par.t_90/2 + t_90_diff + par.t_diff + t_diff_180 + par.t_180/2)
    t_diffecho_2 = 2*(par.t_180/2 + t_180_diff + par.t_diff + t_diff_acq + t_acq/2)
    t_diffecho = max(t_diffecho_1, t_diffecho_2)
    t_90_diff = t_diffecho/2 - (par.t_90/2 + par.t_diff + t_diff_180 + par.t_180/2)
    t_diff_acq = t_diffecho/2 - (par.t_180/2 + t_180_diff + par.t_diff + t_acq/2)
    
    t_cpmg_180_acq = (par.t_echo - t_acq - par.t_180)/2
    t_cpmg_acq_180 = t_acq + t_cpmg_180_acq
    if t_cpmg_180_acq < 1e-6:
        raise Exception('t_echo too short or t_dw*n_samples too long!')
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    yield seq.shim(par.shim_x, par.shim_y, par.shim_z, par.shim_z2, par.shim_zx, par.shim_zy, par.shim_xy, par.shim_x2y2)
    yield seq.wait(0.01)
    
    for g_diff_i in par.g_diff:
    
        for i_scan in range(par.n_scans):
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]

            yield seq.pulse_start(par.f, p_90, par.a_90)
            yield seq.wait(par.t_90)
            yield seq.pulse_end()

            yield seq.wait(t_90_diff)

            yield seq.gradient(*g_diff_i)
            yield seq.wait(par.t_diff)
            yield seq.gradient(0,0,0)

            yield seq.wait( t_diff_180)

            yield seq.pulse_start(par.f, p_180, par.a_180)
            yield seq.wait(par.t_180)
            yield seq.pulse_end()

            yield seq.wait(t_180_diff)

            yield seq.gradient(*g_diff_i)
            yield seq.wait(par.t_diff)
            yield seq.gradient(0,0,0)

            yield seq.wait(t_diff_acq)

            yield seq.wait(t_cpmg_acq_180)
            
            # all echos are identical, so for effiency just generate this subsequence once per scan
            # use add syntax sugar to concatenate intruction strings
            # use python multiply-integer-by-string syntax sugar to duplicate (n_echo) times
            yield par.n_echo * (
                seq.pulse_start(par.f, p_180, par.a_180)
                + seq.wait(par.t_180)
                + seq.pulse_end()
                + seq.wait(t_cpmg_180_acq)
                + seq.acquire(par.f, p_acq, par.t_dw, par.n_samples)
                + seq.wait(t_cpmg_acq_180)
            )

            # end delay to control repetition time
            yield seq.wait(par.t_end)
