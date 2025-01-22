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

PARDEF = [
    ParDef('n_scans', int, 1, min=1, unit=''),
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('a_90', float, 0, min=0, max=1, unit=''),
    ParDef('t_90', float, 32e-6, unit='s'),
    ParDef('a_180', float, 0, min=0, max=1, unit=''),
    ParDef('t_180', float, 32e-6, unit='s'),
    ParDef('t_sat', float_array, [35e-3, 25e-3, 15e-3, 10e-3]), # trying 7,5,3,2 x 5e-3 prime factor spacings as a guess (T2* is around 5 ms)
    ParDef('t_rec', float_array, [100e-6]),
    ParDef('t_echo', float, 500e-6, unit='s'),
    ParDef('n_echo', int, 1000, unit=''),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 64, min=2, unit=''),
    ParDef('t_end', float, 1, unit='s')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(p: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Repetitions(
        len(p.t_rec),
        datalayout.Scans(
            p.n_scans,
            datalayout.Repetitions(
                p.n_echo,
                datalayout.Acquisition(
                    n_samples=p.n_samples,
                    t_dw=p.t_dw))))


def main(par: ParameterSet):    
    t_acq = par.n_samples * par.t_dw
    t1 = par.t_echo/2 - (par.t_90+par.t_180)/2
    t2 = (par.t_echo - par.t_180 - t_acq)/2
    t3 = t_acq + t2
    
    def pulse(freq, phase, amp, time):
        return (
            seq.pulse_start(freq, phase, amp)
            + seq.wait(time)
            + seq.pulse_end())
    
    n_phase_cycle = 4    
    phase_cycle_sat = [0, 0, 0, 0, 0, 0, 0, 0]
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    for t in par.t_rec:
        t_rec_90 = t - par.t_90
        for i_scan in range(par.n_scans):
            p_sat = phase_cycle_sat[i_scan % n_phase_cycle]
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]

            # saturation pulse (series of 90 pulses)
            for tau in par.t_sat:
                yield pulse(par.f, p_sat, par.a_90, par.t_90)
                yield seq.wait(tau)
            yield pulse(par.f, p_sat, par.a_90, par.t_90)

            yield seq.wait(t_rec_90)

            # excitation pulse
            yield seq.pulse_start(par.f, p_90, par.a_90)
            yield seq.wait(par.t_90)
            yield seq.pulse_end()
            yield seq.wait(t1)

            # all echos are identical, so for effiency just generate this subsequence once per scan
            # use add syntax sugar to concatenate intruction strings
            # use python multiply-integer-by-string syntax sugar to duplicate (n_echo) times
            yield par.n_echo * (
                seq.pulse_start(par.f, p_180, par.a_180)
                + seq.wait(par.t_180)
                + seq.pulse_end()
                + seq.wait(t2)
                + seq.acquire(par.f, p_acq, par.t_dw, par.n_samples)
                + seq.wait(t3)
            )
            
            yield seq.wait(par.t_end)
