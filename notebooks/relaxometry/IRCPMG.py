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
    ParDef('t_inv', float_array, [100e-6], unit='s'),
    ParDef('t_echo', float, 500e-6, unit='s'),
    ParDef('n_echo', int, 1000, min=1, unit=''),
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
        len(p.t_inv),
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
    
    n_phase_cycle = 8
    phase_cycle_90 = [0, 180, 0, 180, 90, 270, 90, 270]
    phase_cycle_180 = [90, 90, 270, 270, 0, 0, 180, 180]
    
    # dummy scan to prevent first scan from having much higher signal at short rep times
    t_inv_90 = par.t_inv[0] - (par.t_90+par.t_180)/2
    yield seq.pulse_start(par.f, 0, par.a_180)
    yield seq.wait(par.t_180)
    yield seq.pulse_end()
    yield seq.wait(t_inv_90)
    yield seq.pulse_start(par.f, 180, par.a_90)
    yield seq.wait(par.t_90)
    yield seq.pulse_end()
    yield seq.wait(t1)
    yield par.n_echo * (
        seq.pulse_start(par.f, 90, par.a_180)
        + seq.wait(par.t_180)
        + seq.pulse_end()
        + seq.wait(t2)
        + seq.wait(t3)
    )
    yield seq.wait(par.t_end)
    
    for t in par.t_inv:
        t_inv_90 = t - (par.t_90+par.t_180)/2
        for i_scan in range(par.n_scans):
            p_90 = p_acq = phase_cycle_90[i_scan % n_phase_cycle]
            p_180 = phase_cycle_180[i_scan % n_phase_cycle]

            # inversion pulse
            yield seq.pulse_start(par.f, 0, par.a_180)
            yield seq.wait(par.t_180)
            yield seq.pulse_end()

            yield seq.wait(t_inv_90)

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
