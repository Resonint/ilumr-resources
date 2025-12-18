from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple

PARDEF = [
    ParDef('f', float, 1e6, unit='Hz'),
    ParDef('t_dw', float, 1e-6, min=0.1e-6, max=80e-6, unit='s'),
    ParDef('n_samples', int, 1000, min=2, unit='')
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(p: ParameterSet):
    return seq.Options(
        amp_enabled=False,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Acquisition(
            n_samples=p.n_samples,
            t_dw=p.t_dw)


def main(p: ParameterSet):
    t_acq = p.n_samples * p.t_dw
    yield seq.acquire(p.f, 0, p.t_dw, p.n_samples)
    yield seq.wait(t_acq)
