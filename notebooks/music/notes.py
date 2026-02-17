from matipo import sequence as seq
from matipo import ParDef
from matipo import datalayout
from collections import namedtuple
import numpy as np

import logging
logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.WARNING)

PARDEF = [ # no parameters
]

ParameterSet = namedtuple('ParameterSet', [pd.name for pd in PARDEF])


def get_options(par: ParameterSet):
    return seq.Options(
        amp_enabled=True,
        rx_gain=7)


def get_datalayout(p: ParameterSet):
    return datalayout.Acquisition(t_dw=1e-6, n_samples=10)


def note(freq, t, volume):
    if freq != 0: # playing a note
        # Calculate gradient vector from volume
        # Limited to 0.5 amplitude to avoid exceeding duty cycle limits
        # as this gradient runs continuously for long periods
        grad = min(np.abs(volume), 1)*np.array([0, 0, 0.5]) # Do not increase the 0.5 amplitude
        t_wait = 0.5/freq
        n = int(t/(2*t_wait))
        
        # Create square wave using positive and negative gradient pulses
        for i in range(n):
            yield seq.gradient(*grad)
            yield seq.wait(t_wait)
            yield seq.gradient(*(-grad))
            yield seq.wait(t_wait)
        yield seq.gradient(0,0,0)
    else: # playing a rest
        yield seq.wait(t)
    

def main(par: ParameterSet):
    
    # All Star, Smash Mouth
    all_star = [ # (beats, frequency)
        (1, 329),
        (0.5, 494),
        (0.5, 415),
        (1, 415),
        (0.5, 370),
        (0.5, 329),
        (0.5, 329),
        (1, 440),
        (0.5, 415),
        (0.5, 415),
        (0.5, 370),
        (0.5, 370),
        (1, 329),
        (0.5, 329),
        (0.5, 494),
        (0.5, 415),
        (0.5, 415),
        (0.5, 370),
        (0.5, 370),
        (0.5, 329),
        (0.5, 329),
        (1, 277),
        (1, 247)
    ]
    
    # Rondo Alla Turca, Mozart
    rondo = [ # (beats, frequency)
        (0.25, 493),
        (0.25, 440),
        (0.25, 415),
        (0.25, 440),
        (0.5, 523),
        (0.5, 0),
        (0.25, 587),
        (0.25, 523),
        (0.25, 493),
        (0.25, 523),
        (0.5, 659),
        (0.5, 0),
        (0.25, 698),
        (0.25, 659),
        (0.25, 622),
        (0.25, 659),
        (0.25, 987),
        (0.25, 880),
        (0.25, 830),
        (0.25, 880),
        (0.25, 987),
        (0.25, 880),
        (0.25, 830),
        (0.25, 880),
        (0.95, 1046),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.45, 1046),
        (0.05, 0),
        (0.05, 784),
        (0.05, 880),
        (0.35, 987),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.45, 784),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.05, 784),
        (0.05, 880),
        (0.35, 987),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.45, 784),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.05, 784),
        (0.05, 880),
        (0.35, 987),
        (0.05, 0),
        (0.45, 880),
        (0.05, 0),
        (0.45, 784),
        (0.05, 0),
        (0.45, 739),
        (0.05, 0),
        (1, 659)
    ]
    
    bpm = 120 # beats per minute
    volume = 1 # between 0 and 1
    beat_time = 60/bpm
    for i in range(2): # repeat twice
        for beats, freq in rondo:
            yield from note(freq, beat_time*beats, volume)
    
    yield seq.wait(1)
    
    bpm = 120
    beat_time = 60/bpm
    for beats, freq in all_star:
        yield from note(freq, beat_time*beats, volume)
        yield seq.wait(0.1) # space out the notes
    
    yield seq.acquire(0, 0, 1e-6, 10) # record some data to avoid the no data error message
    yield seq.wait(0.1)
