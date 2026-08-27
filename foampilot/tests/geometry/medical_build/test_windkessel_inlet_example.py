import numpy as np
from foampilot.model_addon.windkessel import Windkessel


def test_windkessel_inlet_waveform_is_solved_in_si_units():
    t = np.linspace(0.0, 0.8, 65)
    q = 3.0e-4 * np.maximum(0.0, np.sin(2.0 * np.pi * t / 0.8))
    wk = Windkessel(t, q, Rc=1.0e6, Rp=2.0e9, C=2.0e-7, L=5.0e3, Cprox=1.0e-8)
    result = wk.solve(t_start=0.0, t_end=0.8, n_steps=400)
    assert result.success
    assert result.p1.shape == result.t.shape
    assert np.all(np.isfinite(result.p1))
    assert float(np.ptp(result.p1)) > 0.0
