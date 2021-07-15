import unittest
import numpy as np
from qiskit.test.base import QiskitTestCase
import quantum_mc.calibration.fitting as ft
import quantum_mc.calibration.time_series as ts

class TestCalibration(QiskitTestCase):

    def test_from_series(self):
        """Simple end-to-end test of the (semi-classical) multiply and add building block."""
        import sys, os
        correl = ft.get_correl("AAPL", "MSFT")
       
        # AAPL, MSFT, SPX
        ticker = "MSFT"
        #data = ts.get_data(ticker)
        ((cdf_x, cdf_y), sigma) = ft.get_cdf_data(ticker)
        (x, y) = ft.get_fit_data(ticker, norm_to_rel = False)
        (pl, coeffs) = ft.fit_piecewise_linear(x, y)
        int_coeffs = ft.convert_to_integer(pl, coeffs)

