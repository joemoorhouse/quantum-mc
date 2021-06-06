# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A circuit that encodes a discretized normal probability distribution in qubit amplitudes."""

from typing import Tuple, Union, List, Optional, Callable
import numpy as np
from scipy.stats import multivariate_normal, norm
from scipy.misc import derivative
from qiskit.circuit import QuantumCircuit


class GaussianCopula(QuantumCircuit):
    r"""A circuit to encode a discretized Gaussian Copula distribution in qubit amplitudes.
    """

    def __init__(self,
                 num_qubits: Union[int, List[int]],
                 cdfs: Union[Callable[[float], float], List[Callable[[float], float]]], 
                 sigma: Optional[Union[float, List[float]]] = None,
                 bounds: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
                 pdfs: Optional[Union[Callable[[float], float], List[Callable[[float], float]]]] = None, 
                 dx: Optional[float] = 1e-6,
                 upto_diag: bool = False,
                 name: str = 'P(X)') -> None:
        r"""
        Args:
            num_qubits: The number of qubits used to discretize the random variable. For a 1d
                random variable, ``num_qubits`` is an integer, for multiple dimensions a list
                of integers indicating the number of qubits to use in each dimension.
            cdfs: The cumulative marginal probability density functions for each dimension.
            sigma: The parameter :math:`\sigma^2` or :math:`\Sigma`, which is the correlation matrix. 
                Defaults to the identity matrix of appropriate size.
            bounds: The truncation bounds of the distribution as tuples. For multiple dimensions,
                ``bounds`` is a list of tuples ``[(low0, high0), (low1, high1), ...]``.
                If ``None``, the bounds are set to ``(-1, 1)`` for each dimension.
            pdfs: The marginal probability density functions for each dimension. 
                If ''None'', marginal pdf is calculated from the pdf by finite differences, but
                consider using automatic differentiation of cdf. 
            dx: The spacing, if finite differences is used to calculate the pdf.
            upto_diag: If True, load the square root of the probabilities up to multiplication
                with a diagonal for a more efficient circuit.
            name: The name of the circuit.
        """
        _check_dimensions_match(num_qubits, cdfs, pdfs, sigma, bounds)
        _check_bounds_valid(bounds)

        # set default arguments
        dim = 1 if isinstance(num_qubits, int) else len(num_qubits)

        mu = 0 if dim == 1 else [0] * dim

        if sigma is None:
            sigma = 1 if dim == 1 else np.eye(dim)

        if bounds is None:
            bounds = (-1, 1) if dim == 1 else [(-1, 1)] * dim

        if not isinstance(cdfs, list):
            cdfs = [cdfs] 

        if pdfs is None:
            pdfs = [lambda x: derivative(cdf, x, dx=dx) for cdf in cdfs]

        if not isinstance(num_qubits, list):  # univariate case
            super().__init__(num_qubits, name=name)

            x = np.linspace(bounds[0], bounds[1], num=2**num_qubits)
        else:  # multivariate case
            super().__init__(sum(num_qubits), name=name)

            # compute the evaluation points using numpy's meshgrid
            # indexing 'ij' yields the "column-based" indexing
            meshgrid = np.meshgrid(*[np.linspace(bound[0], bound[1], num=2**num_qubits[i])
                                     for i, bound in enumerate(bounds)], indexing='ij')
            # flatten into a list of points
            x = list(zip(*[grid.flatten() for grid in meshgrid]))

        from scipy.stats import multivariate_normal

        # compute the normalized, truncated probabilities
        probabilities = multivariate_normal.pdf(x, mu, sigma)

        xa = np.asarray(x, dtype=float)
        zeta = np.concatenate([norm.ppf(cdfs[i](xa[:, i])).reshape(xa.shape[0], 1) for i in range(dim)], 1) 
        probabilities = multivariate_normal.pdf(zeta, mu, sigma) * np.exp(0.5 * np.sum(np.square(zeta), axis=-1)) * np.sqrt((2 * np.pi)**dim)
        for i in range(xa.shape[1]):
            probabilities = probabilities * pdfs[i](xa[:, i])

        normalized_probabilities = probabilities / np.sum(probabilities)

        # store the values, probabilities and bounds to make them user accessible
        self._values = x
        self._probabilities = normalized_probabilities
        self._bounds = bounds

        # use default the isometry (or initialize w/o resets) algorithm to construct the circuit
        # pylint: disable=no-member
        if upto_diag:
            self.isometry(np.sqrt(normalized_probabilities), self.qubits, None)
        else:
            from qiskit.extensions import Initialize  # pylint: disable=cyclic-import
            initialize = Initialize(np.sqrt(normalized_probabilities))
            circuit = initialize.gates_to_uncompute().inverse()
            self.compose(circuit, inplace=True)

    @property
    def values(self) -> np.ndarray:
        """Return the discretized points of the random variable."""
        return self._values

    @property
    def probabilities(self) -> np.ndarray:
        """Return the sampling probabilities for the values."""
        return self._probabilities

    @property
    def bounds(self) -> Union[Tuple[float, float], List[Tuple[float, float]]]:
        """Return the bounds of the probability distribution."""
        return self._bounds


def _check_dimensions_match(num_qubits, cdfs, pdfs, sigma, bounds):
    num_qubits = [num_qubits] if not isinstance(num_qubits, (list, np.ndarray)) else num_qubits
    dim = len(num_qubits)

    if pdfs is not None:
        sigma = [[sigma]] if not isinstance(sigma, (list, np.ndarray)) else sigma
        if len(sigma) != dim or len(sigma[0]) != dim:
            raise ValueError('Dimension of sigma ({} x {}) does not match the dimension of '
                             'the random variable specified by the number of qubits ({})'
                             ''.format(len(sigma), len(sigma[0]), dim))
    
    if sigma is not None:
        sigma = [[sigma]] if not isinstance(sigma, (list, np.ndarray)) else sigma
        if len(sigma) != dim or len(sigma[0]) != dim:
            raise ValueError('Dimension of sigma ({} x {}) does not match the dimension of '
                             'the random variable specified by the number of qubits ({})'
                             ''.format(len(sigma), len(sigma[0]), dim))

    if bounds is not None:
        # bit differently to cover the case the users might pass `bounds` as a single list,
        # e.g. [0, 1], instead of a tuple
        bounds = [bounds] if not isinstance(bounds[0], tuple) else bounds
        if len(bounds) != dim:
            raise ValueError('Dimension of bounds ({}) does not match the dimension of the '
                             'random variable specified by the number of qubits ({})'
                             ''.format(len(bounds), dim))


def _check_bounds_valid(bounds):
    if bounds is None:
        return

    bounds = [bounds] if not isinstance(bounds[0], tuple) else bounds

    for i, bound in enumerate(bounds):
        if not bound[1] - bound[0] > 0:
            raise ValueError('Dimension {} of the bounds are invalid, must be a non-empty '
                             'interval where the lower bounds is smaller than the upper bound.'
                             ''.format(i))
