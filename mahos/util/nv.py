#!/usr/bin/env python3

"""
Functions for Diamond NV centers.

.. This file is a part of MAHOS project.

"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import eigh
import matplotlib as mpl
import matplotlib.pyplot as plt


# Default parameters
# Note that energy is measured in MHz.
# e.g., Dgs_MHz = Dgs / h, where h is Planck constant

Dgs_MHz = 2870.0
Des_MHz = 1423.0
gamma_MHz_mT = 28.03  # MHz / mT
k_perp_MHz_nm_V = 170.0  # MHz nm / V or kHz um / V


class VectorProjector(object):
    """Project vector along different NV-axes in single crystal diamond."""

    AXIS = np.array(
        [
            [0.0, 0.0, 1.0],
            [-2.0 * np.sqrt(2.0) / 3.0, 0.0, -1.0 / 3.0],
            [+np.sqrt(2.0) / 3.0, -np.sqrt(2.0 / 3.0), -1.0 / 3.0],
            [+np.sqrt(2.0) / 3.0, +np.sqrt(2.0 / 3.0), -1.0 / 3.0],
        ]
    )

    def __init__(self):
        self.Rs, self.Rinvs = self.make_matrices()

    def make_matrices(self):
        Rs = []
        Rinvs = []
        for i, e_z in enumerate(self.AXIS):
            if i < len(self.AXIS) - 1:
                e_x_ = self.AXIS[i + 1]
            else:
                e_x_ = self.AXIS[0]

            e_y_ = np.cross(e_x_, e_z)
            e_y = e_y_ / np.linalg.norm(e_y_)
            e_x_ = np.cross(e_y, e_z)
            e_x = e_x_ / np.linalg.norm(e_x_)

            # General form of base transformation is like this.
            # R = np.vstack((e_x, e_y, e_z)).T
            # Rs.append(R)
            # Rinvs.append(np.linalg.inv(R))

            # Since R is orthogonal matrix, Rinv is R.T
            Rinv = np.vstack((e_x, e_y, e_z))
            R = Rinv.T
            Rs.append(R)
            Rinvs.append(Rinv)
        return Rs, Rinvs

    def project(self, i: int, vec: NDArray) -> NDArray:
        """project given vector ((3,) array) along i-th basis."""

        return self.Rinvs[i] @ vec


projector = VectorProjector()


class _VectorProjector2(object):
    """projector to reproduce meow1's odmr gui.

    This is different from VectorProjector and might be bugged.
    (Z-Axis direction is flipped for i = 1, 2, 3.)

    """

    AXIS = np.array(
        [
            [0, 0, 1],
            [2 * np.sqrt(2) / 3, 0, -1 / 3],
            [-np.sqrt(2) / 3, -np.sqrt(2 / 3), -1 / 3],
            [-np.sqrt(2) / 3, np.sqrt(2 / 3), -1 / 3],
        ]
    )

    def __init__(self):
        self.Rs, self.Rinvs = self.make_matrices()

    def make_matrices(self):
        Rs = []
        Rinvs = []
        for i, d0 in enumerate(self.AXIS):
            d1 = self.AXIS[i - 1]

            d_perp = np.cross(d0, d1)
            d1 = np.cross(d_perp, d0)
            e_x = d1 / np.linalg.norm(d1)
            e_y = d_perp / np.linalg.norm(d_perp)
            e_z = d0 / np.linalg.norm(d0)

            Rinv = np.vstack((e_x, e_y, e_z))
            R = Rinv.T
            Rs.append(R)
            Rinvs.append(Rinv)
        return Rs, Rinvs

    def project(self, i: int, vec: NDArray) -> NDArray:
        """project given vector ((3,) array) along i-th basis."""

        return self.Rinvs[i] @ vec


_projector2 = _VectorProjector2()


class Projector_from100(object):
    def __init__(self):
        self.R, self.Rinv = self.make_matrix()

    def make_matrix(self):
        e_x = np.array([1.0, 1.0, -2.0]) / np.sqrt(6)
        e_z = np.ones(3) / np.sqrt(3)
        e_y = np.cross(e_z, e_x)

        Rinv = np.vstack((e_x, e_y, e_z))
        R = Rinv.T
        return R, Rinv

    def project(self, vec: NDArray) -> NDArray:
        """project given vector ((3,) array) in 100 basis to 111 basis."""

        return self.Rinv @ vec


projector_from100 = Projector_from100()
project_from100 = projector_from100.project


def peaks_of_B(
    B, theta=None, phi=None, num: int = 4, D: float = Dgs_MHz, gamma: float = gamma_MHz_mT
) -> list[float]:
    """Compute peak positions (energies) for NV centers under B field (without E field)."""

    if theta is not None and phi is not None:
        B = B * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    peaks = []
    for i in range(num):
        Bx, By, Bz = projector.project(i, B)
        H = np.zeros((3, 3), dtype=np.complex128)
        H[1, 1] = D - gamma * Bz
        H[2, 2] = D + gamma * Bz
        H[1, 0] = gamma * (Bx + 1j * By) / np.sqrt(2)
        H[2, 0] = gamma * (Bx - 1j * By) / np.sqrt(2)
        # we don't need to set these. only lower triangle is seen.
        # H[0, 1] = np.conjugate(H[1, 0]); H[0, 2] = np.conjugate(H[2, 0])
        w = eigh(H, eigvals_only=True, lower=True)
        peaks.extend([w[1] - w[0], w[2] - w[0]])
    return peaks


def peaks_of_B_es(
    B, theta=None, phi=None, num: int = 4, D: float = Des_MHz, gamma: float = gamma_MHz_mT
) -> list[float]:
    """Compute peak positions (energies) for NV centers under B field (without E field)."""

    if theta is not None and phi is not None:
        B = B * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    peaks = []
    for i in range(num):
        Bx, By, Bz = projector.project(i, B)
        peaks.extend([D - gamma * Bz, D + gamma * Bz])
    return peaks


def peaks_of_BE(
    B,
    E,
    theta_B=None,
    phi_B=None,
    theta_E=None,
    phi_E=None,
    num: int = 4,
    D: float = Dgs_MHz,
    gamma: float = gamma_MHz_mT,
    k_perp: float = k_perp_MHz_nm_V,
) -> list[float]:
    """Compute peak positions (energies) for NV centers under B and E fields.

    Ref. Doherty et al. Phys. Rev. B 85, 205203 (2012).

    """
    if theta_B is not None and phi_B is not None:
        B = B * np.array(
            [np.sin(theta_B) * np.cos(phi_B), np.sin(theta_B) * np.sin(phi_B), np.cos(theta_B)]
        )
    if theta_E is not None and phi_E is not None:
        E = E * np.array(
            [np.sin(theta_E) * np.cos(phi_E), np.sin(theta_E) * np.sin(phi_E), np.cos(theta_E)]
        )
    peaks = []
    for i in range(num):
        Bx, By, Bz = projector.project(i, B)
        Ex, Ey, Ez = projector.project(i, E)
        Bp = np.sqrt(Bx**2 + By**2)
        Ep = np.sqrt(Ex**2 + Ey**2)

        th = np.arctan2(k_perp * Ep, gamma * Bz)
        ph = 2 * np.arctan2(By, Bx) + np.arctan2(Ey, Ex)
        R = np.sqrt((gamma * Bz) ** 2 + (k_perp * Ep) ** 2)
        b = gamma * Bp / np.sqrt(2)
        od1 = b * (np.exp(1j * ph / 2) * np.sin(th / 2) + np.exp(-1j * ph / 2) * np.cos(th / 2))
        od2 = b * (np.exp(1j * ph / 2) * np.cos(th / 2) - np.exp(-1j * ph / 2) * np.sin(th / 2))
        H = np.zeros((3, 3), dtype=np.complex128)
        H[1, 1] = D - R
        H[2, 2] = D + R
        H[1, 0] = od1
        H[2, 0] = od2
        # we don't need to set these. only lower triangle is seen.
        # H[0, 1] = np.conjugate(H[1, 0]); H[0, 2] = np.conjugate(H[2, 0])
        w = eigh(H, eigvals_only=True, lower=True)
        peaks.extend([w[1] - w[0], w[2] - w[0]])
    return peaks


def peaks_of_B_aligned(
    B,
    D: float = Dgs_MHz,
    gamma: float = gamma_MHz_mT,
) -> list[float]:
    return peaks_of_B(B, 0.0, 0.0, num=2, D=D, gamma=gamma)


def peaks_of_B_single(
    B,
    theta,
    phi,
    D: float = Dgs_MHz,
    gamma: float = gamma_MHz_mT,
) -> list[float]:
    return peaks_of_B(B, theta, phi, num=1, D=D, gamma=gamma)


def view_projector_axes(proj, indices=[0, 1, 2, 3], color_points=[0.3, 0.5, 0.8, 1.0], text=True):
    reds = mpl.colormaps.get("Reds")
    greens = mpl.colormaps.get("Greens")
    blues = mpl.colormaps.get("Blues")

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    exs = []
    eys = []
    ezs = []
    for i in indices:
        Rinv = proj.Rinvs[i]
        ex, ey, ez = Rinv
        exs.append(ex)
        eys.append(ey)
        ezs.append(ez)
    exs = np.array(exs)
    eys = np.array(eys)
    ezs = np.array(ezs)
    x = y = z = np.zeros(len(indices))
    cp = color_points
    ax.quiver(x, y, z, exs[:, 0], exs[:, 1], exs[:, 2], colors=[reds(p) for p in cp])
    ax.quiver(x, y, z, eys[:, 0], eys[:, 1], eys[:, 2], colors=[greens(p) for p in cp])
    ax.quiver(x, y, z, ezs[:, 0], ezs[:, 1], ezs[:, 2], colors=[blues(p) for p in cp])
    if text:
        for i, ex, ey, ez in zip(indices, exs, eys, ezs):
            ax.text(*ex, f"x{i}")
            ax.text(*ey, f"y{i}")
            ax.text(*ez, f"z{i}")

    bounds = (-1.0, 1.0)
    ax.set_xlim(bounds)
    ax.set_ylim(bounds)
    ax.set_zlim(bounds)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ticks = [-1, 0, 1]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)
    plt.show()
