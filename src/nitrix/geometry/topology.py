# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mesh topology invariants -- the genus-0 defect gate.

Cheap combinatorial invariants of a triangle mesh: the Euler characteristic
``chi = V - E + F`` and (for a closed orientable surface) the genus
``g = (2 - chi) / 2``.  These are the **defect gate** of the field->mesh route:
a consumer checks ``euler_characteristic(mesh) == 2`` (``genus == 0``) to know
whether a marching-cubes / level-set extraction produced the expected
spherical topology -- i.e. whether the template / geometry-light escape hatch
is safe, or a topology correction is required.

Host-side integer combinatorics (like the icosphere construction); these return
Python ``int``\\ s, not traced arrays, and are not differentiable.

The full topology *corrector* (FreeSurfer ``mris_fix_topology``) is deliberately
deferred to a research track; the field->mesh pipeline keeps the seam open so a
corrector can slot between extraction and inflation later.  See
``docs/design/geometry-suite.md`` (decision D1).

Consequence for the field->mesh route (``recon-all-clinical``): because there is
no corrector yet, raw ``marching_cubes`` output is **gate-only** -- if ``genus``
reports ``> 0`` the suite can detect but not repair it.  The supported route to a
genus-0 surface is a template-correspondence model (e.g. ``topofit``/``synthdist``)
+ ``deform_to_sdf``, which inherits genus-0 from the template; a marching-cubes
mesh with handles must be corrected by an external tool until GS-7 lands.
"""

from __future__ import annotations

import numpy as np

from ..sparse import Mesh

__all__ = [
    'euler_characteristic',
    'genus',
]


def _n_edges(mesh: Mesh) -> int:
    """Number of unique undirected edges of a triangle mesh (host-side)."""
    f = np.asarray(mesh.faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return int(np.unique(e, axis=0).shape[0])


def euler_characteristic(mesh: Mesh) -> int:
    """Euler characteristic ``chi = V - E + F``.

    A closed genus-0 surface (topological sphere) has ``chi == 2``; a torus has
    ``chi == 0``; an open disk has ``chi == 1``.

    Parameters
    ----------
    mesh
        Triangle mesh.

    Returns
    -------
    The integer Euler characteristic.  Host-side; not differentiable.
    """
    return mesh.n_vertices - _n_edges(mesh) + mesh.n_faces


def genus(mesh: Mesh) -> int:
    """Genus ``g = (2 - chi) / 2`` of a closed orientable surface.

    A topological sphere has ``g == 0``, a torus ``g == 1``, etc.

    Parameters
    ----------
    mesh
        Triangle mesh, assumed **closed, connected, and orientable** (the genus
        formula does not apply to a surface with boundary or to a non-manifold
        mesh).

    Returns
    -------
    The integer genus.

    Raises
    ------
    ValueError
        If ``2 - chi`` is odd -- a definitive sign the mesh is not a closed
        orientable surface (it has a boundary or is non-manifold), so the
        genus is not well-defined by this formula.
    """
    two_minus_chi = 2 - euler_characteristic(mesh)
    if two_minus_chi % 2 != 0:
        raise ValueError(
            f'genus: 2 - chi = {two_minus_chi} is odd, so the mesh is not a '
            'closed orientable surface (boundary or non-manifold edges?); '
            'genus is undefined by the Euler formula.'
        )
    return two_minus_chi // 2
