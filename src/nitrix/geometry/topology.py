# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Mesh topology invariants -- the genus-0 defect gate.

Cheap combinatorial invariants of a triangle mesh: the Euler characteristic
:math:`\\chi = V - E + F` and (for a closed orientable surface) the genus
:math:`g = (2 - \\chi) / 2`.  These form a **defect gate** for the field-to-mesh
route: a consumer checks ``euler_characteristic(mesh) == 2`` (equivalently
:func:`genus` ``== 0``) to know whether a marching-cubes / level-set extraction
produced the expected spherical topology -- i.e. whether the template escape
hatch is safe, or a topology correction is required.

These are host-side integer combinatorics; they return Python ``int``\\ s, not
traced arrays, and are not differentiable.

A full topology *corrector* (in the manner of FreeSurfer's ``mris_fix_topology``)
is deliberately out of scope here; the field-to-mesh pipeline keeps the seam open
so that a corrector can slot between extraction and inflation later.

Because there is no corrector yet, raw :func:`marching_cubes` output is
gate-only: if :func:`genus` reports a value ``> 0`` the topology defect can be
detected but not repaired.  The supported route to a genus-0 surface is a
template-correspondence model followed by :func:`deform_to_sdf`, which inherits
genus-0 from the template; a marching-cubes mesh with handles must be corrected
by an external tool.
"""

from __future__ import annotations

import numpy as np

from ..sparse import Mesh

__all__ = [
    'euler_characteristic',
    'genus',
]


def _n_edges(mesh: Mesh) -> int:
    """Number of unique undirected edges of a triangle mesh.

    Each of the three sides of every face contributes a candidate edge; the
    endpoint indices are sorted so that an edge and its reverse coincide, and
    duplicate edges shared by adjacent faces are counted once.  Computed
    host-side on NumPy arrays.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh whose ``faces`` array has shape ``(n_faces, 3)``, giving
        the three vertex indices of each triangle.

    Returns
    -------
    int
        The number of distinct undirected edges :math:`E`.  Host-side; not
        differentiable.
    """
    f = np.asarray(mesh.faces)
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    return int(np.unique(e, axis=0).shape[0])


def euler_characteristic(mesh: Mesh) -> int:
    """Euler characteristic :math:`\\chi = V - E + F`.

    Combines the vertex, edge, and face counts of the mesh into the classical
    topological invariant.  A closed genus-0 surface (topological sphere) has
    :math:`\\chi = 2`; a torus has :math:`\\chi = 0`; an open disk has
    :math:`\\chi = 1`.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh, with vertices, ``(n_faces, 3)`` faces, and the edge set
        induced by those faces.

    Returns
    -------
    int
        The integer Euler characteristic :math:`\\chi`.  Host-side; not
        differentiable.
    """
    return mesh.n_vertices - _n_edges(mesh) + mesh.n_faces


def genus(mesh: Mesh) -> int:
    """Genus :math:`g = (2 - \\chi) / 2` of a closed orientable surface.

    Derives the genus (the number of handles) from the Euler characteristic
    :math:`\\chi` returned by :func:`euler_characteristic`.  A topological
    sphere has :math:`g = 0`, a torus :math:`g = 1`, and so on.

    Parameters
    ----------
    mesh : Mesh
        Triangle mesh, assumed **closed, connected, and orientable** (the genus
        formula does not apply to a surface with boundary or to a non-manifold
        mesh).

    Returns
    -------
    int
        The integer genus :math:`g`.  Host-side; not differentiable.

    Raises
    ------
    ValueError
        If :math:`2 - \\chi` is odd -- a definitive sign that the mesh is not a
        closed orientable surface (it has a boundary or is non-manifold), so the
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
