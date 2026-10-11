"""Exact nonlocal chord proposals for an integer-defined compact phase group.

This module proposes points only; it neither estimates nor lower-bounds reach.
For E=ker(R), an integer z gives exp(2*pi*i*P_E*z) in the identity
component. Its ambient angles equal -2*pi*P_normal*z modulo 2*pi.
LLL of the projected integer lattice proposes short such displacements.
All group membership and integer preimages are checked with exact arithmetic.
"""

from __future__ import annotations

from math import lcm

from sympy import Matrix, Rational, ZZ, floor
from sympy.matrices.normalforms import hermite_normal_form
from sympy.polys.matrices import DomainMatrix
from sympy.polys.matrices.normalforms import smith_normal_decomp


def _integers(matrix):
    if any(not value.is_Integer for value in matrix):
        raise ArithmeticError("integer matrix required")
    return [[int(value) for value in row] for row in matrix.tolist()]


def projected_lattice_chords(relation_rows, ambient_dimension):
    """Return exact rational angles of finitely many distinct points of ann(R).

    Every returned point lies in the identity component before wrapping.
    LLL is a deterministic proposal mechanism, not an exact shortest-vector
    solver or a global manifold separation certificate. Finite components
    require additional chord proposals outside this routine.
    """
    count = int(ambient_dimension)
    relation = Matrix(len(relation_rows), count,
                      [int(value) for row in relation_rows for value in row])
    rank = relation.rank()
    if rank != relation.rows:
        raise ValueError("independent relation rows required")
    if rank in (0, count):
        return dict(status="no_nontrivial_connected_normal_lattice", points=[],
                    relation_rank=rank, continuous_dimension=count-rank)

    normal = relation.T*(relation*relation.T).inv()*relation
    tangent = Matrix.eye(count)-normal
    if (normal != normal.T or normal*normal != normal
            or tangent*tangent != tangent or relation*tangent != Matrix.zeros(rank, count)):
        raise ArithmeticError("exact orthogonal projections failed")
    denominator = 1
    for value in normal:
        denominator = lcm(denominator, int(value.q))
    integer_projection = denominator*normal
    _integers(integer_projection)
    integer_projection = Matrix(count, count, [int(value) for value in integer_projection])
    image_basis = hermite_normal_form(integer_projection)
    if image_basis.shape != (count, rank):
        raise ArithmeticError("incorrect projected-lattice image rank")
    reduced_rows, transform = image_basis.T.lll_transform(delta=Rational(3, 4))
    if (reduced_rows != transform*image_basis.T or abs(int(transform.det())) != 1):
        raise ArithmeticError("LLL image transform is not unimodular")

    # U A V=D, hence z=V D^+ U v is an integer preimage for v in im_Z(A).
    diagonal_d, left_d, right_d = smith_normal_decomp(
        DomainMatrix.from_Matrix(integer_projection).convert_to(ZZ))
    diagonal, left, right = (item.to_Matrix() for item in (diagonal_d, left_d, right_d))
    if (left*integer_projection*right != diagonal
            or abs(int(left.det())) != 1 or abs(int(right.det())) != 1):
        raise ArithmeticError("exact projected-lattice Smith decomposition failed")

    points, skipped, seen = [], [], set()
    for index, values in enumerate(reduced_rows.tolist()):
        vector = Matrix(values)
        first = next((value for value in vector if value), 0)
        if first < 0:
            vector = -vector
        smith_values = left*vector
        coordinates = Matrix.zeros(count, 1)
        for row in range(count):
            if diagonal[row, row]:
                coordinates[row] = smith_values[row]/diagonal[row, row]
                if not coordinates[row].is_Integer:
                    raise ArithmeticError("projected lattice vector has no integral preimage")
            elif smith_values[row]:
                raise ArithmeticError("projected vector lies outside the exact image")
        preimage = right*coordinates
        _integers(preimage)
        if integer_projection*preimage != vector:
            raise ArithmeticError("projected-lattice integer preimage failed")
        displacement = -normal*preimage
        wrapped = displacement.applyfunc(lambda value: value-floor(value+Rational(1, 2)))
        windings = relation*wrapped
        _integers(windings)
        if not any(wrapped):
            skipped.append(dict(basis_index=index, reason="point_is_identity_after_wrapping"))
            continue
        point_key = tuple(wrapped)
        if point_key in seen:
            skipped.append(dict(basis_index=index, reason="same_wrapped_point_already_proposed"))
            continue
        seen.add(point_key)
        if any(not value.is_Integer for value in wrapped-displacement):
            raise ArithmeticError("wrapping changed phase by a noninteger turn")
        point = dict(basis_index=index,
                     integer_preimage=[int(value) for value in preimage],
                     projected_normal_turns=[str(value) for value in -displacement],
                     wrapped_phase_turns=[str(value) for value in wrapped],
                     exact_membership_windings=[int(value) for value in windings],
                     unwrapped_normal_length_in_turns_squared=str((displacement.T*displacement)[0]),
                     identity_component_membership_verified=True,
                     proposed_by="LLL basis of P_normal Z^J, exact rational arithmetic")
        points.append(point)
    return dict(status="exact_chord_proposals_ready", points=points, skipped=skipped,
                relation_rank=rank, continuous_dimension=count-rank,
                common_projector_denominator=str(denominator),
                exact_group_membership_verified=True,
                shortest_projected_vector_certified=False,
                reach_lower_bound_certified=False)
