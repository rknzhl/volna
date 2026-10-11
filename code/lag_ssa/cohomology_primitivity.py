"""Exact small-row primitive-cohomology certificate on ONE finite complex.

Edges define its full 1-skeleton, including the explicit vertex count. C[r,E]
contains CLOSED integer 1-cocycles, with signs in the supplied edge orientation.
Triangle closure and the same-complex mod-p Betti number are EXTERNAL
preconditions: this module has no triangles, barcode, field or noise input.

A spanning tree gauges away tree values. Remaining chord values are periods on
a Z-basis of the GRAPH cycle lattice. Closed cocycles annihilate all 2-cell
boundaries, so the column lattice of the r-row period matrix is precisely the
image H1(K;Z)->Z^r. Column HNF=I_r proves surjectivity. If b1(K;F_p)=r,
these forms are a full primitive basis of H1(K;Z)'s FREE dual H^1(K;Z).
Torsion at other primes and the topology/geometry of any true manifold remain
uncertified. No large boundary-matrix Smith normal form is performed.
"""

from math import prod

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from sympy import Integer, Matrix, isprime
from sympy.matrices.normalforms import hermite_normal_form


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer, Integer)):
        raise ValueError(name+" must contain exact integers, not float/bool values")
    return int(value)


def _integer_array(value, name, ndim):
    array=np.asarray(value, dtype=object)
    if array.ndim != ndim:
        raise ValueError(name+" has wrong array dimension")
    result=np.empty(array.shape, dtype=object)
    for index in np.ndindex(array.shape):
        result[index]=_integer(array[index], name)
    return result


def certify_primitivity(edges, integer_cocycles, b1_modp, *, vertex_count,
                        integer_closure_verified=False, same_complex_verified=False,
                        prime=47, return_cycle_matrix=False):
    """Certify integer primitivity CONDITIONAL on stated finite-complex facts.

    `integer_closure_verified` means integer coboundary is ZERO on every
    triangle/2-cell of the same complex, not merely zero modulo p.
    `same_complex_verified` means b1_modp is the TOTAL Betti number of that
    complex over F_p, not the number of selected bars. Both default False.
    Connectedness is computed here, retaining isolated vertices. Cocycles
    may use arbitrary-precision signed integers and either edge orientation.
    """
    if not isinstance(integer_closure_verified, (bool,np.bool_)) or not isinstance(same_complex_verified, (bool,np.bool_)):
        raise ValueError("external verification flags must be boolean")
    n=_integer(vertex_count,"vertex_count")
    betti=_integer(b1_modp,"b1_modp")
    p=_integer(prime,"prime")
    if n < 1 or betti < 0 or not isprime(p):
        raise ValueError("positive vertex count, nonnegative Betti number and prime required")
    edge=_integer_array(edges,"edges",2)
    c=_integer_array(integer_cocycles,"integer_cocycles",2)
    if edge.shape[1] != 2 or c.shape[1] != len(edge):
        raise ValueError("edges[E,2] and integer_cocycles[r,E] must match")
    r=c.shape[0]
    lookup={}
    for index,(u,v) in enumerate(edge):
        if not 0 <= u < n or not 0 <= v < n or u == v:
            raise ValueError("edge endpoints must be distinct vertices of the full complex")
        pair=(min(u,v),max(u,v))
        if pair in lookup:
            raise ValueError("duplicate undirected edge in finite 1-skeleton")
        lookup[pair]=index
    endpoints=np.asarray(edge,dtype=np.int64)
    row=np.concatenate((endpoints[:,0],endpoints[:,1]))
    col=np.concatenate((endpoints[:,1],endpoints[:,0]))
    graph=csr_matrix((np.ones(len(row),dtype=np.int8),(row,col)),shape=(n,n))
    components,_=connected_components(graph,directed=False)
    result=dict(status="finite_complex_certificate",vertices=n,edges=len(edge),
        connected_components=int(components),connected=components == 1,
        selected_cocycle_count=r,prime=p,total_same_complex_b1_modp=betti,
        integer_triangle_closure_verified_externally=bool(integer_closure_verified),
        same_complex_total_modp_betti_verified_externally=bool(same_complex_verified),
        no_triangle_closure_inferred_from_modp_values=True,
        image_column_HNF=None,period_matrix_rank=None,
        graph_cycle_rank=len(edge)-n+int(components),
        selected_image_is_Zr=False,primitive_selected_cohomology_summand_certified=False,
        full_primitive_free_cohomology_basis_certified=False,
        torsion_in_H1_at_other_primes_certified_absent=False,
        true_manifold_topology_or_phase_faithfulness_certified=False,
        no_large_boundary_matrix_SNF=True)
    if components != 1:
        result["status"]="unresolved_disconnected_complex"
        return result
    order,parent=breadth_first_order(graph,0,directed=False,return_predecessors=True)
    potential=np.zeros((r,n),dtype=object)
    tree_edges=set()
    for vertex in order[1:]:
        predecessor=int(parent[vertex]); vertex=int(vertex)
        index=lookup[(min(vertex,predecessor),max(vertex,predecessor))]
        u,v=edge[index]
        sign=1 if (u,v) == (predecessor,vertex) else -1
        potential[:,vertex]=potential[:,predecessor]+sign*c[:,index]
        tree_edges.add(index)
    chord_indices=[index for index in range(len(edge)) if index not in tree_edges]
    periods=np.empty((r,len(chord_indices)),dtype=object)
    for position,index in enumerate(chord_indices):
        u,v=edge[index]
        # root->u, supplied oriented chord u->v, then v->root.
        periods[:,position]=potential[:,u]+c[:,index]-potential[:,v]
    for index in tree_edges:
        u,v=edge[index]
        if any(potential[:,u]+c[:,index]-potential[:,v]):
            raise ArithmeticError("tree gauge periods do not vanish exactly")
    if len(chord_indices) != len(edge)-n+1:
        raise ArithmeticError("fundamental graph cycle count is inconsistent")
    a=Matrix(r,len(chord_indices),[int(value) for value in periods.ravel()])
    hnf=hermite_normal_form(a) if r else Matrix.zeros(0,0)
    rank=hnf.cols
    onto=hnf == Matrix.eye(r)
    index=abs(int(hnf.det())) if rank == r else None
    if integer_closure_verified and same_complex_verified and rank > betti:
        raise ArithmeticError("closed independent integer classes exceed total same-complex mod-p b1")
    conditional=bool(integer_closure_verified and same_complex_verified)
    result.update(image_column_HNF=[[int(value) for value in row] for row in hnf.tolist()],
        period_matrix_rank=rank,image_index_in_Zr=index,
        selected_image_is_Zr=bool(onto),
        primitive_selected_cohomology_summand_certified=bool(onto and integer_closure_verified),
        full_primitive_free_cohomology_basis_certified=bool(onto and conditional and betti == r),
        graph_cycle_period_matrix_shape=[r,len(chord_indices)],
        chord_edge_indices=chord_indices,
        graph_cycle_basis_orientation="root->edge_tail, edge_tail->edge_head, edge_head->root",
        spanning_tree_root=0,
        total_rational_b1_equals_selected_count_certified=bool(conditional and rank == r and betti == r),
        certificate_scope="exact free integral cohomology of supplied connected finite complex only")
    if not integer_closure_verified:
        result["status"]="unresolved_integer_closure_not_verified"
    elif not onto:
        result["status"]="nonprimitive_or_dependent_selected_classes"
    elif not same_complex_verified:
        result["status"]="primitive_summand_full_betti_unverified"
    elif betti != r:
        result["status"]="primitive_summand_not_full_cohomology_basis"
    else:
        result["status"]="full_primitive_free_cohomology_basis"
    if return_cycle_matrix:
        result["graph_cycle_period_matrix_integer"]=periods
    return result


def _torus_complex(size=4):
    """Exact triangulated periodic torus, used by finite algebra controls only."""
    vertex=lambda i,j:(i%size)*size+(j%size)
    edges,values,lookup=[],[],{}
    for i in range(size):
        for j in range(size):
            u=vertex(i,j)
            for dx,dy in ((1,0),(0,1),(1,1)):
                v=vertex(i+dx,j+dy); pair=(min(u,v),max(u,v));sign=1 if u<v else -1
                value=(sign*int(dx and i == size-1),sign*int(dy and j == size-1))
                if pair in lookup:
                    raise ArithmeticError("duplicate edge in analytic torus")
                lookup[pair]=len(edges);edges.append(pair);values.append(value)
    triangles=[]
    for i in range(size):
        for j in range(size):
            a,b,c,d=vertex(i,j),vertex(i+1,j),vertex(i+1,j+1),vertex(i,j+1)
            triangles.extend(((a,b,c),(a,c,d)))
    cocycle=np.asarray(values,dtype=object).T
    for triangle in triangles:
        closure=np.zeros(2,dtype=object)
        for u,v in zip(triangle,triangle[1:]+triangle[:1]):
            closure+=(1 if u<v else -1)*cocycle[:,lookup[(min(u,v),max(u,v))]]
        if any(closure):raise ArithmeticError("analytic torus integer triangle closure failed")
    return np.asarray(edges,dtype=int),cocycle,size*size


def analytic_self_test():
    """Small exact graph/cell controls; no field, noise, benchmark or truth."""
    circle=np.asarray([[0,1],[1,2],[2,3],[0,3]])
    fundamental=np.asarray([[1,0,0,0]],dtype=object)
    options=dict(vertex_count=4,integer_closure_verified=True,same_complex_verified=True)
    ordinary=certify_primitivity(circle,fundamental,1,**options)
    doubled=certify_primitivity(circle,2*fundamental,1,**options)
    assert ordinary["full_primitive_free_cohomology_basis_certified"]
    assert not doubled["full_primitive_free_cohomology_basis_certified"]
    assert doubled["image_index_in_Zr"] == 2
    # Figure-eight graph: [2,3] is a primitive ONE-class summand, not a full
    # two-class basis. Neither independent class is silently divided by gcd.
    figure_eight=np.asarray([[0,1],[1,2],[0,2],[0,3],[3,4],[0,4]])
    pair=np.asarray([[0,0,2,0,0,3]],dtype=object)
    pair_result=certify_primitivity(figure_eight,pair,2,vertex_count=5,
        integer_closure_verified=True,same_complex_verified=True)
    assert pair_result["selected_image_is_Zr"]
    assert pair_result["primitive_selected_cohomology_summand_certified"]
    assert not pair_result["full_primitive_free_cohomology_basis_certified"]
    dependent=certify_primitivity(circle,np.vstack((2*fundamental,3*fundamental)),1,**options)
    assert dependent["period_matrix_rank"] == 1 and not dependent["selected_image_is_Zr"]
    edges,cocycle,n=_torus_complex()
    torus_options=dict(vertex_count=n,integer_closure_verified=True,same_complex_verified=True)
    identity=certify_primitivity(edges,cocycle,2,**torus_options)
    unimodular=certify_primitivity(edges,np.asarray([[2,3],[1,2]],dtype=object)@cocycle,2,**torus_options)
    finite_index=certify_primitivity(edges,np.asarray([[1,1],[1,-1]],dtype=object)@cocycle,2,**torus_options)
    assert identity["full_primitive_free_cohomology_basis_certified"]
    assert unimodular["full_primitive_free_cohomology_basis_certified"]
    assert finite_index["image_index_in_Zr"] == 2
    assert not finite_index["full_primitive_free_cohomology_basis_certified"]
    # Edge reversal must change the integer cochain sign, not its periods.
    reversed_edges=edges[:,::-1]
    reversed_result=certify_primitivity(reversed_edges,-cocycle,2,**torus_options)
    assert reversed_result["image_column_HNF"] == identity["image_column_HNF"]
    unknown=certify_primitivity(edges,cocycle,2,vertex_count=n)
    assert not unknown["full_primitive_free_cohomology_basis_certified"]
    isolated=certify_primitivity(circle,fundamental,1,vertex_count=5,
        integer_closure_verified=True,same_complex_verified=True)
    assert not isolated["connected"] and not isolated["full_primitive_free_cohomology_basis_certified"]
    zero=certify_primitivity(np.empty((0,2),dtype=int),np.empty((0,0),dtype=int),0,
        vertex_count=1,integer_closure_verified=True,same_complex_verified=True)
    assert zero["full_primitive_free_cohomology_basis_certified"]
    try:
        certify_primitivity(edges,cocycle,1,**torus_options)
    except ArithmeticError:
        inconsistent=True
    else:
        raise AssertionError("inconsistent total Betti assumption accepted")
    return dict(status="passed",exact_circle_degree1_degree2_controls=2,
        primitive_2_3_summand_and_dependent_pair_controls=2,
        triangulated_T2_integer_closure_and_basis_controls=3,edge_orientation_control=1,
        unknown_preconditions_disconnected_zero_and_inconsistent_controls=4,
        no_field_noise_or_evaluator_data_read=True)


if __name__ == "__main__":
    print(analytic_self_test())
