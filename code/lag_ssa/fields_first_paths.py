"""Label fine chord paths by a frozen finite complex's full integer cohomology.

No beta values, fields, physical phases, frequencies or circle coordinates
are used. Returned labels certify classes in H1 of THIS finite complex.
"""
import numpy as np
from scipy.spatial.distance import cdist


def edge_integer_labels(X, fineedges, cx, *, max_depth=12,
                        relative_coverage_margin=1e-12):
    """Return (Cfine[E,r], valid[E], metadata) in supplied edge orientation.

    Fixed vertex charts are nearest active landmark balls.  A vertex that
    is itself a stored landmark is assigned its OWN chart.  Each chord is
    subdivided until every subsegment has one ball containing BOTH endpoints;
    convexity then covers that entire straight subsegment.  At switches, two
    charts overlap at a shared point, and their oriented edge label is added.

    Accumulate these labels along the same fine paths used for beta integrals.
    For a coarse oriented edge u->v between landmarks, equality to C[:,edge]
    certifies matching integral homology class, conditional on the frozen full
    primitive finite-complex certificate.  It is not a true-manifold claim.
    Reverse a stored oriented path by NEGATING its integer and beta integrals.
    """
    x=np.asarray(X,dtype=float); edges=np.asarray(fineedges)
    centers=np.asarray(cx['landmarks'],dtype=float)
    c=np.asarray(cx['integer_cocycles'])
    coarse=np.asarray(cx['edges'],dtype=np.int64)
    certificate=cx.get('certificate',{})
    if not certificate.get('full_primitive_free_cohomology_basis_certified',False):
        raise ValueError('full primitive cohomology certificate of this same finite K required')
    if (x.ndim!=2 or centers.ndim!=2 or x.shape[1]!=centers.shape[1]
        or not np.isfinite(x).all() or not np.isfinite(centers).all()):
        raise ValueError('finite X[vertices,ambient] and compatible landmark centers required')
    if (edges.ndim!=2 or edges.shape[1]!=2 or not np.issubdtype(edges.dtype,np.integer)
        or (edges.size and (edges.min()<0 or edges.max()>=len(x)))):
        raise ValueError('fineedges[E,2] must contain valid integer vertex indices')
    if not np.issubdtype(c.dtype,np.integer) or c.shape[1]!=len(coarse):
        raise ValueError('exact integer cocycles in the frozen coarse edge order required')
    if int(max_depth)!=max_depth or max_depth<0:
        raise ValueError('nonnegative integer max_depth required')
    if not 0<=relative_coverage_margin<1:
        raise ValueError('coverage margin must be in [0,1)')
    n,r=len(centers),c.shape[0]
    radius=float(cx['cover_radius'])
    margin=max(float(relative_coverage_margin)*radius,
               128*np.finfo(float).eps*max(1.,radius))
    safe_radius=radius-margin
    if not safe_radius>0:
        raise ValueError('positive robust covering radius required')
    safe2=safe_radius**2
    distance2=cdist(x,centers,'sqeuclidean')
    active=distance2<safe2
    covered=active.any(axis=1)
    roots=np.argmin(distance2,axis=1).astype(np.int64)
    roots[~covered]=-1
    landmark_indices=np.asarray(cx['landmark_indices'],dtype=np.int64)
    if (landmark_indices.shape!=(n,) or np.any(landmark_indices<0)
        or np.any(landmark_indices>=len(x))):
        raise ValueError('frozen landmark indices must refer to this SAME fine vertex array')
    discrepancy=np.linalg.norm(x[landmark_indices]-centers,axis=1)
    if np.max(discrepancy)>128*np.finfo(float).eps*max(1.,float(np.max(abs(centers)))):
        raise ValueError('landmark coordinates do not match X at frozen landmark indices')
    roots[landmark_indices]=np.arange(n)
    adjacency=np.zeros((n,n),dtype=bool)
    labels=np.zeros((n,n,r),dtype=np.int64)
    for k,(u,v) in enumerate(coarse):
        adjacency[u,v]=adjacency[v,u]=True
        labels[u,v]=c[:,k];labels[v,u]=-c[:,k]
    out=np.zeros((len(edges),r),dtype=np.int64)
    valid=np.zeros(len(edges),dtype=bool)
    counters=dict(midpoints_checked=0,certified_chord_subsegments=0,
                  deepest_subdivision=0,chart_transitions=0)
    failures=[]; max_hops=0

    def cover_chain(a,b,aa,ab,depth):
        counters['deepest_subdivision']=max(counters['deepest_subdivision'],depth)
        common=np.flatnonzero(aa&ab)
        if len(common):
            counters['certified_chord_subsegments']+=1
            return [int(common[0])],None
        if depth>=max_depth:
            return None,'unresolved_no_common_ball_at_maximum_depth'
        middle=(a+b)/2
        am=np.einsum('np,np->n',centers-middle,centers-middle)<safe2
        counters['midpoints_checked']+=1
        if not am.any():
            return None,'chord_contains_uncovered_midpoint'
        left,reason=cover_chain(a,middle,aa,am,depth+1)
        if left is None:return None,reason
        right,reason=cover_chain(middle,b,am,ab,depth+1)
        if right is None:return None,reason
        return left+right,None

    for index,(u,v) in enumerate(edges):
        u,v=int(u),int(v)
        if not covered[u] or not covered[v]:
            reason='uncovered_fine_vertex'
        else:
            chain,reason=cover_chain(x[u],x[v],active[u],active[v],0)
            if chain is not None:
                chain=[int(roots[u])]+chain+[int(roots[v])]
                reduced=[chain[0]]
                for chart in chain[1:]:
                    if chart!=reduced[-1]:reduced.append(chart)
                label=[0]*r
                for a,b in zip(reduced[:-1],reduced[1:]):
                    if not adjacency[a,b]:
                        reason='coactive_chart_transition_missing_from_frozen_complex'
                        break
                    for j in range(r):label[j]+=int(labels[a,b,j])
                else:
                    if any(abs(value)>np.iinfo(np.int64).max for value in label):
                        reason='integer_path_label_overflow'
                    else:
                        out[index]=label;valid[index]=True
                        max_hops=max(max_hops,len(reduced)-1)
                        counters['chart_transitions']+=len(reduced)-1
                        continue
        if len(failures)<32:
            failures.append(dict(edge_index=index,tail=u,head=v,reason=reason))
    metadata=dict(status='finite_complex_chord_class_labels' if valid.all()
        else 'unresolved_some_fine_chords',fine_edges=len(edges),r=r,
        valid_edge_count=int(valid.sum()),invalid_edge_count=int((~valid).sum()),
        covered_vertex_count=int(covered.sum()),vertex_count=len(x),
        vertex_charts=roots,covered_vertices=covered,
        cover_radius=radius,robust_cover_radius=safe_radius,
        coverage_margin=margin,maximum_depth=int(max_depth),
        maximum_chart_transitions_on_one_chord=max_hops,
        failure_examples=failures,**counters,
        chord_coverage_rule='one convex landmark ball contains both endpoints of every accepted subsegment',
        landmark_vertices_assigned_to_their_own_charts=True,
        label_orientation='same tail->head orientation as supplied fineedges',
        periods_or_phases_used=False,true_manifold_homotopy_certified=False,
        certificate_scope='integral H1 class in the supplied finite complex only',
        equality_to_coarse_integer_label_is_homology_not_general_pi1_certificate=True)
    return out,valid,metadata


if __name__=='__main__':
    import json
    if __package__:
        from .fields_first_periods import certify_primitivity
    else:
        from fields_first_periods import certify_primitivity
    size=64; landmark=np.arange(0,size,8)
    theta=np.arange(size)*2*np.pi/size
    x=np.c_[np.cos(theta),np.sin(theta)]
    coarse=np.array([[i,i+1] for i in range(7)]+[[0,7]],dtype=np.int64)
    c=np.zeros((1,8),dtype=np.int64);c[0,-1]=-1
    cert=certify_primitivity(coarse,c,1,vertex_count=8,
        integer_closure_verified=True,same_complex_verified=True)
    cx=dict(landmarks=x[landmark],landmark_indices=landmark,edges=coarse,
            integer_cocycles=c,cover_radius=.5,certificate=cert)
    fine=np.array([[i,(i+1)%size] for i in range(size)]+[[0,size//2]],dtype=np.int64)
    value,valid,meta=edge_integer_labels(x,fine,cx)
    assert valid[:-1].all() and not valid[-1]
    assert int(value[:-1].sum())==1
    # The positive arc from landmark0 to landmark1 has the coarse edge class.
    assert int(value[:8].sum())==int(c[0,0])
    # Taking a full turn before that edge must be detected as a different class.
    assert int(value[:-1].sum()+value[:8].sum())!=int(c[0,0])
    print(json.dumps(dict(status='passed',full_turn_label=int(value[:-1].sum()),
        local_coarse_edge_class_matches=True,extra_full_turn_is_detected=True,
        uncovered_diameter_chord_rejected=True,
        exact_chord_coverage_by_convex_ball_subsegments=True,
        scope=meta['certificate_scope'])))
