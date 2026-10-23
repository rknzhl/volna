"""Finite tangent-transport phase model, in the order of phase_ssa_method.pdf.

Y has shape (N,m). Joint windows have m*L coordinates; their retained numerical
linear span has dimension D. Q and W have shape (D,r); beta has shape (r,D).
Local geometry and persistence jointly estimate r under the torus hypothesis.
AFTER W, integer topology selects full period classes; no initial phases enter.
The inverse integrates the estimated fields themselves, with no decoder fit.
Approximate closure, commutation and inverse consistency require diagnostics.
"""
import time
import numpy as np
from scipy import linalg, sparse
from scipy.sparse.csgraph import dijkstra, connected_components
from sklearn.neighbors import NearestNeighbors
from fields_first_frame import fit_projected_W, evaluate_W
from fields_first_periods import prepare_period_complex, fit_periods, reduce_period_basis
from fields_first_tangent_atlas import (fit_atlas, evaluate_projector,
    evaluate_tangent_frame, retract_to_atlas, _poly)
from fields_first_paths import edge_integer_labels
from fields_first_implicit_geometry import fit_implicit_geometry
from fields_first_span import select_span
from tangent_overlap import windows, fold

def auto_dimension(x,neighbors=24,max_dimension=4,gap_threshold=4.):
    count=min(neighbors+1,len(x));ids=NearestNeighbors(n_neighbors=count).fit(x).kneighbors(x,return_distance=False)[:,1:]
    size=min(max_dimension+1,x.shape[1],count-2)
    vals=np.array([linalg.svdvals(x[js]-x[js].mean(0))[:size] for js in ids])
    ratios=np.median(vals[:,:-1]/np.maximum(vals[:,1:],1e-14),axis=0)
    eligible=np.flatnonzero(ratios>=gap_threshold)
    diag=dict(rule='first median local singular gap at least 4',median_singular_gaps=ratios,
        automatic_dimension_is_heuristic=True,neighbours=count-1)
    return (int(eligible[0]+1) if len(eligible) else None),diag


def overlap_maps(q,channels,basis=None):
    a=[];s=[];res=[]
    for i in range(len(q)-1):
        nextq=q[i+1] if basis is None else basis@q[i+1]
        previousq=q[i] if basis is None else basis@q[i]
        j=nextq[:-channels];b=previousq[channels:]
        ss=linalg.svdvals(j);a0=linalg.lstsq(j,b)[0]
        if j.shape[0]<q.shape[2] or np.linalg.matrix_rank(j)<q.shape[2]:
            raise ValueError('overlap_prefix_is_not_injective_on_estimated_tangent')
        a.append(a0);s.append(ss[-1]);res.append(linalg.norm(j@a0-b)/max(linalg.norm(b),1e-15))
    return np.asarray(a),dict(minimum_prefix_singular=float(min(s)),overlap_relative_residual_quantiles=np.quantile(res,[.1,.5,.9]))


def field(model,query):
    x=np.asarray(query);single=x.ndim==1
    if single:x=x[None]
    raw=evaluate_W(model['frame'],x)
    try:
        geom=evaluate_tangent_frame(model['atlas'],x)
    except ValueError as exc:
        if model.get('strict_flows',False) and 'compact tangent-atlas support' in str(exc):
            raise ValueError('flow_outside_geometric_support') from exc
        raise
    q=geom['Q']
    w=np.einsum('npa,nar->npr',q,np.einsum('npa,npr->nar',q,raw))
    if model.get('strict_flows',False):
        singular=np.linalg.svd(w,compute_uv=False)
        nearest=float(np.max(geom['nearest_distance']));gap=float(np.min(geom['gap']))
        stats=model.setdefault('flow_domain_diagnostics',dict(maximum_nearest_distance=0.,minimum_atlas_gap=1.,minimum_frame_singular=float('inf')))
        stats['maximum_nearest_distance']=max(stats['maximum_nearest_distance'],nearest)
        stats['minimum_atlas_gap']=min(stats['minimum_atlas_gap'],gap)
        stats['minimum_frame_singular']=min(stats['minimum_frame_singular'],float(singular[:,-1].min()))
        if gap<.1:raise ValueError('flow_tangent_projector_gap_lost')
        if nearest>3*model['atlas']['bandwidth']:raise ValueError('flow_outside_geometric_support')
        if singular[:,-1].min()<.01*model['diagnostics']['frame_min_singular'] or np.min(singular[:,-1]/singular[:,0])<1e-3:
            raise ValueError('flow_invariant_frame_degenerate')
    return w[0] if single else w


def beta(model,query):
    w=field(model,query)
    return np.linalg.solve(np.swapaxes(w,-1,-2)@w,np.swapaxes(w,-1,-2))


def beta_action(model,query,direction):
    """Return beta_x(v) without materializing the ambient projector.

    If W=Q B and Q has orthonormal columns, beta_x(v)=B^{-1}Q^T v.
    This is the same dual form; no new approximation or phase input is used.
    """
    x=np.atleast_2d(np.asarray(query,dtype=float))
    v=np.atleast_2d(np.asarray(direction,dtype=float))
    if v.shape!=x.shape:raise ValueError('one ambient direction per query required')
    q=evaluate_tangent_frame(model['atlas'],x)['Q']
    raw=evaluate_W(model['frame'],x)
    b=np.einsum('npa,npr->nar',q,raw)
    rhs=np.einsum('npa,np->na',q,v)
    return np.linalg.solve(b,rhs[...,None])[...,0]


def edge_curve(model,edges,s):
    atlas=model['atlas'];x=model['X'];delta=x[edges[:,1]]-x[edges[:,0]]
    positions=[];velocities=[]
    for endpoint in (0,1):
        ids=edges[:,endpoint];q=atlas['Q'][ids];radii=atlas['radius'][ids]
        v=np.einsum('np,npa->na',delta,q)/radii[:,None]
        u=(s-endpoint)*v;poly,der,_=_poly(u,atlas['powers'])
        lin=np.einsum('npa,na->np',q,u)
        pos=x[ids]+radii[:,None]*(lin+np.einsum('nt,ntp->np',poly,atlas['coefficients'][ids]))
        dpoly=np.einsum('nta,na->nt',der,v)
        vel=radii[:,None]*(np.einsum('npa,na->np',q,v)+np.einsum('nt,ntp->np',dpoly,atlas['coefficients'][ids]))
        positions.append(pos);velocities.append(vel)
    return ((1-s)*positions[0]+s*positions[1],
        -positions[0]+(1-s)*velocities[0]+positions[1]+s*velocities[1])


def fine_integrals(model,cx,neighbors=24,order=3,curved=True):
    x=model['X'];n=len(x);r=model['r']
    curved=bool(curved and model['atlas'].get('kind')!='implicit_quadratic')
    dd,jj=NearestNeighbors(n_neighbors=min(neighbors+1,n)).fit(x).kneighbors(x)
    pairs=sorted({tuple(sorted((i,int(j)))) for i,js in enumerate(jj[:,1:]) for j in js if i!=j})
    edges=np.asarray(pairs)
    integer_labels,valid,path_class_diagnostic=edge_integer_labels(x,edges,cx)
    path_class_diagnostic={k:v for k,v in path_class_diagnostic.items() if not isinstance(v,np.ndarray)}
    edges=edges[valid];integer_labels=integer_labels[valid]
    delta=x[edges[:,1]]-x[edges[:,0]]
    length=np.linalg.norm(delta,axis=1)
    graph=sparse.csr_matrix((np.r_[length,length],(np.r_[edges[:,0],edges[:,1]],np.r_[edges[:,1],edges[:,0]])),shape=(n,n))
    if connected_components(graph,directed=False,return_labels=False)!=1:
        raise ValueError('fine integration graph disconnected')
    landmark=cx['landmark_indices'];dist,pred=dijkstra(graph,directed=False,indices=landmark,return_predecessors=True)
    lookup={tuple(e):i for i,e in enumerate(edges)}
    coarse=cx['edges'];u,v=coarse.T
    destinations=[set() for _ in landmark]
    for a,b in coarse:
        destinations[a].add(int(b));destinations[b].add(int(a))
    source_vertices=[];used_edges=set()
    integer_potentials=np.zeros((len(landmark),n,r),dtype=np.int64)
    for source,targets in enumerate(destinations):
        needed=np.zeros(n,dtype=bool)
        for target in targets:
            vertex=int(landmark[target])
            while vertex!=landmark[source] and not needed[vertex]:
                needed[vertex]=True;previous=int(pred[source,vertex])
                used_edges.add(lookup[(min(previous,vertex),max(previous,vertex))])
                vertex=previous
        vertices=np.argsort(dist[source])[1:]
        vertices=vertices[needed[vertices]];source_vertices.append(vertices)
        for vertex in vertices:
            previous=int(pred[source,vertex]);pair=(min(previous,vertex),max(previous,vertex))
            sign=1 if previous<vertex else -1
            integer_potentials[source,vertex]=integer_potentials[source,previous]+sign*integer_labels[lookup[pair]]
    integer_defect=integer_potentials[u,landmark[v]]-cx['integer_cocycles'].T
    if np.any(integer_defect):
        raise ValueError('fine integration paths do not match the declared finite period classes')
    # Only these edges contribute to the requested coarse paths. The shortest
    # paths, their orientations and their summation order stay unchanged.
    integration_indices=np.array(sorted(used_edges),dtype=int)
    integration_edges=edges[integration_indices];delta=delta[integration_indices]
    curve_valid=np.zeros(len(integration_edges),dtype=bool)
    if curved:
        from math import comb
        degree=int(np.max(np.sum(model['atlas']['powers'],axis=1)))+1
        ss=np.linspace(0,1,degree+1)
        B=np.array([[comb(degree,k)*s0**k*(1-s0)**(degree-k) for k in range(degree+1)] for s0 in ss])
        values=np.stack([edge_curve(model,integration_edges,s0)[0] for s0 in ss],axis=1)
        controls=np.einsum('ab,nbp->nap',np.linalg.inv(B),values)
        # One convex cover ball containing the entire control polygon certifies
        # that the polynomial path and the original chord have the same class.
        for start in range(0,len(integration_edges),128):
            cp=controls[start:start+128]
            d2=np.sum((cp[:,:,None,:]-cx['landmarks'][None,None,:,:])**2,axis=3)
            curve_valid[start:start+len(cp)]=np.any(np.max(d2,axis=1)<cx['cover_radius']**2,axis=1)
    if order==3:
        s=np.array([.5-np.sqrt(15)/10,.5,.5+np.sqrt(15)/10]);weights=np.array([5/18,4/9,5/18])
    else:s=np.array([.5]);weights=np.array([1.])
    integral=np.zeros((len(edges),r))
    for pos,weight in zip(s,weights):
        for start in range(0,len(integration_edges),256):
            sl=slice(start,start+256);pts=x[integration_edges[sl,0]]+pos*delta[sl]
            direction=delta[sl].copy()
            if curved:
                curve,velocity=edge_curve(model,integration_edges[sl],pos);good=curve_valid[sl]
                pts[good]=curve[good];direction[good]=velocity[good]
            integral[integration_indices[sl]]+=weight*beta_action(model,pts,direction)
    potentials=np.zeros((len(landmark),n,r))
    for source in range(len(landmark)):
        for vertex in source_vertices[source]:
            previous=int(pred[source,vertex]);pair=(min(previous,vertex),max(previous,vertex))
            sign=1 if previous<vertex else -1
            potentials[source,vertex]=potentials[source,previous]+sign*integral[lookup[pair]]
    raw=potentials[u,landmark[v]]
    reverse_disagreement=potentials[u,landmark[v]]+potentials[v,landmark[u]]
    stretch=dist[u,landmark[v]]/np.maximum(np.linalg.norm(x[landmark[v]]-x[landmark[u]],axis=1),1e-15)
    return raw,dict(fine_edges=len(edges),integrated_fine_edges=len(integration_edges),
        unused_fine_edges_skipped=len(edges)-len(integration_edges),quadrature_order=order,
        curve_integration=bool(curved),curved_edges_with_convex_cover_homotopy_certificate=int(np.sum(curve_valid)),
        chord_fallback_edges=int(np.sum(~curve_valid)),
        curve_certificate_counts_apply_to_integrated_edges=True,
        maximum_fine_edge_length=float(length.max()),fine_edge_length_quantiles=np.quantile(length,[.5,.9,1.]),
        path_stretch_quantiles=np.quantile(stretch,[.5,.9,1.]),
        invalid_fine_chords_removed=int(np.sum(~valid)),path_cover_and_class_check=path_class_diagnostic,
        coarse_path_integer_class_defect_max=int(np.max(np.abs(integer_defect))),
        reverse_path_integral_disagreement_relative=float(linalg.norm(reverse_disagreement)/max(linalg.norm(raw),1e-15)),
        path_homotopy_equivalence_to_true_manifold_certified=False)


def encode(model,query,derivative=False):
    x=np.asarray(query);single=x.ndim==1
    if single:x=x[None]
    cx=model['complex'];pm=model['period_model'];centers=cx['landmarks']
    edges=cx['edges'];r=model['r'];p=x.shape[1];n=len(centers)
    C=np.zeros((n,n,r));C[edges[:,0],edges[:,1]]=cx['integer_cocycles'].T
    C[edges[:,1],edges[:,0]]=-cx['integer_cocycles'].T
    q=pm['normalized_potential'].T;slopes=model['center_normalized_beta']
    radius2=cx['cover_radius']**2
    result=np.full((len(x),r),np.nan);jac=np.full((len(x),r,p),np.nan);covered=np.zeros(len(x),dtype=bool)
    for start in range(0,len(x),128):
        xx=x[start:start+128];diff=xx[:,None]-centers[None]
        d2=np.sum(diff*diff,axis=2);active=d2<radius2;cover=active.any(axis=1)
        root=np.argmax(active,axis=1)
        logs=np.full_like(d2,-np.inf);logs[active]=radius2/(d2[active]-radius2)
        cardinal=model.get('cardinal_phase_weights',False)
        at_center=d2==0
        if cardinal:
            logs[active]-=2*np.log(np.maximum(d2[active],np.finfo(float).tiny))
            exact=at_center.any(axis=1)
            logs[exact]=np.where(at_center[exact],0.,-np.inf)
        logs-=np.where(cover,np.max(logs,axis=1),0.)[:,None]
        w=np.exp(logs);w/=np.maximum(w.sum(axis=1,keepdims=True),np.finfo(float).tiny)
        local=q[None]+C[root]+np.einsum('lrp,nlp->nlr',slopes,diff)
        value=2*np.pi*np.einsum('nl,nlr->nr',w,local)-model.get('origin',0.)
        value[~cover]=np.nan;result[start:start+len(xx)]=np.angle(np.exp(1j*value));covered[start:start+len(xx)]=cover
        if derivative:
            factor=np.zeros_like(d2);factor[active]=-2*radius2/(d2[active]-radius2)**2
            if cardinal:
                nonzero=active&~at_center
                factor[nonzero]-=4/d2[nonzero]
                factor[exact]=0.
            loggrad=factor[:,:,None]*diff;mean=np.einsum('nl,nlp->np',w,loggrad)
            dw=w[:,:,None]*(loggrad-mean[:,None])
            jj=2*np.pi*(np.einsum('nlp,nlr->nrp',dw,local)+np.einsum('nl,lrp->nrp',w,slopes))
            jj[~cover]=np.nan;jac[start:start+len(xx)]=jj
    return dict(phases=result[0] if single else result,covered=covered[0] if single else covered,
                jacobian=(jac[0] if single else jac) if derivative else None)


def phase_field(model,x):
    return np.einsum('...pa,ab->...pb',field(model,x),model['period_model']['Pi'])/(2*np.pi)


def commutator_diagnostics(model,query=None):
    if model['r']<2:return dict(pairs=[],source_probe_count=0)
    if query is None:
        query=model['X'][np.linspace(0,len(model['X'])-1,min(32,len(model['X'])),dtype=int)]
    x=np.asarray(query);raw,jraw=evaluate_W(model['frame'],x,derivative=True)
    mat=model['period_model']['Pi']/(2*np.pi)
    rawE=np.einsum('npa,ab->npb',raw,mat)
    rawJ=np.einsum('npqa,ab->npqb',jraw,mat)
    geom=evaluate_projector(model['atlas'],x);P=geom['P'];E=np.einsum('npq,nqr->npr',P,rawE)
    deriv=[]
    for a in range(model['r']):
        dP=evaluate_projector(model['atlas'],x,directions=E[:,:,a])['dP']
        deriv.append(np.einsum('npq,nqr->npr',dP,rawE)+np.einsum('npq,nqkr,nk->npr',P,rawJ,E[:,:,a]))
    pairs=[];state_scale=float(np.sqrt(np.mean(np.sum(model['X']**2,axis=1))))
    for a in range(model['r']):
        for b in range(a+1,model['r']):
            bracket=deriv[a][:,:,b]-deriv[b][:,:,a]
            normal=bracket-np.einsum('npq,nq->np',P,bracket)
            scale=np.sqrt(np.mean(np.sum(E[:,:,a]**2,axis=1))*np.mean(np.sum(E[:,:,b]**2,axis=1)))
            pairs.append(dict(axes=[a,b],bracket_rms=float(np.sqrt(np.mean(np.sum(bracket**2,axis=1)))),
                normal_bracket_rms=float(np.sqrt(np.mean(np.sum(normal**2,axis=1)))),
                dimensionless_relative_bracket=float(np.sqrt(np.mean(np.sum(bracket**2,axis=1)))*state_scale/max(scale,1e-15))))
    return dict(source_probe_count=len(x),pairs=pairs)


def retract_flow_states(model, x):
    states, report = retract_to_atlas(model['atlas'], x)
    stats = model.setdefault('flow_retraction_diagnostics', dict(calls=0,
        maximum_iterations=0, maximum_normal_residual_relative=0.,
        maximum_displacement_relative=0.))
    stats['calls'] += 1
    stats['maximum_iterations'] = max(stats['maximum_iterations'], report['iterations'])
    for key in ('maximum_normal_residual_relative', 'maximum_displacement_relative'):
        stats[key] = max(stats[key], report[key])
    return states


def enable_geometry_constrained_flows(model):
    """Refine only the finite inverse, after the field/period fit.

    The constraint uses window geometry, never phase targets or a decoder.
    Projecting each RK4 step changes the finite flow model: it approximates
    motion constrained to the estimated surface, not the old ambient ODE.
    """
    if model['atlas'].get('kind')=='implicit_quadratic':
        raise ValueError('normal_graph_constraint_requires_local_graph_geometry')
    if 'flow_domain_diagnostics' in model:
        model['ambient_flow_domain_diagnostics'] = model.pop('flow_domain_diagnostics')
    reference = retract_flow_states(model, model['X'][[0]])
    shift = encode(model, reference)['phases'][0]
    if not np.isfinite(shift).all():
        raise ValueError('retracted_reference_outside_phase_cover')
    model['origin'] = model.get('origin', 0.) + shift
    model['phases'] = np.angle(np.exp(1j*(model['phases']-shift)))
    model['inverse_reference'] = reference[0]
    model['flow_geometry'] = 'normal_graph_constraint'
    model['flow_step'] = .12
    model['diagnostics']['commutator_scope'] = 'ambient transported field; constrained inverse checked by flow order'
    model['diagnostics']['flow_reference_displacement_relative'] = float(
        linalg.norm(reference-model['X'][[0]]) /
        max(np.sqrt(np.mean(np.sum(model['X']**2, axis=1))), 1e-15))


def flow(model,x,angle,axis,step=None):
    if step is None:step=model.get('flow_step', .12)
    if not np.isfinite(step) or step <= 0:raise ValueError('positive finite flow step required')
    states=np.atleast_2d(x).copy();angles=np.broadcast_to(angle,(len(states),))
    if not np.isfinite(angles).all():raise ValueError('finite flow angles required')
    constrained=model.get('flow_geometry') == 'normal_graph_constraint'
    if constrained:states=retract_flow_states(model,states)
    count=max(1,int(np.ceil(np.max(np.abs(angles))/step)));dt=angles/count
    for _ in range(count):
        k1=phase_field(model,states)[:,:,axis]
        k2=phase_field(model,states+dt[:,None]*k1/2)[:,:,axis]
        k3=phase_field(model,states+dt[:,None]*k2/2)[:,:,axis]
        k4=phase_field(model,states+dt[:,None]*k3)[:,:,axis]
        states+=dt[:,None]*(k1+2*k2+2*k3+k4)/6
        if not np.isfinite(states).all():raise FloatingPointError('flow left finite model domain')
        if constrained:states=retract_flow_states(model,states)
    return states


def inverse(model,phases,step=None,reverse=False):
    ph=np.atleast_2d(phases);states=np.tile(model.get('inverse_reference',model['X'][0]),(len(ph),1))
    for j in list(range(model['r']))[::(-1 if reverse else 1)]:
        states=flow(model,states,ph[:,j],j,step)
    return states


def constrained_flow_duality(model, points, probe_count=16, epsilon=1e-4):
    """Test phase speed of the actual projected step, including retraction.

    D Phi times the old ambient field alone misses the geometric correction.
    The central difference uses a fixed model and no observation/phase truth.
    """
    x=np.atleast_2d(points)
    take=np.unique(np.linspace(0,len(x)-1,min(probe_count,len(x)),dtype=int))
    x=retract_flow_states(model,x[take]);columns=[]
    for j in range(model['r']):
        plus=encode(model,flow(model,x,epsilon,j))['phases']
        minus=encode(model,flow(model,x,-epsilon,j))['phases']
        columns.append(np.angle(np.exp(1j*(plus-minus)))/(2*epsilon))
    derivative=np.stack(columns,axis=2)
    return float(linalg.norm(derivative-np.eye(model['r'])[None])/
        np.sqrt(len(take)*model['r'])),int(len(take))


def fit(y,lag=2,dimension=None,field_landmarks=64,period_landmarks=192,neighbors=24,dimension_strategy='persistent_count',span_rtol='auto',relation_tolerance='auto'):
    began=time.time();y=np.asarray(y,dtype=float)
    if y.ndim==1:y=y[:,None]
    if y.ndim!=2 or not np.isfinite(y).all():raise ValueError('finite synchronized Y[times,channels] required')
    if not 2<=lag<len(y):raise ValueError('lag must satisfy 2 <= lag < record length')
    mean=y.mean(0);rawscale=y.std(0)
    if rawscale.max()<1e-12:raise ValueError('constant readings contain no observable dynamics')
    scale=np.where(rawscale>rawscale.max()*1e-12,rawscale,1.)
    normal=np.sqrt(lag*y.shape[1]);fullx=windows((y-mean)/scale,lag)/normal
    basis,x,span_diagnostic=select_span(fullx,span_rtol=span_rtol)
    guessed,dimdiag=auto_dimension(x)
    atlas=None
    if dimension is None and dimension_strategy=='persistent_count':
        from fields_first_periods import infer_torus_dimension
        r,dimdiag=infer_torus_dimension(x,nland=period_landmarks)
        period_landmarks=dimdiag['landmarks']
    elif dimension is None:
        candidates=[];r=None
        for candidate in range(1,min(3,x.shape[1]-1)+1):
            attempt=fit_atlas(x,candidate,k=neighbors)
            res=attempt['diagnostics']['graph_relative_residual_quantiles']
            record=dict(dimension=candidate,median_cubic_graph_relative_residual=float(res[1]),p90_cubic_graph_relative_residual=float(res[2]))
            candidates.append(record)
            if res[1]<.01:
                r=candidate;atlas=attempt;break
        dimdiag.update(rule='smallest dimension with median cubic graph residual <0.01; tail residual is reported as a geometry diagnostic',candidates=candidates)
    else:r=int(dimension)
    diagnostics=dict(dimension_selected=r,dimension_hypothesis_supplied=dimension is not None,
        dimension_diagnostic=dimdiag,all_fit_steps_use_readings_only=True,
        independent_initial_phases_computed=False,stages=[],
        full_window_dimension=fullx.shape[1],retained_numerical_linear_span=basis.shape[1],
        linear_span_relative_residual=span_diagnostic['projection_relative'],
        span_selection=span_diagnostic,
        span_rule=span_diagnostic['rule'])
    model=dict(status='fitting',X=x,r=r,mean=mean,scale=scale,normal=normal,lag=lag,channels=y.shape[1],basis=basis,diagnostics=diagnostics,
        cardinal_phase_weights=True,
        source_full_rms=float(np.sqrt(np.mean(np.sum(fullx**2,axis=1)))),
        source_projection_residual_squared=np.sum((fullx-x@basis.T)**2,axis=1))
    if r is None:model['status']='unresolved_local_dimension';return model
    if not 1<=r<x.shape[1]:
        model['status']='unresolved_tangent_dimension';return model
    print('ATLAS',x.shape,'r',r,flush=True)
    implicit,implicit_diagnostic=fit_implicit_geometry(x,r,relation_tolerance=relation_tolerance)
    if implicit is not None:atlas=implicit
    elif atlas is None:atlas=fit_atlas(x,r,k=neighbors)
    diagnostics['implicit_geometry']=implicit_diagnostic
    diagnostics['geometry_method']=atlas.get('kind','local_polynomial_graphs')
    diagnostics['geometry_neighbors']=int(neighbors)
    geom=evaluate_projector(atlas,x);q=geom['Q']
    a,overlap=overlap_maps(q,y.shape[1],basis);diagnostics.update(overlap=overlap,atlas_min_gap=float(np.min(geom['gap'])))
    diagnostics['stages'].append('tangent_spaces_and_overlap_transfer')
    if diagnostics['atlas_min_gap']<.1:
        model['status']='unresolved_tangent_atlas_gap';return model
    if overlap['minimum_prefix_singular']<1e-3:
        model['status']='unresolved_overlap_prefix_conditioning';return model
    print('FRAME',flush=True)
    frame=fit_projected_W(x,q,a,landmarks=field_landmarks,ridge=1e-6,spatial_weight=1e-3)
    model.update(atlas=atlas,frame=frame)
    w=field(model,x);c=np.einsum('npa,npb->nab',q,w);sv=np.linalg.svd(w,compute_uv=False)
    residual=c[1:]-np.einsum('nab,nbc->nac',a,c[:-1])
    diagnostics.update(frame_anchor_error=float(linalg.norm(w[0]-q[0])),
        frame_transfer_relative=float(linalg.norm(residual)/linalg.norm(c[1:])),
        frame_min_singular=float(sv[:,-1].min()),frame_min_relative_singular=float(np.min(sv[:,-1]/sv[:,0])))
    diagnostics['stages'].append('global_invariant_frame_fit')
    if sv[:,-1].min()<1e-5 or np.min(sv[:,-1]/sv[:,0])<1e-3:
        model['status']='unresolved_degenerate_frame';return model
    print('PERIOD_COMPLEX',flush=True)
    cx=prepare_period_complex(x,r,nland=period_landmarks,
        filtration_cap=dimdiag.get('filtration_cap'));model['complex']=cx
    if not cx['status'].startswith('prepared_'):
        model['status']=cx['status'];return model
    diagnostics['stages'].append('integer_full_period_basis_after_frame')
    print('BETA_INTEGRALS',flush=True)
    raw,pathdiag=fine_integrals(model,cx,neighbors=24,order=3)
    pm=fit_periods(raw,cx);model['period_model']=pm;diagnostics['integration']=pathdiag
    if not pm['status'].startswith('fitted_'):
        model['status']=pm['status'];return model
    frame_gram=np.einsum('npa,npb->ab',w,w)/len(w)
    pm,cx=reduce_period_basis(pm,cx,frame_gram)
    model.update(period_model=pm,complex=cx)
    diagnostics['stages'].append('dual_form_closed_projection_and_periods')
    diagnostics['periods']=pm['diagnostics']
    model['center_normalized_beta']=np.einsum('ab,lbp->lap',np.linalg.inv(pm['Pi']),beta(model,cx['landmarks']))
    model['origin']=encode(model,x[[0]])['phases'][0]
    encoded=encode(model,x,derivative=True);ph=encoded['phases'];E=phase_field(model,x)
    product=np.einsum('nap,npb->nab',encoded['jacobian'],E)
    diagnostics['encoder_field_duality_relative']=float(linalg.norm(product-np.eye(r)[None])/np.sqrt(len(x)*r))
    diagnostics['train_encoder_coverage']=float(np.mean(encoded['covered']))
    diagnostics['phase_step']=np.angle(np.mean(np.exp(1j*np.diff(ph,axis=0)),axis=0))
    diagnostics['train_phase_step_rmse']=float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(ph,axis=0)-diagnostics['phase_step'])))**2)))
    diagnostics['stages'].append('integrate_field_derived_angles')
    diagnostics['commutators']=commutator_diagnostics(model)
    model['phases']=ph;model['status']='conditional_fields_first_model';diagnostics['elapsed_fit_seconds']=time.time()-began
    return model

