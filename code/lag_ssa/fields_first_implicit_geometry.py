"""Validated quadratic window relations, with a local-geometry fallback.

No phase coordinates enter either relation fitting or tangent evaluation.
"""
import numpy as np
from scipy import linalg
from scipy.spatial import cKDTree


def _features(x,rows,cols):
    return np.column_stack((np.ones(len(x)),x,x[:,rows]*x[:,cols]))


def _eigensystem(model,x):
    gradient=2*np.einsum('spq,nq->nsp',model['C'],x)+model['A'][None]
    gram=np.einsum('nsp,nsq->npq',gradient,gradient)
    value,vector=np.linalg.eigh(gram)
    return gradient,value,vector


def fit_implicit_geometry(X,r,relation_tolerance=1e-10):
    """Return (model, diagnostic), or (None, diagnostic) when unsupported.

    Every fifth source row is held out from one relation SVD. The default
    retains the exact-relation checks. 'auto' tries that default first, then
    seven near-relation cutoffs through .0064. Near-relations must validate
    within three times their cutoff and separate r tangent directions on all
    source rows. Their cutoff is selected by worst held-out normalized gap.
    These finite numerical checks do not certify the true state manifold.
    """
    x=np.asarray(X,dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('finite source states by ambient coordinates required')
    n,p=x.shape
    if isinstance(r,(bool,np.bool_)) or int(r)!=r or not 1<=r<p:
        raise ValueError('require integer 1 <= r < ambient dimension')
    automatic=isinstance(relation_tolerance,str) and relation_tolerance=='auto'
    if not automatic:
        if isinstance(relation_tolerance,(bool,np.bool_)):
            raise ValueError('positive finite relation tolerance or auto required')
        try:tolerance=float(relation_tolerance)
        except (TypeError,ValueError) as exc:
            raise ValueError('positive finite relation tolerance or auto required') from exc
        if not np.isfinite(tolerance) or tolerance<=0:
            raise ValueError('positive finite relation tolerance or auto required')
    cutoffs=[1e-10]+[1e-4*2**j for j in range(7)] if automatic else [tolerance]
    r=int(r);rows,cols=np.triu_indices(p)
    count=1+p+len(rows)
    diagnostic=dict(method='validated exact quadratic relations',
        feature_count=count,states=n,ambient_dimension=p,dimension=r,
        requested_relation_tolerance=relation_tolerance,
        relation_tolerance=1e-10 if automatic else tolerance,
        gradient_relative_rank_tolerance=1e-8,no_phase_coordinates_used=True,
        true_manifold_geometry_certified=False,finite_training_checks_only=True,
        candidates=[])

    def refuse(reason):
        diagnostic.update(status='unsupported_implicit_geometry',reason=reason)
        return None,diagnostic

    if count>min(n//2,1024):return refuse('quadratic_feature_budget_exceeded')
    heldout=np.arange(n)%5==0;features=_features(x,rows,cols)
    _,singular,right=linalg.svd(features[~heldout],full_matrices=False,
                               lapack_driver='gesdd')
    diagnostic.update(training_count=int(np.sum(~heldout)),
        validation_count=int(np.sum(heldout)))
    row_norm=np.linalg.norm(features,axis=1)

    def candidate(cutoff):
        approximate=cutoff>1e-10;null=right[singular<=singular[0]*cutoff]
        report=dict(relation_tolerance=float(cutoff),approximate_relations=approximate,
            relation_count=len(null),status='rejected')
        if not len(null):return None,report|dict(reason='no_exact_quadratic_relations')
        residual=np.max(abs(features@null.T),axis=1)/row_norm
        limit=cutoff*(3 if approximate else 1)
        report.update(training_relative_residual=float(np.max(residual[~heldout])),
            validation_relative_residual=float(np.max(residual[heldout])),
            relation_residual_limit=float(limit))
        if np.max(residual)>limit:
            return None,report|dict(reason='quadratic_relations_fail_validation')
        C=np.zeros((len(null),p,p));quadratic=null[:,1+p:]
        C[:,rows,cols]=quadratic/2;C[:,cols,rows]+=quadratic/2
        model=dict(kind='implicit_quadratic',r=r,C=C,
            A=null[:,1:1+p].copy(),b=null[:,0].copy())
        normal_min=1.;gap_min=1.;validation_separation=float('inf')
        for start in range(0,n,64):
            _,value,_=_eigensystem(model,x[start:start+64])
            largest=np.maximum(value[:,-1],np.finfo(float).tiny)
            normal=value[:,r]/largest
            gap=1-np.maximum(value[:,r-1],0)/np.maximum(value[:,r],np.finfo(float).tiny)
            normal_min=min(normal_min,float(np.min(normal)))
            gap_min=min(gap_min,float(np.min(gap)))
            if not approximate:
                rank=np.sum(value>largest[:,None]*1e-8,axis=1)
                if np.any(rank!=p-r):
                    return None,report|dict(reason='quadratic_gradient_kernel_has_wrong_dimension',
                        gradient_rank_min=int(np.min(rank)),gradient_rank_max=int(np.max(rank)))
            selected=heldout[start:start+64]
            if np.any(selected):
                separation=(value[:,r]-value[:,r-1])/largest
                validation_separation=min(validation_separation,float(np.min(separation[selected])))
        report.update(minimum_normal_relative_eigenvalue=normal_min,
            minimum_source_gap=gap_min,
            worst_validation_normalized_separation=validation_separation)
        if approximate and (normal_min<1e-8 or gap_min<.1):
            return None,report|dict(reason='quadratic_gradient_tangent_gap_not_separated')
        report['status']='validated'
        return model,report

    accepted=[]
    for cutoff in cutoffs:
        model,report=candidate(cutoff);diagnostic['candidates'].append(report)
        if model is not None:
            accepted.append((model,report))
            if cutoff<=1e-10:break
    if not accepted:
        if len(cutoffs)==1:diagnostic.update(diagnostic['candidates'][0])
        return refuse(diagnostic['candidates'][0]['reason'] if len(cutoffs)==1
            else 'no_validated_quadratic_geometry')
    model,report=max(accepted,key=lambda item:(
        item[1]['worst_validation_normalized_separation'],-item[1]['relation_tolerance']))
    diagnostic.update(report)
    tree=cKDTree(x,copy_data=True)
    distance=tree.query(x,k=min(9,n))[0][:,-1]
    positive=distance[distance>0]
    if not len(positive):return refuse('no_positive_source_bandwidth')
    bandwidth=float(np.median(positive))
    model.update(X=x.copy(),tree=tree,bandwidth=bandwidth,kernel_radius=4*bandwidth)
    diagnostic.update(status='fitted_implicit_geometry',
        method='validated approximate quadratic relations' if report['approximate_relations']
            else 'validated exact quadratic relations',source_bandwidth=bandwidth)
    model['diagnostics']=dict(diagnostic)
    return model,diagnostic


def evaluate_implicit_geometry(model,query,directions=None,frame_only=False):
    """Evaluate bottom-r gradient-Gram eigenspaces and optional analytic dP.

    Query support is the union of open source balls with radius four times
    the frozen source eighth-neighbour bandwidth. Returned arrays keep a
    leading query axis even for one query vector.
    """
    x=np.asarray(query,dtype=float)
    if x.ndim==1:x=x[None]
    p=model['X'].shape[1];r=model['r']
    if x.ndim!=2 or x.shape[1]!=p or not np.isfinite(x).all():
        raise ValueError('finite queries of the source ambient dimension required')
    derivative=directions is not None
    if derivative:
        directions=np.asarray(directions,dtype=float)
        if directions.ndim==1:directions=np.broadcast_to(directions,x.shape)
        if directions.shape!=x.shape or not np.isfinite(directions).all():
            raise ValueError('finite directions must match query coordinates')
    nearest=model['tree'].query(x,k=1)[0]
    outside=nearest>=model['kernel_radius']
    if np.any(outside):
        raise ValueError(f"query {int(np.flatnonzero(outside)[0])} outside compact tangent-atlas support")
    Q=np.empty((len(x),p,r));gap=np.empty(len(x))
    P=None if frame_only else np.empty((len(x),p,p))
    dP=np.empty((len(x),p,p)) if derivative else None
    for start in range(0,len(x),64):
        stop=min(start+64,len(x));gradient,value,vector=_eigensystem(model,x[start:stop])
        tangent=vector[:,:,:r];normal=vector[:,:,r:]
        if np.any(value[:,r]<=0):raise ValueError('implicit_geometry_normal_rank_lost')
        Q[start:stop]=tangent
        gap[start:stop]=1-np.maximum(value[:,r-1],0)/value[:,r]
        if P is not None:P[start:stop]=np.einsum('npa,nqa->npq',tangent,tangent)
        if derivative:
            gdot=2*np.einsum('spq,nq->nsp',model['C'],directions[start:stop])
            half=np.einsum('nsp,nsq->npq',gradient,gdot)
            dgram=half+half.transpose(0,2,1)
            cross=np.einsum('npi,npq,nqa->nia',normal,dgram,tangent)
            denominator=value[:,None,:r]-value[:,r:,None]
            if np.any(denominator==0):raise ValueError('implicit_geometry_tangent_gap_lost')
            half=np.einsum('npi,nia,nqa->npq',normal,cross/denominator,tangent)
            dP[start:stop]=half+half.transpose(0,2,1)
    result=dict(Q=Q,gap=gap,nearest_distance=nearest)
    if P is not None:result['P']=P
    if derivative:result['dP']=dP
    return result
