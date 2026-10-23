"""Anchored invariant-frame collocation from tangent overlap maps.

Evaluate the fitted ambient field with evaluate_W, then apply the same frozen
tangent projector used for fitting. No phase coordinates enter the frame fit.
"""
import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors


def fps(x,count):
    ids=[0];d=np.sum((x-x[0])**2,axis=1)
    while len(ids)<count:
        j=int(np.argmax(d));ids.append(j);d=np.minimum(d,np.sum((x-x[j])**2,axis=1))
    return np.asarray(ids)

def dictionary(x,q,centers,centerq,h,affine=True):
    dx=x[:,None]-centers[None]
    w=np.exp(-np.sum(dx*dx,axis=2)/(2*h*h));w/=w.sum(axis=1,keepdims=True)
    feature=np.ones((len(x),len(centers),1))
    if affine:feature=np.concatenate((feature,np.einsum('nlp,lpa->nla',dx,centerq)/h),axis=2)
    cross=np.einsum('npa,lpb->nlab',q,centerq)
    return np.einsum('nl,nlab,nlc->nalbc',w,cross,feature).reshape(len(x),q.shape[-1],-1)

def global_affine_dictionary(x,q):
    """Projected ambient affine fields, with columns ordered by output/feature."""
    feature=np.column_stack((np.ones(len(x)),x))
    return np.einsum('npa,nb->napb',q,feature).reshape(len(x),q.shape[-1],-1)


def fit_projected_W(trainX,Q,A,landmarks=64,ridge=1e-6,spatial_weight=1e-3,
                    include_affine='auto'):
    """Fit one anchored overlap problem; optionally include global affine fields.

    Local patches remain available when the global block is included. Evaluate
    the raw field with evaluate_W, then apply the frozen tangent projector.
    """
    x=np.asarray(trainX);q=np.asarray(Q);a=np.asarray(A);n,p=x.shape;r=q.shape[-1]
    ids=fps(x,min(landmarks,n));centers=x[ids];cq=q[ids]
    h=float(np.median(NearestNeighbors(n_neighbors=min(4,len(ids))).fit(centers).kneighbors(x)[0][:,-1]))
    Z=dictionary(x,q,centers,cq,h,True);global_columns=0
    if include_affine=='auto':
        # Bound the extra work by the size of the existing local dictionary.
        include_affine=p*(p+1)<=Z.shape[-1]
    if include_affine:
        global_part=global_affine_dictionary(x,q)
        global_columns=global_part.shape[-1]
        Z=np.concatenate((global_part,Z),axis=2)
    k=Z.shape[-1]
    R=(Z[1:]-np.einsum('nab,nbk->nak',a,Z[:-1])).reshape(-1,k)/np.sqrt((n-1)*r)
    F=np.einsum('npa,nak->npk',q,Z)
    near=NearestNeighbors(n_neighbors=2).fit(x).kneighbors(x,return_distance=False)[:,1]
    S=(F-F[near]).reshape(-1,k)/np.sqrt(n*p);Sr=linalg.qr(S,mode='economic')[1]
    anchor=Z[0];c0=anchor.T@linalg.solve(anchor@anchor.T,np.eye(r),assume_a='pos');null=linalg.null_space(anchor)
    M=np.vstack((R,np.sqrt(spatial_weight)*Sr,np.sqrt(ridge/k)*np.eye(k)))
    z=linalg.lstsq(M@null,-M@c0,cond=1e-10,lapack_driver='gelsy')[0];c=c0+null@z
    model=dict(centers=centers,centerq=cq,h=h,affine=True,C=c[global_columns:],landmark_indices=ids,r=r,p=p,
        ridge=ridge,spatial_weight=spatial_weight,requires_tangent_projection=True,
        training_X=x,training_Q=q,dictionary_columns=int(k),
        global_affine_columns=int(global_columns))
    if include_affine:model['global_affine_C']=c[:global_columns].reshape(p,p+1,r)
    return model

def evaluate_W(model,queryX,derivative=False):
    """Return [...,p,r]; derivative=True also returns [...,p,p,r] analytic Jacobian."""
    x=np.asarray(queryX);single=x.ndim==1
    if single:x=x[None]
    centers,q,h=model['centers'],model['centerq'],model['h'];r=model['r']
    dx=x[:,None]-centers[None]
    logs=-np.sum(dx*dx,axis=2)/(2*h*h);logs-=logs.max(axis=1,keepdims=True)
    w=np.exp(logs);w/=w.sum(axis=1,keepdims=True)
    b=1+r if model['affine'] else 1
    c=model['C'].reshape(len(centers),r,b,r)
    local=np.broadcast_to(c[None,:,:,0,:],(len(x),len(centers),r,r)).copy()
    if model['affine']:
        u=np.einsum('nlp,lpa->nla',dx,q)/h
        local+=np.einsum('nlb,labr->nlar',u,c[:,:,1:,:])
    patch=np.einsum('lpa,nlar->nlpr',q,local)
    W=np.einsum('nl,nlpr->npr',w,patch)
    global_c=model.get('global_affine_C')
    if global_c is not None:
        W+=global_c[:,0,:][None]+np.einsum('ni,pir->npr',x,global_c[:,1:,:])
    if not derivative:return W[0] if single else W
    loggrad=-dx/(h*h);mean=np.einsum('nl,nlp->np',w,loggrad)
    dw=w[:,:,None]*(loggrad-mean[:,None])
    J=np.einsum('nli,nlor->noir',dw,patch)
    if model['affine']:J+=np.einsum('nl,loa,labr,lib->noir',w,q,c[:,:,1:,:],q)/h
    if global_c is not None:J+=global_c[None,:,1:,:]
    return (W[0],J[0]) if single else (W,J)
