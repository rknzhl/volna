"""Reproducible finite audit of a proposed unified phase method.

Run from the repository with its existing virtual environment, for example:
  .venv/bin/python results/phase_method_audit_2026_10_20/run_audit.py --case warped

Cases: circle, harmonic, warped, weak. The weak case is an intended negative test.
This is a research prototype; it does not certify global embedding or infer a
spatial field at locations without observations. Original PDF/notebooks unchanged.
"""
import os
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/takers_coframe_mpl")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import sys, json, time, itertools, argparse, warnings
from pathlib import Path
import numpy as np
from scipy import linalg
from sklearn.neighbors import NearestNeighbors
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code" / "lag_ssa"))
from geometric_phase_frontend import fit_hankel_topology, FrontendSettings
from tangent_overlap import windows, fold
from evaluate_tangent_overlap_angles import match_phases
warnings.filterwarnings("ignore", category=RuntimeWarning,
    message=r"(divide by zero|overflow|invalid value) encountered in matmul")


def make_data(n=2100, seed=31, case="warped"):
    rng=np.random.default_rng(seed)
    theta=np.arange(n)[:,None]*np.array([.41*np.sqrt(2),.27*np.sqrt(3)])[None,:]+[.31,.84]
    a,b=theta.T
    if case=="circle":
        a=theta[:,0];theta=theta[:,:1]
        base=np.array([np.cos(a)+.3*np.cos(2*a),np.sin(a)+.2*np.sin(3*a),
            .2*np.cos(3*a),.2*np.sin(2*a),.1*np.cos(4*a),.1*np.sin(4*a)]).T
    elif case=="warped":
        qa=a+.45*np.sin(a)+.10*np.sin(a+b)
        qb=b+.35*np.sin(b)+.10*np.sin(a+b)
        base=np.array([np.cos(qa),np.sin(qa),np.cos(qb),np.sin(qb),
            .15*np.cos(2*a+b),.15*np.sin(2*a+b),.15*np.cos(a-2*b),.15*np.sin(a-2*b)]).T
    elif case=="harmonic":
        base=np.array([np.cos(a)+.15*np.cos(3*a),np.sin(a)+.15*np.sin(3*a),
            np.cos(b)+.15*np.cos(3*b),np.sin(b)+.15*np.sin(3*b),
            .15*np.cos(2*a+b),.15*np.sin(2*a+b),.15*np.cos(a-2*b),.15*np.sin(a-2*b)]).T
    elif case=="weak":
        base=np.array([np.cos(2*a),np.sin(2*a),np.cos(2*b),np.sin(2*b),
            .15*np.cos(a),.15*np.sin(a),.15*np.cos(b),.15*np.sin(b)]).T
    else:
        raise ValueError(case)
    mix=linalg.qr(rng.normal(size=(base.shape[1],base.shape[1])))[0]
    return base@mix.T,theta

def geometry(x,r,k=24):
    n,p=x.shape
    dist,js=NearestNeighbors(n_neighbors=k+1).fit(x).kneighbors(x)
    dist,js=dist[:,1:],js[:,1:]
    Q=[]; gaps=[]
    for i,idx in enumerate(js):
        delta=x[idx]-x[idx].mean(axis=0)
        _,s,vh=linalg.svd(delta,full_matrices=False)
        Q.append(vh[:r].T); gaps.append(s[r-1]/max(s[r],1e-12))
    Q=np.asarray(Q)
    A=[];cond=[];err=[]
    m=p//2
    for i in range(n-1):
        a0=Q[i+1,:-m];a1=Q[i,m:]
        ai=linalg.lstsq(a0,a1)[0]
        A.append(ai); cond.append(linalg.svdvals(a0)[-1])
        err.append(linalg.norm(a0@ai-a1)/linalg.norm(a1))
    return Q,np.asarray(A),js,dist,dict(local_gap_quantiles=np.quantile(gaps,[0,.1,.5]),
        overlap_min_singular_quantiles=np.quantile(cond,[0,.1,.5]),
        overlap_relative_residual_quantiles=np.quantile(err,[.1,.5,.9]))

def structured_decode(phi,x,delta,Q=None,D=None,degree=4,lag=2):
    n,p=x.shape;m=p//lag
    angles=np.concatenate([phi+ell*delta for ell in range(lag)])
    targets=np.concatenate([x[:,ell*m:(ell+1)*m] for ell in range(lag)])
    B=features(angles,degree)
    if Q is not None and D is not None:
        ders=[];dertargets=[]
        r=phi.shape[1]
        for ell in range(lag):
            _,db=features(phi+ell*delta,degree,derivative=True)
            # J(Psi) D = Q, including the same Q basis and scaled window metric.
            ders.append(np.einsum("nkb,nba->nak",db,D).reshape(n*r,-1))
            dertargets.append(Q[:,ell*m:(ell+1)*m,:].transpose(0,2,1).reshape(n*r,m))
        B=np.vstack((B,np.concatenate(ders)/np.sqrt(r)))
        targets=np.vstack((targets,np.concatenate(dertargets)/np.sqrt(r)))
    coef=linalg.solve(B.T@B+1e-6*len(B)*np.eye(B.shape[1]),B.T@targets,assume_a="pos")
    return coef

def structured_predict(phi,coef,delta,degree=4,lag=2,derivative=False):
    out=[];jac=[]
    for ell in range(lag):
        b,db=features(phi+ell*delta,degree,derivative=True)
        out.append(b@coef);jac.append(np.einsum("nkr,kp->npr",db,coef))
    return np.concatenate(out,axis=1),np.concatenate(jac,axis=1)

def features(phi,degree=4,derivative=False):
    r=phi.shape[1]
    labels=np.asarray([t for t in itertools.product(range(-degree,degree+1),repeat=r)
        if any(t) and next(v for v in t if v)!=0 and next(v for v in t if v)>0])
    arg=phi@labels.T
    M=np.column_stack((np.ones(len(phi)),np.cos(arg),np.sin(arg)))
    if not derivative:return M
    d=np.zeros((len(phi),M.shape[1],r))
    for a in range(r):
        d[:,1:1+len(labels),a]=-np.sin(arg)*labels[:,a]
        d[:,1+len(labels):,a]=np.cos(arg)*labels[:,a]
    return M,d

def evaluate(angles,truth,match):
    target=truth@np.asarray(match["winding_candidate"]).T+match["origins"]
    err=np.angle(np.exp(1j*(angles-target)))
    return dict(phase_mae_rad=float(np.mean(np.abs(err))),phase_max_rad=float(np.max(np.abs(err))))

def charts(model):
    r,nland=model["integrals"].shape
    H=np.zeros((r,nland,nland))
    e=model["edges"];ind=model["edge_indices"]
    for a in range(r):
        H[a,e[:,0],e[:,1]]=model["harmonics"][a,ind]
        H[a,e[:,1],e[:,0]]=-model["harmonics"][a,ind]
    return H

def encoder_base(model,x,H=None):
    """Return phi0[N,r], weights[N,L], their analytic derivatives."""
    if H is None:H=charts(model)
    centers=model["landmark_columns"].T
    diff=x[:,None,:]-centers[None,:,:]
    d2=np.sum(diff*diff,axis=2);rad2=model["cover_radius"]**2
    active=d2<rad2
    raw=np.zeros_like(d2)
    raw[active]=np.exp(rad2/(d2[active]-rad2))
    total=raw.sum(axis=1)
    covered=total>0
    w=raw/np.maximum(total[:,None],np.finfo(float).tiny)
    loggrad=np.zeros_like(diff)
    factor=np.zeros_like(d2)
    factor[active]=-2*rad2/(d2[active]-rad2)**2
    loggrad=factor[:,:,None]*diff
    mean=np.einsum("nl,nlp->np",w,loggrad)
    dw=w[:,:,None]*(loggrad-mean[:,None,:])
    root=np.argmax(active,axis=1)
    local=H[:,root,:].transpose(1,0,2)
    native=model["integrals"][:,root].T+np.einsum("nrl,nl->nr",local,w)
    phi=2*np.pi*np.einsum("ab,nb->na",model["change_basis"],native)
    grad=2*np.pi*np.einsum("ab,nbl,nlp->nap",model["change_basis"],local,dw)
    return phi,w,grad,dw,covered

def affine_dictionary(model, x, w, dw, landmark_q):
    diff = x[:, None, :] - model["landmark_columns"].T[None, :, :]
    radius = model["cover_radius"]
    coord = np.einsum("nlp,lpa->nla", diff, landmark_q) / radius
    val = np.concatenate((w[:, :, None], w[:, :, None] * coord), axis=2)
    grad = np.concatenate((dw[:, :, None, :],
        dw[:, :, None, :] * coord[:, :, :, None]
        + w[:, :, None, None] * landmark_q.transpose(0, 2, 1)[None] / radius), axis=2)
    return val.reshape(len(x), -1), grad.reshape(len(x), -1, x.shape[1])

def query_tangents(train, query, r, k=24):
    ids = NearestNeighbors(n_neighbors=k).fit(train).kneighbors(query, return_distance=False)
    return np.asarray([linalg.svd(train[ix]-train[ix].mean(0), full_matrices=False)[2][:r].T
                       for ix in ids])

def maps_from_q(q):
    m = q.shape[1] // 2
    return np.asarray([linalg.lstsq(q[i+1, :-m], q[i, m:])[0] for i in range(len(q)-1)])

def differential_checks(D, A, reference_scale):
    s = np.linalg.svd(D, compute_uv=False)
    orientation = np.r_[1., np.cumprod(np.sign(np.linalg.det(A)))]
    det = np.linalg.det(D) * orientation
    consistent = bool(np.all(det > 0) or np.all(det < 0))
    rel = float(np.min(s[:, -1] / np.maximum(s[:, 0], 1e-15)))
    scaled = float(s[:, -1].min()/max(reference_scale,1e-15))
    return dict(minimum_singular=float(s[:, -1].min()), minimum_relative_singular=rel,
                minimum_singular_relative_to_initial_rms=scaled,
                orientation_consistent=consistent, accepted=bool(consistent and rel > 1e-3 and scaled > 1e-3))

def run(case="warped", seed=31, ntrain=1400):
    started = time.time(); lag = 2; nval = 200; ntest = 300
    Y, truth = make_data(ntrain+nval+ntest, seed, case)
    train, val, test = Y[:ntrain], Y[ntrain:ntrain+nval], Y[ntrain+nval:]
    mean, scale = train.mean(0), train.std(0)
    normal = np.sqrt(lag*Y.shape[1])
    def preprocess(y): return windows((y-mean)/scale, lag)/normal
    x, xv, xt = map(preprocess, (train, val, test))
    topo = fit_hankel_topology(x.T, FrontendSettings(n_landmarks=192))
    out = dict(case=case, seed=seed, lag=lag,
        counts=dict(train=ntrain, validation=nval, test=ntest),
        protocol="fixed local-affine global real potential; exact analytic encoder derivatives; rank-revealing LS cond=1e-8; lambda selected by validation structured window error among train+validation coframe gates; structured Hermite decoder", 
        topology=topo["diagnostics"], no_truth_used_to_fit_or_select=True, results=[])
    if topo.get("train_phases") is None:
        out["failure"] = "unresolved_initial_toroidal_coordinates"
        return out
    if not topo.get("certificate", {}).get("full_primitive_free_cohomology_basis_certified", False):
        out["failure"] = "unresolved_primitive_finite_complex_basis"
        return out
    H = charts(topo)
    p0, w, grad, dw, cov = encoder_base(topo, x, H)
    pv, wv, gradv, dwv, cv = encoder_base(topo, xv, H)
    pt, wt, gradt, dwt, ct = encoder_base(topo, xt, H)
    if not (cov.all() and cv.all() and ct.all()):
        out["failure"] = "uncovered_states"; return out
    r = p0.shape[1]
    Q, A, _, _, geom = geometry(x, r, k=24)
    Qv = query_tangents(x, xv, r); Qt = query_tangents(x, xt, r)
    Av, At = maps_from_q(Qv), maps_from_q(Qt)
    out["geometry"] = geom
    landmark_indices = NearestNeighbors(n_neighbors=1).fit(x).kneighbors(topo["landmark_columns"].T, return_distance=False).ravel()
    landmark_q = Q[landmark_indices]
    V, dV = affine_dictionary(topo, x, w, dw, landmark_q)
    Vv, dVv = affine_dictionary(topo, xv, wv, dwv, landmark_q)
    Vt, dVt = affine_dictionary(topo, xt, wt, dwt, landmark_q)
    B = np.einsum("nbp,npa->nab", grad, Q)
    Z = np.einsum("nkp,npa->nak", dV, Q)
    constantZ=np.einsum("nlp,npa->nal",dw,Q)
    constantS=np.linalg.svd(constantZ,compute_uv=False)
    out["rejected_constant_dictionary_diagnostic"]=dict(
        minimum_positive_weight_count=int(np.min(np.count_nonzero(w>0,axis=1))),
        locally_rank_deficient_state_count=int(np.count_nonzero(constantS[:,-1] < 1e-10*constantS[:,0])),
        minimum_local_singular=float(constantS[:,-1].min()),
        note="A constant-per-chart correction factors through dw and cannot repair these local rank deficiencies.")
    Bv = np.einsum("nbp,npa->nab", gradv, Qv)
    Zv = np.einsum("nkp,npa->nak", dVv, Qv)
    Bt = np.einsum("nbp,npa->nab", gradt, Qt)
    Zt = np.einsum("nkp,npa->nak", dVt, Qt)
    out["minimum_local_dictionary_singular"] = float(np.linalg.svd(Z, compute_uv=False)[:, -1].min())
    take = np.linspace(0, len(x)-1, 12, dtype=int); eps = 1e-6
    e = np.zeros_like(x[take]); e[:, 0] = eps
    pp, wp, _, dwp, _ = encoder_base(topo, x[take]+e, H)
    pm, wm, _, dwm, _ = encoder_base(topo, x[take]-e, H)
    vp, _ = affine_dictionary(topo, x[take]+e, wp, dwp, landmark_q)
    vm, _ = affine_dictionary(topo, x[take]-e, wm, dwm, landmark_q)
    out["dictionary_finite_difference_max"] = float(np.max(np.abs((vp-vm)/(2*eps)-dV[take, :, 0])))
    dynZ = np.einsum("nab,nbk->nak", A.transpose(0,2,1), Z[1:])-Z[:-1]
    dynB = np.einsum("nab,nbc->nac", A.transpose(0,2,1), B[1:])-B[:-1]
    e0 = max(float(np.mean(B*B)), 1e-12); d0 = max(float(np.mean(dynB*dynB)), 1e-12)
    n, _, k = Z.shape
    # Fix one constant coefficient; rank-revealing least squares handles further redundant features.
    trutrain, trutest = truth[:len(x)], truth[ntrain+nval:ntrain+nval+len(xt)]
    frozen = []
    for lam in [0., .1, 1., 10., 100., 1000.]:
        design = np.vstack((Z.reshape(n*r,k)/np.sqrt(n*r*e0),
            dynZ.reshape((n-1)*r,k)*np.sqrt(lam/((n-1)*r*d0))))[:,1:]
        target = -np.vstack((B.reshape(n*r,r)/np.sqrt(n*r*e0),
            dynB.reshape((n-1)*r,r)*np.sqrt(lam/((n-1)*r*d0))))
        c = np.zeros((k,r))
        c[1:], _, rank, _ = linalg.lstsq(design, target, cond=1e-8, lapack_driver="gelsy")
        D = (B+np.einsum("nak,kb->nab", Z,c)).transpose(0,2,1)
        Dv = (Bv+np.einsum("nak,kb->nab", Zv,c)).transpose(0,2,1)
        Dt = (Bt+np.einsum("nak,kb->nab", Zt,c)).transpose(0,2,1)
        phi, phiv, phit = p0+V@c, pv+Vv@c, pt+Vt@c
        delta = np.angle(np.mean(np.exp(1j*np.diff(phi,axis=0)),axis=0))
        coef = structured_decode(phi,x,delta,Q,D)
        vpred, _ = structured_predict(phiv,coef,delta)
        tpred, _ = structured_predict(phit,coef,delta)
        def raw(z): return z*normal*np.tile(scale,lag)+np.tile(mean,lag)
        vw, tw = windows(val,lag), windows(test,lag)
        _, jac = structured_predict(phi,coef,delta)
        _, vjac = structured_predict(phiv,coef,delta)
        _, tjac = structured_predict(phit,coef,delta)
        row = dict(lambda_value=lam, design_rank=int(rank), design_columns=k-1,
            coefficient_norm=float(linalg.norm(c)),
            train_coframe=differential_checks(D,A,np.sqrt(e0)),
            validation_coframe=differential_checks(Dv,Av,np.sqrt(e0)),
            test_coframe=differential_checks(Dt,At,np.sqrt(e0)),
            validation_window_nrmse=float(linalg.norm(raw(vpred)-vw)/linalg.norm(vw)),
            test_window_nrmse=float(linalg.norm(raw(tpred)-tw)/linalg.norm(tw)),
            test_snapshot_nrmse=float(linalg.norm(fold(raw(tpred),Y.shape[1],lag)-test)/linalg.norm(test)),
            train_field_consistency=float(linalg.norm(np.einsum("npb,nba->npa",jac,D)-Q)/linalg.norm(Q)),
            validation_field_consistency=float(linalg.norm(np.einsum("npb,nba->npa",vjac,Dv)-Qv)/linalg.norm(Qv)),
            test_field_consistency=float(linalg.norm(np.einsum("npb,nba->npa",tjac,Dt)-Qt)/linalg.norm(Qt)),
            test_phase_step_rmse=float(np.sqrt(np.mean(np.angle(np.exp(1j*(np.diff(phit,axis=0)-delta)))**2))),
            estimated_phase_step=delta)
        dyn = np.einsum("nab,nbc->nac", A.transpose(0,2,1), D.transpose(0,2,1)[1:])-D.transpose(0,2,1)[:-1]
        row["normalized_smooth_energy"] = float(np.mean(D*D)/e0)
        row["normalized_dynamic_energy"] = float(np.mean(dyn*dyn)/d0)
        grid = (np.stack(np.meshgrid(np.arange(40)*2*np.pi/40,np.arange(40)*2*np.pi/40),axis=-1).reshape(-1,2)
                if r==2 else np.arange(100)[:,None]*2*np.pi/100)
        gx,jg=structured_predict(grid,coef,delta)
        sh,_=structured_predict(grid+delta,coef,delta)
        row["decoder_grid_min_singular"] = float(np.linalg.svd(jg,compute_uv=False)[:,-1].min())
        row["structured_overlap_max"] = float(np.max(np.abs(gx[:,Y.shape[1]:]-sh[:,:-Y.shape[1]])))
        out["results"].append(row); frozen.append((phi,phit,c))
        print("RESULT",case,lam,"val",row["validation_window_nrmse"],"test",row["test_window_nrmse"],
            "gates",row["train_coframe"]["accepted"],row["validation_coframe"]["accepted"],row["test_coframe"]["accepted"],flush=True)
    eligible = [i for i,t in enumerate(out["results"]) if t["train_coframe"]["accepted"] and t["validation_coframe"]["accepted"]]
    selected = min(eligible,key=lambda i:out["results"][i]["validation_window_nrmse"]) if eligible else None
    out["selected_result_index"]=selected
    out["status"]="conditional_e2e_fit" if selected is not None else "unresolved_coframe"
    # A no-correction baseline uses the same frozen topology and structured decoder.
    baseD=B.transpose(0,2,1)
    basedelta=np.angle(np.mean(np.exp(1j*np.diff(p0,axis=0)),axis=0))
    basecoef=structured_decode(p0,x,basedelta,Q,baseD)
    basepred,_=structured_predict(pt,basecoef,basedelta)
    out["uncorrected_baseline_test_window_nrmse"]=float(linalg.norm(raw(basepred)-tw)/linalg.norm(tw))
    if selected is not None:
        phi,phit,c=frozen[selected]
        D=(B+np.einsum("nak,kb->nab",Z,c)).transpose(0,2,1)
        delta=out["results"][selected]["estimated_phase_step"]
        coef=structured_decode(phi,x,delta,Q,D)
        gx,jg=structured_predict(grid,coef,delta)
        gp,gw,gg,gdw,gcov=encoder_base(topo,gx,H)
        gv,gdv=affine_dictionary(topo,gx,gw,gdw,landmark_q)
        cycle=np.angle(np.exp(1j*(gp+gv@c-grid)))
        # This is a sampled model check, never a proof of global embedding.
        result_checks=dict(decoder_grid_points=len(grid),decoder_grid_encoder_coverage=float(gcov.mean()),
            sampled_check_is_not_global_embedding_certificate=True)
        if gcov.all():
            gj=gg+np.einsum("nkp,kb->nbp",gdv,c)
            product=np.einsum("nbp,npa->nba",gj,jg)
            result_checks.update(phase_cycle_mae_rad=float(np.mean(np.abs(cycle))),
                phase_cycle_max_rad=float(np.max(np.abs(cycle))),
                differential_cycle_relative=float(linalg.norm(product-np.eye(r)[None])/np.sqrt(len(grid)*r)))
        correction_fd=(vp-vm)@c/(2*eps)
        corrected_fd=np.angle(np.exp(1j*(pp-pm)))/(2*eps)+correction_fd
        corrected_j=grad[take,:,0]+np.einsum("nk,kb->nb",dV[take,:,0],c)
        result_checks["corrected_encoder_gradient_finite_difference_max"]=float(np.max(np.abs(corrected_fd-corrected_j)))
        out["selected_model_checks"]=result_checks
    # Hidden generator phases are used only for evaluation after the encoder and selection are frozen.
    for row, (phi,phit,c) in zip(out["results"],frozen):
        match = match_phases(phi,trutrain)
        row["train_phase"] = match
        row["test_phase"] = evaluate(phit,trutest,match)
    out["elapsed_seconds"] = time.time()-started
    return out

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=["circle", "harmonic", "warped", "weak"], default="warped")
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--ntrain", type=int, default=1400)
    args = parser.parse_args()
    result = run(args.case, args.seed, args.ntrain)
    destination = Path(__file__).with_name(f"{args.case}_{args.seed}_results.json")
    destination.write_text(json.dumps(result, indent=2, default=lambda value: np.asarray(value).tolist()))
    print("SAVED", destination, flush=True)
