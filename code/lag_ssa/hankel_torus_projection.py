"""Observed Hankel-window projection onto a learned Fourier torus.

All parameters are fitted before this observer runs.  The state is estimated
from an observed field window, initialized by measured SSA phase products.
The complete finite-window character Gram is kept; frequencies are never
assumed orthogonal.  No absolute-time clock phase initializes the observer.
"""

from dataclasses import dataclass
from collections import deque

import numpy as np
from scipy.linalg.blas import dgemm, dgemv
from scipy.linalg import svd
from sympy import Matrix


@dataclass(frozen=True)
class ProjectionSettings:
    window: int = 128
    weights: str = "uniform"
    max_iterations: int = 25
    max_line_search: int = 8
    max_multistarts: int = 12
    max_phase_step: float = .5
    gradient_tolerance: float = 1e-7
    initial_damping: float = 1e-4
    free_linear_rcond: float = 1e-10


def _deck_starts(labels, coefficients, rank, cap):
    count = len(labels)
    energy = np.sum(coefficients[1:1+count]**2+coefficients[1+count:]**2,axis=1)
    selected=[]
    for index in np.argsort(-energy,kind="stable"):
        trial=selected+[list(map(int,labels[index]))]
        if Matrix(trial).rank()>len(selected): selected=trial
        if len(selected)==rank:break
    if len(selected)!=rank:
        return np.zeros((1,rank)),{"status":"insufficient learned independent characters","starts":1}
    integer=Matrix(selected);inverse=integer.inv()
    determinant=abs(int(integer.det()))
    generators=[tuple(inverse[i,j] % 1 for i in range(rank)) for j in range(rank)]
    zero=tuple([0]*rank);queue=deque([zero]);seen={zero};representatives=[]
    while queue and len(representatives)<cap:
        value=queue.popleft();representatives.append(value)
        for generator in generators:
            following=tuple((first+second)%1 for first,second in zip(value,generator))
            if following not in seen:seen.add(following);queue.append(following)
    offsets=2*np.pi*np.asarray([[float(value) for value in row] for row in representatives])
    return offsets,{"strong_independent_character_rows":selected,"deck_order":determinant,
                    "starts":len(offsets),"deck_enumeration_complete":len(offsets)==determinant,
                    "selection":"descending learned spatial mass character energy; exact independent row selection"}


class HankelTorusProjector:
    def __init__(self,labels,coefficients,frequencies,settings=ProjectionSettings()):
        self.labels=np.asarray(labels,dtype=int)
        self.coefficients=np.asarray(coefficients,dtype=float)
        self.frequencies=np.asarray(frequencies,dtype=float)
        self.settings=settings
        self.rank=len(self.frequencies);self.count=len(self.labels)
        if self.coefficients.shape[0]!=1+2*self.count or self.labels.shape[1]!=self.rank:
            raise ValueError("matching Fourier labels, coefficients and phase dimension required")
        if settings.weights!="uniform":raise ValueError("this declared assay uses uniform weights")
        self.weights=np.full(settings.window,1/settings.window)
        line_frequency=np.einsum("ij,j->i",self.labels,self.frequencies,optimize=False)
        angles=line_frequency[:,None]*np.arange(settings.window)[None,:]
        a=self.coefficients[1:1+self.count];b=self.coefficients[1+self.count:]
        cos,sin=np.cos(angles),np.sin(angles)
        first=a[:,:,None]*cos[:,None,:]+b[:,:,None]*sin[:,None,:]
        second=-a[:,:,None]*sin[:,None,:]+b[:,:,None]*cos[:,None,:]
        bank=np.concatenate((first,second),axis=0)
        self.bank=bank
        weighted=bank*np.sqrt(self.weights)[None,None,:]
        matrix=np.asarray(weighted.reshape(2*self.count,-1),order="F")
        self.gram=dgemm(1.,matrix,matrix,trans_b=True)
        self.weighted_bank=matrix
        self.offsets,self.multistart_metadata=_deck_starts(self.labels,self.coefficients,self.rank,settings.max_multistarts)

    def sufficient_statistics(self,mass_values,starts):
        values=np.asarray(mass_values,dtype=float)-self.coefficients[0,:,None]
        if values.ndim!=2 or values.shape[1]<self.settings.window:
            raise ValueError("at least one full field window is required")
        windows=np.lib.stride_tricks.sliding_window_view(values,self.settings.window,axis=1)[:,:starts,:]
        if windows.shape[1]!=starts:raise ValueError("not enough measured windows for the requested starts")
        weighted=windows*np.sqrt(self.weights)[None,None,:]
        columns=np.asarray(weighted.transpose(0,2,1).reshape(-1,starts),order="F")
        matrix=np.asarray((self.bank*np.sqrt(self.weights)[None,None,:]).reshape(2*self.count,-1),order="F")
        right=dgemm(1.,matrix,columns)
        constants=np.einsum("in,in->n",columns,columns,optimize=False)
        return right,constants

    def feature(self,phase):
        angle=np.einsum("ij,j->i",self.labels,phase,optimize=False)
        cos,sin=np.cos(angle),np.sin(angle)
        feature=np.r_[cos,sin]
        jacobian=np.vstack((-sin[:,None]*self.labels,cos[:,None]*self.labels))
        return feature,jacobian

    def cost_gradient(self,phase,right,constant=0.):
        feature,jacobian=self.feature(phase)
        residual=dgemv(1.,self.gram,feature)-right
        cost=float(constant+np.dot(feature,residual-right))
        gradient=2*np.einsum("ij,i->j",jacobian,residual,optimize=False)
        return cost,gradient,jacobian

    def _optimize(self,initial,right,constant):
        phase=np.asarray(initial,dtype=float).copy()
        damping=self.settings.initial_damping
        initial_cost=self.cost_gradient(phase,right,constant)[0]
        accepted=0;converged=False
        for iteration in range(self.settings.max_iterations):
            cost,gradient,jacobian=self.cost_gradient(phase,right,constant)
            if np.max(abs(gradient))<=self.settings.gradient_tolerance:
                converged=True;break
            gj=dgemm(1.,self.gram,np.asarray(jacobian,order="F"))
            hessian=2*np.einsum("di,dj->ij",jacobian,gj,optimize=False)
            regularizer=np.maximum(np.diag(hessian),1e-10)
            try:step=np.linalg.solve(hessian+damping*np.diag(regularizer),-gradient)
            except np.linalg.LinAlgError:damping*=10;continue
            step*=min(1.,self.settings.max_phase_step/max(np.max(abs(step)),1e-15))
            changed=False
            for scale in (2.**(-index) for index in range(self.settings.max_line_search)):
                trial=phase+scale*step
                candidate=self.cost_gradient(trial,right,constant)[0]
                if candidate<cost:
                    phase=trial;accepted+=1;damping=max(1e-10,damping*.5);changed=True;break
            if not changed:
                damping*=10
                if damping>1e10:break
        final_cost,gradient,_=self.cost_gradient(phase,right,constant)
        return phase,{"initial_cost":initial_cost,"final_cost":final_cost,
                      "iterations":iteration+1,"accepted_steps":accepted,"converged_gradient":converged,
                      "final_gradient_max":float(np.max(abs(gradient)))}

    def project(self,mass_values,seed_phase):
        seed=np.asarray(seed_phase,dtype=float)
        right,constant=self.sufficient_statistics(mass_values,seed.shape[1])
        phases=[];diagnostics=[]
        for index in range(seed.shape[1]):
            attempts=[self._optimize(seed[:,index]+offset,right[:,index],constant[index]) for offset in self.offsets]
            chosen=min(range(len(attempts)),key=lambda which:attempts[which][1]["final_cost"])
            phase,info=attempts[chosen]
            phases.append(np.angle(np.exp(1j*phase)))
            diagnostics.append({**info,"selected_start":chosen,
                                "seed_cost":self.cost_gradient(seed[:,index],right[:,index],constant[index])[0],
                                "cost_not_above_product_seed":info["final_cost"]<=self.cost_gradient(seed[:,index],right[:,index],constant[index])[0]+1e-10})
        phase=np.asarray(phases).T
        angles=np.einsum("ij,jt->it",self.labels,phase,optimize=False)
        design=np.column_stack((np.ones(seed.shape[1]),np.cos(angles.T),np.sin(angles.T)))
        field=np.einsum("nj,jq->qn",design,self.coefficients,optimize=False)
        return {"phase":phase,"mass_field":field,"diagnostics":diagnostics,
                "multistart":self.multistart_metadata}

    def compression_check(self,mass_values,phase):
        right,constant=self.sufficient_statistics(mass_values,1)
        compressed,gradient,_=self.cost_gradient(phase,right[:,0],constant[0])
        features,_=self.feature(phase)
        reconstructed=self.coefficients[0,:,None]+np.einsum("d,dql->ql",features,self.bank,optimize=False)
        difference=mass_values[:,:self.settings.window]-reconstructed
        explicit=float(np.einsum("ql,l,ql->",difference,self.weights,difference,optimize=False))
        numeric=[];epsilon=1e-6
        for index in range(self.rank):
            delta=np.zeros(self.rank);delta[index]=epsilon
            numeric.append((self.cost_gradient(phase+delta,right[:,0],constant[0])[0]
                           -self.cost_gradient(phase-delta,right[:,0],constant[0])[0])/(2*epsilon))
        return {"compressed_cost":compressed,"explicit_window_cost":explicit,
                "cost_absolute_difference":abs(compressed-explicit),
                "analytic_gradient_max_difference_from_finite_difference":float(np.max(abs(gradient-numeric)))}

    def free_linear_projection(self,mass_values,starts):
        """Free coefficients in exactly the same learned Hankel span."""
        values=np.asarray(mass_values,dtype=float)-self.coefficients[0,:,None]
        windows=np.lib.stride_tricks.sliding_window_view(values,self.settings.window,axis=1)[:,:starts,:]
        weighted=windows*np.sqrt(self.weights)[None,None,:]
        columns=np.asarray(weighted.transpose(0,2,1).reshape(-1,starts),order="F")
        u,singular,vh=svd(self.weighted_bank.T,full_matrices=False,lapack_driver="gesdd")
        keep=singular>singular[0]*self.settings.free_linear_rcond
        projected=dgemm(1.,np.asarray(u[:,keep].T,order="F"),columns)
        free=dgemm(1.,np.asarray(vh[keep].T,order="F"),np.asarray(projected/singular[keep,None],order="F"))
        field=self.coefficients[0,:,None]+np.einsum("dq,dn->qn",self.coefficients[1:],free,optimize=False)
        return {"mass_field":field,"free_coefficients":free,
                "metadata":{"rcond":self.settings.free_linear_rcond,"numerical_rank":int(np.sum(keep)),
                    "real_dictionary_columns":len(singular),"largest_singular":float(singular[0]),
                    "smallest_singular":float(singular[-1]),
                    "untruncated_condition":float(singular[0]/singular[-1]) if singular[-1]>0 else None,
                    "mean_fixed_to_same_learned_field":True,"same_dictionary_gram_and_windows":True,
                    "nonlinear_phase_constraints_imposed":False}}
