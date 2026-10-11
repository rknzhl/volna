"""Independent full-JOINT audit of all frozen local predictive models.

No core/runner import, Schur implementation, nonlinear refit, model selection
change or dense new-observation covariance is used. Frequency columns are
scaled by Ntrain for conditioning. All numeric checks and three strict choices
freeze before parsing previously frozen physical character/kernel metadata.
"""

from project_paths import RESULTS_DIR

from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sympy import Matrix


HERE = Path(__file__).resolve().parent
SOURCE = RESULTS_DIR / 'local_group_predictive.json'
OUTPUT = RESULTS_DIR / 'local_group_predictive_audit.json'
NUMERIC = RESULTS_DIR / 'local_group_predictive_numeric_audit.json'
PROTOCOL = RESULTS_DIR / 'local_group_predictive_audit_protocol.json'
TRUTH = RESULTS_DIR / 'ledger_beam_group_selection_audit.json'
SCORES = ("negative_log_predictive_density", "point_FUTURE_raw_sensor_mse", "local_linear_mean_FUTURE_raw_sensor_mse")


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path,allow_pickle=False) as archive:return {key:archive[key] for key in archive.files}
def save(path,payload):
    temporary=path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n");temporary.replace(path)


def declaration():
    source=json.loads(SOURCE.read_text())
    if (source["status"]!="finished_public_only_predictive_comparison" or len(source["rows"])!=6
            or not source["all_six_outputs_frozen_before_any_physical_evaluation"]
            or not source["source_input_and_all_original_model_hashes_unchanged"]):
        raise ValueError("explicit completed all-six immutable predictive outputs required")
    protected={str(SOURCE):digest(SOURCE),source["source_beam"]:source["source_beam_sha256"],
               str(RESULTS_DIR / 'local_group_predictive_protocol.json'):source["protocol_sha256"]}
    protected.update({str(HERE/name):sha for name,sha in source["source_fingerprints"].items()})
    protected.update({item["path"]:item["sha256"] for item in source["protected_original_artifacts"]})
    for entry,case in zip(source["manifest"],source["rows"]):
        if entry["run"]!=case["run"]:raise ValueError("manifest/case order mismatch")
        for key in ("input","ledger_parameters"):protected[entry[key]]=entry[key+"_sha256"]
        for model in entry["models"]:protected[model["parameters"]]=model["parameters_sha256"]
        for model in case["rows"]:protected[model["archive"]]=model["archive_sha256"]
    if sum(len(case["rows"]) for case in source["rows"])!=237:raise ValueError("all 237 models required")
    for path,sha in protected.items():
        if digest(path)!=sha:raise ValueError("changed frozen artifact: "+path)
    return dict(status="prepared_no_data_read",source=str(SOURCE),source_sha256=digest(SOURCE),
        audit_source_sha256=digest(__file__),protected=protected,models=237,
        method="FULL joint H/Hplus; sensor-major amplitude blocks; derivative D/Ntrain; no core import",
        numeric_tolerances=dict(score_atol=1e-8,score_rtol=1e-8,array_rtol=1e-6),
        numeric_stage_frozen_before_truth=True,physical_assessment="cached old kernel/visible-module only; no new evaluation scoring",
        posthoc_selection_block_not_independent_test=True,no_new_fit_threshold_model_or_candidate=True)


def temporal(parameters,times,basis):
    nu=parameters["frequencies"];c=parameters["coefficients_sensor_center"]
    tau=np.asarray(times)-float(parameters["time_center"]);angles=tau[:,None]*nu[None,:];j=len(nu)
    x=np.column_stack((np.ones(len(times)),np.cos(angles),np.sin(angles)))
    point=np.einsum("np,pq->nq",x,c,optimize=False)
    complex_amplitude=c[1:1+j]-1j*c[1+j:]
    ambient=np.real(1j*tau[:,None,None]*complex_amplitude[None,:,:]*np.exp(1j*angles)[:,:,None])
    derivative=np.einsum("njq,jr->nqr",ambient,basis,optimize=False)
    return x,derivative,point,tau


def information(x,d,n_train):
    n,p=x.shape;q,r=d.shape[1:];size=p*q+r;h=np.zeros((size,size))
    g=np.einsum("np,nk->pk",x,x,optimize=False)
    for sensor in range(q):h[sensor*p:(sensor+1)*p,sensor*p:(sensor+1)*p]=g
    cross=np.einsum("np,nqr->pqr",x,d,optimize=False).transpose(1,0,2).reshape(p*q,r)/n_train
    h[:p*q,p*q:]=cross;h[p*q:,:p*q]=cross.T
    h[p*q:,p*q:]=np.einsum("nqr,nqs->rs",d,d,optimize=False)/(n_train*n_train)
    return h


def rhs(x,d,error,n_train):
    return np.r_[np.einsum("np,nq->pq",x,error,optimize=False).T.ravel(),
                 np.einsum("nqr,nq->r",d,error,optimize=False)/n_train]


def joint_reference(x,d,error,new_x,new_d,new_error,sigma):
    n,p=x.shape;m,q=new_error.shape;r=d.shape[2];boundary=p*q
    h=information(x,d,n);plus=h+information(new_x,new_d,n)
    hf=cho_factor(h,lower=True);pf=cho_factor(plus,lower=True)
    delta=cho_solve(hf,rhs(x,d,error,n));dc=delta[:boundary].reshape(q,p).T;de=delta[boundary:]/n
    shift=np.einsum("mp,pq->mq",new_x,dc,optimize=False)+np.einsum("mqr,r->mq",new_d,de,optimize=False)
    shifted=new_error-shift;b=rhs(new_x,new_d,shifted,n)
    quad=(float(np.sum(shifted**2))-float(np.dot(b,cho_solve(pf,b))))/(sigma*sigma)
    logdet=float(2*np.sum(np.log(np.diag(pf[0])))-2*np.sum(np.log(np.diag(hf[0]))))
    columns=np.zeros((boundary+r,r));columns[boundary:]=np.eye(r)
    inverse_columns=cho_solve(hf,columns) if r else np.empty((boundary,0))
    cf=(sigma*sigma)*inverse_columns[boundary:]/(n*n)
    cross=((sigma*sigma)*inverse_columns[:boundary]/n).reshape(q,p,r).transpose(1,0,2)
    return dict(shift=shift,dc=dc,de=de,cf=cf,cross=cross,quadratic=quad,logdet=logdet,
                nll=.5*(m*q*math.log(2*math.pi*sigma*sigma)+logdet+quad))


def compare(first,second,atol=1e-10,rtol=1e-6):
    a,b=np.asarray(first),np.asarray(second)
    if a.shape!=b.shape or not np.allclose(a,b,atol=atol,rtol=rtol):raise ArithmeticError("independent full-joint comparison failed")
    return float(np.max(abs(a-b),initial=0.))


def numeric_case(entry,case):
    public=load(entry["input"]);n=entry["n_train"];start=n+entry["n_validation"];end=start+entry["n_future"]
    train,future=public["Y"][:,:n].T,public["Y"][:,start:end].T;sigma=float(entry["noise_estimate_train_only"]["sigma"])
    rows=[]
    for metadata,published in zip(entry["models"],case["rows"]):
        if metadata["model_index"]!=published["model_index"]:raise ValueError("model order differs")
        parameters,stored=load(metadata["parameters"]),load(published["archive"])
        if not np.array_equal(public["s"],parameters["sensor_positions"]):raise ValueError("changed sensor geometry")
        basis=stored.get("orthonormal_frequency_basis",np.empty((len(parameters["frequencies"]),0)))
        relation=Matrix(len(metadata["relation_rows_HNF"]),len(parameters["frequencies"]),[int(v) for row in metadata["relation_rows_HNF"] for v in row])
        if published["status"]=="scored_and_frozen":
            if basis.shape[1]!=len(parameters["frequencies"])-relation.rows:raise ArithmeticError("wrong frequency dimension")
            compare(np.einsum("jr,js->rs",basis,basis,optimize=False),np.eye(basis.shape[1]),atol=1e-9)
            if relation.rows:compare(np.einsum("ij,jr->ir",np.asarray(relation.tolist(),float),basis,optimize=False),np.zeros((relation.rows,basis.shape[1])),atol=1e-8)
        x,d,point,_=temporal(parameters,public["t"][:n],basis)
        new_x,new_d,new_point,tau=temporal(parameters,public["t"][start:end],basis)
        error,new_error=train-point,future-new_point
        check=dict(point_prediction=compare(new_point.T,stored["point_prediction_future"]),
                   point_mse=compare(np.mean(new_error**2),published["point_FUTURE_raw_sensor_mse"]))
        record=dict(model_index=metadata["model_index"],status=published["status"],checks=check,
                    point_FUTURE_raw_sensor_mse=float(np.mean(new_error**2)))
        if published["status"]=="scored_and_frozen":
            exact=joint_reference(x,d,error,new_x,new_d,new_error,sigma)
            for key,value in (("negative_log_predictive_density",exact["nll"]),("standardized_quadratic",exact["quadratic"]),("logdet_inflation",exact["logdet"])):
                check[key]=compare(value,published[key],atol=1e-8,rtol=1e-8)
            for key,value,atol in (("posterior_coefficient_mean_shift",exact["dc"],1e-10),("posterior_frequency_mean_shift",exact["de"],1e-12),
                                  ("posterior_frequency_covariance",exact["cf"],1e-15),("posterior_coefficient_frequency_covariance",exact["cross"],1e-14)):
                check[key]=compare(value,stored[key],atol=atol)
            linear=new_point+exact["shift"];check["local_linear_mean"]=compare(linear.T,stored["local_linear_prediction_mean_future"])
            j=len(parameters["frequencies"]);coef=parameters["coefficients_sensor_center"]+exact["dc"]
            nu=parameters["frequencies"]+np.einsum("jr,r->j",basis,exact["de"],optimize=False)
            cov=np.einsum("jr,rs,ks->jk",basis,exact["cf"],basis,optimize=False)
            cross=np.einsum("pqr,jr->pqj",exact["cross"],basis,optimize=False)
            nonlinear=np.broadcast_to(coef[0],future.shape).copy()
            for k in range(j):
                angle=tau*nu[k];damping=np.exp(-.5*tau*tau*cov[k,k])
                nonlinear+=damping[:,None]*(np.cos(angle)[:,None]*(coef[1+k]+tau[:,None]*cross[1+j+k,:,k])
                    +np.sin(angle)[:,None]*(coef[1+j+k]-tau[:,None]*cross[1+k,:,k]))
            check["nonlinear_Gaussian_pushforward_mean"]=compare(nonlinear.T,stored["local_Gaussian_nonlinear_pushforward_mean_diagnostic_future"])
            record.update(negative_log_predictive_density=float(exact["nll"]),local_linear_mean_FUTURE_raw_sensor_mse=float(np.mean((future-linear)**2)))
        rows.append(record)
    selections={}
    for score in SCORES:
        eligible=[row for row in rows if score in row];best=min(eligible,key=lambda row:row[score]) if eligible else None
        chosen=case["declared_selection"][score]
        if (best is None)!=(chosen is None) or (best and best["model_index"]!=chosen["model_index"]):raise ArithmeticError("strict comparison selected another model")
        selections[score]=None if best is None else dict(model_index=best["model_index"],value=best[score],stored_score_difference=compare(best[score],chosen["value"],atol=1e-8,rtol=1e-8))
    return dict(run=entry["run"],models=rows,selections=selections,all_three_strict_choices_replayed=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--run",action="store_true");args=parser.parse_args()
    prepared=declaration()
    if not args.run:
        if PROTOCOL.exists():raise FileExistsError("preserve audit protocol")
        save(PROTOCOL,prepared);print(json.dumps(dict(status=prepared["status"],models=237,protocol=str(PROTOCOL))));return
    old=json.loads(PROTOCOL.read_text())
    if old!=prepared:raise ValueError("audit source or frozen inputs changed since source gate")
    if NUMERIC.exists() or OUTPUT.exists():raise FileExistsError("preserve independent audit outputs")
    source=json.loads(SOURCE.read_text());started=time.perf_counter();rows=[]
    for entry,case in zip(source["manifest"],source["rows"]):
        rows.append(numeric_case(entry,case));print(json.dumps(dict(run=entry["run"],numeric_models=len(rows[-1]["models"]),selections=rows[-1]["selections"])),flush=True)
    numeric=dict(prepared,status="finished_all_numeric_checks_before_truth",rows=rows,all237_numeric_checks_frozen_before_truth=True)
    save(NUMERIC,numeric)  # independent numeric identities and strict selections NOW freeze
    truth=json.loads(TRUTH.read_text());known={row["run"]:row for row in truth["rows"]};beam=json.loads(Path(source["source_beam"]).read_text())
    if truth["source_sha256"]!=source["source_beam_sha256"]:raise ValueError("cached physical audit refers to another beam")
    assessments=[]
    for entry,case,original in zip(source["manifest"],rows,beam["rows"]):
        prior=known[entry["run"]];by_id={row["model_index"]:row for row in prior["all_stored_models"]};selected={};gaps={}
        for score,choice in case["selections"].items():
            model=by_id[choice["model_index"]];selected[score]={key:model[key] for key in ("model_index","continuous_dimension","equals_true_observed_image_kernel_truth_only","full_visible_module_and_image_correct_truth_only","every_relation_physically_valid_truth_only")}
            correct=[row for row in case["models"] if score in row and by_id[row["model_index"]]["equals_true_observed_image_kernel_truth_only"]]
            best=min(correct,key=lambda row:row[score]) if correct else None
            gaps[score]=None if best is None else dict(best_correct_model_index=best["model_index"],best_correct_score=best[score],score_gap_to_selected=best[score]-choice["value"])
        assessments.append(dict(run=entry["run"],selected=selected,best_correct_alternative_gaps=gaps,complete_visible_module_truth_only=prior["complete_visible_module_truth_only"]))
    for path,sha in prepared["protected"].items():
        if digest(path)!=sha:raise ValueError("changed frozen artifact during independent audit")
    output=dict(prepared,status="finished_independent_audit",rows=rows,physical_assessment_after_numeric_freeze=assessments,
        numeric_stage=str(NUMERIC),numeric_stage_sha256=digest(NUMERIC),truth_audit=str(TRUTH),truth_audit_sha256=digest(TRUTH),
        all_original_source_input_prediction_hashes_unchanged=True,elapsed_seconds=time.perf_counter()-started,
        summary={score:dict(exact_observed_image=sum(case["selected"][score]["equals_true_observed_image_kernel_truth_only"] for case in assessments),
            complete_module_and_image=sum(case["selected"][score]["full_visible_module_and_image_correct_truth_only"] for case in assessments)) for score in SCORES})
    save(OUTPUT,output);print(json.dumps(dict(status=output["status"],summary=output["summary"])),flush=True)


if __name__=="__main__":main()
