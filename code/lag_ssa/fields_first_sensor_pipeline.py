"""One observation-only pipeline for synchronized sensor data.

The same fixed conditioning and training-only (lag, neighborhood) search are
used for every input. A candidate is accepted only by the reported finite
diagnostics. Reconstruction conditions on an observed block; it is not a
forecast or an identification of a unique physical trend.
"""
import hashlib
import time
import traceback
from pathlib import Path
import numpy as np
from scipy import linalg
import fields_first_phase as core
from fields_first_preprocess import fit_preprocess, highpass_decompose
from tangent_overlap import windows, fold

LAGS=(4,8,16,32,48)
NEIGHBORS=(24,48,96)
GATES=dict(closed_projection_relative=.05,duality_rms=.15,phase_step_rmse=.05,
    inverse_window_relative=.05,full_turn_relative=.05,flow_order_relative=.05,
    minimum_atlas_gap=.1,query_projection_relative=.1,flow_refinement_relative=.005)

def _finite_diagnostics(values):
    return all(np.isfinite(np.asarray(value, dtype=float)).all()
               for value in values)

def _hashes():
    names = ['fields_first_phase.py', 'fields_first_frame.py',
             'fields_first_tangent_atlas.py', 'fields_first_periods.py',
             'fields_first_implicit_geometry.py', 'fields_first_span.py',
             'fields_first_bounded_persistence.py',
             'fields_first_paths.py', 'fields_first_preprocess.py',
             'fields_first_sensor_pipeline.py']
    base = Path(__file__).resolve().parent
    return {name: hashlib.sha256((base/name).read_bytes()).hexdigest()
            for name in names}


def _source_rms(model):
    return float(model.get('source_full_rms',
        np.sqrt(np.mean(np.sum(model['X']**2,axis=1)))))

def _training_gates_once(model):
    d=model['diagnostics'];result=dict(core_status=model['status'],passed=False)
    if model['status']!='conditional_fields_first_model':return result
    result.update(closed_projection_relative=float(d['periods']['projection_relative']),
        duality_rms=float(d['encoder_field_duality_relative']),
        phase_step_rmse=float(d['train_phase_step_rmse']),
        minimum_atlas_gap=float(d['atlas_min_gap']))
    constrained=model.get('flow_geometry')=='normal_graph_constraint'
    result['flow_geometry']=model.get('flow_geometry','ambient_tangent_extension')
    if constrained:
        result['ambient_field_duality_rms']=result['duality_rms']
        result['duality_rms'],result['flow_duality_probe_count']=core.constrained_flow_duality(model,model['X'])
    checks=dict(closed_projection=result['closed_projection_relative']<GATES['closed_projection_relative'],
        duality=result['duality_rms']<GATES['duality_rms'],
        phase_step=result['phase_step_rmse']<GATES['phase_step_rmse'],
        atlas_gap=result['minimum_atlas_gap']>GATES['minimum_atlas_gap'],
        train_encoder_full_coverage=d['train_encoder_coverage']==1.)
    result['checks']=checks
    if constrained:
        result['reference_displacement_relative']=d['flow_reference_displacement_relative']
        checks['reference_projection']=result['reference_displacement_relative']<GATES['inverse_window_relative']
    if not all(checks.values()):return result
    model['strict_flows']=True
    take=np.unique(np.linspace(0,len(model['X'])-1,16,dtype=int))
    source_rms=_source_rms(model)
    starts=model['X'][[0]]
    if constrained:
        loop_take=np.unique(np.linspace(0,len(model['X'])-1,8,dtype=int))
        starts=core.retract_flow_states(model,model['X'][loop_take])
    trials=[]
    for step in ((.12,.06,.03) if constrained else (.12,)):
        if constrained:
            model['flow_step']=step
            model.pop('flow_domain_diagnostics',None)
        loops=[];refinements=[]
        try:
            for j in range(model['r']):
                end=core.flow(model,starts,2*np.pi,j)
                loops.append(float(np.max(linalg.norm(end-starts,axis=1))/source_rms))
                if constrained:
                    refined=core.flow(model,starts,2*np.pi,j,step=step/2)
                    refinements.append(float(np.max(linalg.norm(refined-end,axis=1))/source_rms))
        except ValueError as exc:
            geometric = str(exc) in ('flow_outside_geometric_support',
                'flow_tangent_projector_gap_lost','flow_invariant_frame_degenerate',
                'normal_graph_retraction_did_not_converge') or 'compact tangent-atlas support' in str(exc)
            if not constrained or not geometric or step==.03:raise
            trials.append(dict(step=step,exception=str(exc)))
            continue
        trials.append(dict(step=step,full_turn_relative=loops,
            refinement_relative=refinements))
        if not constrained or (max(loops)<GATES['full_turn_relative'] and
                max(refinements)<GATES['flow_refinement_relative']):break
    result['full_turn_relative']=loops
    result['full_turn_probe_count']=len(starts)
    if constrained:
        result['flow_step']=model['flow_step']
        result['flow_step_trials']=trials
        result['flow_step_refinement_relative']=refinements
        checks['flow_refinement']=max(refinements)<GATES['flow_refinement_relative']
    checks['full_turn']=max(loops)<GATES['full_turn_relative']
    if not all(checks.values()):return result
    reconstructed=core.inverse(model,model['phases'][take])
    reduced_error_squared=float(np.sum((reconstructed-model['X'][take])**2))
    discarded_error_squared=float(np.sum(model.get('source_projection_residual_squared',
        np.zeros(len(model['X'])))[take]))
    result['inverse_window_relative']=float(np.sqrt((reduced_error_squared+discarded_error_squared)/
        (len(take)*source_rms**2)))
    result['inverse_discarded_span_relative']=float(np.sqrt(discarded_error_squared/
        (len(take)*source_rms**2)))
    if model['r']>1:
        reverse=core.inverse(model,model['phases'][take],reverse=True)
        result['flow_order_relative']=float(linalg.norm(reverse-reconstructed)/
            (np.sqrt(len(take))*source_rms))
    else:result['flow_order_relative']=0.
    checks.update(inverse=result['inverse_window_relative']<GATES['inverse_window_relative'],
        flow_order=result['flow_order_relative']<GATES['flow_order_relative'])
    if constrained:result['flow_retraction']=dict(model['flow_retraction_diagnostics'])
    result['source_rms']=source_rms
    result['passed']=bool(all(checks.values()))
    return result


def _training_gates(model):
    """Try a geometry-constrained inverse only after ambient flows fail.

    All original tolerances are retained; the retry additionally checks the
    normal constraint, actual phase speed, eight starts, and step refinement.
    It is independent of sensor count, phase dimension and generator truth.
    """
    try:
        initial=_training_gates_once(model)
    except ValueError as exc:
        if str(exc) not in ('flow_outside_geometric_support',
                'flow_tangent_projector_gap_lost','flow_invariant_frame_degenerate'):
            raise
        initial=dict(passed=False,ambient_flow_exception=str(exc))
    if initial['passed'] or model.get('flow_geometry')=='normal_graph_constraint':
        return initial
    failed=[k for k,v in initial.get('checks',{}).items() if not v]
    if 'ambient_flow_exception' not in initial and not any(
            k in failed for k in ('inverse','full_turn')):
        return initial
    try:
        core.enable_geometry_constrained_flows(model)
        refined=_training_gates_once(model)
        refined['ambient_flow_attempt']=initial
        return refined
    except (ValueError,FloatingPointError,np.linalg.LinAlgError) as exc:
        initial['geometry_constrained_retry']=dict(passed=False,
            exception_type=type(exc).__name__,exception=str(exc))
        return initial

class SensorPipeline:
    """Fixed conditioning/(lag,neighborhood) search. Sole fit input is Y.

    The legacy persistent_count option cross-checks a local geometric dimension
    with persistent H1 under the torus model; it computes no circular coordinates.
    Integer period bases and phase construction follow W.
    """
    def __init__(self,dimension_strategy='persistent_count',neighbors=NEIGHBORS,
                 max_candidates=None):
        self.dimension_strategy=str(dimension_strategy)
        self.neighbors=tuple(int(k) for k in neighbors)
        self.max_candidates=None if max_candidates is None else int(max_candidates)
        if self.max_candidates is not None and self.max_candidates<1:
            raise ValueError('max_candidates must be positive')

    def fit(self,Y,progress_callback=None):
        began=time.time();y=np.asarray(Y,dtype=float)
        if y.ndim==1:y=y[:,None]
        self.status='fitting';self.model=None;self.attempts=[]
        if y.ndim!=2 or y.shape[1]<1 or not np.isfinite(y).all():
            raise ValueError('finite synchronized Y[times,channels] required')
        self.train_count=len(y);self.channels=y.shape[1]
        self.train_std=y.std(0);self.provenance=_hashes()
        if len(y)<192:
            self.status='unresolved_training_block_too_short';return self
        if np.max(self.train_std)<1e-12:
            self.status='unresolved_constant_readings';return self
        self.preprocess=fit_preprocess(y,lag=32,maxrank=24,trend_degrees=(0,))
        split=highpass_decompose(self.preprocess,y,window=129,polyorder=2)
        mask=split['valid_mask'];self.train_raw_indices=np.flatnonzero(mask)
        if not len(self.train_raw_indices):
            self.status='unresolved_training_block_too_short';return self
        self.train_residual=split['residual'][mask]
        self.preprocess_diagnostics=dict(rank=self.preprocess['rank'],
            cv_selected=self.preprocess['cv']['selected'],lag=32,window=129,polyorder=2,
            boundary_trim_each_side=split['boundary_trim_each_side'],
            retained_training_count=len(self.train_residual),
            scope=split['scope'])
        if progress_callback is not None:progress_callback(self.summary())
        attempted_candidates=0
        for lag in LAGS:
            print('PIPELINE_TRY_LAG',lag,flush=True)
            if lag*self.channels>128:
                attempt=dict(lag=lag,status='unresolved_resource_budget',
                    full_window_dimension=lag*self.channels,maximum_full_window_dimension=128,
                    resource_rule='conservative full mL cap; not a test of intrinsic dimension or observability')
                self.attempts.append(attempt)
                print('PIPELINE_LAG_RESULT',lag,attempt['status'],flush=True)
                continue
            for neighbors in self.neighbors:
                if self.max_candidates is not None and attempted_candidates>=self.max_candidates:
                    self.status='unresolved_candidate_budget_exhausted'
                    self.elapsed_fit_seconds=time.time()-began
                    if progress_callback is not None:progress_callback(self.summary())
                    return self
                attempted_candidates+=1
                print('PIPELINE_TRY_NEIGHBORS',neighbors,flush=True)
                attempt=dict(lag=lag,neighbors=neighbors,dimension_strategy=self.dimension_strategy)
                self.attempts.append(attempt)
                try:
                    candidate=core.fit(self.train_residual,lag=lag,neighbors=neighbors,
                        dimension_strategy=self.dimension_strategy)
                    attempt['diagnostics']=candidate['diagnostics']
                    attempt['period_complex_diagnostics']=candidate.get('complex',{}).get('diagnostics')
                    gate=_training_gates(candidate);attempt['gates']=gate
                    if candidate['r'] is None and self.dimension_strategy=='persistent_count':
                        attempt['neighbor_retry_skipped']='dimension diagram is independent of atlas-neighbor count'
                        print('PIPELINE_DIMENSION_UNRESOLVED',lag,flush=True)
                        attempt['completed']=True
                        if progress_callback is not None:progress_callback(self.summary())
                        break
                    if not gate['passed'] and candidate['diagnostics'].get('geometry_method')=='implicit_quadratic':
                        attempt['neighbor_retry_skipped']='validated global geometry is independent of local-neighbor count'
                        attempt['completed']=True
                        if progress_callback is not None:progress_callback(self.summary())
                        break
                    if gate['passed']:
                        self.model=candidate;self.status='accepted_conditionally_by_training_gates'
                        self.train_window_raw_indices=self.train_raw_indices[:len(candidate['X'])]
                        attempt['completed']=True
                        if progress_callback is not None:progress_callback(self.summary())
                        break
                except Exception as exc:
                    attempt.update(status='unresolved_candidate_exception',
                        exception_type=type(exc).__name__,exception=str(exc),
                        traceback=traceback.format_exc(limit=4))
                attempt['completed']=True
                if progress_callback is not None:progress_callback(self.summary())
                print('PIPELINE_CANDIDATE_RESULT',lag,neighbors,attempt.get('gates',attempt.get('exception')),flush=True)
            if self.model is not None:break
        if self.model is None:self.status='unresolved_no_lag_passed_training_gates'
        self.elapsed_fit_seconds=time.time()-began
        if progress_callback is not None:progress_callback(self.summary())
        return self

    def summary(self):
        return dict(status=self.status,preprocessing=getattr(self,'preprocess_diagnostics',None),
            attempts=self.attempts,selected_lag=None if self.model is None else self.model['lag'],
            dimension_strategy=self.dimension_strategy,neighbor_candidates=self.neighbors,
            maximum_candidates=self.max_candidates,
            gates=GATES,provenance=self.provenance,
            elapsed_fit_seconds=getattr(self,'elapsed_fit_seconds',None),
            scope=dict(inputs_are_readings_only=True,phase_initialization_used=False,
                regression_decoder_used=False,physical_trend_identified=False,
                reconstruction_mode='conditioning on observed query block; not forecasting',
                inverse_geometry='ambient field or training-gated normal-graph constraint; no endpoint closure target',
                acceptance_is_empirical_not_global_certificate=True,
                source_span_rule='training singular-gap selection with dimension and discarded-error checks; full-window inverse error includes discarded directions'))

    def transform(self,newY,start_index=0):
        if self.model is None:return dict(status='unresolved_fit',fit_status=self.status)
        y=np.asarray(newY,dtype=float)
        if y.ndim==1:y=y[:,None]
        if y.ndim!=2 or y.shape[1]!=self.channels or not np.isfinite(y).all():
            raise ValueError('query must have the fitted number of finite sensor channels')
        model=self.model;lag=model['lag']
        try:
            split=highpass_decompose(self.preprocess,y,start_index=start_index,window=129,polyorder=2)
            take=np.flatnonzero(split['valid_mask']);residual=split['residual'][take]
            if len(residual)<lag+1:return dict(status='unresolved_query_block_too_short')
            rawindices=split['time_index'][take].astype(int)
            fullx=windows((residual-model['mean'])/model['scale'],lag)/model['normal']
            xt=fullx@model['basis']
            projection=float(linalg.norm(fullx-xt@model['basis'].T)/max(linalg.norm(fullx),1e-15))
            diagnostic=dict(query_projection_relative=projection)
            if not np.isfinite(projection) or projection>GATES['query_projection_relative']:
                return dict(status='unresolved_query_projection',diagnostics=diagnostic)
            enc=core.encode(model,xt,derivative=True)
            diagnostic['encoder_coverage']=float(enc['covered'].mean())
            if not enc['covered'].all():return dict(status='unresolved_query_coverage',diagnostics=diagnostic)
            geom=core.evaluate_projector(model['atlas'],xt)
            diagnostic['minimum_atlas_gap']=float(geom['gap'].min())
            if geom['gap'].min()<=GATES['minimum_atlas_gap']:
                return dict(status='unresolved_query_atlas_gap',diagnostics=diagnostic)
            xhat=core.inverse(model,enc['phases'])
            rawhat=(xhat@model['basis'].T)*model['normal']*np.tile(model['scale'],lag)+np.tile(model['mean'],lag)
            overlap=rawhat[:-1,model['channels']:]-rawhat[1:,:-model['channels']]
            diagnostic['prefold_overlap_relative']=float(linalg.norm(overlap)/max(linalg.norm(rawhat),1e-15))
            diagnostic['query_residual_window_relative']=float(linalg.norm(rawhat-windows(residual,lag))/
                max(linalg.norm(windows(residual,lag)),1e-15))
            e=core.phase_field(model,xt)
            product=np.einsum('nap,npb->nab',enc['jacobian'],e)
            diagnostic['encoder_field_duality_rms']=float(linalg.norm(product-np.eye(model['r'])[None])/
                np.sqrt(len(xt)*model['r']))
            if model.get('flow_geometry')=='normal_graph_constraint':
                diagnostic['ambient_field_duality_rms']=diagnostic['encoder_field_duality_rms']
                diagnostic['encoder_field_duality_rms'],diagnostic['flow_duality_probe_count']=core.constrained_flow_duality(model,xt)
                diagnostic['flow_geometry']=model['flow_geometry']
                diagnostic['flow_retraction']=dict(model['flow_retraction_diagnostics'])
            diagnostic['phase_step_rmse_against_frozen_training_step']=float(np.sqrt(np.mean(
                np.angle(np.exp(1j*(np.diff(enc['phases'],axis=0)-model['diagnostics']['phase_step'])))**2)))
            diagnostic['flow_domain']=dict(model.get('flow_domain_diagnostics',{}))
            numeric=[v for v in diagnostic.values() if isinstance(v,(float,int,np.number))]
            if not _finite_diagnostics(numeric):
                return dict(status='unresolved_nonfinite_query_diagnostic',diagnostics=diagnostic)
            if diagnostic['query_residual_window_relative']>GATES['inverse_window_relative']:
                return dict(status='unresolved_query_inverse',diagnostics=diagnostic)
            if diagnostic['encoder_field_duality_rms']>GATES['duality_rms']:
                return dict(status='unresolved_query_duality',diagnostics=diagnostic)
            if diagnostic['phase_step_rmse_against_frozen_training_step']>GATES['phase_step_rmse']:
                return dict(status='unresolved_query_dynamics',diagnostics=diagnostic)
            if diagnostic['prefold_overlap_relative']>.1:
                return dict(status='unresolved_query_overlap',diagnostics=diagnostic)
            recovered_residual=fold(rawhat,model['channels'],lag)
            recovered=recovered_residual+split['slow'][take]
            return dict(status='transformed_conditionally',reconstruction=recovered,
                residual_reconstruction=recovered_residual,slow=split['slow'][take],
                conditioned_observations=split['denoised'][take],raw_time_indices=rawindices,
                phases=enc['phases'],window_raw_time_indices=rawindices[:len(xt)],
                diagnostics=diagnostic)
        except Exception as exc:
            return dict(status='unresolved_transform_exception',exception_type=type(exc).__name__,
                exception=str(exc),traceback=traceback.format_exc(limit=4))
