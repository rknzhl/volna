"""Generic observation-only sensor cases for the tangent-transport benchmark.

generate(case, n, seed) returns Y[times, channels], theta[times, hidden_r].
The theta output and observation_factory are evaluator material only: pass Y
and a chosen lag to model fitting, never theta or generator derivatives.
No circular/TDA coordinates, tangent labels, or hidden frequencies are fit inputs.
"""
import numpy as np
import sys,importlib.util
from pathlib import Path
from functools import lru_cache

CASE_SETTINGS = {
    'scalarcircle': dict(lag=4, ntrain=900, nevaluation=200),
    'scalar2torus': dict(lag=12, ntrain=2400, nevaluation=200),
    'nonlinear3sensor': dict(lag=4, ntrain=2400, nevaluation=200),
    'threephase6sensor': dict(r=3, lag=4, ntrain=2400, nevaluation=600),
    'threephase4sensor': dict(r=3, lag=4, ntrain=2400, nevaluation=600),
    'fundamental2waves': dict(lag=4, ntrain=2400, nevaluation=200),
    'opticalsteady1d_homogeneous': dict(lag=4, ntrain=2400, nevaluation=200),
    'opticalsteady1d_layered': dict(lag=4, ntrain=2400, nevaluation=200),
    'opticallegacy1d': dict(lag=4, ntrain=1200, nevaluation=200),
    'opticallegacy2d_lens': dict(lag=4, ntrain=800, nevaluation=200),
}

ALIASES = {
    'scalar_circle':'scalarcircle', 'scalar_torus':'scalar2torus',
    'scalar2torus':'scalar2torus', 'nonlinear3torus':'nonlinear3sensor',
    'nonlinear3sensor_torus':'nonlinear3sensor', 'two_waves':'fundamental2waves',
    'optical1d':'opticallegacy1d','optical_lens2d':'opticallegacy2d_lens',
}

def _case(case):
    value=ALIASES.get(case,case)
    if value not in CASE_SETTINGS: raise ValueError(f'Unknown sensor case {case!r}')
    return value

@lru_cache(None)
def _legacy_modules():
    """Load actual legacy sources without replacing repository/stdlib modules."""
    base=Path(__file__).resolve().parents[1]/'waves'/'synthetic'
    spec=importlib.util.spec_from_file_location('_takers_legacy_wave',base/'wave.py')
    wave=importlib.util.module_from_spec(spec);spec.loader.exec_module(wave)
    previous=sys.modules.get('wave');sys.modules['wave']=wave
    try:
        ss=importlib.util.spec_from_file_location('_takers_legacy_scenarios',base/'scenarios.py')
        scenarios=importlib.util.module_from_spec(ss);ss.loader.exec_module(scenarios)
    finally:
        if previous is None:sys.modules.pop('wave',None)
        else:sys.modules['wave']=previous
    return wave,scenarios

@lru_cache(None)
def optical_steady_response(case):
    """Frequency-domain steady response of the ORIGINAL 1D finite-difference solver.

    Same grid, material, source spreading and discrete absorbing boundary as
    legacy wave.solve; no transient simulation or fitted frequency estimates.
    Sampling uses .01*sqrt(3), avoiding the legacy rational first rotation.
    The complex response is generator truth, never a fitting input.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    case=_case(case)
    if not case.startswith('opticalsteady1d_'):raise ValueError('not a steady optical case')
    wave,sc=_legacy_modules();R,h=wave.grid(sc.GRID[1],1)
    kind='слоистая' if case.endswith('layered') else 'однородная'
    index=wave.medium(kind,R);c=1/index;c2=c*c;size=len(index)
    dt_out=.01*np.sqrt(3)
    every=int(np.ceil(dt_out/(.5*h/c.max())));dt=dt_out/every
    sensors=wave.lattice(np.linspace(.4,.75,8),1)
    source=sc.point(.25,1);responses=[];residuals=[]
    for nu,amp in zip(sc.NU,sc.AMP):
        omega=2*np.pi*nu;z=np.exp(1j*omega*dt)
        ratio=dt*dt*c2/(h*h)
        main=(z-2+1/z)+2*ratio
        matrix=sparse.diags((-ratio[1:],main,-ratio[:-1]),(-1,0,1),format='lil',dtype=complex)
        # wave.absorb(new,old,courant) overwrites both boundary rows.
        courant=c*dt/h
        matrix[0,:]=0;matrix[0,0]=z-1+courant[0];matrix[0,1]=-courant[0]
        matrix[-1,:]=0;matrix[-1,-1]=z-1+courant[-1];matrix[-1,-2]=-courant[-1]
        rhs=np.zeros(size,dtype=complex)
        for node,weight in wave.spread(source,h):rhs[node[0]]+=dt*dt*weight*amp*omega/h
        matrix=matrix.tocsr();response=spsolve(matrix,rhs)
        residuals.append(float(np.linalg.norm(matrix@response-rhs)/np.linalg.norm(rhs)))
        responses.append(np.array([response[wave.node(p,h)] for p in sensors]).ravel())
    return np.column_stack(responses),dict(source_files=['code/waves/synthetic/wave.py','code/waves/synthetic/scenarios.py'],
        construction='steady frequency-domain solution of the same 1D discrete PDE and absorbing boundaries',
        material=kind,grid_size=size,dt_out=dt_out,dt_internal=dt,
        relative_resolvent_residuals=residuals,transients_removed_by_construction=True,
        discrete_forcing_nonresonant=True)

def _legacy_generate(case,n,seed):
    wave,sc=_legacy_modules();dimension=2 if case=='opticallegacy2d_lens' else 1
    scenario=dict(d=dimension)
    if dimension==2:scenario['medium']='линза'
    # Extend the legacy source horizon when a longer record is requested;
    # all solver physics/grid/ramp/forcing/default sampling remain unchanged.
    previous_duration=sc.DURATION[dimension]
    sc.DURATION[dimension]=max(previous_duration,(n+3)*.01)
    try:run=sc.simulate(scenario,seed=seed,t_end=sc.START+(n+2)*.01)
    finally:sc.DURATION[dimension]=previous_duration
    Y=run['Y'].T[:n];times=run['t'][:n]+sc.START
    theta=2*np.pi*times[:,None]*np.asarray(sc.NU)[None]
    if len(Y)!=n:raise RuntimeError('legacy simulator returned fewer requested observations')
    return Y,theta

def observation_factory(case, seed=31):
    """Evaluator callable theta[...,r] -> readings[...,m], with arbitrary phases.

    A phase vector gives one sensor vector; an array of phase vectors gives
    one sensor vector per row. A 1D batch is also accepted for scalarcircle.
    """
    case=_case(case)
    if case.startswith('opticallegacy'):
        raise NotImplementedError('Legacy PDE has transient state; no exact static observation(theta) factory. generate returns forcing-phase references only.')
    if case.startswith('opticalsteady'):
        response,_=optical_steady_response(case)
        def observe_steady(theta):
            t=np.asarray(theta,dtype=float)
            if t.ndim==0 or t.shape[-1]!=2:raise ValueError('steady optics requires theta[...,2]')
            return np.real(np.einsum('...r,mr->...m',np.exp(1j*t),response))
        return observe_steady
    if case=='fundamental2waves':
        rng=np.random.default_rng(seed)
        positions=rng.uniform(0,1,(12,2))
        wavevectors=2*np.pi*np.array([[1.,.5],[-.4,1.1]])
        offsets=positions@wavevectors.T
    def observe(theta):
        t=np.asarray(theta,dtype=float)
        if case=='scalarcircle':
            if t.ndim==0: t=t.reshape(1)
            elif t.ndim==1 and t.shape[0]!=1: t=t[:,None]
            if t.shape[-1]!=1: raise ValueError('scalarcircle requires theta[...,1]')
            a=t[...,0]
            return (np.exp(.4*np.cos(a))+.25*np.sin(a)+.15*np.cos(2*a))[...,None]
        if case in ('threephase6sensor','threephase4sensor'):
            if t.ndim==0 or t.shape[-1]!=3:raise ValueError(f'{case} requires theta[...,3]')
            if case=='threephase6sensor':
                return np.stack((np.cos(t),np.sin(t)),axis=-1).reshape(t.shape[:-1]+(6,))
            a,b,c=t[...,0],t[...,1],t[...,2]
            # Positive radii identify a,b; their two deviations identify c.
            radius_a=2+np.cos(c);radius_b=2+np.sin(c)
            return np.stack((radius_a*np.cos(a),radius_a*np.sin(a),
                radius_b*np.cos(b),radius_b*np.sin(b)),axis=-1)
        if t.ndim==0 or t.shape[-1]!=2: raise ValueError('torus cases require theta[...,2]')
        a,b=t[...,0],t[...,1]
        if case=='scalar2torus':
            return (np.cos(a)+.65*np.cos(b)+.15*np.sin(a+b))[...,None]
        if case=='nonlinear3sensor':
            return np.stack((np.exp(.3*np.cos(a))+.2*np.sin(b),
                np.tanh(np.sin(a)+.3*np.cos(b)),np.cos(b)+.2*np.sin(2*a)),axis=-1)
        return np.cos(a[...,None]-offsets[:,0])+.6*np.cos(b[...,None]-offsets[:,1])
    return observe

def generate(case,n,seed=31):
    """Return observations plus separated evaluator truth; no noise is added."""
    case=_case(case)
    if isinstance(n,bool) or int(n)!=n or n<1: raise ValueError('n must be a positive integer')
    if case.startswith('opticallegacy'):return _legacy_generate(case,int(n),seed)
    r=CASE_SETTINGS[case].get('r',1 if case=='scalarcircle' else 2)
    # A seed changes orbit origin; for the wave case it also changes sensors.
    rng=np.random.default_rng(int(seed)+12345)
    origin=rng.uniform(0,2*np.pi,r)
    if case.startswith('opticalsteady'):
        _,legacy=_legacy_modules();increments=2*np.pi*np.asarray(legacy.NU)*(.01*np.sqrt(3))
    else:increments=np.array([.41*np.sqrt(2),.27*np.sqrt(3),.37*np.sqrt(5)])[:r]
    theta=origin+np.arange(int(n))[:,None]*increments
    Y=observation_factory(case,seed)(theta)
    return np.asarray(Y,dtype=float),theta

def generate_variant(case,n,seed=31,*,noise=0.,trend=None,trend_strength=.3,
                     phase_mode='rigid',rate_profile='default',train_count=None):
    """Return observed Y plus a SEPARATE evaluator dictionary.

    noise is sigma_noise / per-channel clean-signal TRAIN standard deviation:
    .01/.05/.10 correspond to nominal 40/26.0206/20 dB SNR. Trend amplitude
    is its TRAIN standard deviation divided by clean signal TRAIN std.
    No preprocessing, detrending, denoising, or model fit occurs here.
    """
    case=_case(case);n=int(n)
    if noise<0:raise ValueError('noise must be nonnegative')
    count=min(int(train_count or CASE_SETTINGS[case]['ntrain']),n)
    if case.startswith('optical'):
        if phase_mode!='rigid' or rate_profile!='default':raise ValueError('optical sampling/forcing is fixed by its explicitly named generator')
        clean,theta=generate(case,n,seed)
    else:
        r=CASE_SETTINGS[case].get('r',1 if case=='scalarcircle' else 2)
        profiles={'default':[.41*np.sqrt(2),.27*np.sqrt(3),.37*np.sqrt(5)],
            'wide':[.17*np.sqrt(3),.31*np.sqrt(2),.29*np.sqrt(5)],
            'slow':[.2*.41*np.sqrt(2),.2*.27*np.sqrt(3),.2*.37*np.sqrt(5)],
            'near_resonant':[.43,.86+1e-3*np.sqrt(2),1.29+1e-3*np.sqrt(3)]}
        if rate_profile not in profiles:raise ValueError('unknown rate profile')
        alpha=np.asarray(profiles[rate_profile])[:r]
        origin=np.random.default_rng(seed+12345).uniform(0,2*np.pi,r)
        time=np.arange(n,dtype=float)
        if phase_mode=='rigid':theta=origin+time[:,None]*alpha
        elif phase_mode=='chirp':theta=origin+time[:,None]*alpha+.2*time[:,None]**2/max(count,1)*alpha
        elif phase_mode=='coupled':
            if r!=2:raise ValueError('coupled probe requires two phases')
            theta=np.empty((n,r));theta[0]=origin
            def velocity(z):return alpha+.09*np.array([np.sin(z[1]-z[0]),np.sin(z[0]-z[1])])
            for i in range(n-1):
                z=theta[i];a=velocity(z);b=velocity(z+a/2);c=velocity(z+b/2);d=velocity(z+c)
                theta[i+1]=z+(a+2*b+2*c+d)/6
        else:raise ValueError('unknown phase mode')
        clean=observation_factory(case,seed)(theta)
    signal_std=clean[:count].std(axis=0)
    trend_array=np.zeros_like(clean)
    if trend is not None:
        tt=np.arange(n,dtype=float)/max(count,1);rng=np.random.default_rng(seed+20003)
        if trend=='linear':shape=np.repeat(tt[:,None],clean.shape[1],axis=1)
        elif trend=='quadratic':shape=np.repeat((tt**2)[:,None],clean.shape[1],axis=1)
        elif trend in ('smooth','smooth_slow'):
            phase=rng.uniform(0,2*np.pi,clean.shape[1])
            shape=np.sin(2*np.pi*tt[:,None]/8+phase)+.3*np.cos(2*np.pi*tt[:,None]/13-phase)
        else:raise ValueError('unknown trend')
        shape-=shape[:count].mean(axis=0)
        shape/=np.maximum(shape[:count].std(axis=0),1e-12)
        signs=rng.choice([-1.,1.],clean.shape[1])
        trend_array=shape*(trend_strength*signal_std*signs)
    sigma=noise*signal_std
    perturbation=np.random.default_rng(seed+40001).normal(size=clean.shape)*sigma
    total=clean+trend_array;observed=total+perturbation
    rigid=(phase_mode=='rigid' and trend is None and not case.startswith('opticallegacy'))
    truth=dict(theta=theta,clean_signal=clean,trend=trend_array,clean_total=total,
        noise=perturbation,sigma_noise=sigma,signal_train_std=signal_std,
        nominal_snr_db=None if noise==0 else float(-20*np.log10(noise)),
        nominal_snr_label='infinite (no noise)' if noise==0 else f'{-20*np.log10(noise):.6g} dB',
        metadata=dict(case=case,seed=seed,noise_level=noise,trend=trend,
            trend_strength=trend_strength,phase_mode=phase_mode,rate_profile=rate_profile,
            sigma_calibration_count=count,rigid_rotation_clean_model=rigid,
            regime='positive noiseless model' if rigid and noise==0 else 'noise robustness' if rigid else 'outside stationary rigid-torus model / stress',
            phase_truth_scope='source forcing reference only; transient PDE state not captured' if case.startswith('opticallegacy') else
                'nonuniform coupled generator angles, not rigid uniform phases; GLZ phase matching alone is not a correctness test' if phase_mode=='coupled' else
                'generator phases; use only after model freeze',
            legacy_rational_sampling_resonance=bool(case.startswith('opticallegacy'))))
    return observed,truth

def benchmark_matrix(include_expensive_2d=False):
    """Small fixed diverse schedule, not a Cartesian explosion of settings."""
    rows=[]
    def add(case,seed=31,**kw):
        rows.append(dict(case=case,seed=seed,**CASE_SETTINGS[case],generator_kwargs=kw))
    add('scalarcircle',47,rate_profile='wide')
    add('scalar2torus')
    add('nonlinear3sensor',47)
    add('fundamental2waves',47,rate_profile='wide')
    add('opticalsteady1d_homogeneous')
    add('opticalsteady1d_layered',47)
    add('opticallegacy1d')
    for level in [.01,.05,.1]:add('nonlinear3sensor',noise=level)
    add('scalarcircle',trend='linear',trend_strength=.3)
    add('nonlinear3sensor',trend='quadratic',trend_strength=.3)
    add('scalar2torus',trend='smooth_slow',trend_strength=.3)
    add('scalar2torus',47,phase_mode='chirp')
    add('nonlinear3sensor',47,phase_mode='coupled')
    add('scalar2torus',47,rate_profile='near_resonant')
    if include_expensive_2d:add('opticallegacy2d_lens')
    return rows

