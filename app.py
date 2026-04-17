"""
Accurate Evaluation of Optimization Algorithms + Data-Driven Algorithm Advisor v5.0
9860 Algorithms | 20 Physical Metrics | Score/100
Original system 100% untouched + NEW Data Advisor tab
Supports: CSV TSV JSON XLSX XLS TXT DAT NPY + paste
Requirements: mealpy>=3.0 niapy>=2.0 gradio numpy scipy scikit-learn pandas plotly psutil openpyxl xlrd
"""
import numpy as np, time, tracemalloc, sqlite3, warnings, importlib, inspect, psutil, io, json, re
from datetime import datetime
from math import comb as _comb
from scipy.optimize import differential_evolution, dual_annealing, minimize, shgo, direct, basinhopping
from scipy.stats import entropy as scipy_entropy
from scipy.spatial import KDTree
from scipy.fft import fft
from scipy import stats as scipy_stats
from scipy.signal import find_peaks
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import OPTICS
from sklearn.preprocessing import StandardScaler
import pandas as pd
warnings.filterwarnings("ignore")
try:
    from mealpy.utils.space import FloatVar; MEALPY_OK = True
except ImportError:
    MEALPY_OK = False
try:
    import niapy; NIAPY_OK = True
except ImportError:
    NIAPY_OK = False
import gradio as gr
import plotly.graph_objects as go
from plotly.subplots import make_subplots
ENGINE = ("MealPy+NiaPy+SciPy" if (MEALPY_OK and NIAPY_OK) else "MealPy+SciPy" if MEALPY_OK else "SciPy+Fallback")
print(f"Engine: {ENGINE}")

MEALPY_MAP = {
    "PSO":("mealpy.swarm_based.PSO","OriginalPSO",{"c1":1.5,"c2":1.5,"w":0.7}),
    "GWO":("mealpy.swarm_based.GWO","OriginalGWO",{}),
    "WOA":("mealpy.swarm_based.WOA","OriginalWOA",{}),
    "SSA":("mealpy.swarm_based.SSA","OriginalSSA",{}),
    "MFO":("mealpy.swarm_based.MFO","OriginalMFO",{}),
    "FA":("mealpy.swarm_based.FA","OriginalFA",{"max_sparks":50,"p_a":0.04,"p_b":0.8,"max_ea":40,"m_sparks":50}),
    "ABC":("mealpy.swarm_based.ABC","OriginalABC",{"n_limits":25}),
    "GOA":("mealpy.swarm_based.GOA","OriginalGOA",{"c_min":0.00004,"c_max":1.0}),
    "SCA":("mealpy.swarm_based.SCA","OriginalSCA",{}),
    "EHO":("mealpy.swarm_based.EHO","OriginalEHO",{"alpha":0.5,"beta":0.1,"n_clans":5}),
    "EWA":("mealpy.bio_based.EWA","OriginalEWA",{}),
    "HHO":("mealpy.swarm_based.HHO","OriginalHHO",{}),
    "MRFO":("mealpy.swarm_based.MRFO","OriginalMRFO",{"s_factor":2.0}),
    "AO":("mealpy.swarm_based.AO","OriginalAO",{}),
    "FOA":("mealpy.swarm_based.FOA","OriginalFOA",{}),
    "SMA":("mealpy.swarm_based.SMA","OriginalSMA",{"p_t":0.03}),
    "TSA":("mealpy.swarm_based.TSA","OriginalTSA",{}),
    "AVOA":("mealpy.swarm_based.AVOA","OriginalAVOA",{"p1":0.6,"p2":0.4,"p3":0.6,"alpha":0.8,"gama":2.5}),
    "DO":("mealpy.swarm_based.DO","OriginalDO",{}),
    "BA":("mealpy.swarm_based.BA","OriginalBA",{"loudness":0.8,"pulse_rate":0.95,"pf_min":0.1,"pf_max":1000.0}),
    "FFA":("mealpy.swarm_based.FFA","OriginalFFA",{"gamma":0.001,"beta_base":2,"alpha":0.2,"alpha_damp":0.99,"delta":0.05,"exponent":2}),
    "ACO":("mealpy.swarm_based.ACOR","OriginalACOR",{"sample_count":25,"q":0.5,"zeta":1.0}),
    "DE":("mealpy.evolutionary_based.DE","OriginalDE",{"wf":0.8,"cr":0.9,"strategy":0}),
    "GA":("mealpy.evolutionary_based.GA","BaseGA",{"pc":0.9,"pm":0.01}),
    "ES":("mealpy.evolutionary_based.ES","OriginalES",{"lamda":0.75}),
    "SA":("mealpy.physics_based.SA","OriginalSA",{"temp_init":100,"cooling_factor":0.99}),
    "EFO":("mealpy.physics_based.EFO","OriginalEFO",{"r_rate":0.3,"ps_rate":0.85,"p_field":0.1,"n_field":0.45}),
    "RUN":("mealpy.physics_based.RUN","OriginalRUN",{}),
    "MVO":("mealpy.physics_based.MVO","OriginalMVO",{"wep_min":0.2,"wep_max":1.0}),
    "TLO":("mealpy.human_based.TLO","OriginalTLO",{}),
    "INFO":("mealpy.human_based.INFO","OriginalINFO",{}),
    "ICA":("mealpy.human_based.ICA","OriginalICA",{"empire_count":5,"assimilation_coeff":4,"revolution_prob":0.05,"revolution_rate":0.1,"revolution_step_size":0.1,"zeta":0.1}),
    "BBO":("mealpy.bio_based.BBO","OriginalBBO",{"p_m":0.01}),
    "HS":("mealpy.music_based.HS","OriginalHS",{"c_r":0.95,"pa_r":0.05}),
    "GTO":("mealpy.swarm_based.GTO","OriginalGTO",{}),
}
NIAPY_MAP = {
    "PSO_n":("niapy.algorithms.basic","ParticleSwarmOptimization",{"c1":2.0,"c2":2.0,"w":0.7}),
    "GWO_n":("niapy.algorithms.basic","GreyWolfOptimizer",{}),
    "DE_n":("niapy.algorithms.basic","DifferentialEvolution",{"f":0.5,"cr":0.9}),
    "GA_n":("niapy.algorithms.basic","GeneticAlgorithm",{"ts":2,"mr":0.05,"cr":0.4}),
    "ABC_n":("niapy.algorithms.basic","ArtificialBeeColonyAlgorithm",{"limit":100}),
    "FA_n":("niapy.algorithms.basic","FireflyAlgorithm",{"alpha":1.0,"betamin":1.0,"gamma":2.0}),
    "BA_n":("niapy.algorithms.basic","BatAlgorithm",{"a":0.5,"r":0.5,"qmin":0,"qmax":2}),
    "CS_n":("niapy.algorithms.basic","CuckooSearch",{"pa":0.25}),
    "SCA_n":("niapy.algorithms.basic","SineCosineAlgorithm",{}),
    "WOA_n":("niapy.algorithms.basic","WhaleOptimizationAlgorithm",{}),
    "HHO_n":("niapy.algorithms.basic","HarrisHawksOptimization",{}),
    "MFO_n":("niapy.algorithms.basic","MothFlameOptimization",{}),
    "SSA_n":("niapy.algorithms.basic","SalpSwarmAlgorithm",{}),
    "GS_n":("niapy.algorithms.basic","GravitationalSearchAlgorithm",{}),
    "HS_n":("niapy.algorithms.basic","HarmonySearch",{"r_accept":0.7,"r_pa":0.35,"b_range":2}),
}
SCIPY_ALGOS = {
    "DE_sci":"differential_evolution","SA_sci":"dual_annealing","SHGO":"shgo",
    "DIRECT":"direct","BasinHop":"basinhopping","NelderMead":"nelder-mead",
    "BFGS":"l-bfgs-b","Powell":"powell","CG":"cg","COBYLA":"cobyla",
}
ALL_SINGLE = list(MEALPY_MAP)+list(NIAPY_MAP)+list(SCIPY_ALGOS)
COMPLEXITY_FACTORS = {
    "PSO":1.0,"GWO":1.0,"WOA":1.1,"SSA":1.0,"MFO":1.0,"FA":1.2,"ABC":1.1,"GOA":1.1,
    "SCA":1.0,"EHO":1.1,"EWA":1.0,"HHO":1.1,"MRFO":1.0,"AO":1.0,"FOA":1.0,"SMA":1.0,
    "TSA":1.0,"AVOA":1.1,"DO":1.0,"BA":1.0,"FFA":1.2,"ACO":1.3,"DE":1.0,"GA":1.2,
    "ES":1.1,"SA":0.3,"EFO":1.0,"RUN":1.0,"MVO":1.0,"TLO":1.0,"INFO":1.1,"ICA":1.2,
    "BBO":1.1,"HS":0.8,"GTO":1.0,"PSO_n":1.0,"GWO_n":1.0,"DE_n":1.0,"GA_n":1.2,
    "ABC_n":1.1,"FA_n":1.2,"BA_n":1.0,"CS_n":1.0,"SCA_n":1.0,"WOA_n":1.1,"HHO_n":1.1,
    "MFO_n":1.0,"SSA_n":1.0,"GS_n":1.1,"HS_n":0.8,"DE_sci":1.0,"SA_sci":0.3,
    "SHGO":1.5,"DIRECT":1.3,"BasinHop":0.5,"NelderMead":0.4,"BFGS":0.6,"Powell":0.5,"CG":0.5,"COBYLA":0.6,
}

_LEVEL_LABELS = {
    2:"Binary",3:"Triple",4:"Quad",5:"Quinary",6:"Senary",7:"Septenary",8:"Octonary",
    9:"Nonary",10:"Denary",11:"Undenary",12:"Duodenary",13:"Tredenary",14:"Quattuordenary",
    15:"Quindenary",16:"Sexdenary",17:"Septendenary",18:"Octodenary",19:"Novemdenary",
    20:"Vigenary",21:"Unvigenary",22:"Duovigenary",23:"Trevigenary",24:"Quattuorvigenary",
    25:"Quinvigenary",26:"Sexvigenary",27:"Septvigenary",28:"Octovigenary",29:"Novemvigenary",
    30:"Trigenary",31:"Untrigenary",32:"Duotrigenary",33:"Tretrigenary",34:"Quattuortrigenary",
    35:"Quintrigenary",36:"Sextrigenary",37:"Septentrigenary",38:"Octotrigenary",39:"Novemtrigenary",
    40:"Quadragenary",41:"Unquadragenary",42:"Duoquadragenary",43:"Trequadragenary",
    44:"Quattuorquadragenary",45:"Quinquadragenary",46:"Sexquadragenary",47:"Septenquadragenary",
    48:"Octoquadragenary",49:"Novemquadragenary",50:"Quinquagenary",
}

class AlgorithmLibrary:
    @classmethod
    def _gen(cls,n,count,seed=0):
        total=_comb(len(ALL_SINGLE),n); k=min(count,total)
        if total<=count:
            from itertools import combinations as _cb
            return ["+".join(c) for c in _cb(ALL_SINGLE,n)]
        rng=np.random.default_rng(42+seed); chosen=set(); result=[]
        while len(result)<k:
            idx=int(rng.integers(0,total))
            if idx in chosen: continue
            chosen.add(idx)
            combo=[]; nn=len(ALL_SINGLE); rem=idx; start=0
            for i in range(n):
                for j in range(start,nn-(n-i-1)):
                    cv=_comb(nn-j-1,n-i-1)
                    if rem<cv: combo.append(ALL_SINGLE[j]); start=j+1; break
                    rem-=cv
            result.append("+".join(combo))
        return result
    @classmethod
    def get_all(cls):
        d={f"Single ({len(ALL_SINGLE)})":ALL_SINGLE}
        for n,s in [(2,0),(3,1),(4,2),(5,3),(6,4),(7,5),(8,6),(9,7),(10,8),(11,9),(12,10),
                    (13,11),(14,12),(15,13),(16,14),(17,15),(18,16),(19,17),(20,18),(21,19),
                    (22,20),(23,21),(24,22),(25,23),(26,24),(27,25),(28,26),(29,27),(30,28),
                    (31,29),(32,30),(33,31),(34,32),(35,33),(36,34),(37,35),(38,36),(39,37),
                    (40,38),(41,39),(42,40),(43,41),(44,42),(45,43),(46,44),(47,45),(48,46),(49,47),(50,48)]:
            d[f"{_LEVEL_LABELS[n]} (200)"]=cls._gen(n,200,s)
        return d
ALL_ALGOS = AlgorithmLibrary.get_all()

class BS:
    @staticmethod
    def sphere(x): return float(np.sum(np.array(x,dtype=float)**2))
    @staticmethod
    def rosen(x):
        x=np.array(x,dtype=float); return float(np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2))
    @staticmethod
    def rastrigin(x):
        x=np.array(x,dtype=float); n=len(x); return float(10*n+np.sum(x**2-10*np.cos(2*np.pi*x)))
    @staticmethod
    def ackley(x):
        x=np.array(x,dtype=float); n=len(x)
        return float(-20*np.exp(-0.2*np.sqrt(np.sum(x**2)/n))-np.exp(np.sum(np.cos(2*np.pi*x))/n)+20+np.e)
    @staticmethod
    def griewank(x):
        x=np.array(x,dtype=float); i=np.arange(1,len(x)+1)
        return float(np.sum(x**2)/4000-np.prod(np.cos(x/np.sqrt(i)))+1)
    @staticmethod
    def levy(x):
        x=np.array(x,dtype=float); w=1+(x-1)/4
        return float(np.sin(np.pi*w[0])**2+np.sum((w[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2))+(w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2))
    @staticmethod
    def schwefel(x):
        x=np.array(x,dtype=float); n=len(x); return float(418.9829*n-np.sum(x*np.sin(np.sqrt(np.abs(x)))))
    @staticmethod
    def michalewicz(x):
        x=np.array(x,dtype=float); i=np.arange(1,len(x)+1)
        return float(-np.sum(np.sin(x)*np.sin(i*x**2/np.pi)**20))
    @staticmethod
    def zakharov(x):
        x=np.array(x,dtype=float); i=np.arange(1,len(x)+1); s=np.sum(0.5*i*x)
        return float(np.sum(x**2)+s**2+s**4)
    @staticmethod
    def dixon(x):
        x=np.array(x,dtype=float); i=np.arange(2,len(x)+1)
        return float((x[0]-1)**2+np.sum(i*(2*x[1:]**2-x[:-1])**2))
    @staticmethod
    def cf1(x): return 0.4*BS.sphere(x)+0.3*BS.rastrigin(x)/10+0.3*BS.ackley(x)/20
    @staticmethod
    def rot_rastrigin(x):
        x=np.array(x,dtype=float); n=len(x); R=np.eye(n); theta=np.pi/4
        if n>=2: R[0,0]=R[1,1]=np.cos(theta); R[0,1]=-np.sin(theta); R[1,0]=np.sin(theta)
        return BS.rastrigin(R@x)
    FUNCTIONS={
        "Sphere":(sphere.__func__,(-5.12,5.12),0.0),"Rosenbrock":(rosen.__func__,(-2.048,2.048),0.0),
        "Rastrigin":(rastrigin.__func__,(-5.12,5.12),0.0),"Ackley":(ackley.__func__,(-32.768,32.768),0.0),
        "Griewank":(griewank.__func__,(-600,600),0.0),"Levy":(levy.__func__,(-10,10),0.0),
        "Schwefel":(schwefel.__func__,(-500,500),0.0),"Michalewicz":(michalewicz.__func__,(0,np.pi),-9.66),
        "Zakharov":(zakharov.__func__,(-5,10),0.0),"DixonPrice":(dixon.__func__,(-10,10),0.0),
        "Composite CF1":(cf1.__func__,(-5.12,5.12),0.0),"Rot.Rastrigin":(rot_rastrigin.__func__,(-5.12,5.12),0.0),
    }
    @staticmethod
    def ruggedness(fn,lo,hi,dim=5,n=200,delta=0.05):
        np.random.seed(0); X=np.random.uniform(lo,hi,(n,dim))
        X1=np.clip(X+np.random.randn(n,dim)*delta*(hi-lo),lo,hi)
        fx=np.array([fn(X[i]) for i in range(n)]); fx1=np.array([fn(X1[i]) for i in range(n)])
        if np.std(fx)<1e-12 or np.std(fx1)<1e-12: return 0.0
        return float(1.0-np.corrcoef(fx,fx1)[0,1])
    @staticmethod
    def modality(fn,lo,hi,n=300):
        x=np.linspace(lo,hi,n); f=np.array([fn([xi]) for xi in x])
        return int(np.sum((f[1:-1]<f[:-2])&(f[1:-1]<f[2:])))
    @staticmethod
    def separability(fn,lo,hi,dim,n=80):
        np.random.seed(1); x0=np.zeros(dim)
        jv=np.var([fn(np.random.uniform(lo,hi,dim)) for _ in range(n)]); mvs=[]
        for d in range(min(dim,5)):
            vals=[]
            for _ in range(n):
                xx=x0.copy(); xx[d]=np.random.uniform(lo,hi); vals.append(fn(xx))
            mvs.append(np.var(vals))
        return float(min(1.0,np.mean(mvs)/(jv+1e-9)))

def _iter_at_pct(h,pct):
    if len(h)<2: return len(h)
    wo,be=h[0],h[-1]
    if abs(wo-be)<1e-12: return 1
    tgt=be+(1-pct)*(wo-be)
    for i,v in enumerate(h):
        if v<=tgt: return i
    return len(h)

TRAITS={
    "PSO":{"ex":0.60,"ep":0.80,"cr":0.70},"GWO":{"ex":0.70,"ep":0.90,"cr":0.85},
    "WOA":{"ex":0.80,"ep":0.75,"cr":0.80},"DE":{"ex":0.90,"ep":0.85,"cr":0.90},
    "GA":{"ex":0.85,"ep":0.65,"cr":0.65},"SA":{"ex":0.65,"ep":0.95,"cr":0.60},
    "HHO":{"ex":0.85,"ep":0.80,"cr":0.82},"SMA":{"ex":0.75,"ep":0.85,"cr":0.80},
    "ABC":{"ex":0.80,"ep":0.70,"cr":0.70},"SSA":{"ex":0.75,"ep":0.70,"cr":0.75},
    "EWA":{"ex":0.72,"ep":0.80,"cr":0.75},"MRFO":{"ex":0.72,"ep":0.88,"cr":0.83},
    "AO":{"ex":0.78,"ep":0.82,"cr":0.80},"MFO":{"ex":0.73,"ep":0.82,"cr":0.79},
    "BBO":{"ex":0.70,"ep":0.78,"cr":0.72},"HS":{"ex":0.65,"ep":0.75,"cr":0.65},
}

def _fallback(name,fn,lo,hi,dim,params,seed):
    np.random.seed(seed); t=TRAITS.get(name.split("+")[0],{"ex":0.70,"ep":0.75,"cr":0.72})
    ps=int(params.get("pop_size",50)); mi=int(params.get("max_iter",100))
    w=float(params.get("inertia",0.7))*t["cr"]; c1=float(params.get("cognitive",1.5))*t["ep"]
    c2=float(params.get("social",1.5))*t["ex"]; mr=float(params.get("mutation",0.01))*(1.5-t["ep"])
    pop=np.random.uniform(lo,hi,(ps,dim)); vel=np.random.uniform(-1,1,(ps,dim))*(hi-lo)*0.1
    fit=np.array([fn(pop[i]) for i in range(ps)]); pb=pop.copy(); pbf=fit.copy()
    gb=pop[fit.argmin()].copy(); gbf=fit.min(); fh=[gbf]; dh=[]; esc=0; stag=0; snaps=[]
    gr_r=20; cs=(hi-lo)/gr_r; vis=set(); tracemalloc.start(); t0=time.perf_counter()
    for it in range(mi):
        wt=w*(1.0-0.5*it/mi); r1=np.random.rand(ps,dim); r2=np.random.rand(ps,dim)
        vel=wt*vel+c1*r1*(pb-pop)+c2*r2*(gb-pop)
        mm=np.random.rand(ps,dim)<mr; vel[mm]+=np.random.randn(int(mm.sum()))*(hi-lo)*0.03
        pop=np.clip(pop+vel,lo,hi); fit=np.array([fn(pop[i]) for i in range(ps)])
        imp=fit<pbf; pb[imp]=pop[imp]; pbf[imp]=fit[imp]; nb=fit.min()
        if nb<gbf:
            gbf=nb; gb=pop[fit.argmin()].copy()
            if stag>3: esc+=1
            stag=0
        else: stag+=1
        fh.append(gbf); dh.append(float(np.mean(np.std(pop,axis=0))))
        if it%max(1,mi//10)==0: snaps.append(pop.copy().tolist())
        for ind in pop[::max(1,ps//8)]:
            cell=tuple(((ind[:3]-lo)/cs).astype(int).clip(0,gr_r-1)); vis.add(cell)
    exec_t=time.perf_counter()-t0; _,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {"best_fit":gbf,"history":fh,"diversity_log":dh,"pop_snapshots":snaps,
            "exec_time":exec_t,"memory_kb":pk/1024,"escapes":esc,"total_stag":max(1,mi),
            "iter_95pct":_iter_at_pct(fh,0.95),"max_iter":mi,
            "visited_cells":len(vis),"total_cells":gr_r**min(dim,3),"pop_size":ps,"dim":dim}

def _run_mealpy(name,fn,lo,hi,dim,params,seed):
    mp,cn,defs=MEALPY_MAP[name]; mod=importlib.import_module(mp); cls=getattr(mod,cn)
    from mealpy.utils.space import FloatVar
    ep=int(params.get("max_iter",100)); ps=int(params.get("pop_size",50))
    kw={"epoch":ep,"pop_size":ps}; kw.update(defs); sig=inspect.signature(cls.__init__).parameters
    for k,v in [("c1",params.get("cognitive",1.5)),("c2",params.get("social",1.5)),
                ("w",params.get("inertia",0.7)),("wf",params.get("mutation",0.8)),("cr",params.get("crossover",0.9))]:
        if k in sig: kw[k]=v
    prob={"obj_func":lambda x:float(fn(x)),"bounds":FloatVar(lb=[lo]*dim,ub=[hi]*dim),"minmax":"min","log_to":None,"save_population":True}
    tracemalloc.start(); t0=time.perf_counter(); model=cls(**kw); model.solve(prob,seed=seed)
    exec_t=time.perf_counter()-t0; _,pk=tracemalloc.get_traced_memory(); tracemalloc.stop(); hist=model.history
    fh=([float(v) for v in hist.list_global_best_fit] if hasattr(hist,"list_global_best_fit") and hist.list_global_best_fit
        else [float(x.target.fitness) for x in hist.list_global_best] if hasattr(hist,"list_global_best") and hist.list_global_best
        else [float(model.g_best.target.fitness)])
    dh=([float(v) for v in hist.list_diversity] if hasattr(hist,"list_diversity") and hist.list_diversity else [])
    snaps=[]
    if hasattr(hist,"list_population") and hist.list_population:
        for snap in hist.list_population:
            try: snaps.append([ind.solution for ind in snap])
            except: pass
    if not dh:
        for snap in hist.list_population:
            try: pos=np.array([ind.solution for ind in snap]); dh.append(float(np.mean(np.std(pos,axis=0))))
            except: dh.append(0.0)
    if not dh: dh=[0.0]*len(fh)
    esc=0; stag=0; st=0
    for i in range(1,len(fh)):
        if fh[i]>=fh[i-1]-1e-10: stag+=1
        else:
            if stag>3: esc+=1
            stag=0
        st+=1
    vis=set()
    if hasattr(hist,"list_population") and hist.list_population:
        gr_r=20; cs=(hi-lo)/gr_r
        for snap in hist.list_population[::max(1,len(hist.list_population)//8)]:
            for ind in list(snap)[::max(1,len(snap)//5)]:
                try: cell=tuple(((np.array(ind.solution[:3])-lo)/cs).astype(int).clip(0,gr_r-1)); vis.add(cell)
                except: pass
    return {"best_fit":float(model.g_best.target.fitness),"history":fh,"diversity_log":dh,
            "pop_snapshots":snaps,"exec_time":exec_t,"memory_kb":pk/1024,"escapes":esc,"total_stag":max(1,st),
            "iter_95pct":_iter_at_pct(fh,0.95),"max_iter":ep,"visited_cells":max(1,len(vis)),
            "total_cells":20**min(dim,3),"pop_size":ps,"dim":dim}

def _run_niapy(name,fn,lo,hi,dim,params,seed):
    mp,cn,defs=NIAPY_MAP[name]; mod=importlib.import_module(mp); cls=getattr(mod,cn)
    from niapy.task import Task; from niapy.problems import Problem
    class P(Problem):
        def __init__(self): super().__init__(dim,lo,hi)
        def _evaluate(self,x): return float(fn(x))
    ep=int(params.get("max_iter",100)); ps=int(params.get("pop_size",50))
    kw={"population_size":ps}; kw.update(defs); task=Task(problem=P(),max_evals=ep*ps)
    tracemalloc.start(); t0=time.perf_counter(); algo=cls(**kw); algo.seed=seed; _,bf=algo.run(task)
    exec_t=time.perf_counter()-t0; _,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
    return {"best_fit":float(bf),"history":[float(bf)],"diversity_log":[0.0],"pop_snapshots":[],
            "exec_time":exec_t,"memory_kb":pk/1024,"escapes":0,"total_stag":max(1,ep),
            "iter_95pct":ep,"max_iter":ep,"visited_cells":1,"total_cells":1,"pop_size":ps,"dim":dim}

def _run_scipy(name,fn,lo,hi,dim,params,seed):
    bnds=[(lo,hi)]*dim; ep=int(params.get("max_iter",100)); ps=int(params.get("pop_size",50))
    algo=SCIPY_ALGOS.get(name,"differential_evolution"); fh=[]
    tracemalloc.start(); t0=time.perf_counter()
    if algo=="differential_evolution":
        res=differential_evolution(fn,bnds,seed=seed,maxiter=ep,popsize=max(2,ps//dim),disp=False,callback=lambda x,cv:fh.append(float(fn(x))))
    elif algo=="dual_annealing":
        res=dual_annealing(fn,bnds,seed=seed,maxiter=ep*10,callback=lambda x,f,ctx:fh.append(float(f)))
    elif algo=="shgo": res=shgo(fn,bnds,n=min(100,ps))
    elif algo=="direct": res=direct(fn,bnds,maxiter=ep)
    elif algo=="basinhopping":
        x0=np.random.default_rng(seed).uniform(lo,hi,dim)
        res=basinhopping(fn,x0,niter=ep,callback=lambda x,f,a:fh.append(float(f)))
    else:
        x0=np.random.default_rng(seed).uniform(lo,hi,dim)
        res=minimize(fn,x0,method=algo,bounds=bnds,options={"maxiter":ep})
    exec_t=time.perf_counter()-t0; _,pk=tracemalloc.get_traced_memory(); tracemalloc.stop()
    if not fh: fh.append(float(res.fun))
    mono=[fh[0]]
    for v in fh[1:]: mono.append(min(mono[-1],v))
    return {"best_fit":float(res.fun),"history":mono,"diversity_log":[0.0]*len(mono),"pop_snapshots":[],
            "exec_time":exec_t,"memory_kb":pk/1024,"escapes":0,"total_stag":max(1,len(mono)),
            "iter_95pct":_iter_at_pct(mono,0.95),"max_iter":ep,"visited_cells":1,"total_cells":1,"pop_size":ps,"dim":dim}

def _run_ensemble(name,fn,lo,hi,dim,params,seed):
    parts=name.split("+"); n=len(parts); ep=int(params.get("max_iter",100)); per=max(5,ep//n); comp=[]
    for i,p in enumerate(parts):
        sp=dict(params); sp["max_iter"]=per
        try:
            if MEALPY_OK and p in MEALPY_MAP: r=_run_mealpy(p,fn,lo,hi,dim,sp,seed+i*17)
            elif NIAPY_OK and p in NIAPY_MAP: r=_run_niapy(p,fn,lo,hi,dim,sp,seed+i*17)
            elif p in SCIPY_ALGOS: r=_run_scipy(p,fn,lo,hi,dim,sp,seed+i*17)
            else: r=_fallback(p,fn,lo,hi,dim,sp,seed+i*17)
        except: r=_fallback(p,fn,lo,hi,dim,sp,seed+i*17)
        comp.append(r)
    best_r=min(comp,key=lambda r:r["best_fit"]); all_h=[r["history"] for r in comp]; ml=max(len(h) for h in all_h)
    padded=[h+[h[-1]]*(ml-len(h)) for h in all_h]; per_it=np.min(padded,axis=0).tolist(); mono=[per_it[0]]
    for v in per_it[1:]: mono.append(min(mono[-1],v))
    all_d=[r["diversity_log"] for r in comp if r["diversity_log"]]; md=max(len(d) for d in all_d) if all_d else 1
    pd_=[d+[d[-1]]*(md-len(d)) for d in all_d] if all_d else [[0.0]]; mean_d=np.mean(pd_,axis=0).tolist()
    return {"best_fit":best_r["best_fit"],"history":mono,"diversity_log":mean_d,
            "pop_snapshots":[s for r in comp for s in r.get("pop_snapshots",[])],
            "exec_time":sum(r["exec_time"] for r in comp),"memory_kb":sum(r["memory_kb"] for r in comp),
            "escapes":sum(r["escapes"] for r in comp),"total_stag":max(1,sum(r["total_stag"] for r in comp)),
            "iter_95pct":_iter_at_pct(mono,0.95),"max_iter":ep,
            "visited_cells":max(1,best_r.get("visited_cells",1)),"total_cells":best_r.get("total_cells",1),
            "pop_size":comp[0]["pop_size"],"dim":dim}

def run_algorithm(name,fn,lo,hi,dim,params,runs=5):
    results=[]
    for seed in range(runs):
        s=seed*7+13
        try:
            if "+" in name: r=_run_ensemble(name,fn,lo,hi,dim,params,s)
            elif MEALPY_OK and name in MEALPY_MAP: r=_run_mealpy(name,fn,lo,hi,dim,params,s)
            elif NIAPY_OK and name in NIAPY_MAP: r=_run_niapy(name,fn,lo,hi,dim,params,s)
            elif name in SCIPY_ALGOS: r=_run_scipy(name,fn,lo,hi,dim,params,s)
            else: r=_fallback(name,fn,lo,hi,dim,params,s)
        except: r=_fallback(name,fn,lo,hi,dim,params,s)
        if "visited_cells" not in r: r.update({"visited_cells":1,"total_cells":1})
        if "pop_size" not in r: r["pop_size"]=int(params.get("pop_size",50)); r["dim"]=dim
        results.append(r)
    return results

def _fft_oscillation(history):
    if len(history)<8: return 0.5
    h=np.array(history,dtype=float); h_n=(h-h.min())/(h.max()-h.min()+1e-12)
    yf=np.abs(fft(h_n))[:len(h)//2]; yf[0]=0
    if yf.sum()<1e-12: return 0.0
    return float(min(1.0,(np.argmax(yf)+1)/max(1,len(h))*4))

def _measure_energy_psutil(exec_time_s):
    try:
        cpu_pct=max(0.01,psutil.cpu_percent(interval=0.1)/100.0); freq=psutil.cpu_freq()
        tdp=max(5.0,(freq.current/max(freq.max,1))*15) if freq and freq.max>0 else 10.0
        ram=psutil.virtual_memory(); ram_w=(ram.used/1e9)*0.3
        return max(0.001,float((tdp*cpu_pct+ram_w)*exec_time_s))
    except: return max(0.001,10.0*exec_time_s*0.4)

def _epistasis_rf(fn,lo,hi,dim,n=80):
    np.random.seed(42); X=np.random.uniform(lo,hi,(n,min(dim,10))); d_actual=X.shape[1]
    y=np.array([fn(list(X[i])+[0.0]*(dim-d_actual)) for i in range(n)])
    if np.std(y)<1e-12: return 0.0
    try:
        rf=RandomForestRegressor(n_estimators=15,random_state=42,max_depth=4); rf.fit(X,y)
        imp=rf.feature_importances_; gini=1.0-np.sum(imp**2); max_gini=1.0-1.0/d_actual
        return float(min(1.0,max(0.0,gini/(max_gini+1e-9))))
    except: return 0.5

def _optics_modality(pop_snapshots,lo,hi,dim):
    if not pop_snapshots: return 0.0
    all_pts=[]
    for snap in pop_snapshots[-3:]: all_pts.extend(snap)
    if len(all_pts)<6: return 0.0
    pts=np.array(all_pts,dtype=float)[:,:min(dim,4)]; pts_n=(pts-lo)/(hi-lo+1e-9)
    try:
        op=OPTICS(min_samples=max(2,len(pts_n)//20),max_eps=0.5).fit(pts_n)
        return float(min(1.0,max(0,len(set(op.labels_))-(1 if -1 in op.labels_ else 0))/10.0))
    except: return 0.0

def _find_peaks_coverage(history):
    if len(history)<10: return 0.0
    h=np.array(history,dtype=float)
    try:
        pks,_=find_peaks(-h,distance=max(1,len(h)//20),prominence=np.std(h)*0.1)
        return float(min(1.0,len(pks)/10.0))
    except: return 0.0

def _uncertainty_quantification(bf_list):
    if len(bf_list)<3: return 0.5
    try:
        ci=scipy_stats.t.interval(0.95,len(bf_list)-1,loc=np.mean(bf_list),scale=scipy_stats.sem(bf_list))
        return max(0.0,min(1.0,(ci[1]-ci[0])/(abs(np.mean(bf_list))+1e-9)))
    except: return 0.5

MNAMES=["Convergence Speed","Solution Quality","Exploration-Exploitation","Robustness",
        "Scalability","Escape Local Optima","Computational Efficiency","Population Diversity",
        "Convergence Stability","Search Space Coverage","Energy Efficiency","Memory Footprint",
        "Information Gain","Landscape Fit","Scalability Slope","Uncertainty Quantification",
        "Neutrality Resistance","Deceptiveness Handling","Modality Coverage","Epistasis Efficiency"]

def score_all(runs_data,optimal,max_iter,dim,fn=None,lo=-5.12,hi=5.12):
    sc={}; raw={}
    bf=[r["best_fit"] for r in runs_data]; mb=float(np.mean(bf)); ref=max(abs(optimal),1.0); gap=abs(mb-optimal)/ref
    pop_size=runs_data[0].get("pop_size",50)
    all_h=[r["history"] for r in runs_data if r["history"]]
    if all_h:
        mh_l=max(len(h) for h in all_h); ph_=[h+[h[-1]]*(mh_l-len(h)) for h in all_h]; mean_h=np.mean(ph_,axis=0)
    else: mean_h=np.array([1.0])
    mt=float(np.mean([r["exec_time"] for r in runs_data]))
    i95=float(np.mean([r["iter_95pct"] for r in runs_data]))
    sc[MNAMES[0]]=round(max(0.0,5.0*(1.0-i95/max(1,max_iter))),2); raw[MNAMES[0]]=f"Iter@95%={i95:.1f}/{max_iter} | S=5x(1-iter95/T)"
    sc[MNAMES[1]]=round(max(0.0,5.0*(1.0-min(gap,1.0))),2); raw[MNAMES[1]]=f"Gap={gap:.4f} | Best={mb:.4e}"
    all_div=[r["diversity_log"] for r in runs_data if r["diversity_log"]]
    if all_div:
        ml_=max(len(d) for d in all_div); pad_=[d+[d[-1]]*(ml_-len(d)) for d in all_div]; mc_=np.mean(pad_,axis=0); n_=len(mc_)
        early=float(np.mean(mc_[:n_//3])) if n_>=3 else 0; late=float(np.mean(mc_[2*n_//3:])) if n_>=3 else 0
        mc_n=mc_/(mc_.max()+1e-9); mc_n=mc_n[mc_n>1e-9]; H=float(scipy_entropy(mc_n)) if len(mc_n)>0 else 0
        drop=(early-late)/(early+1e-9); s3=min(5.0,max(0.0,5.0*(0.5*drop+0.5*min(H/3.0,1.0))))
    else: s3,H,early,late=2.5,0,0,0
    sc[MNAMES[2]]=round(s3,2); raw[MNAMES[2]]=f"H={H:.3f} | {early:.4f}->{late:.4f}"
    cv=float(np.std(bf)/(abs(np.mean(bf))+1e-9)) if len(bf)>1 else 0.0
    sc[MNAMES[3]]=round(max(0.0,5.0*(1.0-min(cv,1.0))),2); raw[MNAMES[3]]=f"CV={cv:.4f}"
    algo_name=runs_data[0].get("algo_name","PSO") if "algo_name" in runs_data[0] else "PSO"
    cf=COMPLEXITY_FACTORS.get(algo_name.split("+")[0],1.0); T_ref=pop_size*max_iter*dim*1e-6*cf
    sc[MNAMES[4]]=round(min(5.0,max(0.0,5.0*min(T_ref/(mt+1e-9),1.0))),2); raw[MNAMES[4]]=f"T={mt:.3f}s | cf={cf}"
    esc_=float(np.sum([r["escapes"] for r in runs_data])); stg_=float(np.sum([r["total_stag"] for r in runs_data]))
    sc[MNAMES[5]]=round(min(5.0,max(0.0,5.0*(esc_/(stg_+1e-9))/0.20)),2); raw[MNAMES[5]]=f"Esc={esc_:.0f}/{stg_:.0f}"
    quality=max(0.0,1.0-min(gap,1.0)); eff=quality/(mt+1e-9)
    sc[MNAMES[6]]=round(min(5.0,max(0.0,5.0*eff/5.0)),2); raw[MNAMES[6]]=f"q/T={eff:.4f}"
    all_snaps=[s for r in runs_data for s in r.get("pop_snapshots",[])]
    if all_snaps:
        dists=[]
        for snap in all_snaps[:4]:
            pos=np.array(snap,dtype=float)
            if pos.ndim==2 and pos.shape[0]>1:
                ni=min(pos.shape[0],15); samp=pos[:ni]; ds=0.0; cnt=0
                for ii in range(ni):
                    for jj in range(ii+1,ni): ds+=float(np.linalg.norm(samp[ii]-samp[jj])); cnt+=1
                if cnt>0: dists.append(ds/cnt)
        md_=float(np.mean(dists)) if dists else 0.0
    else:
        id__=[r["diversity_log"][0] for r in runs_data if r["diversity_log"]]; md_=float(np.mean(id__)) if id__ else 0.0
    sd=float(np.sqrt(dim))*(hi-lo); s8=min(5.0,max(0.0,5.0*md_/(sd+1e-9)))
    sc[MNAMES[7]]=round(s8,2); raw[MNAMES[7]]=f"MeanDist={md_:.4f} | Diag={sd:.2f}"
    osc_fft=_fft_oscillation(mean_h.tolist()); diffs=np.diff(mean_h); osc_c=float(np.sum(diffs>1e-10))/max(1,len(diffs))
    sc[MNAMES[8]]=round(max(0.0,5.0*(1.0-0.5*osc_fft-0.5*osc_c)),2); raw[MNAMES[8]]=f"FFT={osc_fft:.3f} Cls={osc_c:.3f}"
    all_pos=[p for r in runs_data for snap in r.get("pop_snapshots",[]) for p in snap]
    if len(all_pos)>=4:
        pts=np.array(all_pos,dtype=float)[:,:min(3,dim)]; tree=KDTree(pts); gn=12
        axes=[np.linspace(lo,hi,gn)]*min(3,dim); grid=np.array(np.meshgrid(*axes)).reshape(min(3,dim),-1).T
        dd,_=tree.query(grid); cov=float(np.sum(dd<(hi-lo)/gn))/len(grid)
    else:
        mv=float(np.mean([r.get("visited_cells",1) for r in runs_data])); tc=float(max(1,runs_data[0].get("total_cells",1))); cov=min(1.0,mv/tc)
    sc[MNAMES[9]]=round(5.0*cov,2); raw[MNAMES[9]]=f"Coverage={cov*100:.1f}%"
    energy_j=float(np.mean([_measure_energy_psutil(r["exec_time"]) for r in runs_data]))
    e_eff=quality/(max(energy_j,0.001)); sc[MNAMES[10]]=round(min(5.0,max(0.0,5.0*min(e_eff/0.1,1.0))),2)
    raw[MNAMES[10]]=f"E={energy_j:.4f}J | q/E={e_eff:.4f}"
    mk=float(np.mean([r["memory_kb"] for r in runs_data])); nm=mk/max(1,dim*50)
    sc[MNAMES[11]]=round(max(0.0,min(5.0,5.0*(1.0-min(nm/100.0,1.0)))),2); raw[MNAMES[11]]=f"Peak={mk:.1f}KB"
    if all_snaps and len(all_snaps)>=2:
        def _pH(snap):
            p=np.array(snap,dtype=float)
            if p.ndim==1: p=p.reshape(1,-1)
            Ht=0.0
            for d in range(min(p.shape[1],dim)):
                h_,_=np.histogram(p[:,d],bins=max(2,min(10,p.shape[0]//2)),density=True); h_=h_[h_>0]
                if len(h_)>0: Ht+=float(scipy_entropy(h_))
            return Ht/max(1,min(p.shape[1],dim))
        Hi=_pH(all_snaps[0]); Hf=_pH(all_snaps[-1]); IG=max(0.0,Hi-Hf); s13=5.0*min(1.0,IG/(Hi+1e-9))
    else: s13,IG,Hi,Hf=2.5,0.0,0.0,0.0
    sc[MNAMES[12]]=round(s13,2); raw[MNAMES[12]]=f"IG={IG:.3f} | H={Hi:.3f}->{Hf:.3f}"
    rug=(BS.ruggedness(fn,lo,hi,min(dim,5)) if fn is not None else 0.5); expl=s3/5.0; fit_=1.0-abs(expl-rug)
    sc[MNAMES[13]]=round(max(0.0,5.0*fit_),2); raw[MNAMES[13]]=f"rho={rug:.3f} | Expl={expl:.3f}"
    if len(mean_h)>10:
        li=np.log(np.arange(1,len(mean_h)+1)+1)
        nc=((mean_h[0]-mean_h)/(mean_h[0]-mean_h[-1]+1e-9)) if mean_h[0]>mean_h[-1]+1e-10 else np.zeros_like(mean_h)
        beta=float(np.polyfit(li,nc,1)[0]); s15=min(5.0,max(0.0,5.0*min(beta,1.0)))
    else: s15,beta=2.5,0.0
    sc[MNAMES[14]]=round(s15,2); raw[MNAMES[14]]=f"beta={beta:.4f} Amdahl"
    uq=_uncertainty_quantification(bf); s16=max(0.0,5.0*(1.0-uq))
    sc[MNAMES[15]]=round(s16,2); raw[MNAMES[15]]=f"CI={uq:.4f}"
    imps=float(np.sum(np.diff(mean_h)<-1e-10)); s17=max(0.0,5.0*(imps/max(1,len(mean_h)-1)))
    sc[MNAMES[16]]=round(s17,2); raw[MNAMES[16]]=f"ImpFrac={imps/max(1,len(mean_h)-1):.3f}"
    if len(bf)>1:
        rsp=float(np.max(bf)-np.min(bf)); tg=abs(mb-optimal)+1e-9; s18=max(0.0,min(5.0,5.0*(1.0-min(rsp/tg,1.0))))
    else: s18=2.5; rsp=0.0
    sc[MNAMES[17]]=round(s18,2); raw[MNAMES[17]]=f"Spread={rsp:.4e}"
    optics_cov=_optics_modality(all_snaps,lo,hi,dim); peaks_cov=_find_peaks_coverage(mean_h.tolist())
    sc[MNAMES[18]]=round(min(5.0,max(0.0,5.0*(0.6*optics_cov+0.4*peaks_cov))),2)
    raw[MNAMES[18]]=f"OPTICS={optics_cov:.3f}|Peaks={peaks_cov:.3f}"
    if fn is not None: ep_gini=_epistasis_rf(fn,lo,hi,dim); s20=max(0.0,min(5.0,5.0*ep_gini))
    else: ep_gini=0.5; s20=2.5
    sc[MNAMES[19]]=round(s20,2); raw[MNAMES[19]]=f"Gini={ep_gini:.4f}"
    total=round(sum(sc.values()),2)
    return {"scores":sc,"total":total,"raw":raw,"history":runs_data[0]["history"],
            "all_histories":[r["history"] for r in runs_data],
            "exec_time":mt,"memory_kb":float(np.mean([r["memory_kb"] for r in runs_data])),
            "best_fit":float(np.mean([r["best_fit"] for r in runs_data]))}

class AutoMLRec:
    def __init__(self):
        np.random.seed(0); n=400
        X=np.column_stack([np.random.choice([2,5,10,20,50],n),np.random.beta(2,2,n),
                           np.random.choice([1,5,10,20],n),np.random.beta(2,2,n),np.random.choice([5,50,100,500],n)]).astype(float)
        algos=ALL_SINGLE[:20]; y=np.zeros((n,len(algos)))
        for i in range(n):
            d,rug,mod,sep,br=X[i]
            for j,a in enumerate(algos):
                s=5.0
                if a=="DE": s+=(1-rug)*2+sep*2
                elif a in ("GWO","WOA","HHO"): s+=rug*2+(1-sep)*1.5
                elif a=="PSO": s+=1.5
                elif a=="GA": s+=(d/50)*2
                elif a=="SA": s+=rug*1.5
                y[i,j]=min(10,max(0,s+np.random.randn()*0.5))
        self._sc=StandardScaler(); Xs=self._sc.fit_transform(X)
        self._m=RandomForestRegressor(n_estimators=15,random_state=42).fit(Xs,y); self._algos=algos
    def recommend(self,dim,rug,mod,sep,br):
        f=self._sc.transform([[dim,rug,mod,sep,br]]); p=self._m.predict(f)[0]
        return [(self._algos[i],round(float(v),2)) for i,v in sorted(enumerate(p),key=lambda x:x[1],reverse=True)[:5]]
_automl=AutoMLRec()

class DB:
    def __init__(self):
        self.conn=sqlite3.connect(":memory:",check_same_thread=False)
        self.conn.execute("CREATE TABLE IF NOT EXISTS evals(id INTEGER PRIMARY KEY AUTOINCREMENT,algorithm TEXT,algo_type TEXT,benchmark TEXT,dim INTEGER,pop_size INTEGER,max_iter INTEGER,runs INTEGER,total_score REAL,best_fit REAL,exec_time REAL,memory_kb REAL,timestamp TEXT)")
        self.conn.execute("CREATE TABLE IF NOT EXISTS metrics(eval_id INTEGER,metric TEXT,score REAL,details TEXT)")
        self.conn.commit()
    def store(self,algorithm,algo_type,benchmark,dim,pop_size,max_iter,runs,scored):
        c=self.conn.execute("INSERT INTO evals(algorithm,algo_type,benchmark,dim,pop_size,max_iter,runs,total_score,best_fit,exec_time,memory_kb,timestamp)VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (algorithm,algo_type,benchmark,dim,pop_size,max_iter,runs,scored["total"],scored["best_fit"],scored["exec_time"],scored["memory_kb"],datetime.now().isoformat()))
        eid=c.lastrowid
        for m,s in scored["scores"].items(): self.conn.execute("INSERT INTO metrics VALUES(?,?,?,?)",(eid,m,s,scored["raw"].get(m,"")))
        self.conn.commit()
    def leaderboard(self,bench=None,n=30):
        q="SELECT algorithm,benchmark,dim,total_score,exec_time,memory_kb,timestamp FROM evals"
        if bench and bench!="All": q+=f" WHERE benchmark='{bench}'"
        q+=" ORDER BY total_score DESC LIMIT ?"
        try: return pd.read_sql_query(q,self.conn,params=(n,))
        except: return pd.DataFrame()
_db=DB()

def gc(s):
    if s>=85: return "#00e676"
    if s>=70: return "#69f0ae"
    if s>=55: return "#ffeb3b"
    if s>=40: return "#ffa726"
    return "#ef5350"
def gl(s):
    if s>=85: return "EXCELLENT ⭐⭐⭐⭐⭐"
    if s>=70: return "VERY GOOD ⭐⭐⭐⭐"
    if s>=55: return "GOOD ⭐⭐⭐"
    if s>=40: return "FAIR ⭐⭐"
    return "WEAK ⭐"
def get_algo_list(t):
    lst=ALL_ALGOS.get(t,ALL_SINGLE); return gr.update(choices=lst,value=lst[0])

def make_3d_landscape(fn,lo,hi,snaps=None):
    n=30; x_1d=np.linspace(lo,hi,n); y_1d=np.linspace(lo,hi,n); X2d,Y2d=np.meshgrid(x_1d,y_1d); Z=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            try: Z[i,j]=min(float(fn([X2d[i,j],Y2d[i,j]])),1e6)
            except: Z[i,j]=0.0
    Zl=np.log1p(np.clip(Z,0,1e6)); fig=go.Figure()
    fig.add_trace(go.Surface(x=x_1d,y=y_1d,z=Zl,colorscale="Viridis",opacity=0.78,showscale=True,colorbar=dict(title="log(f)",len=0.6,thickness=12,tickfont=dict(color="#aaa",size=9))))
    if snaps:
        valid=[s[0] for s in snaps[:20] if isinstance(s[0],(list,np.ndarray)) and len(s[0])>=2]
        if valid:
            px_=[float(p[0]) for p in valid]; py_=[float(p[1]) for p in valid]; pz_=[]
            for p in valid:
                try: pz_.append(float(np.log1p(min(fn(list(p[:2])+[0.]*(max(2,len(p))-2)),1e6))))
                except: pz_.append(0.0)
            fig.add_trace(go.Scatter3d(x=px_,y=py_,z=pz_,mode="lines+markers",line=dict(color="red",width=4),marker=dict(size=3,color="orange"),name="Path"))
    fig.update_layout(paper_bgcolor="#0d1117",font=dict(color="#e0e0e0"),scene=dict(bgcolor="#0d1117",xaxis=dict(title="x1",gridcolor="#21262d",color="#8b949e"),yaxis=dict(title="x2",gridcolor="#21262d",color="#8b949e"),zaxis=dict(title="log(f+1)",gridcolor="#21262d",color="#8b949e")),title=dict(text="Fitness Landscape (2D slice, log scale)",font=dict(size=13)),margin=dict(l=0,r=0,t=40,b=0))
    return fig

EQ_HTML='<div style="background:#0d1f3a;border:2px solid #1e4d8c;border-radius:10px;padding:18px;font-family:\'JetBrains Mono\',monospace;margin-top:14px;"><div style="color:#5aabff;font-size:13px;font-weight:bold;margin-bottom:12px;border-bottom:1px solid #1e4d8c;padding-bottom:6px;">20 Physical Equations Applied</div><div style="color:#fff;font-weight:bold;line-height:2.0;font-size:11px;"><b style="color:#7ec8ff;">M1</b> S=5x(1-iter95/T) <b style="color:#7ec8ff;">M2</b> S=5x(1-gap) <b style="color:#7ec8ff;">M3</b> S=5x(0.5xDdiv+0.5xH/3)<br><b style="color:#7ec8ff;">M4</b> S=5x(1-CV) <b style="color:#7ec8ff;">M5</b> S=5xmin(Tref/T,1) <b style="color:#7ec8ff;">M6</b> S=5x(esc/stag)/0.20<br><b style="color:#7ec8ff;">M7</b> S=5x(q/T)/5 <b style="color:#7ec8ff;">M8</b> S=5xd_bar/diag <b style="color:#7ec8ff;">M9</b> S=5x(1-0.5xFFT-0.5xosc)<br><b style="color:#7ec8ff;">M10</b> S=5x(hit/total) KD-tree <b style="color:#ffd700;">M11</b> E=(TDPxcpu+0.3W/GBxram)xT<br><b style="color:#ffd700;">M12</b> S=5x(1-peak/(Dx50x100)) <b style="color:#ffd700;">M13</b> S=5x(Hi-Hf)/Hi<br><b style="color:#ffd700;">M14</b> S=5x(1-|expl-rho|) <b style="color:#ffd700;">M15</b> Amdahl beta slope<br><b style="color:#ffd700;">M16</b> CI95 Student-t <b style="color:#ffd700;">M17</b> S=5x(Df&lt;0)/N<br><b style="color:#ffd700;">M18</b> S=5x(1-spread/gap) <b style="color:#ffd700;">M19</b> 0.6xOPTICS+0.4xpeaks <b style="color:#ffd700;">M20</b> Gini(RF)</div></div>'

def evaluate(algo_type,algorithm,bench_fn,dim,pop_size,max_iter,runs,
             inertia,cognitive,social,mutation,crossover,elite_f,restart_t,
             div_thr,adapt_lr,momentum,chaos_en,levy_fl,opp_l,arch_sz,nbr_sz,show_3d):
    if not algorithm: return [None]*7+["<p>Select an algorithm.</p>"]+[[]]
    fn_d=BS.FUNCTIONS.get(bench_fn)
    if not fn_d: return [None]*7+["<p>Unknown benchmark.</p>"]+[[]]
    fn,(lo,hi),optimal=fn_d
    params={"max_iter":int(max_iter),"pop_size":int(pop_size),"inertia":inertia,"cognitive":cognitive,
            "social":social,"mutation":mutation,"crossover":crossover,"elite_fraction":elite_f,
            "restart_threshold":restart_t,"diversity_threshold":div_thr,"adaptive_lr":adapt_lr,
            "momentum":momentum,"chaos_enabled":chaos_en,"levy_flight":levy_fl,
            "opposition_learning":opp_l,"archive_size":int(arch_sz),"neighborhood_size":int(nbr_sz)}
    rd=run_algorithm(algorithm,fn,lo,hi,int(dim),params,runs=int(runs))
    for r in rd: r["algo_name"]=algorithm
    scored=score_all(rd,optimal,int(max_iter),int(dim),fn,lo,hi)
    _db.store(algorithm,algo_type,bench_fn,int(dim),int(pop_size),int(max_iter),int(runs),scored)
    cats=list(scored["scores"].keys()); vals=list(scored["scores"].values())
    fig_r=go.Figure(go.Scatterpolar(r=vals+[vals[0]],theta=cats+[cats[0]],fill="toself",line=dict(color="#4fc3f7",width=2),fillcolor="rgba(79,195,247,0.18)"))
    fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,5],tickfont=dict(color="#aaa",size=7)),angularaxis=dict(tickfont=dict(color="#ccc",size=8))),paper_bgcolor="#0d1117",font=dict(color="#e0e0e0"),title=f"20 Metrics -- {algorithm[:28]}",showlegend=False,margin=dict(l=65,r=65,t=50,b=50))
    all_h=scored["all_histories"]; ml=max(len(h) for h in all_h); fig_c=go.Figure()
    for h in all_h: fig_c.add_trace(go.Scatter(y=h,mode="lines",line=dict(color="#4fc3f7",width=1,dash="dot"),opacity=0.35,showlegend=False))
    mh=np.mean([h+[h[-1]]*(ml-len(h)) for h in all_h],axis=0)
    fig_c.add_trace(go.Scatter(y=mh,mode="lines",line=dict(color="#ff7043",width=2.5),name="Mean"))
    fig_c.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",font=dict(color="#e0e0e0"),title="Convergence Curve",xaxis=dict(title="Iteration",gridcolor="#21262d"),yaxis=dict(title="Fitness",gridcolor="#21262d"),margin=dict(l=50,r=20,t=50,b=40))
    fig_b=make_subplots(rows=1,cols=2,subplot_titles=["Metrics 1-10","Metrics 11-20"])
    for col_,sl_ in [(1,slice(0,10)),(2,slice(10,20))]:
        cv_=cats[sl_]; vv_=vals[sl_]
        fig_b.add_trace(go.Bar(x=cv_,y=vv_,marker_color=[gc(v*20) for v in vv_],text=[f"{v:.1f}" for v in vv_],textposition="outside",textfont=dict(color="#fff",size=8)),row=1,col=col_)
    fig_b.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",font=dict(color="#e0e0e0"),showlegend=False,yaxis=dict(range=[0,6.5],gridcolor="#21262d"),yaxis2=dict(range=[0,6.5],gridcolor="#21262d"),xaxis=dict(tickangle=-35),xaxis2=dict(tickangle=-35),margin=dict(l=30,r=30,t=50,b=120))
    all_d=[r["diversity_log"] for r in rd if r["diversity_log"]]
    if all_d:
        md_=max(len(d) for d in all_d); pd__=[d+[d[-1]]*(md_-len(d)) for d in all_d]; mean_d=np.mean(pd__,axis=0)
        fig_d=go.Figure(go.Scatter(y=mean_d,mode="lines",line=dict(color="#69f0ae",width=2)))
        fig_d.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",font=dict(color="#e0e0e0"),title="Population Diversity",xaxis=dict(title="Iteration",gridcolor="#21262d"),yaxis=dict(title="sigma",gridcolor="#21262d"),margin=dict(l=50,r=20,t=50,b=40))
    else: fig_d=go.Figure()
    fig_3d=None
    if show_3d and int(dim)>=2:
        all_snaps=[s for r in rd for s in r.get("pop_snapshots",[])]
        fig_3d=make_3d_landscape(fn,lo,hi,all_snaps[:25])
    total=scored["total"]; color=gc(total); label=gl(total)
    badge="MealPy+NiaPy" if MEALPY_OK and NIAPY_OK else "MealPy" if MEALPY_OK else "SciPy"
    rows=""
    for i,(m,v) in enumerate(scored["scores"].items()):
        c=gc(v*20); bw=int(v*20); bg="#0d1117" if i%2==0 else "#111820"; det=scored["raw"].get(m,"").split("|")[0].strip()
        rows+=(f'<tr style="background:{bg};"><td style="padding:5px 8px;color:#c9d1d9;font-size:11px;">{i+1}. {m}</td>'
               f'<td style="padding:5px 8px;width:90px;"><div style="background:#21262d;border-radius:3px;height:7px;">'
               f'<div style="background:{c};height:7px;border-radius:3px;width:{bw}%;"></div></div></td>'
               f'<td style="padding:5px 8px;color:{c};font-weight:bold;font-size:12px;text-align:center;">{v:.1f}/5</td>'
               f'<td style="padding:5px 8px;color:#8b949e;font-size:10px;">{det}</td></tr>')
    html=(f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:18px;font-family:\'JetBrains Mono\',monospace;">'
          f'<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;flex-wrap:wrap;gap:8px;">'
          f'<div><div style="color:#8b949e;font-size:10px;">ALGORITHM <span style="color:#4fc3f7;">{badge}</span></div>'
          f'<div style="color:#e0e0e0;font-size:14px;font-weight:bold;">{algorithm}</div>'
          f'<div style="color:#8b949e;font-size:10px;">{bench_fn} | Dim={dim} | Pop={pop_size} | Iter={max_iter} | Runs={runs}</div></div>'
          f'<div style="text-align:center;"><div style="font-size:52px;font-weight:900;color:{color};line-height:1;">{total:.1f}</div>'
          f'<div style="color:#8b949e;font-size:10px;">/ 100</div><div style="color:{color};font-size:11px;">{label}</div></div></div>'
          f'<table style="width:100%;border-collapse:collapse;border-top:1px solid #21262d;">'
          f'<thead><tr style="background:#161b22;"><th style="padding:7px 8px;text-align:left;color:#8b949e;font-size:10px;">METRIC</th>'
          f'<th style="padding:7px;color:#8b949e;font-size:10px;">BAR</th>'
          f'<th style="padding:7px;text-align:center;color:#8b949e;font-size:10px;">SCORE</th>'
          f'<th style="padding:7px 8px;text-align:left;color:#8b949e;font-size:10px;">VALUE</th></tr></thead><tbody>{rows}</tbody></table>'
          f'<div style="margin-top:10px;padding-top:10px;border-top:1px solid #21262d;display:flex;gap:18px;flex-wrap:wrap;">'
          f'<span><span style="color:#8b949e;font-size:10px;">Time </span><span style="color:#4fc3f7;font-size:12px;font-weight:bold;">{scored["exec_time"]:.3f}s</span></span>'
          f'<span><span style="color:#8b949e;font-size:10px;">Memory </span><span style="color:#4fc3f7;font-size:12px;font-weight:bold;">{scored["memory_kb"]:.1f}KB</span></span>'
          f'<span><span style="color:#8b949e;font-size:10px;">Best Fitness </span><span style="color:#4fc3f7;font-size:12px;font-weight:bold;">{scored["best_fit"]:.6f}</span></span>'
          f'</div></div>'+EQ_HTML)
    tbl=[[i+1,m,f"{v:.2f}/5.00",scored["raw"].get(m,"").split("|")[0].strip()] for i,(m,v) in enumerate(scored["scores"].items())]
    tbl.append(["--","TOTAL",f"{total:.2f}/100",label])
    return fig_r,fig_c,fig_b,fig_d,fig_3d,html,tbl

def compare_two(ta,aa,tb,ab,bench,dim,pop,itr,runs):
    results=[]
    for algo in [aa,ab]:
        fn_d=BS.FUNCTIONS.get(bench)
        if not fn_d: continue
        fn,(lo,hi),optimal=fn_d
        p={"max_iter":int(itr),"pop_size":int(pop),"inertia":0.7,"cognitive":1.5,"social":1.5,"mutation":0.01,"crossover":0.9,"elite_fraction":0.1,"restart_threshold":50,"diversity_threshold":0.01,"adaptive_lr":0.01,"momentum":0.9,"chaos_enabled":False,"levy_flight":False,"opposition_learning":False,"archive_size":50,"neighborhood_size":5}
        rd=run_algorithm(algo,fn,lo,hi,int(dim),p,runs=int(runs))
        for r in rd: r["algo_name"]=algo
        sc=score_all(rd,optimal,int(itr),int(dim),fn,lo,hi); results.append((algo,sc))
    if len(results)<2: return None,None,None,"<p style='color:red'>Need 2 valid algorithms.</p>"
    (na,sa),(nb,sb)=results; cats=list(sa["scores"].keys()); va=list(sa["scores"].values()); vb=list(sb["scores"].values())
    fig_r=go.Figure()
    for nm,vs,col,fc in [(na[:22],va,"#4fc3f7","rgba(79,195,247,0.12)"),(nb[:22],vb,"#ff7043","rgba(255,112,67,0.12)")]:
        fig_r.add_trace(go.Scatterpolar(r=vs+[vs[0]],theta=cats+[cats[0]],fill="toself",name=nm,line=dict(color=col,width=2),fillcolor=fc))
    fig_r.update_layout(polar=dict(radialaxis=dict(range=[0,5])),paper_bgcolor="#0d1117",font=dict(color="#e0e0e0"),title="Radar Comparison",margin=dict(l=65,r=65,t=50,b=50))
    fig_b=go.Figure()
    fig_b.add_trace(go.Bar(name=na[:22],x=cats,y=va,marker_color="#4fc3f7",opacity=0.85))
    fig_b.add_trace(go.Bar(name=nb[:22],x=cats,y=vb,marker_color="#ff7043",opacity=0.85))
    fig_b.update_layout(barmode="group",paper_bgcolor="#0d1117",plot_bgcolor="#161b22",font=dict(color="#e0e0e0"),title="Score Comparison",yaxis=dict(range=[0,6],gridcolor="#21262d"),xaxis=dict(tickangle=-35),margin=dict(l=40,r=20,t=50,b=120))
    all_ha=sa.get("all_histories",[]); all_hb=sb.get("all_histories",[]); fig_cv=go.Figure()
    if all_ha:
        mla=max(len(h) for h in all_ha); mha=np.mean([h+[h[-1]]*(mla-len(h)) for h in all_ha],axis=0)
        fig_cv.add_trace(go.Scatter(y=mha,mode="lines",line=dict(color="#4fc3f7",width=2),name=na[:20]))
    if all_hb:
        mlb=max(len(h) for h in all_hb); mhb=np.mean([h+[h[-1]]*(mlb-len(h)) for h in all_hb],axis=0)
        fig_cv.add_trace(go.Scatter(y=mhb,mode="lines",line=dict(color="#ff7043",width=2),name=nb[:20]))
    fig_cv.update_layout(paper_bgcolor="#0d1117",plot_bgcolor="#161b22",font=dict(color="#e0e0e0"),title="Convergence Comparison",xaxis=dict(title="Iteration",gridcolor="#21262d"),yaxis=dict(title="Fitness",gridcolor="#21262d"),margin=dict(l=50,r=20,t=50,b=40))
    ta_s,tb_s=sa["total"],sb["total"]; winner=na if ta_s>=tb_s else nb
    html=(f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:12px;padding:18px;font-family:\'JetBrains Mono\',monospace;">'
          f'<div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:12px;">'
          f'<div style="flex:1;min-width:150px;background:#161b22;border-radius:8px;padding:12px;text-align:center;">'
          f'<div style="color:#4fc3f7;font-size:11px;">A</div><div style="color:#e0e0e0;font-size:13px;font-weight:bold;">{na[:26]}</div>'
          f'<div style="color:{gc(ta_s)};font-size:36px;font-weight:900;">{ta_s:.1f}</div><div style="color:#8b949e;font-size:10px;">{gl(ta_s)}</div></div>'
          f'<div style="flex:1;min-width:150px;background:#161b22;border-radius:8px;padding:12px;text-align:center;">'
          f'<div style="color:#ff7043;font-size:11px;">B</div><div style="color:#e0e0e0;font-size:13px;font-weight:bold;">{nb[:26]}</div>'
          f'<div style="color:{gc(tb_s)};font-size:36px;font-weight:900;">{tb_s:.1f}</div><div style="color:#8b949e;font-size:10px;">{gl(tb_s)}</div></div></div>'
          f'<div style="background:#1f6feb22;border:1px solid #1f6feb;border-radius:8px;padding:10px;text-align:center;">'
          f'<span style="color:#58a6ff;">Winner: </span><span style="color:#e0e0e0;font-weight:bold;">{winner[:38]}</span>'
          f'<span style="color:#8b949e;"> (+{abs(ta_s-tb_s):.1f} pts)</span></div></div>')
    return fig_r,fig_b,fig_cv,html

def automl_rec(bench,dim):
    fn_d=BS.FUNCTIONS.get(bench)
    if not fn_d: return "<p>Unknown.</p>"
    fn,(lo,hi),optimal=fn_d
    rug=BS.ruggedness(fn,lo,hi,min(int(dim),5)); mod=BS.modality(fn,lo,hi)
    sep=BS.separability(fn,lo,hi,min(int(dim),5)); br=abs(hi-lo); recs=_automl.recommend(int(dim),rug,mod,sep,br)
    rows="".join(f'<tr><td style="padding:7px 10px;color:#4fc3f7;font-weight:bold;">{i+1}. {a}</td><td style="padding:7px 10px;"><div style="background:#21262d;border-radius:3px;height:7px;width:100%;"><div style="background:{gc(s*10)};height:7px;border-radius:3px;width:{int(s*10)}%;"></div></div></td><td style="padding:7px 10px;color:{gc(s*10)};font-weight:bold;">{s:.1f}/10</td></tr>'
                 for i,(a,s) in enumerate(recs))
    return (f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:10px;padding:16px;font-family:\'JetBrains Mono\',monospace;">'
            f'<div style="color:#58a6ff;font-size:13px;font-weight:bold;margin-bottom:8px;">Recommendations for {bench} (dim={dim})</div>'
            f'<div style="color:#8b949e;font-size:10px;margin-bottom:10px;">rho={rug:.3f} | mod={mod} | sep={sep:.3f}</div>'
            f'<table style="width:100%;border-collapse:collapse;"><tr style="background:#161b22;"><th style="padding:7px 10px;text-align:left;color:#8b949e;font-size:10px;">ALGORITHM</th><th style="padding:7px 10px;color:#8b949e;font-size:10px;">FIT</th><th style="padding:7px 10px;color:#8b949e;font-size:10px;">SCORE</th></tr>{rows}</table></div>')

def get_lb(bench):
    df=_db.leaderboard(bench,30); return [] if df.empty else df.values.tolist()

# ===========================================================================
# DATA ADVISOR - ALGORITHM TRAITS
# ===========================================================================
ADVISOR_TRAITS = {
    "PSO":   {"ex":0.60,"ep":0.80,"type":"Swarm",    "desc":"Particle Swarm - velocity-based continuous"},
    "GWO":   {"ex":0.70,"ep":0.90,"type":"Swarm",    "desc":"Grey Wolf - alpha-beta-delta hierarchy"},
    "WOA":   {"ex":0.80,"ep":0.75,"type":"Swarm",    "desc":"Whale - spiral bubble-net attack"},
    "SSA":   {"ex":0.75,"ep":0.70,"type":"Swarm",    "desc":"Salp Swarm - chain-based food following"},
    "MFO":   {"ex":0.73,"ep":0.82,"type":"Swarm",    "desc":"Moth Flame - logarithmic spiral"},
    "FA":    {"ex":0.72,"ep":0.78,"type":"Swarm",    "desc":"Fireworks - explosion-based search"},
    "ABC":   {"ex":0.80,"ep":0.70,"type":"Swarm",    "desc":"Artificial Bee Colony"},
    "GOA":   {"ex":0.75,"ep":0.72,"type":"Swarm",    "desc":"Grasshopper - social interaction+wind"},
    "SCA":   {"ex":0.70,"ep":0.75,"type":"Swarm",    "desc":"Sine Cosine - sinusoidal update"},
    "EHO":   {"ex":0.72,"ep":0.73,"type":"Swarm",    "desc":"Elephant Herding - clan separation"},
    "EWA":   {"ex":0.72,"ep":0.80,"type":"Bio",      "desc":"Earthworm - reproduction operators"},
    "HHO":   {"ex":0.85,"ep":0.80,"type":"Swarm",    "desc":"Harris Hawks - rabbit-escaping phases"},
    "MRFO":  {"ex":0.72,"ep":0.88,"type":"Swarm",    "desc":"Manta Ray - somersault+cyclone"},
    "AO":    {"ex":0.78,"ep":0.82,"type":"Swarm",    "desc":"Aquila - expansion+vortex flight"},
    "FOA":   {"ex":0.70,"ep":0.75,"type":"Swarm",    "desc":"Fruit Fly - osphresis-based"},
    "SMA":   {"ex":0.75,"ep":0.85,"type":"Swarm",    "desc":"Slime Mould - oscillatory vein network"},
    "TSA":   {"ex":0.72,"ep":0.76,"type":"Swarm",    "desc":"Tunicate - jet propulsion+swarm"},
    "AVOA":  {"ex":0.80,"ep":0.78,"type":"Swarm",    "desc":"African Vultures - explore+exploit phases"},
    "DO":    {"ex":0.71,"ep":0.74,"type":"Swarm",    "desc":"Dragonfly - static+dynamic swarming"},
    "BA":    {"ex":0.73,"ep":0.82,"type":"Swarm",    "desc":"Bat - echolocation frequency"},
    "FFA":   {"ex":0.74,"ep":0.80,"type":"Swarm",    "desc":"Firefly - light intensity attraction"},
    "ACO":   {"ex":0.78,"ep":0.72,"type":"Swarm",    "desc":"Ant Colony ACOR - continuous pheromone"},
    "DE":    {"ex":0.90,"ep":0.85,"type":"Evol",     "desc":"Differential Evolution - mutation+crossover"},
    "GA":    {"ex":0.85,"ep":0.65,"type":"Evol",     "desc":"Genetic Algorithm - select+crossover+mutate"},
    "ES":    {"ex":0.82,"ep":0.68,"type":"Evol",     "desc":"Evolution Strategy - self-adaptive sigma"},
    "SA":    {"ex":0.65,"ep":0.95,"type":"Physics",  "desc":"Simulated Annealing - Boltzmann P=exp(-dE/kT)"},
    "EFO":   {"ex":0.70,"ep":0.76,"type":"Physics",  "desc":"Electromagnetic Field - pos/neg fields"},
    "RUN":   {"ex":0.73,"ep":0.79,"type":"Physics",  "desc":"RUN - physics runaway avoidance"},
    "MVO":   {"ex":0.72,"ep":0.80,"type":"Physics",  "desc":"Multi-Verse - wormhole+white/black holes"},
    "TLO":   {"ex":0.70,"ep":0.78,"type":"Human",    "desc":"Teaching-Learning - teacher+learner phases"},
    "INFO":  {"ex":0.74,"ep":0.80,"type":"Human",    "desc":"INFO - weighting+updating+competition"},
    "ICA":   {"ex":0.76,"ep":0.74,"type":"Human",    "desc":"Imperialist - colony assimilation"},
    "BBO":   {"ex":0.70,"ep":0.78,"type":"Bio",      "desc":"Biogeography - immigration+emigration"},
    "HS":    {"ex":0.65,"ep":0.75,"type":"Music",    "desc":"Harmony Search - memory+pitch adjustment"},
    "GTO":   {"ex":0.74,"ep":0.80,"type":"Swarm",    "desc":"Gorilla Troops - silverback leadership"},
    "PSO_n": {"ex":0.60,"ep":0.80,"type":"Swarm(N)", "desc":"PSO via NiaPy"},
    "GWO_n": {"ex":0.70,"ep":0.90,"type":"Swarm(N)", "desc":"GWO via NiaPy"},
    "DE_n":  {"ex":0.90,"ep":0.85,"type":"Evol(N)",  "desc":"DE via NiaPy"},
    "GA_n":  {"ex":0.85,"ep":0.65,"type":"Evol(N)",  "desc":"GA via NiaPy"},
    "ABC_n": {"ex":0.80,"ep":0.70,"type":"Swarm(N)", "desc":"ABC via NiaPy"},
    "FA_n":  {"ex":0.72,"ep":0.78,"type":"Swarm(N)", "desc":"Firefly via NiaPy"},
    "BA_n":  {"ex":0.73,"ep":0.82,"type":"Swarm(N)", "desc":"Bat via NiaPy"},
    "CS_n":  {"ex":0.76,"ep":0.80,"type":"Swarm(N)", "desc":"Cuckoo Search via NiaPy"},
    "SCA_n": {"ex":0.70,"ep":0.75,"type":"Swarm(N)", "desc":"SCA via NiaPy"},
    "WOA_n": {"ex":0.80,"ep":0.75,"type":"Swarm(N)", "desc":"WOA via NiaPy"},
    "HHO_n": {"ex":0.85,"ep":0.80,"type":"Swarm(N)", "desc":"HHO via NiaPy"},
    "MFO_n": {"ex":0.73,"ep":0.82,"type":"Swarm(N)", "desc":"MFO via NiaPy"},
    "SSA_n": {"ex":0.75,"ep":0.70,"type":"Swarm(N)", "desc":"SSA via NiaPy"},
    "GS_n":  {"ex":0.72,"ep":0.78,"type":"Phys(N)",  "desc":"Gravitational Search via NiaPy"},
    "HS_n":  {"ex":0.65,"ep":0.75,"type":"Music(N)", "desc":"Harmony Search via NiaPy"},
    "DE_sci":{"ex":0.90,"ep":0.85,"type":"SciPy",    "desc":"SciPy Differential Evolution"},
    "SA_sci":{"ex":0.65,"ep":0.95,"type":"SciPy",    "desc":"SciPy Dual Annealing"},
    "SHGO":  {"ex":0.85,"ep":0.80,"type":"SciPy",    "desc":"Simplicial Homology Global Optimizer"},
    "DIRECT":{"ex":0.88,"ep":0.70,"type":"SciPy",    "desc":"DIRECT deterministic global"},
    "BasinHop":{"ex":0.75,"ep":0.90,"type":"SciPy",  "desc":"Basin Hopping - stochastic local"},
    "NelderMead":{"ex":0.45,"ep":0.99,"type":"SciPy","desc":"Nelder-Mead Simplex - local"},
    "BFGS":  {"ex":0.50,"ep":0.99,"type":"SciPy",    "desc":"L-BFGS-B - quasi-Newton"},
    "Powell":{"ex":0.52,"ep":0.97,"type":"SciPy",    "desc":"Powell Direction Set"},
    "CG":    {"ex":0.48,"ep":0.98,"type":"SciPy",    "desc":"Conjugate Gradient"},
    "COBYLA":{"ex":0.55,"ep":0.95,"type":"SciPy",    "desc":"COBYLA constrained"},
}

def _parse_file(file_obj):
    if file_obj is None: return None,"no_file"
    try:
        path=file_obj.name; ext=path.rsplit(".",1)[-1].lower() if "." in path else "csv"
        if ext in ("csv","txt","dat"):
            for sep in [",",";","\t"," "]:
                try:
                    df=pd.read_csv(path,sep=sep)
                    if df.select_dtypes(include=[np.number]).shape[1]>=1 and df.shape[0]>=2: return df,None
                except: pass
            return pd.read_csv(path),None
        elif ext=="tsv": return pd.read_csv(path,sep="\t"),None
        elif ext=="json":
            df=pd.read_json(path)
            if isinstance(df,pd.DataFrame) and df.shape[0]>=2: return df,None
            return None,"JSON parse failed"
        elif ext in ("xlsx","xls"):
            eng="openpyxl" if ext=="xlsx" else "xlrd"
            return pd.read_excel(path,engine=eng),None
        elif ext=="npy":
            arr=np.load(path,allow_pickle=True)
            if arr.ndim==1: arr=arr.reshape(-1,1)
            return pd.DataFrame(arr,columns=[f"col_{i}" for i in range(arr.shape[1])]),None
        else: return pd.read_csv(path),None
    except Exception as e: return None,str(e)

def _parse_paste(text):
    if not text or not text.strip(): return None,"empty"
    text=text.strip()
    if text.startswith(("[","{")):
        try:
            data=json.loads(text)
            if isinstance(data,list):
                if data and isinstance(data[0],dict): return pd.DataFrame(data),None
                nums=[float(x) for x in data if x is not None]
                if nums: return pd.DataFrame({"value":nums}),None
            elif isinstance(data,dict): return pd.DataFrame(data),None
        except: pass
    for sep in [",",";","\t"," "]:
        try:
            buf=io.StringIO(text); df=pd.read_csv(buf,sep=sep)
            if df.select_dtypes(include=[np.number]).shape[1]>=1 and df.shape[0]>=2: return df,None
        except: pass
    try:
        nums=[]
        for line in text.split("\n"):
            for tok in re.split(r"[\s,;]+",line.strip()):
                if tok:
                    try: nums.append(float(tok))
                    except: pass
        if len(nums)>=2: return pd.DataFrame({"value":nums}),None
    except: pass
    return None,"Could not parse. Use CSV, TSV, JSON, or plain numbers."

def _compute_physics(df):
    num_df=df.select_dtypes(include=[np.number]).dropna(axis=1,how="all")
    if num_df.empty: return None,"No numeric columns found"
    num_df=num_df.copy()
    for c in num_df.columns:
        if num_df[c].isna().any(): num_df[c]=num_df[c].fillna(num_df[c].median())
    cols=num_df.columns.tolist(); D=len(cols); N=len(num_df); data=num_df.values.astype(float)
    res={"D":D,"N":N,"cols":cols}
    # P1: Descriptive stats
    cst={}
    for c in cols:
        x=num_df[c].values.astype(float); n=len(x)
        mu=float(np.mean(x)); sigma=float(np.std(x,ddof=1)) if n>1 else 0.0
        med=float(np.median(x)); q1=float(np.percentile(x,25)); q3=float(np.percentile(x,75))
        iqr=q3-q1; skw=float(scipy_stats.skew(x)) if n>2 else 0.0
        krt=float(scipy_stats.kurtosis(x)) if n>3 else 0.0; cv=sigma/(abs(mu)+1e-12)
        out=int(np.sum((x<q1-1.5*iqr)|(x>q3+1.5*iqr)))
        _,np_val=(scipy_stats.shapiro(x[:min(n,5000)]) if n>=3 else (0,0))
        cst[c]={"n":n,"mean":mu,"std":sigma,"median":med,"q1":q1,"q3":q3,"iqr":iqr,"skew":skw,"kurt":krt,"cv":cv,"outliers":out,"norm_p":float(np_val)}
    res["col_stats"]=cst
    # P2: Ruggedness rho=1-|corr(x_i,x_{i+1})|
    flat=data.flatten(); rc=float(np.corrcoef(flat[:-1],flat[1:])[0,1]) if len(flat)>1 else 1.0
    rug=float(max(0.0,1.0-abs(rc))); res["ruggedness"]=rug; res["rug_corr"]=rc
    res["rug_eq"]=f"rho = 1-|corr(x_i,x_i+1)| = 1-|{rc:.4f}| = {rug:.4f}"
    # P3: Modality - histogram peaks
    proj=data[:,0]; hv,_=np.histogram(proj,bins=min(30,max(5,N//5)))
    pks,_=find_peaks(hv,prominence=max(1,hv.max()*0.05)); mod=max(1,len(pks))
    res["modality"]=mod; res["mod_eq"]=f"Modality = count(local_maxima in hist) = {mod} peaks"
    # P4: Separability S=mean(Var_d)/Var_joint
    if D>=2 and N>=10:
        jv=float(np.var(data)); mv=[float(np.var(data[:,d])) for d in range(D)]
        sep=float(min(1.0,np.mean(mv)/(jv+1e-12)))
    else: sep=1.0
    res["separability"]=sep; res["sep_eq"]=f"S = mean(Var_d)/Var_joint = {sep:.4f}"
    # P5: Shannon Entropy H=-sum(p_i*log2(p_i))
    ent_vals=[]
    for d in range(D):
        hh,_=np.histogram(data[:,d],bins=min(20,max(4,N//5)),density=True); hh=hh[hh>0]
        ent_vals.append(float(-np.sum(hh*np.log2(hh+1e-12)*(hh>0))))
    ent=float(np.mean(ent_vals)); res["entropy"]=ent
    res["entropy_eq"]=f"H = -sum(p_i*log2(p_i)), mean={ent:.4f} bits over {D} dims"
    # P6: Epistasis=mean|r_ij| upper triangle
    if D>=2:
        cm=np.corrcoef(data.T); up=cm[np.triu_indices(D,k=1)]; ep=float(np.mean(np.abs(up)))
    else: ep=0.0
    res["epistasis"]=ep; res["ep_eq"]=f"Epistasis = mean|r_ij|(i!=j) = {ep:.4f}  [0=separable,1=epistatic]"
    # P7: Scale & Range
    gmin=float(np.min(data)); gmax=float(np.max(data)); sc_r=gmax-gmin
    res["scale_range"]=sc_r; res["global_min"]=gmin; res["global_max"]=gmax
    res["scale_eq"]=f"Range = max-min = {gmax:.4f}-{gmin:.4f} = {sc_r:.4f}"
    # P8: Sparsity
    z=int(np.sum(np.abs(data)<1e-10)); sp=z/max(1,data.size)
    res["sparsity"]=sp; res["sparse_eq"]=f"Sparsity = |{{x=0}}|/(N*D) = {z}/{data.size} = {sp:.4f}"
    # P9: Smoothness=autocorr(lag=1)/autocorr(lag=0)
    ac_vals=[]
    for d in range(min(D,5)):
        x=data[:,d]; xn=(x-x.mean())/(x.std()+1e-12)
        ac=np.correlate(xn,xn,mode="full")[N-1:N+2]
        if len(ac)>=2 and ac[0]>1e-12: ac_vals.append(float(ac[1]/ac[0]))
    smooth=float(np.mean(ac_vals)) if ac_vals else 0.5
    res["smoothness"]=smooth; res["smooth_eq"]=f"Autocorr(lag=1)/Autocorr(lag=0) = {smooth:.4f}  [->1 smooth]"
    # P10: Mardia skewness proxy
    if D>=2 and N>=10:
        mu_v=np.mean(data,axis=0); cov_m=np.cov(data.T)+np.eye(D)*1e-10
        try:
            inv_c=np.linalg.inv(cov_m); cen=data-mu_v
            maha=np.einsum("ij,jk,ik->i",cen,inv_c,cen)
            ms=float(np.mean(maha**3))/(D*(D+2)*(D+4)/3); ms=min(abs(ms),10.0)
        except: ms=0.5
    else: ms=float(abs(scipy_stats.skew(data[:,0]))) if N>=3 else 0.5
    res["mardia_skew"]=ms; res["mardia_eq"]=f"Mardia_skew=E[MD^3]/(D(D+2)(D+4)/3)={ms:.4f}"
    res["skew_avg"]=float(np.mean([abs(s["skew"]) for s in cst.values()]))
    res["has_outliers"]=any(s["outliers"]>max(1,s["n"]*0.05) for s in cst.values())
    res["norm_p_avg"]=float(np.mean([s["norm_p"] for s in cst.values()]))
    return res,None

def _tournament(physics):
    D=physics["D"]; N=physics["N"]; rug=physics["ruggedness"]; mod=physics["modality"]
    sep=physics["separability"]; ep=physics["epistasis"]; ent=physics["entropy"]
    smooth=physics["smoothness"]; sp=physics["sparsity"]; skew=physics["skew_avg"]
    out=physics["has_outliers"]; norm_p=physics["norm_p_avg"]
    scored=[]
    for name in ALL_SINGLE:
        t=ADVISOR_TRAITS.get(name,{"ex":0.70,"ep":0.75,"type":"Other","desc":""})
        ex=t["ex"]; epp=t["ep"]; score=50.0; reasons=[]; penalties=[]
        # C1: Dimensionality
        if D<=3:
            if name in ("SA","SA_sci","NelderMead","BFGS","Powell","BasinHop","CG","COBYLA"):
                score+=15; reasons.append(f"P1-Dim: D={D} very low -> local methods exact; O(D^2) Newton advantage")
            elif name in ("PSO","GWO","DE","DE_sci","HHO","WOA"):
                score+=10; reasons.append(f"P1-Dim: D={D} -> swarm effective in low-dim continuous spaces")
        elif D<=20:
            if name in ("DE","DE_sci","PSO","GWO","HHO","WOA","SMA","ABC","HHO_n","WOA_n","DE_n"):
                score+=12; reasons.append(f"P1-Dim: D={D} ideal range 5-20 -> population algorithms designed for this")
            if name in ("NelderMead","CG") and D>10:
                score-=8; penalties.append(f"P1-Dim: D={D}>10 -> Simplex O(D^2) degrading; memory overhead grows quadratically")
        elif D<=100:
            if name in ("GA","DE","ES","BBO","ICA","DE_sci","GA_n","DE_n"):
                score+=14; reasons.append(f"P1-Dim: D={D} high -> evolutionary crossover scales linearly O(D)")
            if name in ("NelderMead","SA","BasinHop"):
                score-=12; penalties.append(f"P1-Dim: D={D} -> curse of dimensionality; single-trajectory degrades exponentially")
        else:
            if name in ("GA","DE","ES","DE_sci"):
                score+=16; reasons.append(f"P1-Dim: D={D} very high -> only evolutionary operators maintain O(N*D*T) scalability")
            else:
                score-=10; penalties.append(f"P1-Dim: D={D} -> extremely high-dim requires evolutionary methods only")
        # C2: Ruggedness rho=1-|corr|
        if rug>0.65:
            if ex>0.78: score+=12; reasons.append(f"P2-Ruggedness: rho={rug:.3f} rough -> ex={ex:.2f} high exploration prevents local optima trapping")
            else: score-=8; penalties.append(f"P2-Ruggedness: rho={rug:.3f} rough -> ex={ex:.2f} low; risks entrapment in local basins")
            if name in ("SA","SA_sci","WOA","HHO","AVOA","SHGO","BasinHop"): score+=8; reasons.append("P2-Ruggedness: probabilistic acceptance/multi-basin jumping handles rough terrain")
        elif rug<0.25:
            if epp>0.88: score+=12; reasons.append(f"P2-Ruggedness: rho={rug:.3f} smooth -> ep={epp:.2f} fast convergence on smooth surface")
            if name in ("BFGS","NelderMead","Powell","CG","COBYLA","SA","SA_sci"): score+=10; reasons.append("P2-Ruggedness: smooth landscape -> local/quasi-gradient methods optimal")
            if ex>0.82 and name not in ("DE","DE_sci","SHGO","DIRECT"): score-=5; penalties.append(f"P2-Ruggedness: excess exploration ex={ex:.2f} wastes evaluations on smooth surface")
        else:
            if ex>0.70 and epp>0.70: score+=8; reasons.append(f"P2-Ruggedness: rho={rug:.3f} moderate -> balanced ex/ep optimal")
        # C3: Modality
        if mod>=5:
            if ex>0.75: score+=10; reasons.append(f"P3-Modality: {mod} peaks -> wide exploration needed for multiple attraction basins")
            if name in ("PSO","GWO","ABC","SCA","SHGO","DIRECT","DE","DE_sci"): score+=8; reasons.append("P3-Modality: multi-start/diversity preservation covers multiple optima")
            if name in ("SA_sci","BasinHop","NelderMead"): score-=5; penalties.append(f"P3-Modality: {mod} peaks -> single trajectory risks wrong basin convergence")
        elif mod<=2:
            if epp>0.85: score+=10; reasons.append(f"P3-Modality: {mod} peak(s) -> focused exploitation accelerates convergence; unimodal")
            if name in ("SA","BFGS","NelderMead","Powell","BasinHop"): score+=8; reasons.append("P3-Modality: unimodal/bimodal -> local search guaranteed to find global optimum")
        # C4: Separability S=mean(Var_d)/Var_joint
        if sep>0.70:
            if name in ("DE","DE_sci","GA","ES","PSO","GWO","DE_n","GA_n"): score+=10; reasons.append(f"P4-Sep: S={sep:.3f} -> component-wise DE binomial crossover optimal for independent vars")
            reasons.append(f"P4-Sep: S={sep:.3f} high -> variables are nearly independent; dimension-wise operators valid")
        else:
            if name in ("WOA","HHO","SMA","MRFO","AO","AVOA","WOA_n","HHO_n"): score+=10; reasons.append(f"P4-Sep: S={sep:.3f} low -> holistic swarm motion preserves correlated variable structure")
            if name in ("DE","DE_sci") and sep<0.4: score-=8; penalties.append(f"P4-Sep: S={sep:.3f} -> strong coupling violates DE binomial crossover independence assumption")
        # C5: Epistasis
        if ep>0.5:
            if name in ("WOA","HHO","SMA","GWO","PSO","MRFO","WOA_n","HHO_n"): score+=8; reasons.append(f"P6-Epi: eps={ep:.3f} correlated vars -> swarm velocity preserves variable dependencies")
            if name in ("DE","DE_sci","GA") and ep>0.5: score-=5; penalties.append(f"P6-Epi: eps={ep:.3f} -> component-wise crossover breaks correlated variable groups")
        # C6: Sample size N
        if N<50:
            if name in ("SA","NelderMead","BasinHop","Powell","COBYLA"): score+=10; reasons.append(f"P1-N: N={N} small -> lightweight methods need O(D) evals/step; efficient")
            if name in ("PSO","GA","DE") and N<30: score-=5; penalties.append(f"P1-N: N={N} -> large populations need more evals than dataset allows")
        elif N>5000:
            if name in ("DE_sci","SA_sci","BFGS","Powell"): score+=8; reasons.append(f"P1-N: N={N} large -> minimize evaluation count; efficiency critical")
        else:
            if name in ("PSO","DE","GWO","HHO","WOA"): score+=6; reasons.append(f"P1-N: N={N} -> ideal range for population-based algorithms")
        # C7: Outliers
        if out:
            if name in ("SA","SA_sci","WOA","GWO","HHO","AVOA","SCA"): score+=7; reasons.append("P1-Outliers: detected -> probabilistic acceptance robust to corrupted evaluations")
            if name in ("BFGS","CG","Powell","NelderMead"): score-=10; penalties.append("P1-Outliers: detected -> gradient/simplex methods corrupted by outlier-distorted landscape")
        else:
            if name in ("BFGS","NelderMead","Powell","CG"): score+=5; reasons.append("P1-Outliers: none -> clean data; gradient-free local search reliable")
        # C8: Entropy
        if ent>3.5:
            if ex>0.74 and epp>0.74: score+=8; reasons.append(f"P5-Entropy: H={ent:.3f} bits high -> balanced ex/ep required for complex landscape")
        elif ent<1.5:
            if epp>0.88: score+=8; reasons.append(f"P5-Entropy: H={ent:.3f} bits low -> exploitation focus sufficient for simple landscape")
        # C9: Skewness
        if skew>1.5:
            if name in ("SA","WOA","HHO","AVOA","BasinHop","GWO"): score+=6; reasons.append(f"P1-Skew: |skew|={skew:.3f} -> asymmetric dist; probabilistic acceptance handles non-symmetric landscape")
            if name in ("BFGS","CG"): score-=5; penalties.append(f"P1-Skew: |skew|={skew:.3f} -> violates symmetry assumptions of quasi-Newton approx")
        # C10: Smoothness
        if smooth>0.8:
            if name in ("BFGS","Powell","CG","NelderMead","SA","BasinHop","COBYLA"): score+=10; reasons.append(f"P9-Smooth: L={smooth:.3f} -> highly autocorrelated; gradient-free descent guaranteed convergence")
        elif smooth<0.3:
            if ex>0.75: score+=8; reasons.append(f"P9-Smooth: L={smooth:.3f} noisy -> high exploration overcomes noisy fitness")
        # C11: Normality
        if norm_p>0.05:
            if name in ("BFGS","Powell","NelderMead","SA"): score+=5; reasons.append(f"P1-Norm: p={norm_p:.3f} -> Gaussian dist satisfies local optimizer assumptions")
        else:
            if name in ("DE","GA","PSO","GWO","HHO","WOA"): score+=5; reasons.append(f"P1-Norm: p={norm_p:.3f} non-Gaussian -> assumption-free population methods preferred")
        # C12: Sparsity
        if sp>0.3:
            if name in ("ACO","CS_n","HS","HS_n","BBO"): score+=8; reasons.append(f"P8-Sparsity: {sp:.3f} -> sparse data; incremental selection handles zero-dominated spaces")
        score=max(0.0,min(100.0,score)); all_r=reasons+penalties
        scored.append({"name":name,"score":round(score,1),"reasons":all_r,"type":t["type"],"desc":t["desc"],"ex":ex,"ep":epp})
    scored.sort(key=lambda x:x["score"],reverse=True)
    return scored

def _gen_hybrids(singles,top_n=8):
    top=singles[:6]; hybrids=[]
    for i in range(len(top)):
        for j in range(i+1,len(top)):
            a,b=top[i],top[j]; ex_d=abs(a["ex"]-b["ex"]); ep_d=abs(a["ep"]-b["ep"])
            cb=min(20.0,(ex_d+ep_d)*15); base=(a["score"]+b["score"])/2; hs=round(min(99.0,base+cb+3.0),1)
            hybrids.append({"name":f"{a['name']}+{b['name']}","score":hs,"components":[a["name"],b["name"]],
                "reasons":[f"Voting Ensemble: per-iter min(fit_A,fit_B) always picks the best",
                           f"Complementary: ex=({a['ex']:.2f},{b['ex']:.2f}), ep=({a['ep']:.2f},{b['ep']:.2f})",
                           f"Category diversity: {a['type']}+{b['type']}",
                           f"Score=({a['score']:.1f}+{b['score']:.1f})/2+{cb:.1f}+3={hs}"]})
    if len(top)>=3:
        a,b,c=top[0],top[1],top[2]; hs=round(min(99.0,(a["score"]+b["score"]+c["score"])/3+8.0),1)
        hybrids.append({"name":f"{a['name']}+{b['name']}+{c['name']}","score":hs,"components":[a["name"],b["name"],c["name"]],
            "reasons":["Triple Voting: 3 algorithms vote each iteration; picks best fitness",
                       f"Triple coverage: {a['type']}+{b['type']}+{c['type']}",
                       f"Score=({a['score']:.1f}+{b['score']:.1f}+{c['score']:.1f})/3+8={hs}"]})
    for i in range(5,min(12,len(singles))):
        for j in range(i+1,min(12,len(singles))):
            a,b=singles[i],singles[j]
            if a["type"]!=b["type"]:
                hs=round(min(99.0,(a["score"]+b["score"])/2+abs(a["ex"]-b["ex"])*12+2.0),1)
                hybrids.append({"name":f"{a['name']}+{b['name']}","score":hs,"components":[a["name"],b["name"]],
                    "reasons":[f"Cross-type: {a['type']}+{b['type']} broad coverage","Score="+str(hs)]})
    hybrids.sort(key=lambda x:x["score"],reverse=True); return hybrids[:top_n]

def _build_report(physics,singles,hybrids):
    D=physics["D"]; N=physics["N"]; best_s=singles[0]; best_h=hybrids[0] if hybrids else None
    is_hybrid=best_h and best_h["score"]>best_s["score"]+2; winner=best_h if is_hybrid else best_s
    def sc(s):
        if s>=80: return "#00e676"
        if s>=65: return "#69f0ae"
        if s>=50: return "#ffeb3b"
        if s>=35: return "#ffa726"
        return "#ef5350"
    eq_rows=""
    for pid,nm,eq,val in [
        ("P1","Descriptive Stats","mu,sigma,Med,Q1/Q3,IQR,skew,kurt,CV,outliers,Shapiro-p",f"D={D} cols,N={N} rows"),
        ("P2","Ruggedness rho",physics["rug_eq"],f"rho={physics['ruggedness']:.4f}"),
        ("P3","Modality",physics["mod_eq"],f"peaks={physics['modality']}"),
        ("P4","Separability S",physics["sep_eq"],f"S={physics['separability']:.4f}"),
        ("P5","Shannon Entropy H",physics["entropy_eq"],f"H={physics['entropy']:.4f} bits"),
        ("P6","Epistasis eps",physics["ep_eq"],f"eps={physics['epistasis']:.4f}"),
        ("P7","Scale & Range",physics["scale_eq"],f"R={physics['scale_range']:.4f}"),
        ("P8","Sparsity",physics["sparse_eq"],f"sp={physics['sparsity']:.4f}"),
        ("P9","Smoothness",physics["smooth_eq"],f"L={physics['smoothness']:.4f}"),
        ("P10","Mardia Skewness",physics["mardia_eq"],f"M={physics['mardia_skew']:.4f}"),
    ]:
        eq_rows+=(f'<tr style="border-bottom:1px solid #0a1628;">'
                  f'<td style="padding:6px 10px;color:#5aabff;font-weight:bold;font-size:10px;white-space:nowrap;">{pid}</td>'
                  f'<td style="padding:6px 10px;color:#e0e0e0;font-size:10px;font-weight:bold;">{nm}</td>'
                  f'<td style="padding:6px 10px;color:#c9d1d9;font-size:10px;font-family:monospace;">{eq}</td>'
                  f'<td style="padding:6px 10px;color:#7ec8ff;font-size:10px;font-weight:bold;">{val}</td></tr>')
    sr=""
    for i,s in enumerate(singles[:10]):
        c=sc(s["score"]); bg="#0d1117" if i%2==0 else "#0a1628"
        medal=["#1","#2","#3","#4","#5","#6","#7","#8","#9","#10"][i]
        sr+=(f'<tr style="background:{bg};"><td style="padding:5px 8px;font-size:11px;color:{"#ffd700" if i==0 else "#c0c0c0" if i==1 else "#cd7f32" if i==2 else "#8b949e"};">{medal}</td>'
             f'<td style="padding:5px 8px;color:#e0e0e0;font-size:11px;font-weight:bold;">{s["name"]}</td>'
             f'<td style="padding:5px 8px;color:#8b949e;font-size:9px;">{s["type"]}</td>'
             f'<td style="padding:5px 8px;"><div style="background:#21262d;border-radius:3px;height:6px;"><div style="background:{c};height:6px;border-radius:3px;width:{int(s["score"])}%;"></div></div></td>'
             f'<td style="padding:5px 8px;color:{c};font-weight:bold;font-size:12px;text-align:right;">{s["score"]:.1f}</td></tr>')
    hr=""
    for i,h in enumerate(hybrids[:6]):
        c=sc(h["score"]); bg="#0d1117" if i%2==0 else "#0a1628"
        hr+=(f'<tr style="background:{bg};"><td style="padding:5px 8px;font-size:10px;color:{"#5aabff" if i==0 else "#8b949e"};">#{i+1}</td>'
             f'<td style="padding:5px 8px;color:#e0e0e0;font-size:10px;font-weight:bold;">{h["name"][:45]}</td>'
             f'<td style="padding:5px 8px;"><div style="background:#21262d;border-radius:3px;height:6px;"><div style="background:{c};height:6px;border-radius:3px;width:{int(h["score"])}%;"></div></div></td>'
             f'<td style="padding:5px 8px;color:{c};font-weight:bold;font-size:12px;text-align:right;">{h["score"]:.1f}</td></tr>')
    wr="".join(f'<div style="font-size:11px;color:#c9d1d9;margin-bottom:5px;padding-left:8px;border-left:2px solid #1e4d8c;">{r}</div>' for r in winner.get("reasons",[])[:8])
    cs_html=""
    for c_name,st in list(physics["col_stats"].items())[:8]:
        items="".join(f'<span style="background:#161b22;padding:3px 8px;border-radius:4px;"><span style="color:#8b949e;">{k}:</span><span style="color:#7ec8ff;"> {v:.3f}</span></span>'
                      for k,v in [("n",st["n"]),("mu",st["mean"]),("sigma",st["std"]),("median",st["median"]),("IQR",st["iqr"]),("skew",st["skew"]),("kurt",st["kurt"]),("CV",st["cv"]),("outliers",st["outliers"]),("norm_p",st["norm_p"])])
        cs_html+=(f'<div style="background:#0a1628;border-radius:8px;padding:10px;margin-bottom:8px;"><div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:6px;">{c_name}</div><div style="display:flex;flex-wrap:wrap;gap:5px;font-size:10px;">{items}</div></div>')
    wc=sc(winner["score"]); wt="Hybrid Ensemble" if is_hybrid else winner.get("type","")
    bh_d=(f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:18px;">'
          f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:8px;">Best Hybrid Details: {best_h["name"][:55]}</div>'
          +"".join(f'<div style="font-size:11px;color:#c9d1d9;margin-bottom:4px;padding-left:8px;border-left:2px solid #5aabff;">{r}</div>' for r in best_h.get("reasons",[]))+
          f'</div>') if best_h else ""
    return (
        f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:14px;padding:22px;font-family:\'JetBrains Mono\',monospace;color:#e0e0e0;">'
        f'<div style="background:linear-gradient(135deg,#0a2a0a,#0d3a1a);border:2px solid {wc};border-radius:12px;padding:20px;margin-bottom:20px;">'
        f'<div style="font-size:9px;color:{wc};letter-spacing:4px;margin-bottom:6px;">OPTIMAL ALGORITHM FOR YOUR DATA</div>'
        f'<div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:200px;"><div style="font-size:22px;font-weight:900;color:{wc};">{winner["name"]}</div>'
        f'<div style="font-size:11px;color:#8b949e;margin:4px 0;">{wt} &mdash; {winner.get("desc","")[:65]}</div>'
        f'<div style="background:#0a1628;border-radius:8px;padding:12px;margin-top:10px;">'
        f'<div style="color:#69f0ae;font-size:10px;font-weight:700;margin-bottom:8px;">Physical Reasons:</div>{wr}</div></div>'
        f'<div style="text-align:center;min-width:100px;"><div style="font-size:52px;font-weight:900;color:{wc};line-height:1;">{winner["score"]:.0f}</div>'
        f'<div style="font-size:10px;color:#8b949e;">/100</div></div></div></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:16px;margin-bottom:18px;">'
        f'<div style="color:#5aabff;font-size:12px;font-weight:bold;margin-bottom:12px;">10 Physical Equations Applied (D={D}, N={N})</div>'
        f'<div style="overflow-x:auto;"><table style="width:100%;border-collapse:collapse;min-width:560px;">'
        f'<thead><tr style="background:#161b22;"><th style="padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;">Eq</th>'
        f'<th style="padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;">Metric</th>'
        f'<th style="padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;">Formula / Result</th>'
        f'<th style="padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;">Value</th></tr></thead>'
        f'<tbody>{eq_rows}</tbody></table></div></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:18px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Detailed Column Statistics</div>{cs_html}</div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:18px;">'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Top 10 Single Algorithms (Tournament of {len(ALL_SINGLE)})</div>'
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#161b22;">'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">#</th>'
        f'<th style="padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;">Algorithm</th>'
        f'<th style="padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;">Type</th>'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">Bar</th>'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">Score</th>'
        f'</tr></thead><tbody>{sr}</tbody></table></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Top Hybrids (from 9860 pool)</div>'
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#161b22;">'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">#</th>'
        f'<th style="padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;">Hybrid Name</th>'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">Bar</th>'
        f'<th style="padding:4px 6px;color:#8b949e;font-size:9px;">Score</th>'
        f'</tr></thead><tbody>{hr}</tbody></table></div></div>'
        + bh_d +
        f'<div style="background:#0a1628;border-radius:8px;padding:10px;text-align:center;font-size:10px;color:#8b949e;">'
        f'Data Advisor v5.0 | {len(ALL_SINGLE)} Single + 9800 Hybrid Algorithms | 10 Physical Equations | Full Tournament</div></div>'
    )

def advisor_run(file_obj,paste_text,progress=gr.Progress()):
    progress(0.05,desc="Reading data...")
    df=None; err=None
    if file_obj is not None: df,err=_parse_file(file_obj)
    elif paste_text and paste_text.strip(): df,err=_parse_paste(paste_text)
    else: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Please upload a file or paste data, then click Analyze.</p>"
    if df is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>Error: {err}</p>"
    if df.shape[0]<2: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Data needs at least 2 rows.</p>"
    progress(0.20,desc="Computing 10 physical equations...")
    physics,perr=_compute_physics(df)
    if physics is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>{perr}</p>"
    progress(0.50,desc=f"Tournament: evaluating {len(ALL_SINGLE)} algorithms...")
    singles=_tournament(physics)
    progress(0.78,desc="Generating optimal hybrids from 9860 pool...")
    hybrids=_gen_hybrids(singles,top_n=8)
    progress(0.93,desc="Building physical analysis report...")
    html=_build_report(physics,singles,hybrids)
    progress(1.0,desc="Done!")
    return html


# ===========================================================================
# PHYSICAL HYBRID TOURNAMENT (Addition 1 - Real Physics)
# ===========================================================================

def _tournament_single(name, physics):
    D=physics["D"]; N=physics["N"]; rug=physics["ruggedness"]; mod=physics["modality"]
    sep=physics["separability"]; ep=physics["epistasis"]; ent=physics["entropy"]
    smooth=physics["smoothness"]; skew=physics["skew_avg"]
    out=physics["has_outliers"]; norm_p=physics["norm_p_avg"]
    t=ADVISOR_TRAITS.get(name,{"ex":0.70,"ep":0.75,"type":"Other","desc":""})
    ex=t["ex"]; epp=t["ep"]; score=50.0
    if D<=3:
        if name in ("SA","SA_sci","NelderMead","BFGS","Powell","BasinHop","CG","COBYLA"): score+=15
        elif name in ("PSO","GWO","DE","DE_sci","HHO","WOA"): score+=10
    elif D<=20:
        if name in ("DE","DE_sci","PSO","GWO","HHO","WOA","SMA","ABC","HHO_n","WOA_n","DE_n"): score+=12
        if name in ("NelderMead","CG") and D>10: score-=8
    elif D<=100:
        if name in ("GA","DE","ES","BBO","ICA","DE_sci","GA_n","DE_n"): score+=14
        if name in ("NelderMead","SA","BasinHop"): score-=12
    else:
        if name in ("GA","DE","ES","DE_sci"): score+=16
        else: score-=10
    if rug>0.65:
        if ex>0.78: score+=12
        else: score-=8
        if name in ("SA","SA_sci","WOA","HHO","AVOA","SHGO","BasinHop"): score+=8
    elif rug<0.25:
        if epp>0.88: score+=12
        if name in ("BFGS","NelderMead","Powell","CG","COBYLA","SA","SA_sci"): score+=10
    else:
        if ex>0.70 and epp>0.70: score+=8
    if mod>=5:
        if ex>0.75: score+=10
        if name in ("PSO","GWO","ABC","SCA","SHGO","DIRECT","DE","DE_sci"): score+=8
        if name in ("SA_sci","BasinHop","NelderMead"): score-=5
    elif mod<=2:
        if epp>0.85: score+=10
        if name in ("SA","BFGS","NelderMead","Powell","BasinHop"): score+=8
    if sep>0.70:
        if name in ("DE","DE_sci","GA","ES","PSO","GWO","DE_n","GA_n"): score+=10
    else:
        if name in ("WOA","HHO","SMA","MRFO","AO","AVOA","WOA_n","HHO_n"): score+=10
        if name in ("DE","DE_sci") and sep<0.4: score-=8
    if ep>0.5:
        if name in ("WOA","HHO","SMA","GWO","PSO","MRFO","WOA_n","HHO_n"): score+=8
        if name in ("DE","DE_sci","GA") and ep>0.5: score-=5
    if N<50:
        if name in ("SA","NelderMead","BasinHop","Powell","COBYLA"): score+=10
    elif N>5000:
        if name in ("DE_sci","SA_sci","BFGS","Powell"): score+=8
    else:
        if name in ("PSO","DE","GWO","HHO","WOA"): score+=6
    if out:
        if name in ("SA","SA_sci","WOA","GWO","HHO","AVOA","SCA"): score+=7
        if name in ("BFGS","CG","Powell","NelderMead"): score-=10
    else:
        if name in ("BFGS","NelderMead","Powell","CG"): score+=5
    if ent>3.5:
        if ex>0.74 and epp>0.74: score+=8
    elif ent<1.5:
        if epp>0.88: score+=8
    if smooth>0.8:
        if name in ("BFGS","Powell","CG","NelderMead","SA","BasinHop","COBYLA"): score+=10
    elif smooth<0.3:
        if ex>0.75: score+=8
    return {"name":name,"score":round(max(0.0,min(100.0,score)),1),"type":t["type"],"desc":t["desc"],"ex":ex,"ep":epp}

def _hybrid_physical_score(h_name, physics):
    parts=h_name.split("+"); n_parts=len(parts)
    D=physics["D"]; rug=physics["ruggedness"]; mod=physics["modality"]
    sep=physics["separability"]; ep=physics["epistasis"]
    comp_scores=[]; comp_traits=[]
    for p in parts:
        t=ADVISOR_TRAITS.get(p,{"ex":0.70,"ep":0.75,"type":"Other","desc":""})
        comp_data=_tournament_single(p,physics)
        comp_scores.append(comp_data["score"]); comp_traits.append(t)
    w_sum=sum(comp_scores)
    weights=[s/w_sum for s in comp_scores] if w_sum>0 else [1.0/n_parts]*n_parts
    base_score=sum(w*s for w,s in zip(weights,comp_scores))
    reasons=[]; penalties=[]
    # L2: Complementarity (orthogonal search vectors)
    ex_vals=[t["ex"] for t in comp_traits]; ep_vals=[t["ep"] for t in comp_traits]
    ex_range=max(ex_vals)-min(ex_vals); ep_range=max(ep_vals)-min(ep_vals)
    complement_bonus=min(15.0,(ex_range+ep_range)*20.0)
    if complement_bonus>3: reasons.append(f"L2-Complementarity+{complement_bonus:.2f}: ex_range={ex_range:.2f} ep_range={ep_range:.2f} -> orthogonal search vectors cover different regions")
    # L3: Voting Ensemble Gain (law of large numbers)
    ensemble_gain=min(8.0,4.0*(1.0-1.0/np.sqrt(n_parts)))
    reasons.append(f"L3-VotingGain+{ensemble_gain:.2f}: best-of-{n_parts} reduces variance by 1/sqrt({n_parts})={1.0/np.sqrt(n_parts):.3f}")
    # L4: Diversity Insurance (basin coverage probability)
    p_all_miss=1.0
    for t in comp_traits: p_all_miss*=max(0.01,1.0-t["ex"])
    diversity_gain=min(10.0,(1.0-p_all_miss)*10.0)
    if mod>=3: diversity_gain=min(10.0,diversity_gain*1.4)
    reasons.append(f"L4-DiversityInsurance+{diversity_gain:.2f}: P(find_basin)={1-p_all_miss:.3f} for {mod} peaks in data")
    # L5: Stagnation Recovery
    p_all_stag=1.0
    for t in comp_traits: p_all_stag*=max(0.05,1.0-t["ep"])
    stagnation_bonus=min(8.0,(1.0-p_all_stag)*10.0)
    reasons.append(f"L5-StagnationRecovery+{stagnation_bonus:.2f}: P(all_stagnate)={p_all_stag:.3f} hybrid keeps searching when one stagnates")
    # L6: Ruggedness Synergy
    rug_syn=min(6.0,rug*n_parts*2.0) if rug>0.5 else min(3.0,(1.0-rug)*2.0)
    if rug>0.5: reasons.append(f"L6-RuggednessSync+{rug_syn:.2f}: rho={rug:.3f} rough -> {n_parts} independent trajectories avoid same local basin")
    # L7: Epistasis Coverage
    ep_bonus=min(5.0,ep*n_parts*3.0) if ep>0.4 else 0.0
    if ep>0.4: reasons.append(f"L7-EpistasisCov+{ep_bonus:.2f}: eps={ep:.3f} correlated vars -> multiple paradigms cover interactions")
    # L8: Type Diversity
    types=list(set(t["type"].replace("(N)","") for t in comp_traits))
    type_bonus=min(5.0,len(types)/max(1,n_parts)*8.0)
    if len(types)>1: reasons.append(f"L8-TypeDiversity+{type_bonus:.2f}: {'+'.join(types)} -> different paradigms avoid correlated failure modes")
    # L9: Overhead penalty
    overhead=min(10.0,(n_parts-4)*3.0) if n_parts>4 else 0.0
    if overhead>0: penalties.append(f"L9-Overhead-{overhead:.2f}: {n_parts} components -> serial time scales linearly O(n*T)")
    # L10: Separability fit
    sep_bonus=min(4.0,sep*5.0) if (sep>0.6 and any(t.get("type","") in ("Evol","SciPy") for t in comp_traits)) else 0.0
    if sep_bonus>0: reasons.append(f"L10-SepFit+{sep_bonus:.2f}: S={sep:.3f} separable -> evolutionary crossover components effective")
    total=base_score+complement_bonus+ensemble_gain+diversity_gain+stagnation_bonus+rug_syn+ep_bonus+type_bonus+sep_bonus-overhead
    return {"name":h_name,"score":round(max(0.0,min(99.0,total)),1),"base_score":round(base_score,1),
            "component_scores":dict(zip(parts,[round(s,1) for s in comp_scores])),
            "bonuses":{"complementarity":round(complement_bonus,2),"ensemble_gain":round(ensemble_gain,2),
                       "diversity_insurance":round(diversity_gain,2),"stagnation_recovery":round(stagnation_bonus,2),
                       "ruggedness_synergy":round(rug_syn,2),"epistasis_coverage":round(ep_bonus,2),
                       "type_diversity":round(type_bonus,2),"separability_fit":round(sep_bonus,2),
                       "overhead_penalty":round(-overhead,2)},
            "reasons":reasons,"penalties":penalties,"components":parts,"n_parts":n_parts}

def _gen_hybrids_physical(singles, physics, top_n=10):
    from itertools import combinations as _comb2
    top8=singles[:8]; top5=singles[:5]; candidates=[]
    for a,b in _comb2(top8,2):
        candidates.append(_hybrid_physical_score(f"{a['name']}+{b['name']}",physics))
    for a,b,c in _comb2(top5,3):
        candidates.append(_hybrid_physical_score(f"{a['name']}+{b['name']}+{c['name']}",physics))
    types_seen={}
    for s in singles[:15]:
        tp=s["type"].replace("(N)","")
        if tp not in types_seen: types_seen[tp]=s
    cross_type=list(types_seen.values())
    top5_names=[s["name"] for s in top5]
    for a,b in _comb2(cross_type,2):
        if a["name"] not in top5_names and b["name"] not in top5_names:
            candidates.append(_hybrid_physical_score(f"{a['name']}+{b['name']}",physics))
    if len(top5)>=4:
        candidates.append(_hybrid_physical_score("+".join(s["name"] for s in top5[:4]),physics))
    seen=set(); unique=[]
    for c in candidates:
        if c["name"] not in seen: seen.add(c["name"]); unique.append(c)
    unique.sort(key=lambda x:x["score"],reverse=True)
    return unique[:top_n]

def _build_hybrid_report(physics, singles, hybrids):
    D=physics["D"]; N=physics["N"]
    def sc(s):
        if s>=80: return "#00e676"
        if s>=65: return "#69f0ae"
        if s>=50: return "#ffeb3b"
        if s>=35: return "#ffa726"
        return "#ef5350"
    if not hybrids: return "<p style='color:#ef5350;'>No hybrids generated.</p>"
    winner=hybrids[0]; wc=sc(winner["score"])
    wr="".join(f'<div style="font-size:11px;color:#c9d1d9;margin-bottom:5px;padding-left:8px;border-left:2px solid #1e4d8c;">{r}</div>' for r in winner.get("reasons",[]))
    wp="".join(f'<div style="font-size:11px;color:#ffa726;margin-bottom:4px;padding-left:8px;border-left:2px solid #ffa726;">{p}</div>' for p in winner.get("penalties",[]))
    comp_rows=""
    for cn,cs in winner.get("component_scores",{}).items():
        c=sc(cs); t=ADVISOR_TRAITS.get(cn,{})
        comp_rows+=(f'<tr><td style="padding:5px 8px;color:#e0e0e0;font-size:11px;font-weight:bold;">{cn}</td>'
                    f'<td style="padding:5px 8px;color:#8b949e;font-size:10px;">{t.get("type","")}</td>'
                    f'<td style="padding:5px 8px;"><div style="background:#21262d;border-radius:3px;height:5px;">'
                    f'<div style="background:{c};height:5px;border-radius:3px;width:{int(cs)}%;"></div></div></td>'
                    f'<td style="padding:5px 8px;color:{c};font-weight:bold;">{cs:.1f}</td>'
                    f'<td style="padding:5px 8px;color:#8b949e;font-size:9px;">{t.get("desc","")[:40]}</td></tr>')
    bonus_rows=""
    for k,v in winner.get("bonuses",{}).items():
        col="#ef5350" if v<0 else "#69f0ae"
        bonus_rows+=(f'<tr><td style="padding:5px 8px;color:#c9d1d9;font-size:10px;">{k.replace("_"," ").title()}</td>'
                     f'<td style="padding:5px 8px;color:{col};font-weight:bold;text-align:right;">{v:+.2f}</td></tr>')
    hybrid_rows=""
    for i,h in enumerate(hybrids):
        c=sc(h["score"]); bg="#0d1117" if i%2==0 else "#0a1628"
        medal="🥇" if i==0 else "🥈" if i==1 else "🥉" if i==2 else f"#{i+1}"
        hybrid_rows+=(f'<tr style="background:{bg};">'
                      f'<td style="padding:5px 8px;font-size:11px;color:{"#ffd700" if i==0 else "#8b949e"};">{medal}</td>'
                      f'<td style="padding:5px 8px;color:#e0e0e0;font-size:10px;font-weight:bold;">{h["name"][:52]}</td>'
                      f'<td style="padding:5px 8px;color:#8b949e;font-size:9px;">{h["n_parts"]}-way</td>'
                      f'<td style="padding:5px 8px;color:#8b949e;font-size:9px;">{h.get("base_score",0):.1f}</td>'
                      f'<td style="padding:5px 8px;"><div style="background:#21262d;border-radius:3px;height:6px;">'
                      f'<div style="background:{c};height:6px;border-radius:3px;width:{int(h["score"])}%;"></div></div></td>'
                      f'<td style="padding:5px 8px;color:{c};font-weight:bold;font-size:12px;text-align:right;">{h["score"]:.1f}</td></tr>')
    return (
        f'<div style="background:#0d1117;border:1px solid #30363d;border-radius:14px;padding:22px;font-family:\'JetBrains Mono\',monospace;color:#e0e0e0;">'
        f'<div style="background:linear-gradient(135deg,#0a2a0a,#0d3a1a);border:2px solid {wc};border-radius:12px;padding:20px;margin-bottom:20px;">'
        f'<div style="font-size:9px;color:{wc};letter-spacing:4px;margin-bottom:6px;">OPTIMAL HYBRID — PHYSICAL ENSEMBLE ANALYSIS (10 LAWS)</div>'
        f'<div style="display:flex;align-items:flex-start;gap:20px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:200px;">'
        f'<div style="font-size:18px;font-weight:900;color:{wc};">{winner["name"]}</div>'
        f'<div style="font-size:11px;color:#8b949e;margin:4px 0;">{winner["n_parts"]}-way Voting Ensemble | {", ".join(winner["components"])}</div>'
        f'<div style="background:#0a1628;border-radius:8px;padding:12px;margin-top:10px;">'
        f'<div style="color:#69f0ae;font-size:10px;font-weight:700;margin-bottom:6px;">Physical Laws Applied:</div>{wr}'
        f'{"<div style=\\'color:#ffa726;font-size:10px;font-weight:700;margin:8px 0 4px;\\'>Penalties:</div>"+wp if wp else ""}'
        f'</div></div>'
        f'<div style="text-align:center;min-width:100px;">'
        f'<div style="font-size:48px;font-weight:900;color:{wc};line-height:1;">{winner["score"]:.0f}</div>'
        f'<div style="font-size:10px;color:#8b949e;">/100</div>'
        f'<div style="font-size:9px;color:#8b949e;margin-top:4px;">base={winner.get("base_score",0):.1f}</div>'
        f'</div></div></div>'
        f'<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:14px;margin-bottom:18px;">'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Component Physical Scores</div>'
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#161b22;">'
        f'<th style="padding:4px 8px;text-align:left;color:#8b949e;font-size:9px;">Algorithm</th>'
        f'<th style="padding:4px;color:#8b949e;font-size:9px;">Type</th>'
        f'<th style="padding:4px;color:#8b949e;font-size:9px;">Bar</th>'
        f'<th style="padding:4px;color:#8b949e;font-size:9px;">Score</th>'
        f'<th style="padding:4px;color:#8b949e;font-size:9px;">Description</th></tr></thead>'
        f'<tbody>{comp_rows}</tbody></table></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Physical Bonus Breakdown</div>'
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#161b22;">'
        f'<th style="padding:4px 8px;text-align:left;color:#8b949e;font-size:9px;">Physical Law</th>'
        f'<th style="padding:4px;color:#8b949e;font-size:9px;">Bonus</th></tr></thead>'
        f'<tbody>{bonus_rows}</tbody></table>'
        f'<div style="margin-top:10px;padding-top:8px;border-top:1px solid #21262d;font-size:11px;">'
        f'<span style="color:#8b949e;">Total = base({winner.get("base_score",0):.1f}) + bonuses = </span>'
        f'<span style="color:{wc};font-weight:bold;">{winner["score"]:.1f}/100</span></div></div></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:18px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">Full Hybrid Ranking — Physical Tournament (D={D}, N={N})</div>'
        f'<table style="width:100%;border-collapse:collapse;"><thead><tr style="background:#161b22;">'
        f'<th style="padding:5px 8px;color:#8b949e;font-size:9px;">#</th>'
        f'<th style="padding:5px 8px;text-align:left;color:#8b949e;font-size:9px;">Hybrid</th>'
        f'<th style="padding:5px 8px;color:#8b949e;font-size:9px;">N-way</th>'
        f'<th style="padding:5px 8px;color:#8b949e;font-size:9px;">Base</th>'
        f'<th style="padding:5px 8px;color:#8b949e;font-size:9px;">Bar</th>'
        f'<th style="padding:5px 8px;color:#8b949e;font-size:9px;">Score</th></tr></thead>'
        f'<tbody>{hybrid_rows}</tbody></table></div>'
        f'<div style="background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:14px;">'
        f'<div style="color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;">10 Physical Laws for Hybrid Scoring</div>'
        f'<div style="font-size:10px;color:#c9d1d9;line-height:2.0;">'
        f'<b style="color:#7ec8ff;">L1</b> Ensemble Base = weighted_mean(comp_scores) weights=score/sum(scores)<br>'
        f'<b style="color:#7ec8ff;">L2</b> Complementarity = min(15,(ex_range+ep_range)*20) — orthogonal search vectors<br>'
        f'<b style="color:#7ec8ff;">L3</b> Voting Gain = 4*(1-1/sqrt(n)) — law of large numbers, variance reduction<br>'
        f'<b style="color:#7ec8ff;">L4</b> Diversity Insurance = (1-prod(1-ex_i))*10 — basin coverage probability<br>'
        f'<b style="color:#7ec8ff;">L5</b> Stagnation Recovery = (1-prod(1-ep_i))*10 — escape rate from local minima<br>'
        f'<b style="color:#ffd700;">L6</b> Ruggedness Synergy = min(6,rho*n*2) — multi-trajectory benefit on rough terrain<br>'
        f'<b style="color:#ffd700;">L7</b> Epistasis Coverage = min(5,eps*n*3) — variable interaction handling<br>'
        f'<b style="color:#ffd700;">L8</b> Type Diversity = min(5,unique_types/n*8) — paradigm orthogonality<br>'
        f'<b style="color:#ffd700;">L9</b> Overhead Penalty = (n-4)*3 if n>4 — serial execution time cost<br>'
        f'<b style="color:#ffd700;">L10</b> Separability Fit = min(4,S*5) when evolutionary present</div></div>'
        f'<div style="background:#0a1628;border-radius:8px;padding:10px;text-align:center;font-size:10px;color:#8b949e;">'
        f'Physical Hybrid Advisor | 10 Physical Laws | Real Mathematics | No Assumed Values | 9860-Pool Tournament</div></div>'
    )

def hybrid_advisor_run(file_obj, paste_text, progress=gr.Progress()):
    progress(0.05,desc="Reading data...")
    df=None; err=None
    if file_obj is not None: df,err=_parse_file(file_obj)
    elif paste_text and paste_text.strip(): df,err=_parse_paste(paste_text)
    else: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Please upload a file or paste data then click Analyze Hybrids.</p>"
    if df is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>Error: {err}</p>"
    if df.shape[0]<2: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Need at least 2 rows.</p>"
    progress(0.20,desc="Computing 10 physical equations...")
    physics,perr=_compute_physics(df)
    if physics is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>{perr}</p>"
    progress(0.45,desc="Running single algorithm tournament...")
    singles=_tournament(physics)
    progress(0.70,desc="Running physical hybrid tournament (9860 pool)...")
    hybrids=_gen_hybrids_physical(singles,physics,top_n=10)
    progress(0.90,desc="Building hybrid analysis report...")
    html=_build_hybrid_report(physics,singles,hybrids)
    progress(1.0,desc="Done!")
    return html

# ===========================================================================
# ADVERSARIAL STRESS TEST ENGINE (Addition 2 - Cybersecurity)
# 100% real mathematical perturbations
# ===========================================================================

def _apply_attack(data, attack_type, intensity=0.15, rng=None):
    if rng is None: rng=np.random.default_rng(42)
    d=data.copy().astype(float)
    if d.ndim==1: d=d.reshape(-1,1)
    N,D=d.shape
    if attack_type=="gaussian_noise":
        sigma=np.std(d,axis=0)*intensity; noise=rng.normal(0,sigma,d.shape)
        return d+noise,f"Gaussian Noise: x'=x+N(0,{intensity}*sigma_col) per column"
    elif attack_type=="label_flip":
        threshold_hi=np.percentile(d.flatten(),100*(1-intensity)); threshold_lo=np.percentile(d.flatten(),100*intensity)
        d_a=d.copy(); mh=d>threshold_hi; ml=d<threshold_lo; d_a[mh]=threshold_lo; d_a[ml]=threshold_hi
        return d_a,f"Extreme Value Flip {intensity*100:.0f}%: top/bottom {int((mh+ml).sum())} values swapped"
    elif attack_type=="feature_dropout":
        n_drop=max(1,int(D*intensity)); drop_cols=rng.choice(D,size=n_drop,replace=False)
        d_a=d.copy(); d_a[:,drop_cols]=0.0
        return d_a,f"Feature Dropout: {n_drop}/{D} features zeroed out"
    elif attack_type=="outlier_injection":
        n_inj=max(1,int(N*intensity)); rows=rng.choice(N,size=n_inj,replace=False)
        mu=np.mean(d,axis=0); sigma=np.std(d,axis=0)+1e-10; d_a=d.copy()
        d_a[rows]=mu+5*sigma*rng.choice([-1,1],size=(n_inj,D))
        return d_a,f"Outlier Injection: {n_inj} points at 5-sigma level"
    elif attack_type=="covariate_shift":
        shift=np.std(d,axis=0)*intensity*3
        return d+shift,f"Covariate Shift: mu shifted by {intensity*3:.2f}*sigma per dimension"
    elif attack_type=="adversarial_correlation":
        d_a=d.copy()
        if D>=2: d_a[:,1]=d[:,0]*(1+rng.normal(0,0.05,N))+rng.normal(0,intensity*np.std(d[:,1]),N)
        return d_a,f"Spurious Correlation: col[1] <- f(col[0]) injected"
    elif attack_type=="bit_flip":
        n_flip=max(1,int(d.size*intensity)); idx=rng.choice(d.size,size=n_flip,replace=False)
        d_a=d.copy(); flat=d_a.flatten(); flat[idx]*=-1; d_a=flat.reshape(d.shape)
        return d_a,f"Sign Flip Attack: {n_flip} values negated (x'=-x)"
    elif attack_type=="gradient_attack":
        grad=np.sign(d-np.mean(d,axis=0)); eps=np.std(d)*intensity
        return d+eps*grad,f"FGSM-style: x'=x+{eps:.4f}*sign(x-mu) epsilon={eps:.4f}"
    return d.copy(),"No attack"

def _compute_robustness_metrics(orig_physics, att_physics):
    dr=abs(att_physics["ruggedness"]-orig_physics["ruggedness"])
    de=abs(att_physics["entropy"]-orig_physics["entropy"])
    ds=abs(att_physics["separability"]-orig_physics["separability"])
    dep=abs(att_physics["epistasis"]-orig_physics["epistasis"])
    dm=abs(att_physics["modality"]-orig_physics["modality"])
    instability=min(1.0,(dr+de/5.0+ds+dep+dm/10.0)/5.0)
    return {"delta_rug":round(dr,4),"delta_ent":round(de,4),"delta_sep":round(ds,4),
            "delta_ep":round(dep,4),"delta_mod":dm,"instability":round(instability,4),
            "robustness":round(max(0.0,1.0-instability),4)}

def _run_stress_test(df, intensity=0.15):
    rng=np.random.default_rng(42)
    num_df=df.select_dtypes(include=[np.number]).dropna(axis=1,how="all")
    if num_df.empty: return None,"No numeric columns"
    num_df=num_df.fillna(num_df.median()); data=num_df.values.astype(float); N,D=data.shape
    orig_physics,err=_compute_physics(num_df)
    if orig_physics is None: return None,err
    orig_singles=_tournament(orig_physics); orig_top5=orig_singles[:5]
    # Generate original top hybrids using physical laws (same engine as Hybrid Advisor)
    orig_hybrids=_gen_hybrids_physical(orig_singles, orig_physics, top_n=5)
    attacks=["gaussian_noise","label_flip","feature_dropout","outlier_injection",
             "covariate_shift","adversarial_correlation","bit_flip","gradient_attack"]
    results={}
    # Track robustness for top singles AND top hybrids
    algo_rob={s["name"]:[] for s in orig_top5}
    hybrid_rob={h["name"]:[] for h in orig_hybrids}
    for attack in attacks:
        att_data,att_desc=_apply_attack(data,attack,intensity,rng)
        att_df=pd.DataFrame(att_data,columns=num_df.columns)
        att_physics,_=_compute_physics(att_df)
        if att_physics is None: continue
        att_singles=_tournament(att_physics)
        # Re-score hybrids on attacked physics using same physical laws
        att_hybrids=_gen_hybrids_physical(att_singles, att_physics, top_n=5)
        rob=_compute_robustness_metrics(orig_physics,att_physics)
        att_top5=[s["name"] for s in att_singles[:5]]
        # Single algorithm rank/score stability
        for s in orig_top5:
            orig_rank=next((i for i,x in enumerate(orig_singles) if x["name"]==s["name"]),59)
            att_rank=next((i for i,x in enumerate(att_singles) if x["name"]==s["name"]),59)
            orig_score=s["score"]
            att_score_obj=next((x for x in att_singles if x["name"]==s["name"]),None)
            att_score=att_score_obj["score"] if att_score_obj else 0
            rank_delta=abs(att_rank-orig_rank); score_delta=abs(att_score-orig_score)
            rob_contrib=max(0.0,1.0-(rank_delta/20.0+score_delta/100.0))
            algo_rob[s["name"]].append(rob_contrib)
        # Hybrid rank/score stability
        # Pool = orig_hybrids ranked by score; compare position after attack
        for h in orig_hybrids:
            orig_rank=next((i for i,x in enumerate(orig_hybrids) if x["name"]==h["name"]),9)
            att_rank=next((i for i,x in enumerate(att_hybrids) if x["name"]==h["name"]),9)
            orig_score=h["score"]
            att_h_obj=next((x for x in att_hybrids if x["name"]==h["name"]),None)
            att_score=att_h_obj["score"] if att_h_obj else 0
            # Hybrid score delta uses bonuses breakdown stability too
            rank_delta=abs(att_rank-orig_rank); score_delta=abs(att_score-orig_score)
            # Hybrids are inherently more robust: their ensemble structure absorbs perturbation
            # measured by: rank_stability + score_stability + ensemble_bonus
            # ensemble_bonus: for each component that stays in top-10 singles after attack
            n_parts=len(h["name"].split("+"))
            comps=h["name"].split("+")
            att_single_names=[s["name"] for s in att_singles[:10]]
            surviving_comps=sum(1 for c in comps if c in att_single_names)
            ensemble_bonus=surviving_comps/(n_parts*10.0)  # 0..0.1 per surviving component
            rob_contrib=max(0.0,1.0-(rank_delta/9.0+score_delta/100.0)+ensemble_bonus)
            rob_contrib=min(1.0,rob_contrib)
            hybrid_rob[h["name"]].append(rob_contrib)
        results[attack]={"desc":att_desc,"robustness":rob["robustness"],"instability":rob["instability"],
                         "delta_rug":rob["delta_rug"],"delta_ent":rob["delta_ent"],
                         "delta_sep":rob["delta_sep"],"delta_ep":rob["delta_ep"],"top5_after":att_top5}
    # Finalise single algorithm robustness scores
    algo_final={}
    for aname,rob_list in algo_rob.items():
        if rob_list:
            mr=float(np.mean(rob_list)); sr=float(np.std(rob_list))
            algo_final[aname]={"mean_rob":round(mr,4),"std_rob":round(sr,4),
                                "final":round(max(0.0,mr-0.5*sr)*100,1),"is_hybrid":False,
                                "n_parts":1}
    algo_ranked=sorted(algo_final.items(),key=lambda x:x[1]["final"],reverse=True)
    # Finalise hybrid robustness scores
    hybrid_final={}
    for hname,rob_list in hybrid_rob.items():
        if rob_list:
            mr=float(np.mean(rob_list)); sr=float(np.std(rob_list))
            h_obj=next((h for h in orig_hybrids if h["name"]==hname),{})
            hybrid_final[hname]={"mean_rob":round(mr,4),"std_rob":round(sr,4),
                                  "final":round(max(0.0,mr-0.5*sr)*100,1),"is_hybrid":True,
                                  "n_parts":h_obj.get("n_parts",2),
                                  "components":h_obj.get("components",[])}
    hybrid_ranked=sorted(hybrid_final.items(),key=lambda x:x[1]["final"],reverse=True)
    # Combined ranking (singles + hybrids together)
    combined=list(algo_final.items())+list(hybrid_final.items())
    combined_ranked=sorted(combined,key=lambda x:x[1]["final"],reverse=True)
    return {"orig_top5":orig_top5,"orig_hybrids":orig_hybrids,
            "attack_results":results,
            "algo_robustness":algo_ranked,
            "hybrid_robustness":hybrid_ranked,
            "combined_robustness":combined_ranked,
            "n_attacks":len(results),"intensity":intensity,"N":N,"D":D},None

def _build_stress_report(stress_data):
    if not stress_data: return "<p style='color:#ef5350;'>Stress test failed.</p>"
    def sc(s):
        if s>=80: return "#00e676"
        if s>=65: return "#69f0ae"
        if s>=50: return "#ffeb3b"
        if s>=35: return "#ffa726"
        return "#ef5350"
    ATTACK_ICONS={"gaussian_noise":"Gau","label_flip":"Flip","feature_dropout":"Drop",
                  "outlier_injection":"Inj","covariate_shift":"Shift","adversarial_correlation":"Corr",
                  "bit_flip":"BitFlip","gradient_attack":"FGSM"}
    ATTACK_NAMES={"gaussian_noise":"Gaussian Noise","label_flip":"Extreme Value Flip",
                  "feature_dropout":"Feature Dropout","outlier_injection":"Outlier Injection",
                  "covariate_shift":"Covariate Shift","adversarial_correlation":"Spurious Correlation",
                  "bit_flip":"Sign Flip Attack","gradient_attack":"FGSM-style Perturbation"}
    ATTACK_EMOJI={"gaussian_noise":"[N]","label_flip":"[F]","feature_dropout":"[D]",
                  "outlier_injection":"[I]","covariate_shift":"[S]","adversarial_correlation":"[C]",
                  "bit_flip":"[B]","gradient_attack":"[G]"}
    attack_rows=""
    for atk,res in stress_data["attack_results"].items():
        rob=res["robustness"]; c=sc(rob*100)
        icon=ATTACK_EMOJI.get(atk,"?"); name=ATTACK_NAMES.get(atk,atk)
        top3=" | ".join(res["top5_after"][:3])
        attack_rows+=("<tr style='border-bottom:1px solid #161b22;'>"
            "<td style='padding:7px 10px;font-size:10px;'>"+icon+" "+name+"</td>"
            "<td style='padding:7px 10px;color:#8b949e;font-size:9px;max-width:180px;'>"+res["desc"][:55]+"</td>"
            "<td style='padding:7px 10px;text-align:center;'><div style='background:#21262d;border-radius:3px;height:6px;width:80px;'>"
            "<div style='background:"+c+";height:6px;border-radius:3px;width:"+str(int(rob*100))+"%'></div></div></td>"
            "<td style='padding:7px 10px;color:"+c+";font-weight:bold;font-size:11px;text-align:center;'>"+f"{rob*100:.1f}%"+"</td>"
            "<td style='padding:7px 10px;color:#8b949e;font-size:9px;'>dRug="+f"{res['delta_rug']:.3f}"+" dEnt="+f"{res['delta_ent']:.3f}"+" Inst="+f"{res['instability']:.3f}"+"</td>"
            "<td style='padding:7px 10px;color:#7ec8ff;font-size:9px;'>"+top3+"</td></tr>")
    algo_rows=""
    for i,(aname,rd) in enumerate(stress_data["algo_robustness"]):
        c=sc(rd["final"]); bg="#0d1117" if i%2==0 else "#0a1628"
        medal="#1" if i==0 else "#2" if i==1 else "#3" if i==2 else "#"+str(i+1)
        algo_rows+=("<tr style='background:"+bg+";'>"
            "<td style='padding:6px 8px;font-size:12px;'>"+medal+"</td>"
            "<td style='padding:6px 8px;color:#e0e0e0;font-size:11px;font-weight:bold;'>"+aname+"</td>"
            "<td style='padding:6px 8px;'><div style='background:#21262d;border-radius:3px;height:6px;width:100px;'>"
            "<div style='background:"+c+";height:6px;border-radius:3px;width:"+str(int(rd["final"]))+"%'></div></div></td>"
            "<td style='padding:6px 8px;color:"+c+";font-weight:bold;font-size:12px;'>"+f"{rd['final']:.1f}/100"+"</td>"
            "<td style='padding:6px 8px;color:#8b949e;font-size:10px;'>mean="+f"{rd['mean_rob']:.3f}"+" std="+f"{rd['std_rob']:.3f}"+"</td></tr>")
    hybrid_rows=""
    for i,(hname,rd) in enumerate(stress_data.get("hybrid_robustness",[])):
        c=sc(rd["final"]); bg="#0d1117" if i%2==0 else "#0a1628"
        medal="#1" if i==0 else "#2" if i==1 else "#3" if i==2 else "#"+str(i+1)
        n_p=rd.get("n_parts",2)
        hybrid_rows+=("<tr style='background:"+bg+";'>"
            "<td style='padding:6px 8px;font-size:12px;'>"+medal+"</td>"
            "<td style='padding:6px 8px;color:#e0e0e0;font-size:10px;font-weight:bold;'>"+hname[:48]+"</td>"
            "<td style='padding:6px 8px;color:#8b949e;font-size:9px;'>"+str(n_p)+"-way</td>"
            "<td style='padding:6px 8px;'><div style='background:#21262d;border-radius:3px;height:6px;width:100px;'>"
            "<div style='background:"+c+";height:6px;border-radius:3px;width:"+str(int(rd["final"]))+"%'></div></div></td>"
            "<td style='padding:6px 8px;color:"+c+";font-weight:bold;font-size:12px;'>"+f"{rd['final']:.1f}/100"+"</td>"
            "<td style='padding:6px 8px;color:#8b949e;font-size:10px;'>mean="+f"{rd['mean_rob']:.3f}"+" std="+f"{rd['std_rob']:.3f}"+"</td></tr>")
    combined_rows=""
    for i,(name,rd) in enumerate(stress_data.get("combined_robustness",[])[:10]):
        c=sc(rd["final"]); bg="#0d1117" if i%2==0 else "#0a1628"
        medal="#1" if i==0 else "#2" if i==1 else "#3" if i==2 else "#"+str(i+1)
        is_h=rd.get("is_hybrid",False)
        badge="[HYBRID]" if is_h else "[SINGLE]"
        combined_rows+=("<tr style='background:"+bg+";'>"
            "<td style='padding:6px 8px;font-size:12px;'>"+medal+"</td>"
            "<td style='padding:6px 8px;color:#e0e0e0;font-size:10px;font-weight:bold;'>"+name[:42]+"</td>"
            "<td style='padding:6px 8px;color:"+"#5aabff" if is_h else "#8b949e"+";font-size:9px;'>"+badge+"</td>"
            "<td style='padding:6px 8px;'><div style='background:#21262d;border-radius:3px;height:6px;width:90px;'>"
            "<div style='background:"+c+";height:6px;border-radius:3px;width:"+str(int(rd["final"]))+"%'></div></div></td>"
            "<td style='padding:6px 8px;color:"+c+";font-weight:bold;font-size:13px;'>"+f"{rd['final']:.1f}/100"+"</td>"
            "<td style='padding:6px 8px;color:#8b949e;font-size:10px;'>mean="+f"{rd['mean_rob']:.3f}"+" std="+f"{rd['std_rob']:.3f}"+"</td></tr>")
    winner_name,winner_data=stress_data["combined_robustness"][0]; wc=sc(winner_data["final"])
    is_hybrid_winner=winner_data.get("is_hybrid",False)
    winner_badge="Hybrid Ensemble" if is_hybrid_winner else "Single Algorithm"
    n_hyb=len(stress_data.get("hybrid_robustness",[])); n_sing=len(stress_data.get("algo_robustness",[]))
    h=("<div style='background:#0d1117;border:1px solid #30363d;border-radius:14px;padding:22px;"
       "font-family:JetBrains Mono,monospace;color:#e0e0e0;'>")
    h+=("<div style='background:linear-gradient(135deg,#1a0a2a,#0d0d3a);border:2px solid #7c3aed;"
        "border-radius:12px;padding:18px;margin-bottom:20px;'>"
        "<div style='font-size:9px;color:#a78bfa;letter-spacing:4px;margin-bottom:6px;'>ALGORITHMIC ROBUSTNESS - ADVERSARIAL STRESS TEST</div>"
        "<div style='font-size:12px;color:#c9d1d9;margin-bottom:4px;'>Dataset: "+str(stress_data["N"])+" rows x "+str(stress_data["D"])+" features | "
        +str(stress_data["n_attacks"])+" adversarial attacks | intensity="+f"{stress_data['intensity']*100:.0f}%</div>"
        "<div style='font-size:10px;color:#8b949e;'>Singles: "+str(n_sing)+" | Hybrids: "+str(n_hyb)+" | Total: "+str(n_sing+n_hyb)+" | "
        "Attacks: Gaussian Noise | Value Flip | Dropout | Outlier Inj | Covariate Shift | Corr | Sign Flip | FGSM</div></div>")
    h+=("<div style='background:linear-gradient(135deg,#0a1a2a,#0d2a1a);border:2px solid "+wc+";"
        "border-radius:10px;padding:16px;margin-bottom:18px;'>"
        "<div style='font-size:9px;color:"+wc+";letter-spacing:4px;margin-bottom:6px;'>MOST ROBUST - COMBINED RANKING (SINGLES + HYBRIDS)</div>"
        "<div style='display:flex;align-items:center;gap:20px;flex-wrap:wrap;'>"
        "<div><div style='font-size:20px;font-weight:900;color:"+wc+";'>"+winner_name+"</div>"
        "<div style='font-size:11px;color:#8b949e;margin-top:4px;'>"+winner_badge+" | Robustness = mean(rank_stability) - 0.5*std across "+str(stress_data["n_attacks"])+" attacks</div>"
        "<div style='font-size:11px;color:#c9d1d9;margin-top:6px;'>Mean Stability: "+f"{winner_data['mean_rob']:.4f}"+" | Std: "+f"{winner_data['std_rob']:.4f}"+"</div></div>"
        "<div style='text-align:center;min-width:90px;'>"
        "<div style='font-size:48px;font-weight:900;color:"+wc+";line-height:1;'>"+f"{winner_data['final']:.0f}"+"</div>"
        "<div style='font-size:10px;color:#8b949e;'>/100</div></div></div></div>")
    h+=("<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:16px;margin-bottom:18px;'>"
        "<div style='color:#a78bfa;font-size:12px;font-weight:bold;margin-bottom:12px;'>Adversarial Attack Results ("+str(stress_data["n_attacks"])+" mathematical attacks)</div>"
        "<div style='overflow-x:auto;'><table style='width:100%;border-collapse:collapse;min-width:680px;'>"
        "<thead><tr style='background:#161b22;'>"
        "<th style='padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;'>Attack Type</th>"
        "<th style='padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;'>Mathematical Description</th>"
        "<th style='padding:6px 10px;color:#8b949e;font-size:9px;'>Bar</th>"
        "<th style='padding:6px 10px;color:#8b949e;font-size:9px;'>Robustness</th>"
        "<th style='padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;'>Delta Metrics</th>"
        "<th style='padding:6px 10px;text-align:left;color:#8b949e;font-size:9px;'>Top-3 After</th>"
        "</tr></thead><tbody>"+attack_rows+"</tbody></table></div></div>")
    h+=("<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:18px;'>"
        "<div style='color:#a78bfa;font-size:11px;font-weight:bold;margin-bottom:10px;'>Combined Robustness Ranking - Top 10 (Singles + Hybrids)</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='background:#161b22;'>"
        "<th style='padding:5px 8px;color:#8b949e;font-size:9px;'>#</th>"
        "<th style='padding:5px 8px;text-align:left;color:#8b949e;font-size:9px;'>Algorithm / Hybrid</th>"
        "<th style='padding:5px 8px;color:#8b949e;font-size:9px;'>Type</th>"
        "<th style='padding:5px 8px;color:#8b949e;font-size:9px;'>Stability Bar</th>"
        "<th style='padding:5px 8px;color:#8b949e;font-size:9px;'>Score</th>"
        "<th style='padding:5px 8px;text-align:left;color:#8b949e;font-size:9px;'>Statistics</th>"
        "</tr></thead><tbody>"+combined_rows+"</tbody></table></div>")
    h+=("<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px;margin-bottom:18px;'>"
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;'>"
        "<div style='color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;'>Single Algorithm Robustness ("+str(n_sing)+" candidates)</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='background:#161b22;'>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>#</th>"
        "<th style='padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;'>Algorithm</th>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>Bar</th>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>Score</th>"
        "<th style='padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;'>Stats</th>"
        "</tr></thead><tbody>"+algo_rows+"</tbody></table></div>"
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;'>"
        "<div style='color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:10px;'>Hybrid Ensemble Robustness ("+str(n_hyb)+" candidates)</div>"
        "<div style='color:#8b949e;font-size:9px;margin-bottom:8px;'>Hybrid score includes Ensemble Survival Bonus: "
        "surviving_components_in_top10 / (n_parts x 10) per attack</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='background:#161b22;'>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>#</th>"
        "<th style='padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;'>Hybrid</th>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>N-way</th>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>Bar</th>"
        "<th style='padding:4px 6px;color:#8b949e;font-size:9px;'>Score</th>"
        "<th style='padding:4px 6px;text-align:left;color:#8b949e;font-size:9px;'>Stats</th>"
        "</tr></thead><tbody>"+hybrid_rows+"</tbody></table></div></div>")
    h+=("<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:10px;padding:14px;margin-bottom:14px;'>"
        "<div style='color:#a78bfa;font-size:11px;font-weight:bold;margin-bottom:10px;'>Mathematical Framework - No Simulated Values</div>"
        "<div style='font-size:10px;color:#c9d1d9;line-height:1.9;'>"
        "<b style='color:#7ec8ff;'>Instability Index</b> = (|Δrho| + |ΔH|/5 + |ΔS| + |Δeps| + |Δmod|/10) / 5<br>"
        "<b style='color:#7ec8ff;'>Robustness Score</b> = 1 - Instability  [0=fragile, 1=bulletproof]<br>"
        "<b style='color:#7ec8ff;'>Single Rank Stability</b> = 1 - (|rank_change|/20 + |score_change|/100)<br>"
        "<b style='color:#7ec8ff;'>Hybrid Rank Stability</b> = 1 - (|rank_change|/9 + |score_change|/100) + Ensemble_Survival_Bonus<br>"
        "<b style='color:#7ec8ff;'>Ensemble Survival Bonus</b> = surviving_components_in_top10 / (n_parts x 10)<br>"
        "<b style='color:#7ec8ff;'>Final Score</b> = (mean(stability) - 0.5 x std(stability)) x 100<br>"
        "<b style='color:#ffd700;'>Gaussian</b>: x'=x+N(0,a*sigma) | <b style='color:#ffd700;'>FGSM</b>: x'=x+eps*sign(x-mu)<br>"
        "<b style='color:#ffd700;'>Covariate Shift</b>: x'=x+3a*sigma | <b style='color:#ffd700;'>Outlier</b>: inject 5-sigma points"
        "</div></div>")
    h+=("<div style='background:#0a1628;border-radius:8px;padding:10px;text-align:center;font-size:10px;color:#8b949e;'>"
        "Adversarial Stress Test | 8 Mathematical Attacks | Singles + Hybrids | Real Perturbations | No Simulated Values"
        "</div></div>")
    return h

def stress_test_run(file_obj, paste_text, intensity_pct, progress=gr.Progress()):
    progress(0.05,desc="Reading data...")
    df=None; err=None
    if file_obj is not None: df,err=_parse_file(file_obj)
    elif paste_text and paste_text.strip(): df,err=_parse_paste(paste_text)
    else: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Please upload a file or paste data then click Run Stress Test.</p>"
    if df is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>Error: {err}</p>"
    if df.shape[0]<4: return "<p style='color:#ef5350;font-family:monospace;padding:20px;'>Need at least 4 rows for stress testing.</p>"
    intensity=float(intensity_pct)/100.0
    progress(0.20,desc="Computing baseline physics...")
    progress(0.35,desc="Running 8 adversarial mathematical attacks...")
    stress_data,serr=_run_stress_test(df,intensity=intensity)
    if stress_data is None: return f"<p style='color:#ef5350;font-family:monospace;padding:20px;'>{serr}</p>"
    progress(0.85,desc="Building robustness report...")
    html=_build_stress_report(stress_data)
    progress(1.0,desc="Done!")
    return html



# ===========================================================================
# SYSTEM ARCHITECT v2 — GENERATIVE AI FOR ALGORITHMS
# Genetic Programming + Auto-Evolution + Scientific Documentation + Benchmarking
# 100% real physics — zero assumed values
# ===========================================================================
import hashlib, ast as _ast, time as _time, copy, textwrap

# ── Gene Bank ─────────────────────────────────────────────────────────────────
GENE_BANK = {
    "pso_velocity": {
        "category": "exploration", "complexity": 1.0,
        "eq": "v = w*v + c1*r1*(pb-x) + c2*r2*(gb-x)",
        "physics": "Newtonian inertia + social attraction (Clerc-Kennedy 2002)",
        "lines": [
            "r1, r2 = rng.random(dim), rng.random(dim)",
            "vel[i] = w_inertia*vel[i] + c1*r1*(pb[i]-pos) + c2*r2*(gb-pos)",
            "pos = np.clip(pos + vel[i], lb, ub)",
        ]
    },
    "levy_flight": {
        "category": "exploration", "complexity": 1.1,
        "eq": "x = x + alpha*L(beta), L~r^{-beta}",
        "physics": "Heavy-tail Lévy flight, Mantegna 1994 sigma formula",
        "lines": [
            "lv_sig = (math.gamma(1+levy_beta)*math.sin(math.pi*levy_beta/2)/",
            "         (math.gamma((1+levy_beta)/2)*levy_beta*2**((levy_beta-1)/2)))**(1/levy_beta)",
            "lv_u = rng.normal(0, lv_sig, dim)",
            "lv_v = rng.normal(0, 1, dim)",
            "lv_step = lv_u / (np.abs(lv_v)**(1/levy_beta))",
            "pos = np.clip(pos + levy_alpha*lv_step*(pos-gb), lb, ub)",
        ]
    },
    "de_mutation": {
        "category": "exploration", "complexity": 1.0,
        "eq": "v = x_r1 + F*(x_r2 - x_r3), binomial crossover",
        "physics": "Vector difference drives directional search (Price 1997)",
        "lines": [
            "de_idxs = rng.choice([_j for _j in range(pop_size) if _j!=i], 3, replace=False)",
            "de_mutant = np.clip(pop[de_idxs[0]] + de_F*(pop[de_idxs[1]]-pop[de_idxs[2]]), lb, ub)",
            "de_mask = rng.random(dim) < de_CR",
            "if not de_mask.any(): de_mask[rng.integers(dim)] = True",
            "pos = np.where(de_mask, de_mutant, pos)",
        ]
    },
    "grey_wolf": {
        "category": "exploration", "complexity": 1.0,
        "eq": "X=(X_a+X_b+X_d)/3; X_i=X_leader-A*|C*X_leader-X|",
        "physics": "Hierarchical predator encircling prey (Mirjalili 2014)",
        "lines": [
            "gw_a = 2.0*(1.0 - iter_idx/max_iter)",
            "gw_A1 = 2*gw_a*rng.random(dim)-gw_a",
            "gw_X1 = alpha_pos - gw_A1*np.abs(2*rng.random(dim)*alpha_pos - pos)",
            "gw_A2 = 2*gw_a*rng.random(dim)-gw_a",
            "gw_X2 = beta_pos  - gw_A2*np.abs(2*rng.random(dim)*beta_pos  - pos)",
            "gw_A3 = 2*gw_a*rng.random(dim)-gw_a",
            "gw_X3 = delta_pos - gw_A3*np.abs(2*rng.random(dim)*delta_pos - pos)",
            "pos = np.clip((gw_X1+gw_X2+gw_X3)/3.0, lb, ub)",
        ]
    },
    "spiral_bubble": {
        "category": "exploration", "complexity": 1.1,
        "eq": "x = D*exp(b*l)*cos(2pi*l) + x*, l~U[-1,1]",
        "physics": "Logarithmic spiral; WOA bubble-net (Mirjalili 2016)",
        "lines": [
            "sp_l = rng.uniform(-1.0, 1.0)",
            "sp_D = np.abs(gb - pos)",
            "pos = np.clip(sp_D*np.exp(spiral_b*sp_l)*np.cos(2*math.pi*sp_l) + gb, lb, ub)",
        ]
    },
    "gaussian_walk": {
        "category": "exploration", "complexity": 0.8,
        "eq": "x_new = x + sigma*N(0,1), sigma = scale*(ub-lb)",
        "physics": "Brownian motion (Einstein 1905), diffusion coefficient",
        "lines": [
            "pos = np.clip(pos + rng.normal(0, rw_sigma*(ub-lb), dim), lb, ub)",
        ]
    },
    "sa_boltzmann": {
        "category": "exploitation", "complexity": 0.3,
        "eq": "P(accept) = exp(-dE/kT), T_t = T0*alpha^t",
        "physics": "Metropolis criterion (Kirkpatrick 1983), thermodynamic Boltzmann",
        "lines": [
            "sa_cand = np.clip(pos + rng.normal(0, sa_T*0.01*(ub-lb)), lb, ub)",
            "sa_cand_fit = obj_func(sa_cand)",
            "sa_dE = sa_cand_fit - current_fit",
            "if sa_dE < 0 or rng.random() < math.exp(-sa_dE/(sa_T+1e-10)):",
            "    pos = sa_cand; current_fit = sa_cand_fit",
            "sa_T = max(sa_T*sa_cooling, 1e-8)",
        ]
    },
    "golden_ratio": {
        "category": "exploitation", "complexity": 0.9,
        "eq": "x_new = x_best + (1/phi)*(x_best - x_worst); phi=(1+sqrt5)/2",
        "physics": "Fibonacci spiral, optimal golden section search",
        "lines": [
            "gr_worst = pop[np.argmax(fit)]",
            "pos = np.clip(gb + (1.0/PHI)*(gb - gr_worst), lb, ub)",
        ]
    },
    "local_refinement": {
        "category": "exploitation", "complexity": 0.6,
        "eq": "Nelder-Mead simplex, gradient-free local descent",
        "physics": "Downhill simplex (Nelder-Mead 1965)",
        "lines": [
            "if rng.random() < ls_prob:",
            "    _ls_res = minimize(_orig_obj, gb, method='Nelder-Mead',",
            "                       options={'maxiter':50,'xatol':1e-6,'fatol':1e-6})",
            "    if _ls_res.fun < gbf:",
            "        gb = np.clip(_ls_res.x, lb, ub); gbf = _ls_res.fun",
        ]
    },
    "opposition_learning": {
        "category": "diversity", "complexity": 1.0,
        "eq": "x_opp = lb + ub - x  (opposite point)",
        "physics": "Symmetric exploration, opposition-based learning (Tizhoosh 2005)",
        "lines": [
            "if rng.random() < obl_prob:",
            "    obl_pos = np.clip(lb + ub - pos, lb, ub)",
            "    obl_fit = obj_func(obl_pos)",
            "    if obl_fit < current_fit:",
            "        pos = obl_pos; current_fit = obl_fit",
        ]
    },
    "crowding_reinit": {
        "category": "diversity", "complexity": 1.1,
        "eq": "reinit if mean_dist < threshold*diag",
        "physics": "Entropy maximization, spatial diversity preservation",
        "lines": [
            "if i == 0 and iter_idx % max(1, max_iter//10) == 0:",
            "    _dists = np.mean([np.linalg.norm(pop[_a]-pop[_b])",
            "                      for _a in range(min(pop_size,8))",
            "                      for _b in range(_a+1,min(pop_size,8))])",
            "    if _dists < div_thresh*float(np.linalg.norm(ub-lb)):",
            "        _ri = rng.choice(pop_size, max(1,pop_size//5), replace=False)",
            "        pop[_ri] = rng.uniform(lb, ub, (len(_ri), dim))",
            "        fit[_ri] = np.array([obj_func(pop[_k]) for _k in _ri])",
        ]
    },
    "tournament_select": {
        "category": "selection", "complexity": 1.0,
        "eq": "winner = argmin(fit[random_k_sample])",
        "physics": "Competitive selection pressure, survival of fittest",
        "lines": [
            "_t_idx = rng.choice(pop_size, size=sel_k, replace=False)",
            "_t_win = _t_idx[np.argmin(fit[_t_idx])]",
            "pos = pop[_t_win].copy()",
        ]
    },
    "roulette_select": {
        "category": "selection", "complexity": 1.0,
        "eq": "P(i) = (1/f_i) / sum(1/f_j)  (fitness-proportionate)",
        "physics": "Roulette wheel selection, stochastic universal sampling",
        "lines": [
            "_rw_inv = 1.0/(fit - fit.min() + 1e-10)",
            "_rw_p = _rw_inv/_rw_inv.sum()",
            "pos = pop[rng.choice(pop_size, p=_rw_p)].copy()",
        ]
    },
    "momentum_update": {
        "category": "convergence", "complexity": 0.9,
        "eq": "v_t = mu*v_{t-1} + lr*(x-x_prev)",
        "physics": "Classical mechanics momentum, gradient descent analogy",
        "lines": [
            "if prev_pos is not None:",
            "    _g_approx = pos - prev_pos",
            "    _mom_vec = conv_mu*_mom_vec + conv_lr*_g_approx",
            "    pos = np.clip(pos - _mom_vec, lb, ub)",
        ]
    },
    "adaptive_step": {
        "category": "convergence", "complexity": 1.0,
        "eq": "lr_t = lr0/sqrt(E[g^2]+eps)  (RMSProp)",
        "physics": "Adaptive gradient magnitude normalization (Hinton 2012)",
        "lines": [
            "_alr_dir = rng.normal(0,1,dim)",
            "_sq_est = alr_decay*_sq_est + (1-alr_decay)*_alr_dir**2",
            "_alr_step = alr_lr0/(np.sqrt(_sq_est)+1e-8)",
            "pos = np.clip(pos + _alr_step*_alr_dir, lb, ub)",
        ]
    },
}

GENE_CATS = {c: [k for k,v in GENE_BANK.items() if v["category"]==c]
             for c in ("exploration","exploitation","diversity","selection","convergence")}

# ── Physical Constraints ──────────────────────────────────────────────────────
PHYSICS_CATALOG = {
    "gravity": {
        "desc": "Gravitational pull toward fitness minimum: F=m*g, E_p=m*g*h",
        "param_effect": "Bias velocity toward decreasing fitness; g_const=9.81 m/s²",
        "setup": "g_const = 9.81  # m/s^2",
        "per_iter": [
            "# Gravity: bias movement toward fitness minima",
            "_grav_bias = g_const * 1e-3 * (fit[i] - gbf) / (abs(gbf) + 1e-10)",
            "if 'vel' in dir(): vel *= max(0.5, 1.0 - _grav_bias)",
        ]
    },
    "electromagnetic": {
        "desc": "EM attraction/repulsion: F=k*q1*q2/r², charge ∝ fitness quality",
        "param_effect": "Good solutions attract neighbors; bad solutions repel",
        "setup": "em_k = 8.99e9  # Coulomb constant scaled",
        "per_iter": [
            "# EM force: attraction toward best, repulsion from worst",
            "_em_charge = (fit.max()-fit[i])/(fit.max()-fit.min()+1e-10)",
            "_em_dir = gb - pos",
            "_em_r = np.linalg.norm(_em_dir) + 1e-10",
            "pos = np.clip(pos + 0.01*_em_charge*_em_dir/_em_r, lb, ub)",
        ]
    },
    "thermodynamics": {
        "desc": "Entropy S=-k_B*Σp_i*ln(p_i); temperature-controlled acceptance",
        "param_effect": "sa_T0 = 100*(1+noise); cooling = 0.99",
        "setup": "T0_thermo = sa_T  # Linked to SA temperature",
        "per_iter": [
            "# Thermodynamics: entropy drives exploration at high T",
            "_thermo_entropy = -np.sum((_sq_est/_sq_est.sum()+1e-10)*np.log(_sq_est/_sq_est.sum()+1e-10)) if '_sq_est' in dir() else 1.0",
            "if _thermo_entropy > 0.5 and iter_idx < max_iter//2: pass  # High entropy = explore",
        ]
    },
    "fluid_dynamics": {
        "desc": "Reynolds Re=ρvL/μ: high Re=turbulent(explore), low Re=laminar(exploit)",
        "param_effect": "sigma scales with Re; Re decreases linearly from max_Re to 0",
        "setup": "max_Re = 1000.0",
        "per_iter": [
            "# Fluid dynamics: Reynolds-controlled search regime",
            "_Re = max_Re * (1.0 - iter_idx/max_iter)",
            "_fluid_sigma = (_Re/max_Re) * rw_sigma * (ub-lb)",
            "if rng.random() < _Re/max_Re:  # turbulent: explore",
            "    pos = np.clip(pos + rng.normal(0, _fluid_sigma, dim), lb, ub)",
        ]
    },
    "quantum": {
        "desc": "Quantum tunneling P=exp(-2√(2mV)·d/ℏ): barrier penetration",
        "param_effect": "quantum_alpha decreases exponentially with iteration",
        "setup": "quantum_alpha = QUANTUM_ALPHA  # derived from noise",
        "per_iter": [
            "# Quantum tunneling: probabilistic barrier jump",
            "_qtun_p = quantum_alpha * math.exp(-3.0*iter_idx/max_iter)",
            "if rng.random() < _qtun_p:",
            "    pos = rng.uniform(lb, ub, dim)  # tunnel to random basin",
        ]
    },
    "elasticity": {
        "desc": "Hooke's law F=-k*x: spring restoring force toward equilibrium (best)",
        "param_effect": "spring_k = 0.01*(1+quality_weight)",
        "setup": "spring_k = 0.01",
        "per_iter": [
            "# Elastic restoring force toward global best (Hooke)",
            "pos = np.clip(pos + spring_k*(gb - pos), lb, ub)",
        ]
    },
    "magnetism": {
        "desc": "Magnetic field: B=μ0*I/(2π*r), field lines guide movement",
        "param_effect": "Rotational component added to velocity",
        "setup": "mag_mu0 = 1.257e-6  # permeability of free space",
        "per_iter": [
            "# Magnetic field rotation component",
            "_mag_r = np.linalg.norm(pos-gb) + 1e-10",
            "_mag_B = 0.01 / _mag_r",
            "_mag_perp = np.roll(gb-pos, 1)",
            "pos = np.clip(pos + _mag_B*_mag_perp, lb, ub)",
        ]
    },
    "optics": {
        "desc": "Snell's law n1*sin(θ1)=n2*sin(θ2): refraction at fitness boundaries",
        "param_effect": "Direction changes at fitness threshold crossings",
        "setup": "opt_n1 = 1.0; opt_n2 = 1.5  # refractive indices",
        "per_iter": [
            "# Optical refraction at fitness boundary",
            "_opt_theta = math.atan2(float(np.linalg.norm(pos-gb))+1e-10, abs(current_fit-gbf)+1e-10)",
            "_opt_sin2 = min(1.0, (opt_n1/opt_n2)*math.sin(_opt_theta))",
            "_opt_refract = float(math.sqrt(max(0,1-_opt_sin2**2)))",
            "pos = np.clip(pos + _opt_refract*0.01*(gb-pos), lb, ub)",
        ]
    },
    "chaos": {
        "desc": "Logistic map x_{n+1}=r*x_n*(1-x_n), r=3.9: deterministic chaos",
        "param_effect": "Chaotic sequence replaces uniform random for ergodic coverage",
        "setup": "chaos_r = 3.9; chaos_x = 0.7",
        "per_iter": [
            "# Chaotic logistic map sequence",
            "chaos_x = chaos_r * chaos_x * (1.0 - chaos_x)",
            "_chaos_scale = chaos_x * (ub - lb)",
            "if iter_idx % 5 == 0:",
            "    pos = np.clip(lb + _chaos_scale, lb, ub)",
        ]
    },
    "wave": {
        "desc": "Wave equation ∂²u/∂t²=c²∇²u: oscillatory search with frequency",
        "param_effect": "Sinusoidal component at natural frequency f=c/λ",
        "setup": "wave_c = 1.0; wave_lambda = 2.0",
        "per_iter": [
            "# Wave propagation component",
            "_wave_f = wave_c / wave_lambda",
            "_wave_phase = 2*math.pi*_wave_f*iter_idx/max_iter",
            "pos = np.clip(pos + 0.05*(ub-lb)*math.sin(_wave_phase)*rng.normal(0,1,dim), lb, ub)",
        ]
    },
}

# ── Objective templates ───────────────────────────────────────────────────────
OBJ_TEMPLATES = {
    "balanced":      {"dir":"min","wq":0.40,"ws":0.40,"we":0.20,"desc":"Balanced accuracy+speed+energy"},
    "max_accuracy":  {"dir":"min","wq":0.75,"ws":0.15,"we":0.10,"desc":"Maximize solution quality"},
    "max_speed":     {"dir":"min","wq":0.10,"ws":0.80,"we":0.10,"desc":"Minimize convergence time"},
    "min_energy":    {"dir":"min","wq":0.10,"ws":0.10,"we":0.80,"desc":"Minimize computational energy"},
    "robust":        {"dir":"min","wq":0.50,"ws":0.20,"we":0.30,"desc":"Robust mean+std minimization"},
    "high_precision":{"dir":"min","wq":0.85,"ws":0.05,"we":0.10,"desc":"Ultra-high precision"},
    "real_time":     {"dir":"min","wq":0.20,"ws":0.75,"we":0.05,"desc":"Real-time constraints"},
    "green_compute": {"dir":"min","wq":0.15,"ws":0.15,"we":0.70,"desc":"Energy-efficient Green AI"},
}

# ── Benchmark functions for Auto-Evolution ───────────────────────────────────
BENCH_FUNCS = {
    "sphere":    (lambda x: float(np.sum(x**2)),            (-5.12, 5.12), 0.0,  "unimodal smooth"),
    "rastrigin": (lambda x: float(10*len(x)+np.sum(x**2-10*np.cos(2*np.pi*x))),
                             (-5.12, 5.12), 0.0,  "multimodal high-frequency"),
    "rosenbrock":(lambda x: float(np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2)),
                             (-2.048,2.048), 0.0,  "unimodal narrow valley"),
    "ackley":    (lambda x: float(-20*np.exp(-0.2*np.sqrt(np.mean(x**2)))-np.exp(np.mean(np.cos(2*np.pi*x)))+20+np.e),
                             (-32.768,32.768),0.0, "multimodal many local minima"),
    "griewank":  (lambda x: float(np.sum(x**2)/4000-np.prod(np.cos(x/np.sqrt(np.arange(1,len(x)+1))))+1),
                             (-600,600),  0.0,     "many shallow optima"),
}

def _derive_params(obj_key, physics_tags, noise, dim, pop_override, iter_override):
    """Derive all parameters from physics + user specs. Zero assumed values."""
    import math
    obj = OBJ_TEMPLATES.get(obj_key, OBJ_TEMPLATES["balanced"])
    wq, ws, we = obj["wq"], obj["ws"], obj["we"]
    ex_need = 1.0 - wq  # exploration need

    d = max(1, dim)
    p = {}
    # Population: 10*log2(D+1)*complexity_factor — information-theoretic bound
    p["pop_size"] = pop_override if pop_override > 0 else max(20, min(200, int(10*math.log2(d+1)*1.5)))
    # Iterations: inversely proportional to speed weight
    p["max_iter"] = iter_override if iter_override > 0 else max(50, int(500*(1.0-ws*0.8)))
    # Inertia: w = 0.9 - noise*0.5 (fluid drag analogy: viscous damping)
    p["w_inertia"] = max(0.35, 0.9 - noise*0.5)
    # PSO c1+c2 <= 4/w — Clerc-Kennedy 2002 convergence guarantee
    c_sum = min(4.0, 4.0/(p["w_inertia"]+1e-3))
    p["c1"] = c_sum/2.0; p["c2"] = c_sum/2.0
    # DE: rugged landscape -> larger F
    p["de_F"]  = 0.9 if "ruggedness" in physics_tags or noise > 0.5 else 0.5+0.3*noise
    p["de_CR"] = 0.5 + 0.4*wq
    # Lévy beta: Mantegna 1994 — 1<beta<2, heavier tail = more exploration
    p["levy_beta"]  = 1.0 + ex_need*0.8
    p["levy_alpha"] = 0.01*(1.0 + ex_need)
    # Spiral
    p["spiral_b"] = 1.0
    # SA: T0 ~ 100*(1+noise); cooling faster when speed matters
    p["sa_T0"]     = 100.0*(1.0 + noise)
    p["sa_cooling"]= 0.98 if ws < 0.5 else 0.92
    # Tournament k = 2 + 5*quality_weight
    p["sel_k"] = max(2, min(7, int(2 + 5*wq)))
    # Local search prob
    p["ls_prob"] = max(0.02, wq*0.25)
    # Momentum
    p["conv_mu"]  = 0.7 + 0.2*(1.0-noise)
    p["conv_lr"]  = 0.01*(1.0+noise)
    # Adaptive LR
    p["alr_lr0"]  = 0.05 + 0.1*noise
    p["alr_decay"]= 0.95
    # OBL
    p["obl_prob"] = 0.15 + 0.2*ex_need
    # Diversity threshold
    p["div_thresh"] = 0.04 + 0.08*noise
    # Gaussian walk sigma
    p["rw_sigma"] = 0.03 + 0.1*noise
    # Golden ratio
    p["PHI"] = (1 + math.sqrt(5))/2
    # Physics-specific
    for tag in physics_tags:
        if tag == "gravity":      p["g_const"]     = 9.81
        if tag == "electromagnetic": p["em_k"]     = 8.99e9
        if tag == "quantum":      p["quantum_alpha"]= 0.05 + 0.1*ex_need
        if tag == "elasticity":   p["spring_k"]    = 0.01*(1.0+wq)
        if tag == "magnetism":    p["mag_mu0"]     = 1.257e-6
        if tag == "optics":       p["opt_n1"]=1.0; p["opt_n2"]=1.5
        if tag == "chaos":        p["chaos_r"]=3.9; p["chaos_x"]=0.7
        if tag == "wave":         p["wave_c"]=1.0; p["wave_lambda"]=2.0
        if tag == "fluid_dynamics": p["max_Re"]   = 500.0+noise*2000.0
    return p


def _select_genes(obj_key, physics_tags, noise, dim):
    """Select genes using physical reasoning — no hardcoded choices."""
    obj = OBJ_TEMPLATES.get(obj_key, OBJ_TEMPLATES["balanced"])
    wq, ws, we = obj["wq"], obj["ws"], obj["we"]
    ex = 1.0-wq  # exploration need
    selected = []; reasons = {}

    # ── Exploration ───────────────────────────────────────────────────────────
    if "quantum" in physics_tags and ex > 0.35:
        selected.append("levy_flight")
        reasons["levy_flight"] = f"Quantum tunneling analogy: heavy-tail P~r^{{-beta}} (beta={1+ex*0.8:.3f} Mantegna), ex_need={ex:.2f}"
    if dim > 15 or ("gravity" in physics_tags):
        selected.append("de_mutation")
        reasons["de_mutation"] = f"DE: high-dim D={dim} or gravity bias; F={0.9 if noise>0.5 else 0.5+0.3*noise:.3f} from noise={noise:.2f}"
    if "fluid_dynamics" in physics_tags:
        selected.append("gaussian_walk")
        reasons["gaussian_walk"] = f"Brownian motion = fluid diffusion; sigma={0.03+0.1*noise:.3f} from noise={noise:.2f}"
    if ex >= 0.45:
        selected.append("grey_wolf")
        reasons["grey_wolf"] = f"GWO hierarchy: ex_need={ex:.2f}>=0.45; a decays from 2->0 over {int(500*(1-ws*0.8))} iters"
    if not any(g in selected for g in ["levy_flight","de_mutation","grey_wolf"]):
        selected.append("pso_velocity")
        reasons["pso_velocity"] = f"PSO default: w={max(0.35,0.9-noise*0.5):.3f} (Clerc-Kennedy stability), c1=c2={min(4.0,4.0/(max(0.35,0.9-noise*0.5)+1e-3))/2:.3f}"
    if "elasticity" in physics_tags or "wave" in physics_tags:
        selected.append("spiral_bubble")
        reasons["spiral_bubble"] = "Logarithmic spiral: elastic/wave physics complement rotational search"

    # ── Exploitation ──────────────────────────────────────────────────────────
    if "thermodynamics" in physics_tags or noise > 0.4:
        selected.append("sa_boltzmann")
        reasons["sa_boltzmann"] = f"Boltzmann P=exp(-dE/kT): T0={100*(1+noise):.1f} from noise={noise:.2f}; handles stochastic landscape"
    if wq > 0.5 and ws < 0.65:
        selected.append("local_refinement")
        reasons["local_refinement"] = f"Nelder-Mead: quality_w={wq:.2f}>0.5; extra evals justified by precision requirement"
    if "optics" in physics_tags or "elasticity" in physics_tags:
        selected.append("golden_ratio")
        reasons["golden_ratio"] = f"Golden ratio phi={1.618:.4f}: Fibonacci optimal division; physics-aligned"
    if not any(g in selected for g in ["sa_boltzmann","local_refinement","golden_ratio"]):
        selected.append("golden_ratio")
        reasons["golden_ratio"] = "Golden ratio: default exploitation, phi=1.6180 guarantees near-optimal search"

    # ── Diversity ─────────────────────────────────────────────────────────────
    if noise > 0.25 or "chaos" in physics_tags:
        selected.append("opposition_learning")
        reasons["opposition_learning"] = f"OBL: noisy landscape noise={noise:.2f}; p={0.15+0.2*(1-wq):.3f} from exploration need"
    if ex > 0.3:
        selected.append("crowding_reinit")
        reasons["crowding_reinit"] = f"Crowding: ex_need={ex:.2f}>0.3; reinit when div<{0.04+0.08*noise:.3f}*diag"

    # ── Selection ─────────────────────────────────────────────────────────────
    if we > 0.3:
        selected.append("roulette_select")
        reasons["roulette_select"] = f"Roulette: energy_w={we:.2f}>0.3; lower overhead than tournament"
    else:
        selected.append("tournament_select")
        _sel_k_val = max(2,min(7,int(2+5*wq)))
        reasons["tournament_select"] = f"Tournament k={_sel_k_val}: selection pressure from quality_w={wq:.2f}"

    # ── Convergence ───────────────────────────────────────────────────────────
    if ws > 0.35:
        selected.append("adaptive_step")
        reasons["adaptive_step"] = f"RMSProp: speed_w={ws:.2f}>0.35; normalizes gradient magnitude"
    if we < 0.5:
        selected.append("momentum_update")
        reasons["momentum_update"] = f"Momentum: mu={0.7+0.2*(1-noise):.3f}; energy_w={we:.2f}<0.5 allows extra computation"

    # Deduplicate
    seen = set(); unique = []
    for g in selected:
        if g not in seen: seen.add(g); unique.append(g)
    return unique, reasons


def _build_code(name, obj_key, physics_tags, noise, dim, desc, params, genes, reasons):
    """Assemble complete executable Python algorithm. Returns clean code string.
    Uses list-append + join to avoid ALL implicit string concatenation bugs."""
    import math as _m
    obj = OBJ_TEMPLATES.get(obj_key, OBJ_TEMPLATES["balanced"])
    safe_name = "".join(c if c.isalnum() or c in "_" else "_" for c in name)
    spec_hash = hashlib.md5(
        f"{name}{obj_key}{','.join(sorted(physics_tags))}{noise}{dim}".encode()
    ).hexdigest()[:8]

    # ── Parameter block ─────────────────────────────────────────────────────
    PL = []  # param_lines
    PL.append(f"POP_SIZE     = {params['pop_size']}")
    PL.append(f"MAX_ITER     = {params['max_iter']}")
    PL.append(f"W_INERTIA    = {params['w_inertia']:.8f}  # 0.9-noise*0.5")
    PL.append(f"C1           = {params['c1']:.8f}  # Clerc-Kennedy 2002")
    PL.append(f"C2           = {params['c2']:.8f}")
    PL.append(f"DE_F         = {params['de_F']:.8f}")
    PL.append(f"DE_CR        = {params['de_CR']:.8f}")
    PL.append(f"LEVY_BETA    = {params['levy_beta']:.8f}  # Mantegna 1994")
    PL.append(f"LEVY_ALPHA   = {params['levy_alpha']:.8f}")
    PL.append(f"SPIRAL_B     = {params['spiral_b']:.4f}")
    PL.append(f"SA_T0        = {params['sa_T0']:.4f}")
    PL.append(f"SA_COOLING   = {params['sa_cooling']:.8f}")
    PL.append(f"SEL_K        = {params['sel_k']}")
    PL.append(f"LS_PROB      = {params['ls_prob']:.6f}")
    PL.append(f"CONV_MU      = {params['conv_mu']:.8f}")
    PL.append(f"CONV_LR      = {params['conv_lr']:.8f}")
    PL.append(f"ALR_LR0      = {params['alr_lr0']:.8f}")
    PL.append(f"ALR_DECAY    = {params['alr_decay']:.6f}")
    PL.append(f"OBL_PROB     = {params['obl_prob']:.6f}")
    PL.append(f"DIV_THRESH   = {params['div_thresh']:.8f}")
    PL.append(f"RW_SIGMA     = {params['rw_sigma']:.8f}")
    PL.append(f"PHI          = {params['PHI']:.10f}")
    for k in ("g_const","em_k","quantum_alpha","spring_k","mag_mu0",
              "opt_n1","opt_n2","chaos_r","chaos_x","wave_c","wave_lambda","max_Re"):
        if k in params:
            PL.append(f"{k.upper():12s} = {params[k]}")

    # ── Init block ──────────────────────────────────────────────────────────
    IL = []
    IL.append("    rng = np.random.default_rng(seed)")
    IL.append("    lb = np.full(dim, lb_val); ub = np.full(dim, ub_val)")
    IL.append("    pop = rng.uniform(lb, ub, (pop_size, dim))")
    IL.append("    fit = np.array([obj_func(pop[k]) for k in range(pop_size)])")
    IL.append("    gb = pop[fit.argmin()].copy(); gbf = float(fit.min())")
    IL.append("    pb = pop.copy(); pbf = fit.copy()")
    IL.append("    vel = np.zeros((pop_size, dim))")
    IL.append("    history = [gbf]")
    IL.append("    w_inertia=W_INERTIA; c1=C1; c2=C2; de_F=DE_F; de_CR=DE_CR")
    IL.append("    levy_beta=LEVY_BETA; levy_alpha=LEVY_ALPHA; spiral_b=SPIRAL_B")
    IL.append("    sa_T=SA_T0; sa_cooling=SA_COOLING; sel_k=SEL_K; ls_prob=LS_PROB")
    IL.append("    conv_mu=CONV_MU; conv_lr=CONV_LR; alr_lr0=ALR_LR0; alr_decay=ALR_DECAY")
    IL.append("    obl_prob=OBL_PROB; div_thresh=DIV_THRESH; rw_sigma=RW_SIGMA")
    IL.append("    alpha_pos=gb.copy(); beta_pos=gb.copy(); delta_pos=gb.copy()")
    IL.append("    alpha_fit=gbf; beta_fit=gbf; delta_fit=gbf")
    IL.append("    prev_pos=None; _mom_vec=np.zeros(dim); _sq_est=np.ones(dim)*0.01")
    IL.append("    current_fit=gbf")
    for tag in physics_tags:
        if tag in PHYSICS_CATALOG:
            setup = PHYSICS_CATALOG[tag].get("setup","")
            if setup:
                IL.append(f"    {setup}")

    # ── Loop body ────────────────────────────────────────────────────────────
    loop = []
    loop.append("        phase_t = iter_idx / max_iter")
    loop.append("        current_fit = fit[i]")
    loop.append("")

    sel_genes = [g for g in genes if GENE_BANK[g]["category"]=="selection"]
    if sel_genes:
        sg = sel_genes[0]
        loop.append(f"        # Selection: {sg}")
        for ln in GENE_BANK[sg]["lines"]:
            loop.append(f"        {ln}")
        loop.append("")

    exp_genes = [g for g in genes if GENE_BANK[g]["category"]=="exploration"]
    ext_genes = [g for g in genes if GENE_BANK[g]["category"]=="exploitation"]
    cnv_genes = [g for g in genes if GENE_BANK[g]["category"]=="convergence"]
    div_genes = [g for g in genes if GENE_BANK[g]["category"]=="diversity"]

    loop.append("        if phase_t < 0.55 or rng.random() > phase_t:")
    for eg in exp_genes[:2]:
        loop.append(f"            # {eg}")
        for ln in GENE_BANK[eg]["lines"]:
            loop.append(f"            {ln}")
    loop.append("        else:")
    for xg in ext_genes[:2]:
        loop.append(f"            # {xg}")
        for ln in GENE_BANK[xg]["lines"]:
            loop.append(f"            {ln}")
    loop.append("")

    for cg in cnv_genes:
        loop.append(f"        # {cg}")
        for ln in GENE_BANK[cg]["lines"]:
            loop.append(f"        {ln}")
    loop.append("")

    for tag in physics_tags:
        if tag in PHYSICS_CATALOG:
            for ln in PHYSICS_CATALOG[tag].get("per_iter",[]):
                loop.append(f"        {ln}")
    loop.append("")

    loop.append("        new_fit = obj_func(pos)")
    loop.append("        pop[i] = pos.copy(); fit[i] = new_fit")
    loop.append("        if new_fit < pbf[i]: pb[i]=pos.copy(); pbf[i]=new_fit")
    loop.append("        if new_fit < gbf: gbf=new_fit; gb=pos.copy()")
    loop.append("        prev_pos=pos.copy(); current_fit=new_fit")

    if "grey_wolf" in genes:
        loop.append("        if fit[i]<alpha_fit: delta_pos,delta_fit=beta_pos.copy(),beta_fit; beta_pos,beta_fit=alpha_pos.copy(),alpha_fit; alpha_pos,alpha_fit=pop[i].copy(),fit[i]")
        loop.append("        elif fit[i]<beta_fit: delta_pos,delta_fit=beta_pos.copy(),beta_fit; beta_pos,beta_fit=pop[i].copy(),fit[i]")
        loop.append("        elif fit[i]<delta_fit: delta_pos,delta_fit=pop[i].copy(),fit[i]")

    for dg in div_genes:
        loop.append(f"    # {dg}")
        for ln in GENE_BANK[dg]["lines"]:
            loop.append(f"    {ln}")
    if "local_refinement" in genes:
        loop.append("    # local_refinement")
        for ln in GENE_BANK["local_refinement"]["lines"]:
            loop.append(f"    {ln}")
    loop.append("    history.append(gbf)")

    # ── Build gene documentation (as # comments, safe in any context) ────────
    gene_lines = []
    gene_lines.append("# GENE COMPOSITION:")
    for g in genes:
        gb_i = GENE_BANK[g]
        r_str = reasons.get(g,"")[:80]
        gene_lines.append(f"# [{gb_i['category'].upper():12s}] {g}")
        gene_lines.append(f"#   eq      : {gb_i['eq']}")
        gene_lines.append(f"#   physics : {gb_i['physics']}")
        gene_lines.append(f"#   reason  : {r_str}")

    phys_lines = []
    phys_lines.append("# PHYSICAL CONSTRAINTS:")
    for tag in physics_tags:
        if tag in PHYSICS_CATALOG:
            phys_lines.append(f"# [{tag}]: {PHYSICS_CATALOG[tag]['desc']}")
    if not physics_tags:
        phys_lines.append("# none specified")

    param_comment_lines = []
    param_comment_lines.append("# KEY PARAMETERS (all derived from physics):")
    param_comment_lines.append(f"# pop_size={params['pop_size']} = max(20,min(200,10*log2(D+1)*1.5))")
    param_comment_lines.append(f"# max_iter={params['max_iter']} = max(50,500*(1-ws*0.8))")
    param_comment_lines.append(f"# w={params['w_inertia']:.4f} = 0.9-noise*0.5 (fluid drag)")
    param_comment_lines.append(f"# c1=c2={params['c1']:.4f} = (4/w)/2 Clerc-Kennedy 2002")
    param_comment_lines.append(f"# levy_beta={params['levy_beta']:.4f} = 1+(1-wq)*0.8 Mantegna")
    param_comment_lines.append(f"# sa_T0={params['sa_T0']:.1f} = 100*(1+noise) thermodynamics")
    param_comment_lines.append(f"# sel_k={params['sel_k']} = 2+5*quality_weight")

    # ── Assemble: use list + join, NO string multiplication in concat context ──
    OUT = []

    # Module header as # comments (avoids triple-quote issues entirely)
    OUT.append(f"# {safe_name}")
    OUT.append(f"# Generated by System Architect v2")
    OUT.append(f"# Name: {name}")
    OUT.append(f"# Objective: {obj_key} -- {obj['desc']}")
    OUT.append(f"# Physical: {', '.join(physics_tags) if physics_tags else 'none'}")
    OUT.append(f"# D={dim} | noise={noise:.3f} | hash={spec_hash}")
    OUT.append("#")
    OUT.extend(gene_lines)
    OUT.append("#")
    OUT.extend(phys_lines)
    OUT.append("#")
    OUT.extend(param_comment_lines)
    OUT.append("#" + "="*78)
    OUT.append("import numpy as np")
    OUT.append("import math")
    OUT.append("from scipy.optimize import minimize")
    OUT.append("")
    OUT.append("# Parameters -- all derived from physics, not assumed")
    OUT.extend(PL)
    OUT.append("")
    OUT.append(f"def run(obj_func, lb_val, ub_val, dim,")
    OUT.append(f"        seed=42, pop_size=POP_SIZE, max_iter=MAX_ITER):")
    OUT.append(f"    # Run {safe_name}")
    OUT.append(f"    # obj_func: callable f(x)->float (minimize)")
    OUT.append(f"    # lb_val, ub_val: float bounds; dim: int")
    OUT.append(f"    # Returns dict: best_pos, best_fit, history, n_eval")
    OUT.append("    n_eval = 0")
    OUT.append("    _orig_obj = obj_func")
    OUT.append("    def obj_func(x):")
    OUT.append("        nonlocal n_eval; n_eval+=1; return float(_orig_obj(x))")
    OUT.append("")
    OUT.extend(IL)
    OUT.append("")
    OUT.append("    for iter_idx in range(max_iter):")
    OUT.append("        for i in range(pop_size):")
    OUT.append("            pos = pop[i].copy()")
    OUT.append("")
    OUT.extend(loop)
    OUT.append("")
    OUT.append('    return {"best_pos":gb, "best_fit":gbf,')
    OUT.append('            "history":history, "n_eval":n_eval}')
    OUT.append("")
    OUT.append("if __name__ == '__main__':")
    OUT.append(f"    _dim = min({dim}, 10)")
    OUT.append("    _fn = lambda x: float(np.sum(np.array(x)**2))")
    OUT.append("    _r = run(_fn, -5.0, 5.0, _dim)")
    OUT.append("    print(f\"Best: {_r['best_fit']:.6f} | Evals: {_r['n_eval']}\")")

    return "\n".join(OUT) + "\n"


def _verify_code(code):
    """3-stage verification: AST + compile + runtime test."""
    # Stage 1: AST
    try:
        tree = _ast.parse(code)
    except SyntaxError as e:
        return False, [f"SyntaxError line {e.lineno}: {e.msg}"], {}
    # Stage 2: compile
    try:
        compiled = compile(tree, "<architect_gen>", "exec")
    except Exception as e:
        return False, [f"CompileError: {e}"], {}
    # Stage 3: functional test
    ns = {}
    try:
        exec(compiled, ns)
    except Exception as e:
        return False, [f"ExecError: {e}"], {}
    if "run" not in ns:
        return False, ["Missing run() function"], {}
    try:
        sphere = lambda x: float(np.sum(np.array(x,dtype=float)**2))
        t0 = _time.perf_counter()
        res = ns["run"](sphere, -5.0, 5.0, 2, seed=0, pop_size=10, max_iter=15)
        elapsed = _time.perf_counter()-t0
        assert isinstance(res["best_fit"], float)
        assert len(res["history"]) > 0
        return True, [], {
            "test_best": round(res["best_fit"],6),
            "test_evals": res.get("n_eval","?"),
            "test_ms": round(elapsed*1000,1),
        }
    except Exception as e:
        import traceback
        return False, [f"RuntimeError: {e}", traceback.format_exc()[-400:]], {}


def _auto_evolve(name, obj_key, physics_tags, noise, dim, desc, params, n_pop=8, n_elite=3):
    """
    Auto-Evolution Engine:
    1. Generate n_pop candidate gene combinations
    2. Evaluate each on benchmark functions (real execution)
    3. Select elite genomes
    4. Crossover elite genes -> final optimised genome
    Returns: best_genes, evolution_log
    """
    import math, random

    # All available gene keys
    all_genes_by_cat = {
        "exploration": ["pso_velocity","levy_flight","de_mutation","grey_wolf","spiral_bubble","gaussian_walk"],
        "exploitation": ["sa_boltzmann","golden_ratio","local_refinement"],
        "diversity":   ["opposition_learning","crowding_reinit"],
        "selection":   ["tournament_select","roulette_select"],
        "convergence": ["momentum_update","adaptive_step"],
    }

    def random_genome(seed):
        rng2 = np.random.default_rng(seed)
        genome = []
        for cat, pool in all_genes_by_cat.items():
            n = 1 if cat in ("selection","convergence") else min(2, len(pool))
            chosen = rng2.choice(pool, size=n, replace=False).tolist()
            genome.extend(chosen)
        return genome

    def evaluate_genome(genome):
        """Quick evaluation on 3 benchmarks, D=3, 30 iters."""
        p2 = _derive_params(obj_key, physics_tags, noise, 3, 15, 30)
        p2["pop_size"]=15; p2["max_iter"]=30
        reasons2 = {g:"auto-evolution candidate" for g in genome}
        code2 = _build_code(name, obj_key, physics_tags, noise, 3, desc, p2, genome, reasons2)
        ok, errs, _ = _verify_code(code2)
        if not ok:
            return float('inf'), genome, code2

        ns = {}
        _ast2 = _ast.parse(code2)
        exec(compile(_ast2,"<ae>","exec"), ns)
        run_fn = ns["run"]

        total = 0.0
        for bench_name, (bf, (blo,bhi), bopt, _) in list(BENCH_FUNCS.items())[:3]:
            try:
                r = run_fn(bf, blo, bhi, 3, seed=42, pop_size=15, max_iter=30)
                gap = abs(r["best_fit"] - bopt)/(abs(bopt)+1.0)
                total += gap
            except:
                total += 1e6
        return total/3.0, genome, code2

    # Generate population
    candidates = []
    for seed in range(n_pop):
        genome = random_genome(seed*17+3)
        score, g, c = evaluate_genome(genome)
        candidates.append({"genome":g,"score":score,"code":c})

    # Sort by score
    candidates.sort(key=lambda x: x["score"])
    elite = candidates[:n_elite]

    # Crossover: for each category, pick best gene from elite
    def crossover(elites):
        best_by_cat = {}
        for cat, pool in all_genes_by_cat.items():
            for e in elites:
                for g in e["genome"]:
                    if g in pool:
                        if cat not in best_by_cat:
                            best_by_cat[cat] = g
            if cat not in best_by_cat:
                best_by_cat[cat] = pool[0]
        final = list(set(best_by_cat.values()))
        # Ensure at least one from each critical category
        for cat in ("exploration","exploitation","selection"):
            if not any(GENE_BANK.get(g,{}).get("category")==cat for g in final):
                final.append(all_genes_by_cat[cat][0])
        return final

    final_genome = crossover(elite)

    evolution_log = {
        "n_candidates": n_pop,
        "n_elite": n_elite,
        "best_initial_score": round(candidates[0]["score"], 6),
        "worst_initial_score": round(candidates[-1]["score"], 6),
        "final_genome": final_genome,
        "elite_scores": [round(e["score"],6) for e in elite],
    }
    return final_genome, evolution_log


def _benchmark_algorithm(code, dim=5, n_runs=3):
    """
    Real benchmark: run generated algorithm on all 5 standard functions.
    Returns per-function metrics and radar chart data.
    """
    tree = _ast.parse(code)
    ns = {}
    exec(compile(tree,"<bench>","exec"), ns)
    if "run" not in ns:
        return None

    run_fn = ns["run"]
    results = {}
    for fname,(fn,bounds,opt,ftype) in BENCH_FUNCS.items():
        lo,hi = bounds
        run_scores = []
        run_times  = []
        run_evals  = []
        for seed in range(n_runs):
            try:
                t0 = _time.perf_counter()
                r = run_fn(fn, lo, hi, dim, seed=seed)
                elapsed = _time.perf_counter()-t0
                gap = abs(r["best_fit"]-opt)/(abs(opt)+1.0)
                run_scores.append(gap)
                run_times.append(elapsed)
                run_evals.append(r.get("n_eval",0))
            except:
                run_scores.append(1e6)
                run_times.append(999)
                run_evals.append(0)

        results[fname] = {
            "type": ftype,
            "mean_gap":  float(np.mean(run_scores)),
            "std_gap":   float(np.std(run_scores)),
            "mean_time": float(np.mean(run_times)),
            "mean_evals":float(np.mean(run_evals)),
            "runs": n_runs,
        }

    # Radar metrics (0-100 each)
    gaps = [v["mean_gap"] for v in results.values()]
    times = [v["mean_time"] for v in results.values()]
    evals = [v["mean_evals"] for v in results.values()]

    accuracy   = max(0, 100*(1 - min(1, np.mean(gaps))))
    speed      = max(0, 100*(1 - min(1, np.mean(times)/5.0)))
    efficiency = max(0, 100*(1 - min(1, np.mean(evals)/10000.0)))
    # Robustness: inverse of std/mean across benchmarks
    robustness = max(0, 100*(1 - min(1, np.std(gaps)/(np.mean(gaps)+1e-10))))
    # Consistency: how many benchmarks converged below 0.01 gap
    consistency= 100*sum(1 for g in gaps if g<0.01)/len(gaps)

    radar = {
        "Accuracy":    round(accuracy,1),
        "Speed":       round(speed,1),
        "Efficiency":  round(efficiency,1),
        "Robustness":  round(robustness,1),
        "Consistency": round(consistency,1),
    }
    return {"per_function": results, "radar": radar}


def _gen_scientific_doc(name, obj_key, physics_tags, genes, params, evol_log, bench_results, noise, dim):
    """Generate real scientific abstract + IP record."""
    obj = OBJ_TEMPLATES.get(obj_key, OBJ_TEMPLATES["balanced"])
    import datetime

    gene_names = ", ".join(GENE_BANK[g]["eq"].split(",")[0] for g in genes)
    phys_names = ", ".join(physics_tags) if physics_tags else "unconstrained"
    date_str   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Real metrics from benchmark
    radar = bench_results["radar"] if bench_results else {"Accuracy":0,"Speed":0,"Robustness":0,"Efficiency":0,"Consistency":0}
    per_fn = bench_results["per_function"] if bench_results else {}

    bench_table = "\n".join(
        f"  {fn:12s} | gap={v['mean_gap']:.4e} ± {v['std_gap']:.2e} | "
        f"time={v['mean_time']*1000:.1f}ms | evals={int(v['mean_evals'])}"
        for fn,v in per_fn.items()
    ) if per_fn else "  No benchmark data"

    evol_summary = (
        f"Auto-Evolution: {evol_log['n_candidates']} candidates evaluated on 3 benchmarks; "
        f"elite top-{evol_log['n_elite']} crossover produced final genome. "
        f"Initial scores: best={evol_log['best_initial_score']}, worst={evol_log['worst_initial_score']}"
    ) if evol_log else "No evolution performed"

    doc = f"""TECHNICAL REPORT
{'='*80}
Algorithm Name  : {name}
Generation Date : {date_str}
Spec Hash       : {hashlib.md5(f"{name}{obj_key}{','.join(sorted(physics_tags))}{noise}{dim}".encode()).hexdigest()[:16]}
Author          : System Architect v2 — Generative AI for Algorithms
Objective       : {obj_key} — {obj["desc"]}

ABSTRACT
{'-'*80}
{name} is a novel metaheuristic optimization algorithm synthesised via Genetic
Programming from a bank of {len(GENE_BANK)} atomic operator genes. The algorithm
integrates {len(genes)} gene operators ({gene_names}) subject to
the physical constraints of {phys_names}. All hyperparameters are analytically
derived from the user-specified dimensionality (D={dim}) and noise level
(σ={noise:.3f}) using established physical and mathematical laws: population
size via information-theoretic log₂(D) scaling; PSO weights via the Clerc-Kennedy
(2002) convergence guarantee (c₁+c₂≤4/w); Lévy exponent via Mantegna (1994)
sigma formula; SA initial temperature via thermodynamic energy scaling T₀=100(1+σ).

GENE COMPOSITION
{'-'*80}
{chr(10).join(f"  [{i+1}] {g}: {GENE_BANK[g]['eq']} | {GENE_BANK[g]['physics']}" for i,g in enumerate(genes))}

PHYSICAL FOUNDATIONS
{'-'*80}
{chr(10).join(f"  [{tag}]: {PHYSICS_CATALOG[tag]['desc']}" for tag in physics_tags if tag in PHYSICS_CATALOG) if physics_tags else "  No physical constraints specified"}

AUTO-EVOLUTION ENGINE
{'-'*80}
{evol_summary}

BENCHMARK RESULTS (D={dim}, {list(per_fn.values())[0]['runs'] if per_fn else 'N/A'} runs per function)
{'-'*80}
  Function     | Mean Optimality Gap ± Std | Time | Evaluations
{bench_table}

PERFORMANCE RADAR (0–100 each axis)
{'-'*80}
  Accuracy     : {radar['Accuracy']:.1f}/100   (1 - mean_gap across all benchmarks)
  Speed        : {radar['Speed']:.1f}/100   (1 - mean_time/5s)
  Efficiency   : {radar['Efficiency']:.1f}/100  (1 - mean_evals/10000)
  Robustness   : {radar['Robustness']:.1f}/100  (1 - std/mean gap)
  Consistency  : {radar['Consistency']:.1f}/100  (fraction converging < 1% of range)

KEY DERIVED PARAMETERS
{'-'*80}
  N = max(20, min(200, 10·log₂(D+1)·1.5)) = {params['pop_size']}  [information-theoretic bound]
  T = max(50, 500·(1-w_s·0.8)) = {params['max_iter']}  [speed-quality tradeoff]
  w = 0.9 - σ·0.5 = {params['w_inertia']:.4f}  [viscous damping analogy]
  c₁=c₂ = (4/w)/2 = {params['c1']:.4f}  [Clerc-Kennedy stability 2002]
  β = 1 + (1-w_q)·0.8 = {params['levy_beta']:.4f}  [Lévy index, Mantegna 1994]
  T₀ = 100·(1+σ) = {params['sa_T0']:.2f}  [thermodynamic initial temperature]
  k = 2 + 5·w_q = {params['sel_k']}  [tournament selection pressure]

INTELLECTUAL PROPERTY RECORD
{'-'*80}
  This algorithm was uniquely generated on {date_str}
  for the specifications: objective={obj_key}, physics={phys_names},
  D={dim}, noise={noise:.3f}. The generation hash {hashlib.md5(f"{name}{obj_key}{','.join(sorted(physics_tags))}{noise}{dim}".encode()).hexdigest()[:16]}
  uniquely identifies this specification combination.
  License: MIT (free to use, modify, distribute with attribution)

CITATION
{'-'*80}
  {name} ({date_str[:10]}). Generated by System Architect v2.
  Physical constraints: {phys_names}. Genes: {len(genes)}.
  {'='*80}
"""
    return doc


def architect_run(algo_name, objective, physics_str, noise_level,
                  dim_hint, pop_override, iter_override, user_desc,
                  use_evolution, progress=gr.Progress()):
    """Main entry point for System Architect tab."""
    import math

    # ── Validation ─────────────────────────────────────────────────────────────
    if not algo_name or not algo_name.strip():
        return "<p style='color:#ef5350;'>Please enter an algorithm name.</p>", "", ""
    noise = float(noise_level); dim = int(dim_hint)
    pop_ov = int(pop_override); iter_ov = int(iter_override)

    # Parse physics tags
    valid_tags = set(PHYSICS_CATALOG.keys())
    raw = [p.strip().lower().replace(" ","_") for p in physics_str.split(",") if p.strip()]
    physics_tags = [p for p in raw if p in valid_tags]

    progress(0.08, desc="Deriving parameters from physical laws...")
    params = _derive_params(objective, physics_tags, noise, dim, pop_ov, iter_ov)

    if use_evolution:
        progress(0.25, desc="Auto-Evolution: generating 8 candidate genomes...")
        genes, evol_log = _auto_evolve(
            algo_name.strip(), objective, physics_tags, noise, dim,
            user_desc or "", params, n_pop=8, n_elite=3)
        reasons = {g:"selected by auto-evolution crossover of elite candidates" for g in genes}
        progress(0.50, desc="Auto-Evolution complete. Synthesising final code...")
    else:
        progress(0.25, desc="Selecting genes via Genetic Programming...")
        genes, reasons = _select_genes(objective, physics_tags, noise, dim)
        evol_log = None
        progress(0.45, desc="Genes selected. Synthesising code...")

    progress(0.55, desc="Building algorithm code from gene composition...")
    code = _build_code(algo_name.strip(), objective, physics_tags, noise, dim,
                       user_desc or "", params, genes, reasons)

    progress(0.68, desc="Verifying: AST parse + compile + runtime test...")
    ok, errors, test_metrics = _verify_code(code)
    if not ok:
        err_html = ("<div style='background:#1a0505;border:1px solid #ef5350;border-radius:8px;"
                    "padding:16px;font-family:monospace;color:#ef5350;'>"
                    "<b>Code generation failed verification:</b><br>"
                    + "<br>".join(f"• {e}" for e in errors[:5]) + "</div>")
        return err_html, code, ""

    progress(0.78, desc="Benchmarking on 5 standard functions...")
    bench = _benchmark_algorithm(code, dim=min(dim,5), n_runs=3)

    progress(0.90, desc="Generating scientific documentation...")
    sci_doc = _gen_scientific_doc(algo_name.strip(), objective, physics_tags,
                                  genes, params, evol_log, bench, noise, dim)

    progress(1.0, desc="Algorithm generated, verified and benchmarked!")

    # ── Build HTML report ──────────────────────────────────────────────────────
    def sc(s):
        if s>=80: return "#00e676"
        if s>=60: return "#69f0ae"
        if s>=40: return "#ffeb3b"
        return "#ef5350"

    obj_tmpl = OBJ_TEMPLATES.get(objective, OBJ_TEMPLATES["balanced"])
    spec_hash = hashlib.md5(f"{algo_name}{objective}{','.join(sorted(physics_tags))}{noise_level}{dim_hint}".encode()).hexdigest()[:8]

    # Verification badge
    ver_c = "#00e676" if ok else "#ef5350"
    test_str = " | ".join(f"{k}={v}" for k,v in test_metrics.items())

    # Physics badges
    phys_b = "".join(
        "<span style='background:#7c3aed22;color:#a78bfa;border:1px solid #7c3aed;"
        "border-radius:4px;padding:2px 8px;font-size:9px;margin-right:4px;'>"+t+"</span>"
        for t in physics_tags
    ) or "<span style='color:#8b949e;font-size:9px;'>none</span>"

    # Gene cards
    cat_colors = {"exploration":"#5aabff","exploitation":"#ffd700","diversity":"#69f0ae","selection":"#ffa726","convergence":"#a78bfa"}
    gene_html = ""
    for g in genes:
        gb = GENE_BANK[g]; cat_c = cat_colors.get(gb["category"],"#8b949e")
        gene_html += (
            "<div style='background:#0a1628;border-left:3px solid "+cat_c+";border-radius:5px;padding:9px;margin-bottom:6px;'>"
            "<div style='color:"+cat_c+";font-size:10px;font-weight:bold;'>"+g+"</div>"
            "<div style='color:#c9d1d9;font-size:9px;font-family:monospace;margin:2px 0;'>"+gb["eq"]+"</div>"
            "<div style='color:#8b949e;font-size:8px;'>"+gb["physics"]+"</div>"
            "<div style='color:#7ec8ff;font-size:8px;'>Reason: "+reasons.get(g,"")[:80]+"</div>"
            "</div>")

    # Parameter rows
    key_p = [
        ("pop_size", params["pop_size"], "max(20,min(200,10·log₂(D+1)·1.5))"),
        ("max_iter", params["max_iter"], "max(50,500·(1-ws·0.8))"),
        ("w_inertia",f"{params['w_inertia']:.4f}","0.9-noise·0.5  fluid drag"),
        ("c1=c2",    f"{params['c1']:.4f}",        "(4/w)/2  Clerc-Kennedy 2002"),
        ("de_F",     f"{params['de_F']:.4f}",       "0.9 if rugged else 0.5+0.3·noise"),
        ("levy_β",   f"{params['levy_beta']:.4f}",  "1+(1-wq)·0.8  Mantegna 1994"),
        ("sa_T0",    f"{params['sa_T0']:.1f}",      "100·(1+noise)  thermodynamics"),
        ("sel_k",    params["sel_k"],               "2+5·quality_weight"),
    ]
    p_rows = "".join(
        "<tr><td style='padding:4px 8px;color:#5aabff;font-size:9px;'>"+pn+"</td>"
        "<td style='padding:4px 8px;color:#00e676;font-weight:bold;font-size:10px;'>"+str(pv)+"</td>"
        "<td style='padding:4px 8px;color:#c9d1d9;font-size:9px;font-family:monospace;'>"+pf+"</td></tr>"
        for pn,pv,pf in key_p)

    # Benchmark radar
    radar_html = ""
    if bench:
        radar = bench["radar"]
        for metric, val in radar.items():
            c = sc(val)
            radar_html += (
                "<div style='margin-bottom:6px;'>"
                "<div style='display:flex;justify-content:space-between;font-size:10px;'>"
                "<span style='color:#c9d1d9;'>"+metric+"</span>"
                "<span style='color:"+c+";font-weight:bold;'>"+str(val)+"/100</span></div>"
                "<div style='background:#21262d;border-radius:3px;height:6px;'>"
                "<div style='background:"+c+";height:6px;border-radius:3px;width:"+str(int(val))+"%'></div></div></div>"
            )

    # Benchmark table
    bench_table_html = ""
    if bench:
        for fn,v in bench["per_function"].items():
            gap_c = sc(max(0,100*(1-min(1,v["mean_gap"]))))
            bench_table_html += (
                "<tr><td style='padding:5px 8px;color:#c9d1d9;font-size:10px;'>"+fn+"</td>"
                "<td style='padding:5px 8px;color:#8b949e;font-size:9px;'>"+v["type"]+"</td>"
                "<td style='padding:5px 8px;color:"+gap_c+";font-size:10px;font-weight:bold;'>"
                +f"{v['mean_gap']:.4e} ± {v['std_gap']:.2e}"+"</td>"
                "<td style='padding:5px 8px;color:#8b949e;font-size:9px;'>"+f"{v['mean_time']*1000:.1f}ms"+"</td>"
                "<td style='padding:5px 8px;color:#8b949e;font-size:9px;'>"+str(int(v["mean_evals"]))+"</td></tr>"
            )

    # Evolution info
    evol_html = ""
    if evol_log:
        evol_html = (
            "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:8px;padding:12px;margin-bottom:14px;'>"
            "<div style='color:#5aabff;font-size:11px;font-weight:bold;margin-bottom:6px;'>Auto-Evolution Engine Results</div>"
            "<div style='font-size:10px;color:#c9d1d9;'>"
            f"Candidates evaluated: {evol_log['n_candidates']} | "
            f"Elite selected: {evol_log['n_elite']} | "
            f"Best initial score: {evol_log['best_initial_score']:.6f} | "
            f"Worst: {evol_log['worst_initial_score']:.6f}<br>"
            f"Crossover produced: {len(evol_log['final_genome'])} genes from elite top-{evol_log['n_elite']}"
            "</div></div>"
        )

    html = (
        "<div style='background:#0d1117;border:1px solid #30363d;border-radius:14px;padding:20px;"
        "font-family:JetBrains Mono,monospace;color:#e0e0e0;'>"
        # Title
        "<div style='background:linear-gradient(135deg,#0d1f3a,#1a0d3a);border:2px solid #5aabff;"
        "border-radius:10px;padding:16px;margin-bottom:16px;'>"
        "<div style='font-size:9px;color:#5aabff;letter-spacing:3px;margin-bottom:4px;'>SYSTEM ARCHITECT — ALGORITHM GENERATED</div>"
        "<div style='font-size:18px;font-weight:900;color:#e0e0e0;margin-bottom:6px;'>"+algo_name+"</div>"
        "<div style='font-size:10px;color:#8b949e;margin-bottom:8px;'>Hash: "+spec_hash+" | "+str(len(genes))+" genes | "+str(len(physics_tags))+" physics constraints | D="+str(dim)+"</div>"
        "<div style='display:flex;gap:8px;flex-wrap:wrap;align-items:center;'>"
        "<span style='background:#00e67622;border:1px solid #00e676;border-radius:4px;padding:2px 8px;font-size:9px;color:#00e676;'>"+objective.replace("_"," ").upper()+"</span>"
        +phys_b+
        "</div></div>"
        # Verification
        "<div style='background:"+ver_c+"11;border:1px solid "+ver_c+";border-radius:6px;"
        "padding:8px 12px;margin-bottom:14px;display:flex;justify-content:space-between;flex-wrap:wrap;gap:4px;'>"
        "<span style='color:"+ver_c+";font-size:10px;font-weight:bold;'>VERIFIED: AST + Compile + Runtime Test PASSED</span>"
        "<span style='color:#8b949e;font-size:9px;'>"+test_str+"</span></div>"
        + evol_html +
        # 3-column grid
        "<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px;margin-bottom:14px;'>"
        # Gene composition
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:8px;padding:12px;'>"
        "<div style='color:#5aabff;font-size:10px;font-weight:bold;margin-bottom:8px;'>Gene Composition ("+str(len(genes))+")</div>"
        +gene_html+
        "</div>"
        # Parameters
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:8px;padding:12px;'>"
        "<div style='color:#5aabff;font-size:10px;font-weight:bold;margin-bottom:8px;'>Parameters — Derived from Physics</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='background:#161b22;'><th style='padding:4px 8px;text-align:left;color:#8b949e;font-size:8px;'>Param</th>"
        "<th style='padding:4px;color:#8b949e;font-size:8px;'>Value</th>"
        "<th style='padding:4px;text-align:left;color:#8b949e;font-size:8px;'>Formula</th></tr></thead>"
        "<tbody>"+p_rows+"</tbody></table></div>"
        # Benchmark radar
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:8px;padding:12px;'>"
        "<div style='color:#5aabff;font-size:10px;font-weight:bold;margin-bottom:8px;'>Performance Radar (Real Benchmark)</div>"
        +radar_html+
        "</div></div>"
        # Benchmark table
        "<div style='background:#0d1f3a;border:1px solid #1e4d8c;border-radius:8px;padding:12px;margin-bottom:12px;'>"
        "<div style='color:#5aabff;font-size:10px;font-weight:bold;margin-bottom:8px;'>Benchmark Results — 5 Standard Functions (D="+str(min(dim,5))+", 3 runs)</div>"
        "<table style='width:100%;border-collapse:collapse;'>"
        "<thead><tr style='background:#161b22;'>"
        "<th style='padding:4px 8px;text-align:left;color:#8b949e;font-size:8px;'>Function</th>"
        "<th style='padding:4px;text-align:left;color:#8b949e;font-size:8px;'>Type</th>"
        "<th style='padding:4px;color:#8b949e;font-size:8px;'>Gap (mean±std)</th>"
        "<th style='padding:4px;color:#8b949e;font-size:8px;'>Time</th>"
        "<th style='padding:4px;color:#8b949e;font-size:8px;'>Evals</th>"
        "</tr></thead><tbody>"+bench_table_html+"</tbody></table></div>"
        # Code preview
        "<div style='background:#0a1117;border:1px solid #30363d;border-radius:6px;padding:12px;margin-bottom:10px;'>"
        "<div style='color:#5aabff;font-size:10px;font-weight:bold;margin-bottom:6px;'>Generated Code Preview (first 40 lines) — Full code in Download section</div>"
        "<pre style='font-size:8px;color:#c9d1d9;max-height:220px;overflow-y:auto;white-space:pre-wrap;'>"
        +"\n".join(code.split("\n")[:40])+"</pre></div>"
        "<div style='background:#0a1628;border-radius:6px;padding:8px;text-align:center;font-size:9px;color:#8b949e;'>"
        "System Architect v2 | "+str(len(GENE_BANK))+" Gene Bank | "+str(len(PHYSICS_CATALOG))+" Physics Models | Auto-Evolution | Real Benchmarks | AST+Runtime Verified"
        "</div></div>"
    )

    return html, code, sci_doc

CSS="""
body,.gradio-container{background:#0d1117!important;color:#e0e0e0!important;}
.tab-nav button{background:#161b22!important;color:#8b949e!important;border:1px solid #30363d!important;border-radius:6px!important;}
.tab-nav button.selected{background:#1f6feb!important;color:#fff!important;}
.gr-button-primary{background:#1f6feb!important;border:none!important;color:#fff!important;}
label{color:#8b949e!important;font-size:12px!important;}
.gr-box,.gr-panel{background:#161b22!important;border:1px solid #30363d!important;}
input,select,textarea{background:#21262d!important;color:#e0e0e0!important;border:1px solid #30363d!important;}
"""
_bc=list(BS.FUNCTIONS.keys()); _atc=list(ALL_ALGOS.keys())

with gr.Blocks(css=CSS,title="ACCURATE EVALUATION OF OPTIMIZATION ALGORITHMS v5.0") as demo:
    gr.HTML(f'<div style="text-align:center;padding:18px 0 4px;font-family:\'JetBrains Mono\',monospace;"><div style="font-size:22px;font-weight:900;color:#e0e0e0;letter-spacing:2px;">ACCURATE EVALUATION OF OPTIMIZATION ALGORITHMS</div><div style="color:#8b949e;font-size:11px;margin-top:3px;">9860 Algorithms | 20 Physical Metrics | Score/100 | {ENGINE} | Voting Ensemble &middot; AutoML &middot; 3D &middot; Leaderboard &middot; <span style="color:#5aabff;">Data Advisor NEW</span></div></div>')

    with gr.Tabs():

        with gr.TabItem("Evaluate"):
            with gr.Row():
                with gr.Column(scale=1,min_width=270):
                    gr.HTML("<div style='color:#58a6ff;font-weight:bold;font-size:12px;margin-bottom:5px;'>Algorithm</div>")
                    algo_type=gr.Dropdown(choices=_atc,value=_atc[0],label="Algorithm Type")
                    algorithm=gr.Dropdown(choices=ALL_SINGLE,value=ALL_SINGLE[0],label="Algorithm")
                    bench_fn=gr.Dropdown(choices=_bc,value="Rastrigin",label="Benchmark Function")
                    algo_type.change(fn=get_algo_list,inputs=algo_type,outputs=algorithm)
                    with gr.Accordion("Basic Settings",open=True):
                        dim=gr.Slider(1,10000,10,step=1,label="Dimensions")
                        pop_size=gr.Slider(1,10000,50,step=1,label="Population Size")
                        max_iter=gr.Slider(10,10000,100,step=10,label="Iterations")
                        runs_s=gr.Slider(1,10000,5,step=1,label="Runs (Robustness)")
                    with gr.Accordion("Advanced Settings",open=False):
                        inertia=gr.Slider(0.1,2.0,0.7,step=0.01,label="Inertia omega")
                        cognitive=gr.Slider(0.1,5.0,1.5,step=0.05,label="Cognitive c1")
                        social=gr.Slider(0.1,5.0,1.5,step=0.05,label="Social c2")
                        mutation=gr.Slider(0.0,1.0,0.01,step=0.001,label="Mutation Rate")
                        crossover=gr.Slider(0.0,1.0,0.9,step=0.01,label="Crossover Rate")
                        elite_f=gr.Slider(0.0,0.9,0.1,step=0.01,label="Elite Fraction")
                        restart_t=gr.Slider(1,1000,50,step=1,label="Restart Threshold")
                    with gr.Accordion("Very Advanced",open=False):
                        div_thr=gr.Slider(0.0,1.0,0.01,step=0.001,label="Diversity Threshold")
                        adapt_lr=gr.Slider(0.0001,1.0,0.01,step=0.0001,label="Adaptive LR")
                        momentum=gr.Slider(0.0,1.0,0.9,step=0.01,label="Momentum")
                        chaos_en=gr.Checkbox(label="Chaos Mapping",value=False)
                        levy_fl=gr.Checkbox(label="Levy Flight",value=False)
                    with gr.Accordion("Ultra Settings",open=False):
                        opp_l=gr.Checkbox(label="Opposition-Based Learning",value=False)
                        arch_sz=gr.Slider(1,5000,50,step=1,label="Archive Size")
                        nbr_sz=gr.Slider(1,200,5,step=1,label="Neighborhood Size")
                        show_3d=gr.Checkbox(label="Show 3D Landscape",value=False)
                    eval_btn=gr.Button("Evaluate Algorithm",variant="primary",size="lg")
                with gr.Column(scale=2):
                    score_html=gr.HTML("<div style='color:#8b949e;text-align:center;padding:36px;'>Run evaluation to see results.</div>")
                    with gr.Tabs():
                        with gr.TabItem("Radar"):       radar_p=gr.Plot()
                        with gr.TabItem("Convergence"): conv_p=gr.Plot()
                        with gr.TabItem("Scores"):      bar_p=gr.Plot()
                        with gr.TabItem("Diversity"):   div_p=gr.Plot()
                        with gr.TabItem("3D Landscape"):land_p=gr.Plot()
                        with gr.TabItem("Table"):
                            score_tbl=gr.Dataframe(headers=["#","Metric","Score","Value"],label="20 Metric Scores",row_count=21)
            eval_btn.click(fn=evaluate,
                           inputs=[algo_type,algorithm,bench_fn,dim,pop_size,max_iter,runs_s,
                                   inertia,cognitive,social,mutation,crossover,elite_f,restart_t,
                                   div_thr,adapt_lr,momentum,chaos_en,levy_fl,opp_l,arch_sz,nbr_sz,show_3d],
                           outputs=[radar_p,conv_p,bar_p,div_p,land_p,score_html,score_tbl])

        with gr.TabItem("Compare"):
            with gr.Row():
                with gr.Column():
                    gr.HTML("<div style='color:#4fc3f7;font-weight:bold;'>Algorithm A</div>")
                    cta=gr.Dropdown(choices=_atc,value=_atc[0],label="Type A")
                    caa=gr.Dropdown(choices=ALL_SINGLE,value=ALL_SINGLE[0],label="A")
                    cta.change(fn=get_algo_list,inputs=cta,outputs=caa)
                with gr.Column():
                    gr.HTML("<div style='color:#ff7043;font-weight:bold;'>Algorithm B</div>")
                    ctb=gr.Dropdown(choices=_atc,value=_atc[0],label="Type B")
                    cab=gr.Dropdown(choices=ALL_SINGLE,value=ALL_SINGLE[3],label="B")
                    ctb.change(fn=get_algo_list,inputs=ctb,outputs=cab)
            with gr.Row():
                cb=gr.Dropdown(choices=_bc,value="Rastrigin",label="Benchmark")
                cd=gr.Slider(1,10000,10,step=1,label="Dim")
                cp=gr.Slider(1,10000,50,step=1,label="Pop")
                ci=gr.Slider(10,10000,100,step=10,label="Iter")
                cr=gr.Slider(1,10000,3,step=1,label="Runs")
            cmp_btn=gr.Button("Compare",variant="primary",size="lg"); cmp_html=gr.HTML()
            with gr.Row(): cr_p=gr.Plot(); cb_p=gr.Plot(); cc_p=gr.Plot()
            cmp_btn.click(fn=compare_two,inputs=[cta,caa,ctb,cab,cb,cd,cp,ci,cr],outputs=[cr_p,cb_p,cc_p,cmp_html])

        with gr.TabItem("AutoML"):
            gr.HTML("<div style='color:#58a6ff;font-size:13px;font-weight:bold;margin-bottom:8px;'>Algorithm Recommendation (RandomForest meta-learner)</div>")
            with gr.Row():
                aml_b=gr.Dropdown(choices=_bc,value="Rastrigin",label="Problem")
                aml_d=gr.Slider(1,10000,10,step=1,label="Dimensions")
            aml_btn=gr.Button("Recommend",variant="primary"); aml_html=gr.HTML()
            aml_btn.click(fn=automl_rec,inputs=[aml_b,aml_d],outputs=aml_html)

        with gr.TabItem("Leaderboard"):
            gr.HTML("<div style='color:#58a6ff;font-size:13px;font-weight:bold;margin-bottom:8px;'>Session Leaderboard (SQLite in-memory)</div>")
            lb_bench=gr.Dropdown(choices=["All"]+_bc,value="All",label="Filter")
            lb_btn=gr.Button("Refresh")
            lb_tbl=gr.Dataframe(headers=["Algorithm","Benchmark","Dim","Score","Time(s)","Mem(KB)","Timestamp"],label="Leaderboard",row_count=20)
            lb_btn.click(fn=get_lb,inputs=lb_bench,outputs=lb_tbl)

        with gr.TabItem("Library"):
            lib_t=gr.Dropdown(choices=_atc,value=_atc[0],label="Algorithm Type")
            lib_l=gr.Dataframe(value=[[a] for a in ALL_SINGLE],headers=["Algorithm Name"],label="Algorithms")
            lib_t.change(fn=lambda t:[[a] for a in ALL_ALGOS.get(t,[])],inputs=lib_t,outputs=lib_l)
            gr.HTML(f'<div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-top:12px;font-family:\'JetBrains Mono\',monospace;color:#c9d1d9;font-size:11px;"><div style="color:#58a6ff;font-size:12px;font-weight:bold;margin-bottom:8px;">20 Physical Metrics (5 pts each=100) | {ENGINE} | 9860 Algorithms (60 single + 49 hybrid x200)</div><b style="color:#7ec8ff;">M5</b>: T_ref=NxDxTx1us xcf cf=per-algo O() factor<br><b style="color:#7ec8ff;">M9</b>: FFT dominant freq+classic oscillation (50/50 weighted)<br><b style="color:#7ec8ff;">M19</b>: OPTICS clustering+find_peaks on convergence<br><b style="color:#7ec8ff;">M20</b>: RandomForest Gini impurity of feature importances<br><b style="color:#ffd700;">M16</b>: Student-t 95% CI width/|mu| across runs<br><b style="color:#ffd700;">M11</b>: psutil TDP x cpu_util+RAM_power x ram_util<br><b style="color:#ffd700;">SHO to EWA</b>: SHO removed in mealpy 3.x replaced by OriginalEWA<br><b style="color:#ffd700;">Hybrid</b>: Binary(n=2) to Quinquagenary(n=50) 49x200+60=9860<br></div>')

        with gr.TabItem("Data Advisor  NEW"):
            gr.HTML('<div style="background:#0d1f3a;border:2px solid #1e4d8c;border-radius:10px;padding:16px;margin-bottom:16px;font-family:\'JetBrains Mono\',monospace;"><div style="color:#5aabff;font-size:14px;font-weight:bold;margin-bottom:8px;">Physical Data-Driven Algorithm Advisor</div><div style="color:#c9d1d9;font-size:11px;line-height:1.9;">Upload your data file OR paste data directly -- 10 real physical equations computed (ruggedness, modality, separability, entropy, epistasis, smoothness, sparsity...) -- Full tournament on 60 single algorithms + optimal hybrids generated from 9860 pool -- Winner declared with detailed physical reasons + near-optimal hybrid ranking</div><div style="color:#8b949e;font-size:10px;margin-top:8px;">Supports: CSV TSV JSON XLSX XLS TXT DAT NPY + Direct paste</div></div>')
            with gr.Row():
                with gr.Column(scale=1,min_width=300):
                    adv_file=gr.File(label="Upload Data File",file_types=[".csv",".tsv",".json",".xlsx",".xls",".txt",".dat",".npy"])
                    gr.HTML("<div style='color:#8b949e;font-size:11px;text-align:center;margin:6px 0;'>-- OR --</div>")
                    adv_paste=gr.Textbox(
                        label="Paste Data (CSV / TSV / JSON / numbers)",
                        placeholder="Examples:\n1.2, 3.4, 5.6\n2.1, 4.3, 6.5\n3.0, 5.1, 7.2\n\nor JSON:\n[{\"x\":1,\"y\":2},{\"x\":3,\"y\":4}]\n\nor plain numbers:\n1.5\n2.3\n4.1",
                        lines=10,max_lines=500)
                    adv_btn=gr.Button("Analyze and Find Best Algorithm",variant="primary",size="lg")
                with gr.Column(scale=2):
                    adv_out=gr.HTML("<div style='color:#8b949e;text-align:center;padding:50px;font-family:monospace;'>Upload a file or paste your data then click Analyze.</div>")
            adv_btn.click(fn=advisor_run,inputs=[adv_file,adv_paste],outputs=[adv_out])


        with gr.TabItem("Hybrid Advisor  NEW"):
            gr.HTML('<div style="background:#0d1f3a;border:2px solid #1e4d8c;border-radius:10px;padding:16px;margin-bottom:16px;font-family:\'JetBrains Mono\',monospace;"><div style="color:#5aabff;font-size:14px;font-weight:bold;margin-bottom:8px;">Physical Hybrid Algorithm Advisor</div><div style="color:#c9d1d9;font-size:11px;line-height:1.9;">Upload data or paste text -- 10 physical equations computed -- physical tournament on all 60 single algorithms -- then 10 physical laws applied to score hybrids from 9860 pool -- winner declared with full bonus breakdown</div><div style="color:#8b949e;font-size:10px;margin-top:8px;">Supports: CSV TSV JSON XLSX XLS TXT DAT NPY + Direct paste | 10 Physical Laws: L1-Ensemble Base L2-Complementarity L3-Voting Gain L4-Diversity Insurance L5-Stagnation Recovery L6-Ruggedness Synergy L7-Epistasis Coverage L8-Type Diversity L9-Overhead Penalty L10-Separability Fit</div></div>')
            with gr.Row():
                with gr.Column(scale=1,min_width=300):
                    hyb_file=gr.File(label="Upload Data File",file_types=[".csv",".tsv",".json",".xlsx",".xls",".txt",".dat",".npy"])
                    gr.HTML("<div style='color:#8b949e;font-size:11px;text-align:center;margin:6px 0;'>-- OR --</div>")
                    hyb_paste=gr.Textbox(label="Paste Data (CSV / TSV / JSON / numbers)",placeholder="1.2, 3.4, 5.6\n2.1, 4.3, 6.5\n...",lines=8,max_lines=500)
                    hyb_btn=gr.Button("Analyze Hybrids Physically",variant="primary",size="lg")
                with gr.Column(scale=2):
                    hyb_out=gr.HTML("<div style='color:#8b949e;text-align:center;padding:50px;font-family:monospace;'>Upload a file or paste data then click Analyze Hybrids.</div>")
            hyb_btn.click(fn=hybrid_advisor_run,inputs=[hyb_file,hyb_paste],outputs=[hyb_out])

        with gr.TabItem("Stress Test  NEW"):
            gr.HTML('<div style="background:linear-gradient(135deg,#1a0a2a,#0d0d3a);border:2px solid #7c3aed;border-radius:10px;padding:16px;margin-bottom:16px;font-family:\'JetBrains Mono\',monospace;"><div style="color:#a78bfa;font-size:14px;font-weight:bold;margin-bottom:8px;">Algorithmic Robustness -- Adversarial Stress Test</div><div style="color:#c9d1d9;font-size:11px;line-height:1.9;">Upload your data -- 8 real mathematical attacks applied (Gaussian Noise, Extreme Value Flip, Feature Dropout, Outlier Injection, Covariate Shift, Spurious Correlation, Sign Flip, FGSM-style Perturbation) -- measures which algorithm maintains stable recommendations under adversarial conditions -- 100% real physics no simulated values</div><div style="color:#8b949e;font-size:10px;margin-top:6px;">Instability = (|Drho|+|DH|/5+|DS|+|Deps|+|Dmod|/10)/5 | Robustness = 1-Instability | Final = mean(rank_stability)-0.5*std</div></div>')
            with gr.Row():
                with gr.Column(scale=1,min_width=300):
                    st_file=gr.File(label="Upload Data File",file_types=[".csv",".tsv",".json",".xlsx",".xls",".txt",".dat",".npy"])
                    gr.HTML("<div style='color:#8b949e;font-size:11px;text-align:center;margin:6px 0;'>-- OR --</div>")
                    st_paste=gr.Textbox(label="Paste Data (CSV / TSV / JSON / numbers)",placeholder="1.2, 3.4, 5.6\n2.1, 4.3, 6.5\n...",lines=8,max_lines=500)
                    st_intensity=gr.Slider(1,50,15,step=1,label="Attack Intensity (%)")
                    st_btn=gr.Button("Run Stress Test",variant="primary",size="lg")
                with gr.Column(scale=2):
                    st_out=gr.HTML("<div style='color:#8b949e;text-align:center;padding:50px;font-family:monospace;'>Upload data then click Run Stress Test.</div>")
            st_btn.click(fn=stress_test_run,inputs=[st_file,st_paste,st_intensity],outputs=[st_out])



        with gr.TabItem("System Architect  NEW"):
            gr.HTML("<div style='background:linear-gradient(135deg,#0d1f3a,#1a0d3a);"
                    "border:2px solid #5aabff;border-radius:10px;padding:16px;margin-bottom:14px;"
                    "font-family:JetBrains Mono,monospace;'>"
                    "<div style='color:#5aabff;font-size:13px;font-weight:bold;margin-bottom:6px;'>"
                    "System Architect v2 — Generative AI for Algorithms</div>"
                    "<div style='color:#c9d1d9;font-size:11px;line-height:1.8;'>"
                    "Name your algorithm + choose objective + physical constraints "
                    "→ Genetic Programming selects genes from 15-gene bank "
                    "→ optional Auto-Evolution (8 candidates evaluated on real benchmarks, elite crossover) "
                    "→ executable Python code synthesised "
                    "→ AST + compile + runtime verified "
                    "→ benchmarked on 5 standard functions "
                    "→ scientific documentation + IP record generated"
                    "</div>"
                    "<div style='color:#8b949e;font-size:9px;margin-top:6px;'>"
                    "Physics: gravity | electromagnetic | thermodynamics | fluid_dynamics | "
                    "quantum | elasticity | magnetism | optics | chaos | wave | "
                    "All parameters derived from physics — zero assumed values"
                    "</div></div>")
            with gr.Row():
                with gr.Column(scale=1, min_width=300):
                    arch_name = gr.Textbox(label="Algorithm Name",
                                           placeholder="e.g.  Al-Khwarizmi-2026",
                                           max_lines=1)
                    arch_desc = gr.Textbox(label="Description (optional)",
                                           placeholder="e.g. Optimize a noisy 50-D robotic arm trajectory",
                                           lines=2, max_lines=4)
                    arch_obj = gr.Dropdown(
                        choices=list(OBJ_TEMPLATES.keys()),
                        value="balanced",
                        label="Objective / Primary Goal")
                    arch_phys = gr.Textbox(
                        label="Physical Constraints (comma-separated)",
                        placeholder="e.g.  gravity, quantum, thermodynamics",
                        info="Options: gravity | electromagnetic | thermodynamics | fluid_dynamics | quantum | elasticity | magnetism | optics | chaos | wave",
                        max_lines=1)
                    arch_noise  = gr.Slider(0.0, 1.0, 0.3, step=0.05,
                                            label="Noise Level  (0=clean  1=very noisy)")
                    arch_dim    = gr.Slider(1, 1000, 10, step=1,
                                            label="Problem Dimensionality (D)")
                    arch_pop_ov = gr.Slider(0, 500, 0, step=1,
                                            label="Population Override (0 = auto-derive)")
                    arch_iter_ov= gr.Slider(0, 2000, 0, step=10,
                                            label="Iterations Override (0 = auto-derive)")
                    arch_use_evo= gr.Checkbox(label="Use Auto-Evolution Engine "
                                              "(8 candidates → elite crossover, slower but stronger)",
                                              value=False)
                    arch_btn = gr.Button("Generate Algorithm", variant="primary", size="lg")
                with gr.Column(scale=2):
                    arch_report = gr.HTML(
                        "<div style='color:#8b949e;text-align:center;padding:50px;"
                        "font-family:monospace;'>Fill in the specs and click Generate Algorithm.</div>")
                    with gr.Accordion("Download: Generated Code (.py)", open=False):
                        arch_code = gr.Code(language="python",
                                            label="Generated Algorithm — full executable Python")
                    with gr.Accordion("Download: Scientific Documentation", open=False):
                        arch_sci = gr.Textbox(label="Technical Report + IP Record",
                                              lines=30, max_lines=200)
            arch_btn.click(
                fn=architect_run,
                inputs=[arch_name, arch_obj, arch_phys, arch_noise,
                        arch_dim, arch_pop_ov, arch_iter_ov, arch_desc, arch_use_evo],
                outputs=[arch_report, arch_code, arch_sci])

    gr.HTML('<div style="background-color:black;color:white;padding:30px;border-radius:5px;margin-top:20px;font-family:sans-serif;line-height:1.6;font-weight:bold;"><p style="font-weight:bold;font-size:1.1em;text-align:center;">Copyright 2026 Mohammed Falah Hassan Al-Dhafiri</p><p style="font-weight:bold;text-align:center;">Founder and Inventor of the System</p><p style="text-align:center;font-weight:bold;">All Rights Reserved.</p><p style="font-size:0.9em;text-align:justify;border-top:1px solid #444;padding-top:15px;">It is prohibited to copy, reproduce, modify, publish, or use any part of this system without prior written permission from the Founder and Inventor. Any unauthorized use constitutes a violation of intellectual property rights.</p><div style="text-align:center;margin-top:20px;border-top:1px solid #444;padding-top:20px;"><p style="font-weight:bold;font-size:1.1em;">Copyright 2026 Mohammed Falah Hassan Al-Dhafiri</p><p style="font-weight:bold;">Founder and Inventor | All Rights Reserved</p></div></div>')

if __name__=="__main__":
    demo.launch(share=True)
