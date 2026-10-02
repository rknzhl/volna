"""Build a vector graphical highlight from one analytical two-wave field."""

from pathlib import Path
import os

os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/takers-matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Polygon, Ellipse, FancyArrowPatch, Rectangle
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output' / 'pdf'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 13, 'axes.labelsize': 12,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.edgecolor': '#a5b1b9', 'axes.linewidth': .75,
    'xtick.color': '#53616c', 'ytick.color': '#53616c',
    'text.color': '#182d3c', 'axes.labelcolor': '#53616c',
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'mathtext.fontset': 'dejavusans',
})

BLUE, ORANGE, INK = '#1677ac', '#d77a2a', '#182d3c'
SENSOR_COLORS = ['#345c9d', '#169386', '#9a5a91']
FREQUENCIES = np.array([.55, .55 * np.sqrt(2)])
AMPLITUDES = np.array([1.5, 1.0])  # centimetres
ANGLES = np.deg2rad([24, -58])
DEPTH, CURRENT = 3.0, .08
WAVENUMBERS = np.array([brentq(lambda k: (2*np.pi*f-k*CURRENT*np.cos(a))**2
    -9.81*k*np.tanh(k*DEPTH), .001, 10.0) for f,a in zip(FREQUENCIES,ANGLES)])
WAVEVECTORS = WAVENUMBERS[:, None] * np.c_[np.cos(ANGLES), np.sin(ANGLES)]
INITIAL_PHASES = np.array([.35, -1.1])
INTRINSIC = 2*np.pi*FREQUENCIES - WAVENUMBERS*CURRENT*np.cos(ANGLES)
BOTTOM_SENSORS = np.array([[5.5, .8], [12.0, 1.5]])
BOTTOM_COLORS = ['#b66331', '#56786e']
SENSORS = np.array([[4, 3], [11, 5], [14, 10], [3, 9], [8, 11], [15, 2], [7, 7], [12, 12]])
SNAPSHOT = 5.0


def phases(points):
    return INITIAL_PHASES - np.einsum('...d,jd->...j', np.asarray(points), WAVEVECTORS)


def field(points, times):
    phase = phases(points)
    return np.sum(AMPLITUDES * np.cos(phase[..., None, :] +
                  2*np.pi*np.asarray(times)[..., None]*FREQUENCIES), axis=-1)


def velocity(points,times,z,include_current=True):
    phase=phases(points)[...,None,:]+2*np.pi*np.asarray(times)[...,None]*FREQUENCIES
    gain=(AMPLITUDES/100)*INTRINSIC*np.cosh(WAVENUMBERS*(z+DEPTH))/np.sinh(WAVENUMBERS*DEPTH)
    harmonic=gain*np.cos(phase)
    result=np.einsum('...j,jd->...d',harmonic,np.c_[np.cos(ANGLES),np.sin(ANGLES)])
    if include_current:
        result[...,0]+=CURRENT
    return result


def project(points):
    p = np.asarray(points)
    return np.stack([.18 + .76*p[..., 0]/18 - .23*p[..., 1]/14,
                     .14 + .18*p[..., 0]/18 + .44*p[..., 1]/14], axis=-1)


def arrow(ax, start, end, color, lw=2, size=15, zorder=10):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>',
                  mutation_scale=size, linewidth=lw, color=color, zorder=zorder))


def build(lang):
    ru = lang == 'ru'
    fig = plt.figure(figsize=(15.6, 9.0), facecolor='white')
    ax = fig.add_axes([.005, .10, .685, .73])
    ax.set_xlim(-.09, 1.07)
    ax.set_ylim(-.15, .86)
    ax.set_aspect('equal')
    ax.axis('off')
    # Wind loading can contribute to buoy drift; no wind-to-drift law is assumed.
    wind_path=MplPath([[.035,.73],[.115,.79],[.260,.74],[.345,.80]],
                      [MplPath.MOVETO]+[MplPath.CURVE4]*3)
    ax.add_patch(FancyArrowPatch(path=wind_path,arrowstyle='-|>',
                 mutation_scale=15,linewidth=1.6,color='#688d9a',zorder=25))
    ax.text(.095,.80,('Ветер, ' if ru else 'Wind, ')+r'$\mathbf{v}_w$',
            fontsize=16,color='#557b89',zorder=25)
    bed = project([[0,0],[18,0],[18,14],[0,14]])+[0,-.22]
    ax.add_patch(Polygon(bed,facecolor='#e7e6df',edgecolor='#b4bfc2',lw=.75,zorder=0))
    ax.add_patch(Polygon(np.r_[bed[:2],(bed[:2]+[0,-.013])[::-1]],
                 facecolor='#c9ceca',edgecolor='none',zorder=0))
    # The front strip is opened to expose the horizontal bed and instrument mounts.
    ax.text(*(project([1.0,.8])+[0,-.217]),'Дно' if ru else 'Bed',fontsize=15,color='#6c7977')
    corners = project([[0,0],[18,0],[18,14],[0,14]])
    xs,ys=np.linspace(0,18,110),np.linspace(0,14,90)
    xx,yy=np.meshgrid(xs,ys)
    points=np.c_[xx.ravel(),yy.ravel()]
    eta=field(points,np.array([SNAPSHOT])).reshape(xx.shape)
    projected_surface=project(np.stack([xx,yy],axis=-1))
    projected_surface[...,1]+=.005*eta
    phase=phases(points)+2*np.pi*FREQUENCIES*SNAPSHOT
    slopes=np.einsum('pj,jd->pd',AMPLITUDES*np.sin(phase),WAVEVECTORS)*(.005/(.22/DEPTH))
    normals=np.c_[-slopes,np.ones(len(points))]
    normals/=np.linalg.norm(normals,axis=1)[:,None]
    light=np.array([-.25,-.55,.80]); light/=np.linalg.norm(light)
    diffuse=np.einsum('pd,d->p',normals,light)
    smooth=(diffuse-diffuse.min())/(diffuse.max()-diffuse.min())
    marine=LinearSegmentedColormap.from_list('marine',['#5d9fbd','#87bdd1','#b2d8e3','#d5e9ef'])
    rgba=marine((.12+.75*smooth).reshape(xx.shape))
    transparency=.14+.86*np.clip((yy-1.0)/5,0,1)**1.2
    rgba[...,:3]=rgba[...,:3]*transparency[...,None]+np.array([.87,.92,.93])*(1-transparency[...,None])
    rgba[...,3]=1
    water=ax.pcolormesh(projected_surface[...,0],projected_surface[...,1],rgba,
                       shading='gouraud',rasterized=False,antialiased=False,zorder=2)
    front_upper=projected_surface[0]
    front_lower=project(np.c_[xs,np.zeros_like(xs)])+[0,-.22]
    ax.add_patch(Polygon(np.r_[front_upper,front_lower[::-1]],facecolor='#a5cede',
                        edgecolor='#aec7d1',lw=.65,alpha=.26,zorder=6))
    right_upper=projected_surface[:,-1]
    right_lower=project(np.c_[np.full_like(ys,18),ys])+[0,-.22]
    ax.add_patch(Polygon(np.r_[right_upper,right_lower[::-1]],facecolor='#9fc5d7',
                        edgecolor='#adc5d0',lw=.65,alpha=.18,zorder=6))
    for j, (origin, color) in enumerate([([1.0, 5.0], BLUE), ([3.8, 12.5], ORANGE)]):
        tip = np.array(origin) + 3.4*np.array([np.cos(ANGLES[j]), np.sin(ANGLES[j])])
        arrow(ax, project(origin), project(tip), color, lw=2.5, size=17)
    ax.text(*(project([1.0,5.0])+[-.055,.025]),'$f_1$',color=BLUE,fontsize=18)
    ax.text(*(project([3.8,12.5])+[-.025,.020]),'$f_2$',color=ORANGE,fontsize=18)
    for i, sensor in enumerate(SENSORS[:3]):
        p = project(sensor)
        p[1] += .005*float(field(sensor,np.array([SNAPSHOT]))[0])
        color = SENSOR_COLORS[i] if i < 3 else INK
        ax.plot([p[0], p[0]], [p[1]-.022, p[1]+.075], color=INK, lw=1.1, zorder=8)
        ax.scatter(*p, s=52 if i < 3 else 20, color=color, edgecolor='white', linewidth=.85, zorder=10)
        if i < 3:
            ax.text(p[0]+.023, p[1]-.034, '$S_'+str(i+1)+'$', color=color, fontsize=16, fontweight='bold', zorder=11)
    buoy = project([9.3, 7.0])
    buoy[1] += .005*float(field([9.3,7.0],np.array([SNAPSHOT]))[0])
    # The float and its motion arrows are schematic, independent of the fixed gauges.
    drift_direction=np.array([.76/18,.18/18])
    drift_direction/=np.linalg.norm(drift_direction)
    lateral=np.array([-drift_direction[1],drift_direction[0]])
    for distance,opacity in [(.077,.92),(.117,.75),(.157,.56)]:
        centre=buoy-drift_direction*distance+[0,-.004]
        width=distance*.35
        upper,lower=centre+lateral*width,centre-lateral*width
        control=centre+drift_direction*.034
        path=MplPath([upper,control,control,lower],[MplPath.MOVETO]+[MplPath.CURVE4]*3)
        ax.add_patch(PathPatch(path,fill=False,color='#4c98b5',lw=1.3,alpha=opacity*.70,zorder=9))
        ax.add_patch(PathPatch(path,fill=False,color='#ecf8fc',lw=.95,alpha=opacity,zorder=10))
    for width in [.13]:
        ax.add_patch(Ellipse(buoy+[0,-.008],width,width*.32,facecolor='none',edgecolor='#e3f5fc',lw=.8,zorder=7))
    ax.add_patch(Ellipse(buoy+[.010,-.007],.106,.032,facecolor='#395469',alpha=.16,edgecolor='none',zorder=7))
    # A spherical wave buoy: float, narrow deck, mast and sensor head.
    float_body=Ellipse(buoy+[0,.021],.100,.086,facecolor='#e4a922',edgecolor='#98752c',lw=.9,zorder=12)
    ax.add_patch(float_body)
    ax.add_patch(Ellipse(buoy+[-.013,.034],.067,.059,facecolor='#f4c64b',edgecolor='none',zorder=13))
    ax.add_patch(Ellipse(buoy+[-.026,.047],.019,.029,angle=-20,facecolor='#ffdf88',edgecolor='none',alpha=.8,zorder=14))
    ax.add_patch(Ellipse(buoy+[0,.060],.042,.012,facecolor='#485c69',edgecolor='#384c59',lw=.6,zorder=15))
    submerged=Rectangle(buoy+[-.052,-.025],.104,.033,facecolor='#81b8cc',edgecolor='none',alpha=.88,zorder=15)
    submerged.set_clip_path(float_body)
    ax.add_patch(submerged)
    ax.add_patch(Ellipse(buoy+[0,.008],.091,.010,facecolor='none',edgecolor='#4f91a5',lw=.9,zorder=16))
    ax.plot([buoy[0],buoy[0]],[buoy[1]+.064,buoy[1]+.124],color='#415d70',lw=1.55,zorder=16)
    ax.add_patch(Ellipse(buoy+[0,.127],.016,.014,facecolor='#d77d34',edgecolor='#9c6331',lw=.6,zorder=17))
    ax.plot([buoy[0],buoy[0]],[buoy[1]+.134,buoy[1]+.154],color='#415d70',lw=.65,zorder=17)
    arrow(ax,buoy+[.063,.029],buoy+[.172,.053],INK,lw=1.4,size=12)
    ax.text(*(buoy+[.111,.001]),r'$\mathbf{v}_b$',fontsize=16)
    ax.add_patch(FancyArrowPatch(buoy+[-.079,.015],buoy+[-.079,.099],arrowstyle='<->',mutation_scale=10,color=INK,lw=.9,zorder=18))
    # Bottom-mounted heads sample velocity just above the bed.
    for j,point in enumerate(BOTTOM_SENSORS):
        p=project(point)+[0,-.22]
        col=BOTTOM_COLORS[j]
        ax.add_patch(Ellipse(p+[.006,-.001],.061,.019,facecolor='#9baba9',alpha=.25,edgecolor='none',zorder=19))
        ax.add_patch(Ellipse(p,.052,.018,facecolor='#c6d1d6',edgecolor=col,lw=.85,zorder=20))
        ax.add_patch(Polygon([p+[-.016,.003],p+[.016,.003],p+[.011,.025],p+[-.011,.025]],
                     facecolor=col,edgecolor=INK,lw=.6,zorder=21))
        for dx in [-.022,0,.022]:
            ax.plot([p[0],p[0]+dx],[p[1]+.025,p[1]+.070],color=col,lw=.7,alpha=.65,zorder=21)
        ax.text(p[0]+.03,p[1]-.012,'$D_'+str(j+1)+'$',fontsize=16,color=col,zorder=22)
    times = np.linspace(0,10,1801)
    measured = field(SENSORS[:3],times)
    signalax=fig.add_axes([.705,.385,.255,.41])
    for i in range(3):
        shift=i*6
        signalax.plot(times,measured[i]+shift,color=SENSOR_COLORS[i],lw=1.65)
        signalax.text(-.12,shift,'$S_'+str(i+1)+'$',color=SENSOR_COLORS[i],fontsize=16,
                      ha='right',va='center',clip_on=False)
    signalax.set_xlim(0,10)
    signalax.set_ylim(-3.1,15.1)
    signalax.set_yticks([])
    signalax.set_xticks([0,5,10])
    signalax.tick_params(labelsize=13,length=2.5)
    signalax.spines['left'].set_visible(False)
    signalax.set_title(r'$\eta(t)$',fontsize=16,loc='left',pad=12)
    speedax=fig.add_axes([.705,.16,.255,.125])
    bottom_velocity=velocity(BOTTOM_SENSORS,times,-DEPTH+.2,include_current=True)[...,0]*100
    for j in range(2):
        speedax.plot(times,bottom_velocity[j],color=BOTTOM_COLORS[j],lw=1.6,ls='-' if j==0 else (0,(4,2)),label='$D_'+str(j+1)+'$')
    speedax.set_xlim(0,10)
    speedax.set_xticks([0,5,10])
    speedax.set_yticks([])
    speedax.tick_params(labelsize=13,length=2.5)
    speedax.spines['left'].set_visible(False)
    speedax.set_xlabel('$t$ [с]' if ru else '$t$ [s]',fontsize=15,labelpad=7)
    speedax.set_title(r"$u_x(t)$",fontsize=16,loc='left',pad=12)
    speedax.legend(loc='upper right',ncol=2,frameon=False,fontsize=15,handlelength=1.1,
                   columnspacing=.75,borderaxespad=0,bbox_to_anchor=(1.02,1.48))
    for ext in ['pdf','svg','png']:
        water.set_rasterized(ext=='svg')
        fig.savefig(OUT/f'wave_measurements_{lang}.{ext}',dpi=260,facecolor='white',bbox_inches='tight',pad_inches=.15,
                    metadata={'Creator':'Analytical wave measurement illustration'} if ext=='pdf' else None)
    plt.close(fig)


if __name__ == '__main__':
    assert np.allclose(INTRINSIC**2,9.81*WAVENUMBERS*np.tanh(WAVENUMBERS*DEPTH),atol=1e-10)
    assert np.all(INTRINSIC>0)
    # A complete least-squares harmonic fit verifies the spectrum and mapped phases.
    t = np.linspace(0, 40, 4001)
    design = np.column_stack([f(t) for nu in FREQUENCIES
               for f in (lambda t,nu=nu: np.cos(2*np.pi*nu*t),
                         lambda t,nu=nu: np.sin(2*np.pi*nu*t))])
    coef = np.linalg.lstsq(design, field(SENSORS,t).T, rcond=None)[0]
    fitted_amp = np.hypot(coef[0::2],coef[1::2]).T
    fitted_phase = np.arctan2(-coef[1::2],coef[0::2]).T
    assert np.allclose(fitted_amp,AMPLITUDES,atol=1e-11)
    assert np.allclose(np.exp(1j*fitted_phase),np.exp(1j*phases(SENSORS)),atol=1e-11)
    for language in ['ru','en']:
        build(language)
    print('Created RU and EN: PDF, SVG, PNG. Dispersion, amplitudes and phases verified.')
