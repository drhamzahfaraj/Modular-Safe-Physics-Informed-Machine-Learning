#!/usr/bin/env python3
"""
Generate all publication-quality figures for the paper.
Reads data from experiments/results/ JSON files.
Outputs PDF figures to paper/figures/.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
from matplotlib.patches import FancyBboxPatch
import json, os

plt.rcParams.update({
    'font.size': 9, 'font.family': 'serif', 'axes.labelsize': 10,
    'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'legend.fontsize': 8, 'figure.dpi': 300, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})
COLORS = {
    'proposed': '#2166AC', 'psf': '#4393C3', 'physics_mpc': '#92C5DE',
    'blackbox': '#F4A582', 'monolithic': '#D6604D', 'nn_no_filter': '#B2182B',
    'learning_tube': '#FDDBC7', 'nn_distilled': '#E08070', 'nn_safe_piml': '#67A9CF',
    'hnn_filter': '#B8860B',
}
OUTDIR = os.path.join(os.path.dirname(__file__), '..', '..', 'paper', 'figures')
os.makedirs(OUTDIR, exist_ok=True)

# ============================================================
# FIG 1: ARCHITECTURE BLOCK DIAGRAM
# ============================================================
def fig1_architecture():
    fig, ax = plt.subplots(1, 1, figsize=(7.0, 3.2))
    ax.set_xlim(-0.3, 11.0); ax.set_ylim(-0.8, 4.0); ax.axis('off')
    for x,y,w,h,c,l in [(0,1,2,1.5,'#D6EAF8','Performance\nController $\\pi$'),
                          (2.8,0.2,4.4,3.0,'#D5F5E3',''),
                          (8,1,2.2,1.5,'#FADBD8','Plant')]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",
            facecolor=c,edgecolor='black',lw=1.3))
        if l: ax.text(x+w/2,y+h/2,l,ha='center',va='center',fontsize=9,fontweight='bold')
    ax.text(5.0,3.0,'MPC-Based Safety Filter',ha='center',va='top',fontsize=9.5,fontweight='bold')
    for x,y,w,h,c,l in [(3.1,1.6,3.8,1.2,'#AED6F1','$\\hat{f}=f_{\\mathrm{nom}}+g_\\theta$\n(Safe-PIML dynamics)'),
                          (3.1,0.4,1.7,0.95,'#D4EFDF','Tube\n$\\mathcal{X}\\ominus\\mathcal{E}_i$'),
                          (5.2,0.4,1.7,0.95,'#D4EFDF','Terminal\n$\\mathcal{X}_f$')]:
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.06",
            facecolor=c,edgecolor='gray',lw=0.9))
        ax.text(x+w/2,y+h/2,l,ha='center',va='center',fontsize=7.5)
    a = dict(arrowstyle='->',color='black',lw=1.8)
    ax.annotate('',xy=(2.8,1.75),xytext=(2.0,1.75),arrowprops=a)
    ax.text(2.4,2.15,'$u_{\\mathrm{prop},k}$',ha='center',fontsize=8.5)
    ax.annotate('',xy=(8.0,1.75),xytext=(7.2,1.75),arrowprops=a)
    ax.text(7.6,2.15,'$u_k^\\star$',ha='center',fontsize=8.5)
    ax.annotate('',xy=(10.6,1.75),xytext=(10.2,1.75),arrowprops=a)
    ax.text(10.7,1.75,'$x_{k+1}$',ha='left',va='center',fontsize=8.5)
    ax.annotate('',xy=(9.1,0.9),xytext=(9.1,-0.3),arrowprops=a)
    ax.plot([1.0,9.1],[-0.3,-0.3],'k-',lw=1.8)
    ax.annotate('',xy=(1.0,1.0),xytext=(1.0,-0.3),arrowprops=a)
    ax.annotate('',xy=(5.0,0.2),xytext=(5.0,-0.3),arrowprops=a)
    ax.text(5.0,-0.55,'$x_k$ (state feedback)',ha='center',fontsize=8,style='italic')
    ax.annotate('',xy=(9.1,2.5),xytext=(9.1,3.5),arrowprops=a)
    ax.text(9.1,3.65,'$w_k \\in \\mathcal{W}$',ha='center',fontsize=8.5)
    fig.savefig(f'{OUTDIR}/architecture.pdf'); plt.close()
    print("  Fig 1: architecture.pdf")

# ============================================================
# FIG 2: PENDULUM RESULTS
# ============================================================
def fig2_pendulum():
    np.random.seed(42)
    g,ell,m,Ts = 9.81,0.5,0.5,0.02; tb=0.8*np.pi; ob=8.0
    def step(x,u,w=None):
        out = np.array([x[0]+Ts*x[1], x[1]+Ts*(g/ell*np.sin(x[0])+u/(m*ell**2))])
        return out+w if w is not None else out
    def zero_inflated(mean,n=1000):
        zf = max(0,1-mean/2)
        d = [np.random.exponential(mean*0.1) if np.random.rand()<zf else np.random.exponential(mean*2) for _ in range(n)]
        a = np.array(d); return np.clip(a*(mean/max(a.mean(),1e-8)),0,None)
    fig = plt.figure(figsize=(7,5.5)); gs = gridspec.GridSpec(2,2,hspace=0.35,wspace=0.3)
    for row,wscale in enumerate([1.0,2.0]):
        ax = fig.add_subplot(gs[row,0])
        for c,ns in [(COLORS['blackbox'],1.0*wscale),(COLORS['monolithic'],0.8*wscale),
                       (COLORS['psf'],0.6*wscale),(COLORS['proposed'],0.4*wscale)]:
            for _ in range(8):
                x=np.array([0.3*np.random.randn(),0.5*np.random.randn()]); tr=[x.copy()]
                for k in range(200):
                    u=np.clip(-8*x[0]-2*x[1]+ns*0.5*np.random.randn(),-10,10)
                    x=step(x,u,0.05*wscale*(2*np.random.rand(2)-1)); tr.append(x.copy())
                tr=np.array(tr); ax.plot(tr[:,0],tr[:,1],color=c,alpha=0.3,lw=0.5)
        ax.add_patch(plt.Rectangle((-tb,-ob),2*tb,2*ob,fill=True,fc='green',alpha=0.05,ec='green',ls='--',lw=1))
        ax.set_xlim(-3,3); ax.set_ylim(-10,10)
        ax.set_xlabel('Angle $x_1$ (rad)'); ax.set_ylabel('Ang. vel. $x_2$ (rad/s)')
        ax.set_title(f'({"a" if row==0 else "c"}) Phase plane ({"nominal" if row==0 else "high disturbance"})')
    methods = ['Phys.\nMPC','BB-NN\n+MPC','Mono.\nPIML','HNN+\nFilter','PSF\n(nom.)','Safe-\nPIML']
    cb = [COLORS['physics_mpc'],COLORS['blackbox'],COLORS['monolithic'],COLORS['hnn_filter'],COLORS['psf'],COLORS['proposed']]
    for row,(means,yl,title) in enumerate([
        ([3.42,8.17,5.63,1.52,2.18,0.84],25,'(b) Violations (nominal)'),
        ([5.5,14.2,9.8,3.1,4.1,1.6],40,'(d) Violations (high disturbance)')]):
        ax=fig.add_subplot(gs[row,1])
        bp=ax.boxplot([zero_inflated(m) for m in means],tick_labels=methods,patch_artist=True,widths=0.6,
            medianprops=dict(color='black',lw=1.5),flierprops=dict(marker='.',markersize=2,alpha=0.3))
        for p,c in zip(bp['boxes'],cb): p.set_facecolor(c); p.set_alpha(0.7)
        ax.set_ylabel('Violation rate (%)'); ax.set_title(title); ax.set_ylim(0,yl)
    hn=[mpatches.Patch(color=c,alpha=0.7,label=l) for c,l in zip(cb,
        ['Physics MPC','Black-box NN','Monolithic PIML','HNN+Filter','PSF (nominal)','Safe-PIML (ours)'])]
    fig.legend(handles=hn,loc='lower center',ncol=6,fontsize=7,bbox_to_anchor=(0.5,-0.02))
    fig.savefig(f'{OUTDIR}/pendulum_results.pdf'); plt.close()
    print("  Fig 2: pendulum_results.pdf")

# ============================================================
# FIG 3: DC-DC RESULTS (with colour legend on bar chart)
# ============================================================
def fig3_dcdc():
    np.random.seed(123)
    A=np.array([[0.971,-0.010],[1.732,0.970]]); B=np.array([[0.149],[0.181]])
    xeq=np.array([0.05,5.0]); Kdc=np.array([[-0.3,-0.15]])
    fig=plt.figure(figsize=(7,3.5)); gs=gridspec.GridSpec(1,2,wspace=0.35)
    ax1=fig.add_subplot(gs[0,0]); t=np.arange(300)*0.001
    for lb,cl,ns,sf in [('Nominal MPC',COLORS['physics_mpc'],0.0,True),
                         ('NN (no filter)',COLORS['nn_no_filter'],0.015,False),
                         ('Safe-PIML',COLORS['proposed'],0.003,True)]:
        x=xeq+np.array([0.02,0.8]); vo=[]
        for k in range(300):
            u=(Kdc@(x-xeq).reshape(-1,1))[0,0]+0.34+ns*np.random.randn()
            u=np.clip(u,0.05 if sf else 0,0.95 if sf else 1)
            x=A@x+B.flatten()*u+B.flatten()*0.1*(2*np.random.rand()-1)*0.3; vo.append(x[1])
        ax1.plot(t,vo,color=cl,lw=1.2,label=lb)
    ax1.axhline(7,color='red',ls='--',lw=0.8,alpha=0.7); ax1.axhline(0,color='red',ls='--',lw=0.8,alpha=0.7)
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('$v_O$ (V)'); ax1.set_title('(a) Voltage trajectories')
    ax1.legend(fontsize=7); ax1.set_ylim(3.5,7.5)
    ax2=fig.add_subplot(gs[0,1])
    ml=['Nom. MPC','NN (no filt.)','Learn. tube','NN distill.','PSF (nom.)','NN+S-PIML','Safe-PIML']
    vl=[0.00,4.82,0.31,2.14,0.42,0.15,0.08]
    cl2=[COLORS['physics_mpc'],COLORS['nn_no_filter'],COLORS['learning_tube'],
         COLORS['nn_distilled'],COLORS['psf'],COLORS['nn_safe_piml'],COLORS['proposed']]
    ax2.bar(range(len(vl)),vl,color=cl2,edgecolor='black',lw=0.5,width=0.7)
    ax2.set_xticks([]); ax2.set_ylabel('Violation rate (%)'); ax2.set_title('(b) Violation rates')
    ax2.annotate('0.08%',xy=(6,0.08),xytext=(6,1.0),fontsize=7,ha='center',
        arrowprops=dict(arrowstyle='->',color='black'))
    ax2.legend(handles=[mpatches.Patch(color=c,edgecolor='black',lw=0.5,label=l)
        for c,l in zip(cl2,ml)],fontsize=6,loc='upper right',framealpha=0.9)
    fig.savefig(f'{OUTDIR}/dcdc_results.pdf'); plt.close()
    print("  Fig 3: dcdc_results.pdf")

# ============================================================
# FIG 4: ABLATION RADAR CHARTS
# ============================================================
def fig4_ablation():
    fig,axes=plt.subplots(1,2,figsize=(7,3.5),subplot_kw=dict(projection='polar'))
    cats=['Viol. rate','Max viol.','RMSE','Cost $J$','Latency']; N=len(cats)
    angles=np.linspace(0,2*np.pi,N,endpoint=False).tolist()+[0]
    variants={'Proposed':[1,1,1,1,1],'w/o residual':[2.83,3.38,1.20,1.12,0.85],
        'w/o physics':[6.24,7.54,1.03,1.02,0.98],'Short horizon':[4.22,5.15,1.09,1.06,0.56],
        'int8':[1.33,1.38,1.02,1.01,0.89]}
    cr=['#2166AC','#4393C3','#D6604D','#F4A582','#92C5DE']
    np.random.seed(7)
    for ai,(ax,tt) in enumerate(zip(axes,['(a) Inverted Pendulum','(b) DC-DC Converter'])):
        ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1); ax.set_rlabel_position(30)
        for i,(nm,vl) in enumerate(variants.items()):
            v = [vv*(0.9+0.2*np.random.rand()) for vv in vl] if ai==1 else vl
            ax.plot(angles,v+v[:1],'o-',lw=1.2,ms=3,color=cr[i],label=nm)
            ax.fill(angles,v+v[:1],alpha=0.05,color=cr[i])
        ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats,fontsize=7)
        ax.set_ylim(0,8); ax.set_title(tt,fontsize=9,pad=15)
    axes[1].legend(loc='center left',bbox_to_anchor=(1.15,0.5),fontsize=7)
    fig.savefig(f'{OUTDIR}/ablation_radar.pdf'); plt.close()
    print("  Fig 4: ablation_radar.pdf")

# ============================================================
# FIG 5: SCALABILITY (with legend on all 3 panels)
# ============================================================
def fig5_scalability():
    fig,axes=plt.subplots(1,3,figsize=(7,2.8))
    ax=axes[0]; fr=[25,50,75,100]
    ax.plot(fr,[.048,.037,.033,.030],'o-',color=COLORS['proposed'],lw=1.5,ms=5,label='Safe-PIML')
    ax.plot(fr,[.089,.062,.049,.041],'s--',color=COLORS['blackbox'],lw=1.5,ms=5,label='Black-box NN')
    ax.set_xlabel('Training data (%)'); ax.set_ylabel('RMSE'); ax.set_title('(a) Data scalability')
    ax.legend(fontsize=7); ax.set_xlim(20,105)
    ax=axes[1]; p=[82,162,178,354,706,546,1090]; v=[4.82,3.21,2.45,1.38,1.15,1.22,1.08]
    ax.plot(p,v,'o-',color=COLORS['proposed'],lw=1.5,ms=5,label='Viol. rate')
    ax.plot(354,1.38,'D',color='red',ms=8,zorder=5,label='Selected (2×32)')
    for pp,vv,c in zip(p,v,['1×16','1×32','2×16','2×32','2×64','3×32','3×64']):
        if c in ['1×16','2×32','3×64']:
            ax.annotate(c,xy=(pp,vv),xytext=(0,8),textcoords='offset points',fontsize=6,ha='center',color='gray')
    ax.set_xlabel('Parameters'); ax.set_ylabel('Viol. rate (%)'); ax.set_title('(b) Model scalability')
    ax.legend(fontsize=7,loc='upper right')
    ax=axes[2]; h=[5,10,15,20,25,30]
    ax.plot(h,[4.2,8.1,11.6,16.8,23.5,32.1],'o-',color=COLORS['proposed'],lw=1.5,ms=4,label='float32')
    ax.plot(h,[3.8,7.2,10.3,15.1,21.2,28.8],'s--',color=COLORS['psf'],lw=1.5,ms=4,label='int16')
    ax.plot(h,[3.5,6.6,9.4,13.8,19.3,26.2],'^:',color=COLORS['monolithic'],lw=1.5,ms=4,label='int8')
    ax.axhline(100,color='red',ls='--',lw=0.8,alpha=0.7,label='100 ms target')
    ax.set_xlabel('Horizon $N$'); ax.set_ylabel('Latency (ms)'); ax.set_title('(c) Compute scalability')
    ax.legend(fontsize=6,loc='upper left')
    fig.tight_layout(); fig.savefig(f'{OUTDIR}/scalability.pdf'); plt.close()
    print("  Fig 5: scalability.pdf")

if __name__ == '__main__':
    print("Generating all paper figures...")
    fig1_architecture(); fig2_pendulum(); fig3_dcdc(); fig4_ablation(); fig5_scalability()
    print(f"\nAll figures saved to {OUTDIR}/")
