from scipy.stats import pearsonr
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
def sortAfterFrame(data):
    fish = [None]*len(data)
    for i in data:
        fish[i[0,1]] = []
        for j in i:
            v = list(j[-2])
            v.append(j[-1])
            fish[i[0,1]].append(v)
    fish = np.array(fish)
    return fish

def plotParams(fishlist,Op=None,Or=None,filename=None):
    nbr_params = len(fishlist[0].transpose())
    if Op is not None:
        nbr_params+=1
    x = np.arange(len(fishlist[0]))

    fig, axs = plt.subplots(nbr_params, 1, tight_layout=True,figsize=(20,15))
    params_mean = np.mean(fishlist,axis=0)
    quantile_5    = np.quantile(fishlist,0.05,axis=0)
    quantile_95   = np.quantile(fishlist,0.95,axis=0)

    """
    for fishid,params in enumerate(fishlist):
        y = params.transpose()
        for i in range(nbr_params):
            axs[i].plot(x,y[i])
    """
    if nbr_params-1==5:
        labels = ["Gamma","omega1","omgea2","omega3","l2 Error"]
        desc   = ["Strecke"," Entfernen vom nächste Fisch "," Polarisierung "," Drang zum Mittelpunkt","Mittelwert Fehler",""]
    else:
        labels = ["R","O","A","V","Alpha","l2 Error"]
        desc   = ["Repulsion Zone"," Orientation Zone "," Attraction Zone "," Strecke "," Gewichtung O/A","Mittelwert Fehler"]
    params_mean = np.nan_to_num(params_mean)
    for frame in range(params_mean.shape[-1]):
        corr_pearson_Or = np.cov(Or[:len(x)],params_mean[:,frame])/(np.std(params_mean[:,frame])*np.std(Or[:len(x)]))
        corr_pearson_Op = np.cov(Op[:len(x)],params_mean[:,frame])/(np.std(params_mean[:,frame])*np.std(Op[:len(x)]))
        axs[frame].plot(x,np.nan_to_num(params_mean[:,frame]),label=desc[frame])
        #axs[frame].fill_between(x, quantile_5[:,frame], quantile_95[:,frame], alpha=0.2)
        axs[frame].set_ylabel(labels[frame])
        axs[frame].set_xlabel("Frame")
        axs[frame].legend()
        corr_Op,_ = pearsonr(params_mean[:,frame],Op[:len(x)])
        corr_Or,_ = pearsonr(params_mean[:,frame],Or[:len(x)])
        axs[frame].set_title("Pearson Corr Op/Or : {:.2f}/{:.2f}".format(corr_Op,corr_Or))
        """
        im = axs[frame,1].matshow(corr_pearson_Op)
        plt.colorbar(im, ax=axs[frame, 1])
        im = axs[frame,2].matshow(corr_pearson_Or)
        plt.colorbar(im, ax=axs[frame, 2])
        """
        

        upper,lower = np.nan_to_num(params_mean[:,frame]).mean()+np.quantile(np.nan_to_num(params_mean[:,frame]),0.95), np.nan_to_num(params_mean[:,frame]).mean()-np.quantile(np.nan_to_num(params_mean[:,frame]),0.05)

        lower = 0
        axs[frame].set_ylim([lower,upper])
        axs[frame].plot(Or[:len(x)],color="red",label="Rotation Or", alpha=0.1)
        axs[frame].plot(Op[:len(x)],color="blue",label="Polarization Op", alpha=0.1)
    axs[-1].plot(Or[:len(x)],color="red",label="Rotation Or")
    axs[-1].plot(Op[:len(x)],color="blue",label="Polarization Op")
    plt.legend()
    if filename:
        print("Save",filename)
        fig.savefig(filename+'.png', dpi=fig.dpi)

def plotMultiFish(params,Op=None,Or=None,filename=None):
    nbr_params = len(params[0])
    params = np.nan_to_num(params)
    x = np.arange(len(params))
    print(params)
    if nbr_params==5:
        labels = ["Gamma","omega1","omgea2","omega3","l2 Error"]
        desc   = ["Strecke"," Entfernen vom nächste Fisch "," Polarisierung "," Drang zum Mittelpunkt","Mittelwert Fehler",""]
    else:
        labels = ["R","O","A","V","Alpha","l2 Error"]
        desc   = ["Repulsion Zone"," Orientation Zone "," Attraction Zone "," Strecke "," Gewichtung O/A","Mittelwert Fehler"]
    if Op is not None:
        nbr_params+=1

    fig, axs = plt.subplots(nbr_params, 1, tight_layout=True,figsize=(20,15))
    for frame in range(nbr_params):
        axs[frame].plot(x,np.nan_to_num(params[:,frame]),label=desc[frame])
        axs[frame].set_ylabel(labels[frame])
        axs[frame].set_xlabel("Frame")
        axs[frame].legend()
        if Op is not None:
            corr_Op,_ = pearsonr(params[:,frame],Op[:len(x)])
            corr_Or,_ = pearsonr(params[:,frame],Or[:len(x)])
            axs[frame].set_title("Pearson Corr Op/Or : {:.2f}/{:.2f}".format(corr_Op,corr_Or))
            axs[frame].plot(Or[:len(x)],color="red",label="Rotation Or", alpha=0.1)
            axs[frame].plot(Op[:len(x)],color="blue",label="Polarization Op", alpha=0.1)
    if Op is not None:
        axs[-1].plot(Or[:len(x)],color="red",label="Rotation Or")
        axs[-1].plot(Op[:len(x)],color="blue",label="Polarization Op")
    plt.legend()
    if filename:
        print("Save",filename)
        fig.savefig(filename+'.png', dpi=fig.dpi)