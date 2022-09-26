import glob
import os
from matplotlib.pyplot import flag
import numpy as np
from obspy import read, Stream
from utils import read_event_file, read_sta_file, mk_dir
from scipy import interpolate
from obspy.core import Trace
from obspy.io.sac import SACTrace

def _stack(tr,npts,dt):
    ## stack ncfs
    half_n = int((npts-1)/2)
    tr_on = tr[0:half_n+1]
    tr_off = tr[half_n:npts]
    tr_on_reverse = tr_on[::-1]
    tr_out = tr_on_reverse + tr_off
    tr = -np.diff(tr_out)/dt
    return tr

def get_rms(records):
    ## cal rms
    return np.sqrt(sum([x**2 for x in records])/len(records))
 
def _snr(data):
    ## cal snr
    max_amplitude = max(abs(data))
    noise = get_rms(data)
    snr = max_amplitude/noise
    return snr

def write_sac(data,sour_id,staid,f1,f2,dist,dt,out_path,flag):
    sacfile = Trace()
    sacfile.data = data
    sacfile.stats.delta = dt
    file_filted = sacfile.filter('bandpass',freqmin=float(f1),freqmax=float(f2),corners=4, zerophase=True)
    file_filted.data = file_filted.data/max(abs(file_filted.data))
    sac = SACTrace.from_obspy_trace(file_filted)
    sac.dist=dist
    sac.delta=dt
    out = out_path + '/VV_' + sour_id + '-' + staid + '_' + str(f1) + '-' + str(f2) + '-' + flag + '.sac'
    sac.write(out)
    #print(out)

def get_obs_waveform(sac_file):
    st = read(sac_file)
    tr = st[0]
    obs = tr.data
    dist = tr.stats.sac.dist
    dt = tr.stats.delta
    npts = tr.stats.npts
    obs_data = _stack(obs,npts,dt)
    nstep = (npts - 1)/2

    return obs_data,dist,dt,nstep

def select_write_data(obs_path,eventid,staid,config):
    half_duration = int(config["forward_info"]["half_duration"])
    dt = float(config["forward_info"]["dt"])
    nstep = int(config["forward_info"]["nstep"])
    snr_min = float(config["select_info"]["snr_min"])

    os.chdir(obs_path)
    #print(eventid,staid)
    sac_file = glob.glob("VV_%s-%s*" % (eventid,staid))
    #print(sac_file)
    if len(sac_file) == 0:
        return 0,0,False

    ## get and select obs waveform
    [obs_data,dist,Dt,npts] = get_obs_waveform(sac_file[0])

    ## cal snr of obs waveform 
    snr = _snr(obs_data)
    if snr < snr_min:
        return 0,0,False

    ## interpolate obs data
    t1 = np.arange(0,Dt*npts,Dt)
    t2 = np.arange(0,dt*(nstep-half_duration),dt)
    #print(npts,nstep)

    f = interpolate.interp1d(t1,obs_data,kind='linear')
    obs_interp = f(t2)
    #print(len(obs_interp))

    ## add zeros (half duration of guass source)
    zeros = np.zeros(half_duration)
    obs_interp = np.concatenate((zeros,obs_interp))
    return obs_interp,dist,True


def get_syn_waveform(syn_path):
    os.chdir(syn_path)
    su_files = glob.glob("*dz_SU")
    syn = []
    for su in su_files:
        st = read(su)
        for tr in st:
            syn.append(tr.data)
    return syn


def bpfilter(config):
    ## get the info needed from config
    eventlst = config["eventlst"]
    archive_path = config["archive_path"]
    obs_path = config["obs_path"]
    freqband = config["bandpass"].split('/')
    dt = config["forward_info"]["dt"]
    vmean = float(config["select_info"]["vmean"])

    eventid = read_event_file(eventlst)
    print(eventid)
    for ie in range(len(eventid)):
        syn_path = archive_path + '/J' + eventid[ie] + '/OUTPUT_FILES/'
        sta_path = archive_path + '/J' + eventid[0] + '/DATA/STATIONS'

        syn = get_syn_waveform(syn_path)
        syn_sta = read_sta_file(sta_path)

        if int(eventid[ie]) < 10:
            eventidnum = '0' + str(int(eventid[ie]))
        elif int(eventid[ie]) < 100:
            eventidnum = str(int(eventid[ie]))
        else:
             eventidnum = eventid[ie]

        for ista in range(len(syn_sta)):
            staid = syn_sta[ista]
            syn_data = syn[ista]
            [obs_data,dist,flag] = select_write_data(obs_path,eventidnum,staid,config)
            if flag == False:
                continue
            obs_syn = archive_path + '/obs_syn/'
            #obs_out = obs_syn + 'obs'
            #syn_out = obs_syn + 'syn'
            mk_dir(obs_syn)
            #mk_dir(syn_out)
            #mk_dir(obs_out)

            #write_sac(obs_data,eventidnum,staid,dist,dt,obs_out)
            #write_sac(syn_data,eventidnum,staid,dist,dt,syn_out)
            for fb in freqband:
                f1 = fb.split('-')[0]
                f2 = fb.split('-')[1]
                Syn_out = obs_syn + '/BP_' + f1 + '-' + f2
                Obs_out = obs_syn + '/BP_' + f1 + '-' + f2
                mk_dir(Syn_out)
                mk_dir(Obs_out)
                if dist > 2*(vmean/float(f1)):
                    write_sac(syn_data,eventidnum,staid,f1,f2,dist,dt,Syn_out,'syn')
                    write_sac(obs_data,eventidnum,staid,f1,f2,dist,dt,Obs_out,'obs')


            