import os
import glob
import matplotlib
import obspy
from obspy import read


def _xcorr_shift(d, s):
    cc = np.correlate(d, s, mode="full")
    time_shift = cc.argmax() - len(d) + 1
    return time_shift


def cc_error(d1, d2, deltat, cc_shift, cc_dlna, sigma_dt_min, sigma_dlna_min):
    """
    Estimate error for dt and dlna with uncorrelation assumption
    """
    nlen_t = len(d1)

    d2_cc_dt = np.zeros(nlen_t)
    d2_cc_dtdlna = np.zeros(nlen_t)

    for index in range(0, nlen_t):
        index_shift = index - cc_shift

        if 0 <= index_shift < nlen_t:
            # corrected by c.c. shift
            d2_cc_dt[index] = d2[index_shift]

            # corrected by c.c. shift and amplitude
            d2_cc_dtdlna[index] = np.exp(cc_dlna) * d2[index_shift]

    # time derivative of d2_cc (velocity)
    d2_cc_vel = np.gradient(d2_cc_dtdlna, deltat)

    # the estimated error for dt and dlna with uncorrelation assumption
    sigma_dt_top = np.sum((d1 - d2_cc_dtdlna)**2)
    sigma_dt_bot = np.sum(d2_cc_vel**2)

    sigma_dlna_top = sigma_dt_top
    sigma_dlna_bot = np.sum(d2_cc_dt**2)

    sigma_dt = np.sqrt(sigma_dt_top / sigma_dt_bot)
    sigma_dlna = np.sqrt(sigma_dlna_top / sigma_dlna_bot)

    if sigma_dt < sigma_dt_min:
        sigma_dt = sigma_dt_min

    if sigma_dlna < sigma_dlna_min:
        sigma_dlna = sigma_dlna_min

    return sigma_dt, sigma_dlna


def subsample_xcorr_shift(d, s):
    """
    Calculate the correlation time shift around the maximum amplitude of the
    synthetic trace with subsample accuracy.
    :param s:
    :param d:
    """
    # Estimate shift and use it as a guideline for the subsample accuracy
    # shift.
    time_shift = _xcorr_shift(d.data, s.data) * d.stats.delta

    # Align on the maximum amplitude of the synthetics.
    pick_time = s.stats.starttime + s.data.argmax() * s.stats.delta

    # Will raise a warning if the trace ids don't match which we don't care
    # about here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return xcorr_pick_correction(
            pick_time, s, pick_time, d, 20.0 * time_shift,
            20.0 * time_shift, 10.0 * time_shift)[0]


def calculate_adjoint_source(observed, synthetic, config, window,
                             adjoint_src, figure):  # NOQA

    ret_val_p = {}
    ret_val_q = {}

    nlen_data = len(synthetic.data)
    deltat = synthetic.stats.delta

    fp = np.zeros(nlen_data)
    fq = np.zeros(nlen_data)

    misfit_sum_p = 0.0
    misfit_sum_q = 0.0

    # ===
    # loop over time windows
    # ===
    for wins in window:
        left_window_border = wins[0]
        right_window_border = wins[1]

        left_sample = int(np.floor(left_window_border / deltat)) + 1
        nlen = int(np.floor((right_window_border -
                             left_window_border) / deltat)) + 1
        right_sample = left_sample + nlen

        d = np.zeros(nlen)
        s = np.zeros(nlen)

        d[0:nlen] = observed.data[left_sample:right_sample]
        s[0:nlen] = synthetic.data[left_sample:right_sample]

        # All adjoint sources will need some kind of windowing taper
        d = window_taper(d, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)
        s = window_taper(s, taper_percentage=config.taper_percentage,
                     taper_type=config.taper_type)

        i_shift = _xcorr_shift(d, s)
        t_shift = i_shift * deltat

        cc_dlna = 0.5 * np.log(sum(d[0:nlen]*d[0:nlen]) /
                               sum(s[0:nlen]*s[0:nlen]))

        sigma_dt, sigma_dlna = cc_error(d, s, deltat, i_shift, cc_dlna,
                                        config.dt_sigma_min,
                                        config.dlna_sigma_min)

        misfit_sum_p += 0.5 * (t_shift/sigma_dt) ** 2
        misfit_sum_q += 0.5 * (cc_dlna/sigma_dlna) ** 2

        dsdt = np.gradient(s, deltat)
        nnorm = simps(y=dsdt*dsdt, dx=deltat)
        fp[left_sample:right_sample] = dsdt[:] * t_shift / nnorm / sigma_dt**2

        mnorm = simps(y=s*s, dx=deltat)
        fq[left_sample:right_sample] =\
            -1.0 * s[:] * cc_dlna / mnorm / sigma_dlna ** 2

    ret_val_p["misfit"] = misfit_sum_p
    ret_val_q["misfit"] = misfit_sum_q

    if adjoint_src is True:
        ret_val_p["adjoint_source"] = fp[::-1]
        ret_val_q["adjoint_source"] = fq[::-1]

    if config.measure_type == "dt":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_p["adjoint_source"],
                                        ret_val_p["misfit"],
                                        window, VERBOSE_NAME)

        return fp,t_shift
        #ret_val_p

    if config.measure_type == "am":
        if figure:
            generic_adjoint_source_plot(observed, synthetic,
                                        ret_val_q["adjoint_source"],
                                        ret_val_q["misfit"],
                                        window, VERBOSE_NAME)

        return ret_val_q

def Cal_adj_src(path,taper):
    os.chdir(path)
    egfs = glob.glob("*obs*")
    for egf in egfs:
        sgf = egf.replace('obs','syn')
        tr = read(egf)[0]
        window = []
        window.append([float(tr.stats.sac.t1),float(tr.stats.sac.t2)])
        print(window)
        [adj,misfit] = calculate_adjoint_source(d, s, config, window, adjoint_src, figure)


def Pick(path):
    os.chdir(path)
    print(path)
    file_data = glob.glob("*sac")
    Dist = {} ; dist = []
    for i in range(len(file_data)):
        st = read(file_data[i])
        tr = st[0]
        dist.append(tr.stats.sac.dist)
        Dist[str(dist[i])] = file_data[i]
    
    print(Dist[str(max(dist))],Dist[str(min(dist))])
    file1 = Dist[str(max(dist))] ; file2 = Dist[str(min(dist))]
    tr1 = read(file1) ; tr2 = read(file2)
    t11 = tr1[0].stats.sac.t1 ; t12 = tr1[0].stats.sac.t2
    t21 = tr2[0].stats.sac.t1 ; t22 = tr2[0].stats.sac.t2
    k1 = (t11 -t21)/(max(dist)-min(dist)) ; k2 = (t12 - t22)/(max(dist)-min(dist))
    
    for i in range(len(file_data)):
        st = read(file_data[i])
        tr = st[0]
        dk = tr.stats.sac.dist - min(dist)
        t1 = dk * k1 + t21
        t2 = dk * k2 + t22
        tr.stats.sac.t1 = t1
        tr.stats.sac.t2 = t2
        tr.write(path + '/' + file_data[i])

def select_window(config):
    archive_path = config["archive_path"]
    obs_syn = archive_path + '/obs_syn/'
    freqband = config["bandpass"].split('/')

    class item:
         def __init__(self):
             self.taper_percentage = 0.4
             self.taper_type = 'cos_p10'
             self.measure_type = 'dt'
             self.dt_sigma_min = 0.0001
             self.dlna_sigma_min = 0
    taper = item()
    print(taper.taper_percentage)
    for freq in freqband:
        freq_dir = obs_syn + 'BP_' + freq
        Pick(freq_dir)
        Cal_adj_src(freq_dir,taper)
