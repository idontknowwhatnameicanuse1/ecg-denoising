import numpy as np
# from ecg_qc import EcgQc

def SNR(input,ground_truth):
    noise=input-ground_truth
    snr=10*np.log10((np.sum(ground_truth**2))/(np.sum(noise**2)))
    return snr

def RMSE(input,ground_truth):
    rmse=np.sqrt(np.mean((ground_truth-input)**2))
    return rmse

def ecg_quanlity_verify(input_data):
    ecg_qc = EcgQc('rfc_norm_2s.pkl',  # 原来是model=这个，运行之后发现不需要写model=
                   sampling_frequency=360,
                   normalized=True)
    signal_quality = ecg_qc.get_signal_quality(input_data)
    return signal_quality
