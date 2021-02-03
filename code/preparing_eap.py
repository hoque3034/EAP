# All nrcessary function for SAP.

import keras
import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import ttest_ind
from statsmodels.stats import weightstats as stests
from keras import backend as K
from sklearn.metrics import mean_absolute_error
from keras.models import model_from_json
from keras import losses
from sklearn import preprocessing

LenHotvect = 8
SinMaskValue = -2
AngleMaskValue = 360

PhyscDict = {'A': [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
             'C': [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
             'D': [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
             'E': [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
             'F': [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
             'G': [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
             'H': [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
             'I': [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
             'K': [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
             'L': [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
             'M': [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
             'N': [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
             'P': [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
             'Q': [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
             'R': [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
             'S': [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
             'T': [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
             'V': [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
             'W': [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
             'Y': [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476]}

HotvectDict = {'B': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'C': [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
               'E': [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'G': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
               'H': [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], 'I': [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
               'S': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], 'T': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]}


# Functions

def ExtractMinMeanMaxStd(mean_std_file):
    with open(mean_std_file) as p:
        lines = p.readlines()
        p.close()
        min_mean, max_std = [], []
        for line in lines:
            line = line.strip()
            split_line = line.split()

            min_mean.append(float(split_line[0]))
            max_std.append(float(split_line[1]))
        min_mean = np.array(min_mean)
        max_std = np.array(max_std)
        min_mean = min_mean.reshape(1, min_mean.shape[0])
        max_std = max_std.reshape(1, max_std.shape[0])
    return min_mean, max_std


def ExtractProteinName(directory):
    with open(directory) as f:
        lines = f.readlines()
        f.close()

        proteins = []
        for line in lines:
            line = line.strip()
            split_line = line.split()
            proteins.append((split_line[0]))
    return proteins


def SAPPreprocessing(DataFolder, Proteins, LenFeatures, LenInputNN, LenOut, WinStep, WinSize, NormVersion, Min_Mean,
                     MaxStd, SinFlag, Reuse):
    FinalInput = np.zeros((0, LenInputNN))
    FinalOutput = np.zeros((0, LenOut))

    for Protein in Proteins:
        ProteinFile = DataFolder + "proteins/" + Protein + "/"
        PSSMFile = ProteinFile + Protein + ".pssm"
        DSSPFile = ProteinFile + Protein + ".dssp"
        TFile = ProteinFile + Protein + ".t"
        SS8PredFile = ProteinFile + Protein + ".out.ss8"

        theta, tau = extract_theta_tau(TFile)
        phi, psi = extract_phi_psi(DSSPFile)
        aa, ss, asa = extract_aa_ss_asa(DSSPFile)
        ss_predicted = extract_ss_predicted(SS8PredFile)
        pssm_mtx = extract_pssm_matrix(PSSMFile)
        ss_8hotvect_mtx = extract_ss_hotvect_mtx(ss_predicted)
        phys_mtx = extract_physc(aa, PhyscDict)
        feature_mtx_each_protein = np.concatenate((pssm_mtx, phys_mtx, ss_8hotvect_mtx), axis=1)

        if (SinFlag == "Sin"):
            phi_sin = sin_cos(phi)
            psi_sin = sin_cos(psi)
            theta_sin = sin_cos(theta)
            tau_sin = sin_cos(tau)
            out_angle = np.concatenate((phi_sin, psi_sin, theta_sin, tau_sin), axis=1)
        else:
            out_angle = np.concatenate((phi, psi, theta, tau), axis=1)

        feature_capture_win_each = window_sliding(Protein, WinStep, WinSize, feature_mtx_each_protein, LenFeatures)
        FinalInput = np.concatenate((FinalInput, feature_capture_win_each), axis=0)
        FinalOutput = np.concatenate((FinalOutput, out_angle), axis=0)

    FinalInput, Min_Mean, MaxStd = Normalization(FinalInput, NormVersion, Reuse, Min_Mean, MaxStd)
    return FinalInput, FinalOutput


def extract_theta_tau(t_file):
    with open(t_file) as p:
        lines = p.readlines()
        theta = []
        tau = []
        for line in lines:
            line = line.strip()
            split_line = line.split()
            if (len(split_line) != 0 and len(split_line[0]) != 0):
                if (split_line[0][0] != '#'):
                    theta.append(float(split_line[2]))
                    tau.append(float(split_line[3]))
        p.close()
    theta = np.array(theta)
    tau = np.array(tau)
    theta = theta.reshape(theta.shape[0], 1)
    tau = tau.reshape(tau.shape[0], 1)
    return theta, tau


def extract_phi_psi(dssp_file):
    with open(dssp_file) as p:
        lines_dssp = p.readlines()
        p.close()
    np_phi = np.array([float(i) for i in lines_dssp[3].strip().split(' ')])
    np_psi = np.array([float(i) for i in lines_dssp[4].strip().split(' ')])
    np_phi = np_phi.reshape(np_phi.shape[0], 1)
    np_psi = np_psi.reshape(np_psi.shape[0], 1)
    return np_phi, np_psi


def extract_aa_ss_asa(dssp_file):
    with open(dssp_file) as p:
        lines_dssp = p.readlines()
        p.close()
    aa_seq = lines_dssp[1]
    aa_seq = aa_seq.strip()
    Secondry_seq = lines_dssp[2]
    Secondry_seq = Secondry_seq.strip()
    np_asa = np.array([float(i) for i in lines_dssp[5].strip().split(' ')])
    np_asa = np_asa.reshape(np_asa.shape[0], 1)
    return aa_seq, Secondry_seq, np_asa


def extract_ss_predicted(fasta_file):
    with open(fasta_file) as p:
        lines_dssp = p.readlines()
        p.close()
    aa_seq = lines_dssp[1]
    ss_seq = aa_seq.strip()
    return ss_seq


def extract_pssm_matrix(pssm_file):
    pssm_row = np.zeros((0, 20))
    pssm_mtx = np.zeros((0, 20))
    with open(pssm_file) as p:
        lines = p.readlines()
        p.close()
    for line in lines:
        line = line.strip()
        split_line = line.split()
        if (len(split_line) == 44) and (split_line[0] != '#'):
            pssm_row = [-float(i) for i in split_line[2:22]]
            pssm_row = np.array(pssm_row).reshape(1, 20)
            pssm_mtx = np.concatenate((pssm_mtx, pssm_row), axis=0)
    return pssm_mtx


def extract_ss_hotvect_mtx(Secondry_seq):
    ss_8_row = np.zeros((0, LenHotvect))
    ss_8_mtrx = np.zeros((0, LenHotvect))
    len_prot = len(Secondry_seq)
    for i in range(0, len_prot):
        ss_8_row = np.array(HotvectDict[Secondry_seq[i]]).reshape(1, LenHotvect)
        ss_8_mtrx = np.concatenate((ss_8_mtrx, ss_8_row), axis=0)
    return ss_8_mtrx


def extract_physc(aa_sequence, phys_dic):
    phys = np.zeros((0, 7))
    phys_mtx = np.zeros((0, 7))
    len_protein = len(aa_sequence)
    for i in range(0, len_protein):
        phys = phys_dic[aa_sequence[i]]
        phys = np.array(phys).reshape(1, 7)
        phys_mtx = np.concatenate((phys_mtx, phys), axis=0)
    return phys_mtx


def sin_cos(angle_mtx):
    angle_sin = np.zeros(angle_mtx.shape[0])
    angle_cos = np.zeros(angle_mtx.shape[0])
    for i in range(0, angle_mtx.shape[0]):
        if (angle_mtx[i] == 360):
            angle_sin[i] = SinMaskValue
            angle_cos[i] = SinMaskValue
        else:
            angle_mtx[i] = angle_mtx[i] * np.pi / 180.
            angle_sin[i] = np.sin(angle_mtx[i])
            angle_cos[i] = np.cos(angle_mtx[i])
    angle_cos = angle_cos.reshape(angle_mtx.shape[0], 1)
    angle_sin = angle_sin.reshape(angle_mtx.shape[0], 1)
    angle_sin_cos = np.concatenate((angle_sin, angle_cos), axis=1)
    return angle_sin_cos


def window_sliding(protein_name, win_step, win_size, input_matrix, len_feature_vect):
    len_input_nn = (2 * win_size + 1) * len_feature_vect
    len_prot = input_matrix.shape[0]
    window_input_mtx = np.zeros([0, len_input_nn])
    window_whole_prot = np.zeros([0, len_input_nn])
    for core in range(0, len_prot, 1):
        left_win = core - win_size
        right_win = core + win_size
        if (right_win >= len_prot):
            right_win = right_win - len_prot
            window_input_mtx = np.concatenate(
                (input_matrix[left_win:, :].ravel(), input_matrix[:right_win + 1, :].ravel()), axis=0)
            window_input_mtx = window_input_mtx.reshape(1, window_input_mtx.shape[0])
        elif (left_win < 0):
            window_input_mtx = np.concatenate(
                (input_matrix[left_win:, :].ravel(), input_matrix[:right_win + 1, :].ravel()), axis=0)
            window_input_mtx = window_input_mtx.reshape(1, window_input_mtx.shape[0])
        else:
            window_input_mtx = input_matrix[left_win:right_win + 1, :]
            window_input_mtx = window_input_mtx.ravel()
            window_input_mtx = window_input_mtx.reshape(1, window_input_mtx.shape[0])
        if (window_whole_prot.shape[1] != window_input_mtx.shape[1]):
            print(window_whole_prot.shape, window_input_mtx.shape)
            print(protein_name)
            print(len_prot, left_win, right_win, core)
        window_whole_prot = np.concatenate((window_whole_prot, window_input_mtx), axis=0)
    return window_whole_prot


def Normalization(input_matrix, norm_version, reuse, min_mean, max_std):
    if (norm_version == "range"):
        if (reuse == 0):
            min_mean = np.min(input_matrix, axis=0).reshape(1, min_mean.shape[0])
            max_std = np.max(input_matrix, axis=0).reshape(1, max_std.shape[0])
        for i in range(0, input_matrix.shape[1]):
            input_matrix[:, i] = (input_matrix[:, i] - float(min_mean[0, i])) / (
            (float(max_std[0, i]) - float(min_mean[0, i])))
    elif (norm_version == "zscore"):
        if (reuse == 0):
            min_mean = np.mean(input_matrix, axis=0).reshape(1, min_mean.shape[0])
            max_std = np.std(input_matrix, axis=0).reshape(1, max_std.shape[0])
        for i in range(0, input_matrix.shape[1]):
            input_matrix[:, i] = (input_matrix[:, i] - float(min_mean[0, i])) / (float(max_std[0, i]))
    return input_matrix, min_mean, max_std


# 1th column sin 2 th colmun cos
def tan_to_angles(input_mtx):
    num_row = input_mtx.shape[0]
    tan_mtx = np.zeros((num_row, 0))
    angles_mtx = np.zeros((num_row, 0))
    tan_mtx = input_mtx[:, 0] / input_mtx[:, 1]
    angles_mtx = np.arctan(tan_mtx)
    angles_mtx = angles_mtx * 180. / np.pi
    return angles_mtx


def CalAEAngles(y_true, y_pred):
    mask = np.zeros(y_true.shape)
    for i in range(0, y_true.shape[0]):
        for j in range(0, y_true.shape[1]):
            if (y_true[i, j] != AngleMaskValue):
                mask[i, j] = 1.0
    y_true = y_true * mask
    y_pred = y_pred * mask
    error = y_pred - y_true
    error = abs(error)
    for i in range(0, error.shape[0]):
        for j in range(0, error.shape[1]):
            if (error[i, j] >= 180):
                error[i, j] = 360 - error[i, j]
    count = y_true.shape[0] - (mask.shape[0] - np.sum(mask))
    AErr = np.sum(error)
    MAEAngles = AErr / count

    return MAEAngles


def CalAESinCos(real_sincos, pred_sincos):
    num_row = real_sincos.shape[0]
    real_angles = np.zeros((num_row, 0))
    pred_angles = np.zeros((num_row, 0))
    mae_value = 0

    real_angles = tan_to_angles(real_sincos)
    pred_angles = tan_to_angles(pred_sincos)
    mask = np.zeros((real_angles.shape[0]))
    for i in range(0, real_sincos.shape[0]):
        if real_sincos[i, 0] != SinMaskValue:
            mask[i] = 1.0
    real_angles = real_angles * mask
    pred_angles = pred_angles * mask
    error = pred_angles - real_angles
    error = abs(error)

    for i in range(0, error.shape[0]):
        if (error[i] >= 180):
            error[i] = 360 - error[i]
    count = np.sum(mask)
    AErr = np.sum(error)
    MAEAngles = AErr / count
    return MAEAngles
































































































































































