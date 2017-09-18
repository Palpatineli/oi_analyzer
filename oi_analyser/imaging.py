#!/usr/bin/env python3
"""analyse in Fourier transformation based continuous optical imaging"""
import colorsys
import math
from typing import Tuple

import h5py as h5
import numpy as np
from tqdm import tqdm

FRAMES_PER_PULSE = 6  # default sync signal
__author__ = "Keji Li"

Rect = Tuple[int, int, int, int]


def circular_transform(file_path: str) -> (np.ndarray, np.ndarray, np.ndarray):
    """calculate the dft at period_in_seconds from a continuous optical imaging file
    Args:
        file_path: the hdf5 file path
    Returns:
        |  dft,   phase_series, average_image|
        |  2D,        1D,            2D      |
        |complex,    float,         float    |
    """
    h5file = h5.File(file_path, 'r')
    cam_frames = h5file['frame_data']
    cam_frame_phases = extract_frame_phases(h5file)
    cam_frame_coeff = np.exp(-1j * cam_frame_phases)

    cam_frame_shape = cam_frames[0].shape
    result = np.zeros(cam_frame_shape, dtype=np.complex)
    ave = np.zeros(cam_frame_shape, dtype=np.float)

    # get rid of first and last cycle
    trial_ends = next(x for x in np.nonzero(np.diff(cam_frame_phases) < 0))
    cam_frame_range = range(trial_ends[0] + 1, trial_ends[-1] + 1)

    for frame_idx in tqdm(cam_frame_range):
        frame = cam_frames[frame_idx]
        result += cam_frame_coeff[frame_idx] * frame
        ave += frame
    ave /= len(cam_frame_range)
    result -= ave * cam_frame_coeff[cam_frame_range].sum()
    h5file.close()
    return result, cam_frame_phases, ave


def detrend(file_path: str, rect: Rect):
    """calculate the dft at period_in_seconds from a continuous optical imaging file, assuming
    pixels within rect stay stable, and use their average to normalize each frame
    Args:
        file_path: the hdf5 file path
        rect: [x0, y0, x1, y1] inclusive top and left, exclusive bottom and right
    Returns:
        (dft complex matrix, phase_series, averaged_image, trend)
    """
    h5file = h5.File(file_path, 'r')
    cam_frame_phases = extract_frame_phases(h5file)

    cam_frames = h5file['frame_data']
    frame_mean = np.zeros(len(cam_frame_phases))
    for idx in tqdm(range(len(cam_frame_phases))):
        frame_mean[idx] = np.mean(cam_frames[idx][rect[0]:rect[2], rect[1]:rect[3]])

    def linregress(x, y):
        coef_matrix = np.vstack([x, np.ones(len(y))]).T
        return np.linalg.lstsq(coef_matrix, y)

    slope, _ = linregress(np.arange(len(frame_mean)), frame_mean)[0]

    trend = np.arange(len(frame_mean)) * slope
    cam_frame_coeff = np.exp(-1j * cam_frame_phases)

    trial_ends = next(x for x in np.nonzero(np.diff(cam_frame_phases) < 0))
    cam_frame_range = range(trial_ends[0] + 1, trial_ends[-1] + 1)

    cam_frame_shape = cam_frames[0].shape
    result = np.zeros(cam_frame_shape, dtype=np.complex)
    ave = np.zeros(cam_frame_shape, dtype=np.float)

    for frame_idx in tqdm(cam_frame_range):
        frame = cam_frames[frame_idx] - trend[frame_idx]
        result += cam_frame_coeff[frame_idx] * frame
        ave += frame
    ave /= len(cam_frame_range)
    result -= ave * cam_frame_coeff[cam_frame_range].sum()
    h5file.close()
    return result, cam_frame_phases, ave, frame_mean


def normalized(file_path: str, rect: Rect):
    h5file = h5.File(file_path, 'r')
    cam_frame_phases = extract_frame_phases(h5file)

    cam_frames = h5file['frame_data']
    inverse_frame_mean = np.zeros(len(cam_frame_phases))
    for idx in tqdm(range(len(cam_frame_phases))):
        inverse_frame_mean[idx] = 1 / np.mean(cam_frames[idx][rect[0]:rect[2], rect[1]:rect[3]])

    cam_frame_coeff = np.exp(-1j * cam_frame_phases)

    # noinspection PyUnresolvedReferences
    trial_ends = np.nonzero(np.diff(cam_frame_phases) < 0)[0]
    cam_frame_range = range(trial_ends[0] + 1, trial_ends[-1] + 1)

    cam_frame_shape = cam_frames[0].shape
    result = np.zeros(cam_frame_shape, dtype=np.complex)
    ave = np.zeros(cam_frame_shape, dtype=np.float)

    for frame_idx in tqdm(cam_frame_range):
        frame = cam_frames[frame_idx]
        result += (cam_frame_coeff[frame_idx] * inverse_frame_mean[frame_idx]) * frame
        ave += inverse_frame_mean[frame_idx] * frame
    ave /= len(cam_frame_range)
    result -= ave * cam_frame_coeff[cam_frame_range].sum()
    h5file.close()
    return result, cam_frame_phases, ave, inverse_frame_mean


def extract_frame_phases(h5file: h5.File) -> np.ndarray:
    """calculate the phase corresponding to each frame, from diode signals
    Args:
        h5file: the hdf5 file
    Returns:
        1-d frame phase (-π to π) time series
    """
    diode_stamps = clean_diode(np.array(h5file['diode_nidaq_time']),
                               np.array(h5file['diode_signal']))[0]
    # add one more diode signal at the end
    diode_stamps = np.append(diode_stamps, [2 * diode_stamps[-1] - diode_stamps[-2]])
    period_in_stim_frames = h5file.attrs['period_in_frames'][0]
    period_in_pulses = int(period_in_stim_frames / h5file.attrs.get('frame_per_pulse', FRAMES_PER_PULSE))
    phases = np.linspace(0, 2 * np.pi, period_in_pulses, endpoint=False)
    image_stamps = np.array(h5file['frame_timestamps'])
    # get rid of frame after stimulus was done, also there might be a fake pulse at 0
    image_stamps = image_stamps[1 if image_stamps[0] == 0
    else 0: np.searchsorted(image_stamps, diode_stamps[-1])]
    post_indices = np.searchsorted(diode_stamps, image_stamps)
    pre_indices = post_indices - 1
    post_phases = phases[post_indices % period_in_pulses]
    post_phases[post_phases == 0] = np.pi * 2
    cam_frame_phases = (phases[pre_indices % period_in_pulses] *
                        (diode_stamps[post_indices] - image_stamps) +
                        post_phases * (image_stamps - diode_stamps[pre_indices])) / \
                       (np.diff(diode_stamps))[pre_indices]
    return cam_frame_phases


def roll_over(h5file: str, throw_out_ends: bool = True) -> np.ndarray:
    """calculate the dft at period_in_seconds from a continuous optical imaging file
    Args:
        h5file: the file path
        throw_out_ends: whether throw out the first and last trials
    Returns:
        2d numpy array of complex reflecting the dft of image
    """
    h5file = h5.File(h5file, 'r')
    cam_frame_per_period = int(h5file.attrs['period_in_seconds'][0] *
                               1000 / h5file.attrs['exposure_time'][0])
    print(cam_frame_per_period)
    frames = h5file['frame_data']
    shape = frames[0].shape
    rolled = np.zeros((cam_frame_per_period, shape[0], shape[1]))
    print(frames.size, frames.size // cam_frame_per_period - 1)
    if throw_out_ends:
        trial_range = range(1, frames.size // cam_frame_per_period - 1)
    else:
        trial_range = range(frames.size // cam_frame_per_period)
    for idx in tqdm(trial_range):
        for idx2 in range(cam_frame_per_period):
            rolled[idx2, :, :] += frames[idx * cam_frame_per_period + idx2]
    h5file.close()
    return rolled


def circular_free(h5file: str):
    rolled = roll_over(h5file, True)
    phases = np.linspace(0, 2 * np.pi, rolled.shape[0], False)
    result = np.sum(np.exp(-1j * phases)[:, np.newaxis, np.newaxis] * rolled, 0)
    return result, rolled


def clean_diode(onset: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """delete diode signals that have small signal
    Args:
        onset: numpy.ndarray
        signal: numpy.ndarray
    Return:
        onset and amplitude of each diode pulse
    """
    sorted_signal = np.sort(signal)
    threshold = sorted_signal[len(sorted_signal) / 50] * 0.75
    index = signal > threshold
    return onset[index], signal[index]


def colorize(vector_mat: np.ndarray, amp_normalizer=None):
    angle_mat = np.angle(vector_mat)
    amp_mat = np.abs(vector_mat)
    if amp_normalizer is not None:
        amp_mat /= np.abs(amp_normalizer)
    hue = (angle_mat + math.pi) / (2 * math.pi) + 0.5
    r_flat = amp_mat.flatten()
    r_max = np.sort(r_flat)[int(len(r_flat) * 0.95)] * 1.25
    lightness = np.minimum(amp_mat / r_max, 1)
    saturation = 1  # 0.8
    temp_c = np.vectorize(colorsys.hls_to_rgb)(hue, lightness, saturation)  # --> tuple
    final_color = np.array(temp_c)  # -->  array of (3,n,m) shape, but need (n,m,3)
    final_color = np.rollaxis(final_color, 1)
    final_color = np.rollaxis(final_color, 2, 1)
    return final_color


def get_colormap(half_size: int = 100) -> np.ndarray:
    scale = np.arange(-half_size, half_size + 1)
    real, imag = np.meshgrid(scale, scale)
    return colorize(np.flipud(np.add(real, imag * 1j)))


NAMING_OPPOSITES = {"left": "right", "right": "left", "up": "down", "down": "up", "0": "180",
                    "180": "0", "90": "270", "270": "90"}


def online_result(file_name: str = None):
    from glob import glob
    from os import path
    if file_name:
        file_list = glob(path.expanduser(r'~\*{0}.h5'.format(file_name)))
    else:
        file_list = glob(path.expanduser(r'~\*.h5'))
    if len(file_list) == 0:
        raise IOError("cannot find h5 file in Home folder")
    latest_file = max(file_list, key=path.getatime)
    h5_file = h5.File(latest_file, 'r')
    roi = h5_file.attrs['roi']
    shape = ((roi[3] - roi[1] + 1) / roi[5], (roi[2] - roi[0] + 1) / roi[4])
    cam_phase = np.array(h5_file['cam_phase'])
    dft = np.reshape(np.multiply(h5_file['online_imag'], 1j) + h5_file['online_real'], shape)
    ave = np.reshape(h5_file['online_ave'], shape) / len(cam_phase)
    dft -= np.exp(-1j * cam_phase).sum() * ave
    return dft, cam_phase, ave


def convert():
    from os import path
    from tkinter import Tk
    import tkinter.filedialog as tkfd
    from scipy.io import savemat
    root = Tk()
    file_path = tkfd.askopenfilename(parent=root, defaultextension='.h5', initialdir=path.expanduser('~/OI-data'))
    save_path = path.splitext(file_path)[0] + '.mat'
    root.destroy()
    dft, phase, average = circular_transform(file_path)
    savemat(save_path, {"dft": dft, "phase": phase, "average_image": average}, do_compression=True)
