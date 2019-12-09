#!/usr/bin/env python3
"""Analyse in Fourier transformation based continuous optical imaging."""
from typing import Tuple
import colorsys
import math
import numexpr as ne

import h5py as h5
import numpy as np
from tqdm import tqdm
from uifunc import FileSelector

FRAMES_PER_PULSE = 6  # default sync signal
__author__ = "Keji Li"

Rect = Tuple[int, int, int, int]


def circular_transform(file_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the dft at period_in_seconds from a continuous optical imaging file.

    Args:
        file_path: the hdf5 file path
    Returns:
        (dft, average frame, phase):
            dft: complex image
            phase: phase of all frames in recording
            ave: average of all frames
    """
    h5file = h5.File(file_path, 'r')
    cam_frames = h5file['frame_data']
    phase = extract_frame_phases(h5file)
    cam_frame_coeff = np.exp(-1j * phase)

    cam_frame_shape = cam_frames[0].shape
    result = np.zeros(cam_frame_shape, dtype=np.complex)
    ave = np.zeros(cam_frame_shape, dtype=np.float)

    # get rid of first and last cycle
    trial_ends = next(x for x in np.nonzero(np.diff(phase) < 0))
    cam_frame_range = range(trial_ends[0] + 1, trial_ends[-1] + 1)

    for frame_idx in tqdm(cam_frame_range):
        frame = cam_frames[frame_idx]
        result += cam_frame_coeff[frame_idx] * frame
        ave += frame
    ave /= len(cam_frame_range)
    result -= ave * cam_frame_coeff[cam_frame_range].sum()
    h5file.close()
    return result, ave, phase[cam_frame_range]

def detrend(file_path: str, rect: Rect, return_trend: bool = False) -> Tuple[np.ndarray, ...]:
    """Calculate the dft at period_in_seconds from a continuous optical imaging file.

    Assuming pixels within rect stay stable, and use their average to deduce a global trend line.
    Args:
        file_path: the hdf5 file path
        rect: [x0, y0, x1, y1] inclusive top and left, exclusive bottom and right
    Returns:
        (dft complex matrix, phase_series, trend, averaged_image)
    """
    h5file = h5.File(file_path, 'r')
    phase = extract_frame_phases(h5file)

    cam_frames = h5file['frame_data']
    trend = np.zeros(len(phase))
    for idx in tqdm(range(len(phase))):
        trend[idx] = np.mean(cam_frames[idx][rect[0]:rect[2], rect[1]:rect[3]])

    def linregress(x, y):
        return np.linalg.lstsq(np.vstack([x, np.ones(len(y))]).T, y, rcond=None)

    slope, _ = linregress(np.arange(len(trend)), trend)[0]

    trend = np.arange(len(trend)) * slope
    cam_frame_coeff = np.exp(-1j * phase)

    trial_ends = next(x for x in np.nonzero(np.diff(phase) < 0))
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
    if return_trend:
        return result, ave, phase, trend
    else:
        return result, ave, phase


def normalized(file_path: str, rect: Rect):
    """Calculate circular summation after normalizing each frame with a null area given by Rect."""
    h5file = h5.File(file_path, 'r')
    phase = extract_frame_phases(h5file)

    cam_frames = h5file['frame_data']
    cam_frame_coeff = np.exp(-1j * phase)

    trial_ends = np.nonzero(np.diff(phase) < 0)[0]
    cam_frame_range = range(trial_ends[0] + 1, trial_ends[-1] + 1)  # from second cycle to the last cycle

    cam_frame_shape = cam_frames[0].shape
    result = np.zeros(cam_frame_shape, dtype=np.complex)
    ave = np.zeros(cam_frame_shape, dtype=np.float)

    for frame_idx in tqdm(cam_frame_range):
        frame = cam_frames[frame_idx]
        normal_coef = 1 / np.mean(frame[rect[0]:rect[2], rect[1]:rect[3]])
        result += (cam_frame_coeff[frame_idx] * normal_coef) * frame
        ave += normal_coef * frame
    ave /= len(cam_frame_range)
    result -= ave * cam_frame_coeff[cam_frame_range].sum()
    h5file.close()
    return result, ave, phase


def extract_frame_phases(h5file: h5.File) -> np.ndarray:
    """Calculate the phase corresponding to each frame, from diode signals.

    Args:
        h5file: the hdf5 file
    Returns:
        1-d frame phase (-π to π) time series
    """
    diode_stamps = clean_diode(np.asarray(h5file['diode_nidaq_time']),
                               np.asarray(h5file['diode_signal']))[0]
    # add one more diode signal at the end
    diode_stamps = np.append(diode_stamps, [2 * diode_stamps[-1] - diode_stamps[-2]])
    period_in_stim_frames = h5file.attrs['period_in_frames'][0]
    period_in_pulses = int(period_in_stim_frames / h5file.attrs.get('frame_per_pulse', FRAMES_PER_PULSE))
    phases = np.linspace(0, 2 * np.pi, period_in_pulses, endpoint=False)
    image_stamps = np.array(h5file['frame_timestamps'])
    # get rid of frame after stimulus was done, also there might be a fake pulse at 0
    image_stamps = image_stamps[1 if image_stamps[0] == 0 else 0: np.searchsorted(image_stamps, diode_stamps[-1])]
    post_indices = np.searchsorted(diode_stamps, image_stamps)
    pre_indices = post_indices - 1
    post_phases = phases[post_indices % period_in_pulses]
    post_phases[post_phases == 0] = np.pi * 2
    phase = (phases[pre_indices % period_in_pulses] *
             (diode_stamps[post_indices] - image_stamps) +
             post_phases * (image_stamps - diode_stamps[pre_indices])) / \
        (np.diff(diode_stamps))[pre_indices]
    return phase


def roll_over(h5file_path: str, throw_out_ends: bool = True) -> np.ndarray:
    """Calculate the dft at period_in_seconds from a continuous optical imaging file.

    Args:
        h5file: the file path
        throw_out_ends: whether throw out the first and last trials
    Returns:
        2d numpy array of complex reflecting the dft of image
    """
    h5file = h5.File(h5file_path, 'r')
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
    """Run circular summation assuming perfect periodicity."""
    rolled = roll_over(h5file, True)
    phases = np.linspace(0, 2 * np.pi, rolled.shape[0], False)
    result = np.sum(np.exp(-1j * phases)[:, np.newaxis, np.newaxis] * rolled, 0)
    return result, rolled


def clean_diode(onset: np.ndarray, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Delete diode signals that are too small.

    Args:
        onset: numpy.ndarray
        signal: numpy.ndarray
    Return:
        onset and amplitude of each diode pulse
    """
    threshold = np.percentile(signal, 2, interpolation='higher') * 0.75
    index = signal > threshold
    return onset[index], signal[index]


def colorize(vector_mat: np.ndarray, amp_normalizer=None):
    """Colorize dft matrix using true phase and amplitude, as hue and brightness."""
    phase_mat = np.angle(vector_mat)
    amp_mat = np.abs(vector_mat) if amp_normalizer is None else np.abs(vector_mat / amp_normalizer)
    r_flat = amp_mat.flatten()
    r_max = np.sort(r_flat)[int(len(r_flat) * 0.99)] * 1.25
    colored = np.vectorize(colorsys.hls_to_rgb)((phase_mat + math.pi) / (2 * math.pi) + 0.5,
                                                np.minimum(amp_mat / r_max, 1), 1)
    return np.moveaxis(np.array(colored), 0, 2)  # -->  array of (3,n,m) shape, but need (n,m,3)


def get_colormap(half_size: int = 100) -> np.ndarray:
    """Show a colormap as a wheel, in accordance to colorize()."""
    scale = np.arange(-half_size, half_size + 1)
    real, imag = np.meshgrid(scale, scale)
    return colorize(np.flipud(np.add(real, imag * 1j)))


NAMING_OPPOSITES = {"left": "right", "right": "left", "up": "down", "down": "up",
                    "0": "180", "180": "0", "90": "270", "270": "90"}


def online_result(file_name: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract the online calculated dft result, equivolent to ciruclar_free but has tail noise.

    Args:
        file_name: recording hdf file
    Returns:
        (dft, average frame, phase from camera sync signal)
    """
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
    return dft, ave, cam_phase


@FileSelector(['.h5', '.hdf'])
def convert(file_path: str):
    """Scritp to calculate dft from hdf recording and save to .mat file."""
    from scipy.io import savemat
    save_path = file_path[0: file_path.rfind('.')] + '.mat'
    dft, average, phase = circular_transform(file_path)
    savemat(save_path, {"dft": dft, "phase": phase, "average_image": average}, do_compression=True)
    return


@FileSelector(['.h5', '.hdf'])
def calculate(file_path: str):
    """Script to calculate dft from hdf recording and display resulting figure."""
    from matplotlib import pyplot as plt
    dft, average, phase = circular_transform(file_path)
    plt.figure(figsize=(12, 12), dpi=200)
    ax = plt.subplot(2, 2, 1)
    ax.imshow(np.angle(dft))
    ax.text(2, -2, 'phase')
    ax = plt.subplot(2, 2, 2)
    ax.imshow(np.abs(dft))
    ax.text(2, -2, 'power')
    ax = plt.subplot(2, 2, 3)
    ax.imshow(colorize(dft))
    ax.text(2, -2, 'combined')
    ax = plt.subplot(2, 2, 4)
    ax.imshow(get_colormap())
    ax.text(2, -2, 'colormap')
    plt.show()
    return dft, average, phase
