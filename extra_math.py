import numpy as np


def rot_matrix(angle):
    return np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]])


def shift_rotate(x, shift, angle):
    assert isinstance(angle, (np.float32, np.float16, float))
    assert shift.shape[-1] == 2
    assert x.shape[-1] == 2
    return (x + shift) @ rot_matrix(angle).T


def rotate_shift(x, shift, angle):
    assert isinstance(angle, (np.float32, np.float16, float))
    assert shift.shape[-1] == 2
    assert x.shape[-1] == 2
    return (x) @ rot_matrix(angle).T + shift
