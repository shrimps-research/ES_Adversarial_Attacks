import numpy as np

def nn_interpolate(mat, new_size):
    """ vectorized Nearest Neighbor Interpolation
        upsample from mat.shape to new_size
    """
    old_size = mat.shape
    row_ratio, col_ratio = np.array(new_size)/np.array(old_size)
    # row wise interpolation 
    row_idx = (np.ceil(range(1, 1 + int(old_size[0]*row_ratio))/row_ratio) - 1).astype(int)
    # column wise interpolation
    col_idx = (np.ceil(range(1, 1 + int(old_size[1]*col_ratio))/col_ratio) - 1).astype(int)
    # get and return final matrix
    final_matrix = mat[:, row_idx][col_idx, :]
    return final_matrix