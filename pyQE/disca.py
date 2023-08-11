import numpy as np
from tqdm import trange, tqdm
import numba as nb

def symmetrize_disca(arr, sym, nkpoints, rotation = None, quiet=False, output_fname=None, rotate_output=False):
    """
    Parameters
    ----------
    rotations : dict
        Dictionary with keys being the axes that have experienced rotation in
        the DISCA calculation, and values being the values of these keys,
        namely the rotation angle in radians
    
    """
    THETA = 2.0*np.pi/float(sym)
    NK_TOT = nkpoints
    str_f = arr#np.loadtxt('structure_factor_all-phonon.dat')

    if rotation is not None: #requires antirotation before and rotation after
        Rmat = rotation['matrix']
        plane_dir = rotation['plane_dir']
        plane_val = rotation['plane_val']
        clipping = rotation['clip_threshold']

        ORIGIN = np.average(str_f[:,:3], axis=0)
        str_f[:,:3] = np.transpose(Rmat @ (str_f[:,:3]-ORIGIN).T) + ORIGIN
        str_f[:,:3][:,plane_dir] = np.where(np.abs(str_f[:,:3][:,plane_dir])<clipping, plane_val, str_f[:,:3][:,plane_dir])

    if not quiet:
        str_fp = np.zeros_like(str_f)
        str_f_new = np.zeros((int(str_fp.shape[0]*float(sym)),4))
        str_fp[0,:] = str_f[0,:]
        for p in range(1,int(NK_TOT)):#trange(1, int(NK_TOT), desc=f"Removing doubles for {label}", position=1):
            if np.arctan(str_f[p,0]/str_f[p,1]) < (2.0*np.pi / (float(sym)-np.finfo(float).eps)):
                str_fp[p,:] = str_f[p,:]

        ctr = 0
        R = np.zeros((2,2))
        pbar = tqdm(total=str_f_new.shape[0])
        for i in range(int(sym)):
            R[0,:] = np.array((np.cos(i*THETA), -np.sin(i*THETA)))
            R[1,:] = np.array((np.sin(i*THETA), np.cos(i*THETA)))
            for p in range(int(NK_TOT)):
                str_f_new[ctr,:2] = R@str_fp[p,:2]
                str_f_new[ctr,2:] = str_fp[p,2:]
                ctr += 1
                pbar.update(1)
    else:
        str_f_new = str_f

    if rotation is not None and not rotate_output: #perform rotation
        Rmat = np.linalg.inv(Rmat)
        ORIGIN = np.average(str_f_new[:,:3], axis=0)
        str_f_new[:,:3] = np.transpose(Rmat @ (str_f_new[:,:3]-ORIGIN).T) + ORIGIN

    if output_fname is not None:
        np.savetxt(output_fname,str_f_new, fmt='%12.6f') #'structure_factor_all-phonon_rot.dat'
    return str_f_new


def get_rotation_matrix(axis_ids, angs):
    Rs = []
    for ida, axis_id in enumerate(axis_ids):
        angle = angs[ida]
        cosang = np.cos(angle)
        sinang = np.sin(angle)
        if axis_id == 0: #x axis rotation
            R = np.array([
                [1, 0, 0],
                [0, cosang, -sinang],
                [0, sinang, cosang]
                ])
            Rs.append(R)
        elif axis_id == 1: #y axis rotation
            R = np.array([
                [cosang, 0, sinang],
                [0, 1, 0],
                [-sinang, 0, cosang]
            ])
            Rs.append(R)
        elif axis_id == 2: #z axis rotation
            R = np.array([
                [cosang, -sinang, 0],
                [sinang, cosang, 0],
                [0, 0, 1]
            ])
            Rs.append(R)
    res = np.eye(3)
    for R in Rs:
        res = res@R
    return res

def get_bounded_scattering_vectors(data, dim1=0, dim2=1, plane_dim=2, plane_val=0., fill_value = 0., resolution=2048, square=False):
    """
    Takes rotated coordinates and determines largest interior
    rectangle (or square depending on passed arguments) so that a
    Gaussian broadening data reduction scheme can be applied to 
    a uniform grid.

    Returns
    -------
    lir : `numpy.ndarray`, shape(4,)
        Largest inscribed rectangle in pixel coordinates of the interpolated grid

    corners : tuple
        Corners of the LIR given in the same units as the input data
    """
    from scipy.interpolate import griddata
    x = np.linspace(data[:,dim1].min(), data[:,dim1].max(), resolution)
    y = np.linspace(data[:,dim2].min(), data[:,dim2].max(), resolution)
    xx, yy = np.meshgrid(x, y)
    interp = np.rint(griddata(
        points = (data[:,dim1], data[:,dim2]),
        values = data[:,plane_dim]-plane_val+1.,
        xi = (xx, yy),
        fill_value = fill_value
    )).astype(bool)
    lir = largest_interior_rectangle(interp)
    find_id = lambda total, val: np.argmin(np.abs(total-val))
    
    RX = xx[lir[1]:lir[1]+lir[3],lir[0]:lir[0]+lir[2]]
    RY = yy[lir[1]:lir[1]+lir[3],lir[0]:lir[0]+lir[2]]
    if square:
        WIDTH = min(RX.max()-RX.min(), RY.max()-RY.min())
        WIDTH_ID = min(lir[3],lir[2])
        AVG_X = (RX.max()+RX.min())/2
        AVG_Y = (RY.max()+RY.min())/2
        corners = (AVG_X-WIDTH//2, AVG_X+WIDTH//2, AVG_Y-WIDTH//2, AVG_Y+WIDTH//2)
        lir[0] = find_id(y, AVG_Y)-WIDTH_ID//2
        lir[1] = find_id(x, AVG_X)-WIDTH_ID//2
        lir[2:] = WIDTH_ID
    else:
        corners = (RX.min(), RX.max(), RY.min(), RY.max())
    return (x, y, interp), lir, corners

def largest_interior_rectangle(grid):
    h_adjacency = horizontal_adjacency(grid)
    v_adjacency = vertical_adjacency(grid)
    s_map = span_map(grid, h_adjacency, v_adjacency)
    return biggest_span_in_span_map(s_map)


@nb.njit('uint32[:,::1](boolean[:,::1])', parallel=True, cache=True)
def horizontal_adjacency(grid):
    result = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.uint32)
    for y in nb.prange(grid.shape[0]):
        span = 0
        for x in range(grid.shape[1]-1, -1, -1):
            if grid[y, x]:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit('uint32[:,::1](boolean[:,::1])', parallel=True, cache=True)
def vertical_adjacency(grid):
    result = np.zeros((grid.shape[0], grid.shape[1]), dtype=np.uint32)
    for x in nb.prange(grid.shape[1]):
        span = 0
        for y in range(grid.shape[0]-1, -1, -1):
            if grid[y, x]:
                span += 1
            else:
                span = 0
            result[y, x] = span
    return result


@nb.njit('uint32(uint32[:])', cache=True)
def predict_vector_size(array):
    zero_indices = np.where(array == 0)[0]
    if len(zero_indices) == 0:
        if len(array) == 0:
            return 0
        return len(array)
    return zero_indices[0]


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def h_vector(h_adjacency, x, y):
    vector_size = predict_vector_size(h_adjacency[y:, x])
    h_vector = np.zeros(vector_size, dtype=np.uint32)
    h = np.Inf
    for p in range(vector_size):
        h = np.minimum(h_adjacency[y+p, x], h)
        h_vector[p] = h
    h_vector = np.unique(h_vector)[::-1]
    return h_vector


@nb.njit('uint32[:](uint32[:,::1], uint32, uint32)', cache=True)
def v_vector(v_adjacency, x, y):
    vector_size = predict_vector_size(v_adjacency[y, x:])
    v_vector = np.zeros(vector_size, dtype=np.uint32)
    v = np.Inf
    for q in range(vector_size):
        v = np.minimum(v_adjacency[y, x+q], v)
        v_vector[q] = v
    v_vector = np.unique(v_vector)[::-1]
    return v_vector


@nb.njit('uint32[:,:](uint32[:], uint32[:])', cache=True)
def spans(h_vector, v_vector):
    spans = np.stack((h_vector, v_vector[::-1]), axis=1)
    return spans


@nb.njit('uint32[:](uint32[:,:])', cache=True)
def biggest_span(spans):
    if len(spans) == 0:
        return np.array([0, 0], dtype=np.uint32)
    areas = spans[:, 0] * spans[:, 1]
    biggest_span_index = np.where(areas == np.amax(areas))[0][0]
    return spans[biggest_span_index]


@nb.njit('uint32[:, :, :](boolean[:,::1], uint32[:,::1], uint32[:,::1])',
         parallel=True, cache=True)
def span_map(grid, h_adjacency, v_adjacency):

    y_values, x_values = grid.nonzero()
    span_map = np.zeros(grid.shape + (2,), dtype=np.uint32)

    for idx in nb.prange(len(x_values)):
        x, y = x_values[idx], y_values[idx]
        h_vec = h_vector(h_adjacency, x, y)
        v_vec = v_vector(v_adjacency, x, y)
        s = spans(h_vec, v_vec)
        s = biggest_span(s)
        span_map[y, x, :] = s

    return span_map


@nb.njit('uint32[:](uint32[:, :, :])', cache=True)
def biggest_span_in_span_map(span_map):
    areas = span_map[:, :, 0] * span_map[:, :, 1]
    largest_rectangle_indices = np.where(areas == np.amax(areas))
    x = largest_rectangle_indices[1][0]
    y = largest_rectangle_indices[0][0]
    span = span_map[y, x]
    return np.array([x, y, span[0], span[1]], dtype=np.uint32)