import numpy as np
import math
from PIL import Image


# reference facial points, a list of coordinates (x,y)
REFERENCE_FACIAL_POINTS = [
    [30.29459953,  51.69630051],
    [65.53179932,  51.50139999],
    [48.02519989,  71.73660278],
    [33.54930115,  92.3655014],
    [62.72990036,  92.20410156]
]

DEFAULT_CROP_SIZE = (96, 112)


def get_ref_facial_points(face_size=(112, 112)):
    assert face_size in [(96, 112), (112, 112)]
    tmp_5pts = np.array(REFERENCE_FACIAL_POINTS, dtype=np.float32)
    
    if face_size[0] == face_size[1]:
        tmp_crop_size = np.array(DEFAULT_CROP_SIZE)
        size_diff = max(tmp_crop_size) - tmp_crop_size
        tmp_5pts += size_diff / 2
        tmp_crop_size += size_diff
    return tmp_5pts


def find_nonreflective_similarity(uv, xy):
    options = {"K": 2}

    K = options["K"]
    M = xy.shape[0]
    x = xy[:, 0].reshape((-1, 1))   # use reshape to keep a column vector
    y = xy[:, 1].reshape((-1, 1))   # use reshape to keep a column vector

    tmp1 = np.hstack((x, y, np.ones((M, 1)), np.zeros((M, 1))))
    tmp2 = np.hstack((y, -x, np.zeros((M, 1)), np.ones((M, 1))))
    X = np.vstack((tmp1, tmp2))

    u = uv[:, 0].reshape((-1, 1))
    v = uv[:, 1].reshape((-1, 1))
    U = np.vstack((u, v))

    if np.linalg.matrix_rank(X) >= 2 * K:
        r, _, _, _ = np.linalg.lstsq(X, U, rcond=None)
        r = np.squeeze(r)
    
    sc = r[0]
    ss = r[1]
    tx = r[2]
    ty = r[3]

    Tinv = np.array([
        [sc, -ss, 0],
        [ss, sc, 0],
        [tx, ty, 1]
    ])

    T = np.linalg.inv(Tinv)
    T[:, 2] = np.array([0, 0, 1])
    return T, Tinv


def tformfwd(trans, uv):
    uv = np.hstack((uv, np.ones((uv.shape[0], 1))))
    xy = np.dot(uv, trans)
    xy = xy[:, :-1]
    return xy


def find_similarity(uv, xy):
    # solve for trans1
    trans1, trans1_inv = find_nonreflective_similarity(uv, xy)

    xyR = xy
    xyR[:, 0] = -1 * xyR[:, 0]

    trans2r, _ = find_nonreflective_similarity(uv, xyR)

    # manually reflect the tform to undo the reflection done on xyR
    TreflectY = np.array([
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])

    trans2 = np.dot(trans2r, TreflectY)

    # figure out if trans1 or trans2 is better
    xy1 = tformfwd(trans1, uv)
    norm1 = np.linalg.norm(xy1 - xy)
    xy2 = tformfwd(trans2, uv)
    norm2 = np.linalg.norm(xy2 - xy)

    if norm1 <= norm2:
        return trans1, trans1_inv
    else:
        trans2_inv = np.linalg.inv(trans2)
        return trans2, trans2_inv


def get_similarity_transform(src_pts, dst_pts):
    """Find Similarity Transform Matrix "cv2_trans" which could be directly used by cv2.warpAffine():
        u = src_pts[:, 0]
        v = src_pts[:, 1]
        x = dst_pts[:, 0]
        y = dst_pts[:, 1]
        [x, y].T = cv_trans * [u, v, 1].T

    Parameters:
        src_pts: Kx2 np.array 
        dst_pts: Kx2 np.array
    Returns:
        cv2_trans: 2x3 np.array
    """
    trans, _ = find_similarity(src_pts, dst_pts)
    cv2_trans = trans[:, :2].T
    return cv2_trans


def euclidean_distance(src_pts, dst_pts):
    src_pts, dst_pts = np.array(src_pts), np.array(dst_pts)
    dist = src_pts - dst_pts
    dist = np.sum(np.multiply(dist, dist))
    return np.sqrt(dist)

def alignment(img, left_eye, right_eye, nose):
    upside_down = False
    if nose[1] < left_eye[1] or nose[1] < right_eye[1]:
        upside_down = True

    # find rotation direction
    if left_eye[1] > right_eye[1]:
        t_point = (right_eye[0], left_eye[1])
        direction = -1  # clockwise
    else:
        t_point = (left_eye[0], right_eye[1])
        direction = 1   # counterclockwise

    # find length of triangle edges
    a = euclidean_distance(left_eye, t_point)
    b = euclidean_distance(right_eye, t_point)
    c = euclidean_distance(right_eye, left_eye)

    if b != 0 and c != 0:
        cos_a = (b**2 + c**2 - a**2) / (2*b*c)
        cos_a = min(1.0, max(-1.0, cos_a))
        angle = np.arccos(cos_a)
        angle = (angle * 180) / math.pi

        # rotate base image
        if direction == -1:
            angle = 90 - angle
        if upside_down:
            angle = angle + 90
        
        img = Image.fromarray(img)
        img = np.array(img.rotate(direction * angle))
    return img