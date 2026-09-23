from __future__ import annotations

import numpy as np

EPS = 1.0e-12


def csr_sources(ptr: np.ndarray) -> np.ndarray:
    ptr = np.asarray(ptr, dtype=np.int64)
    return np.repeat(np.arange(len(ptr) - 1, dtype=np.int64), np.diff(ptr))


def compute_kabsch_rotations_numpy(
    X0: np.ndarray,
    Xt: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    chunk_size: int = 8192,
) -> tuple[np.ndarray, np.ndarray]:
    """Production-V1-compatible node-wise Kabsch rotations.

    Finds R_i such that (Xt_j-Xt_i) ~= R_i (X0_j-X0_i) over each fixed CSR stencil.
    """
    X0 = np.asarray(X0, dtype=np.float64)
    Xt = np.asarray(Xt, dtype=np.float64)
    ptr = np.asarray(ptr, dtype=np.int64)
    idx = np.asarray(idx, dtype=np.int64)
    n = X0.shape[0]
    src = csr_sources(ptr)
    A = X0[idx] - X0[src]
    B = Xt[idx] - Xt[src]
    H = np.zeros((n, 3, 3), dtype=np.float64)
    np.add.at(H, src, A[:, :, None] * B[:, None, :])
    R = np.empty((n, 3, 3), dtype=np.float64)
    S_all = np.empty((n, 3), dtype=np.float64)
    for start in range(0, n, int(chunk_size)):
        end = min(start + int(chunk_size), n)
        U, S, VT = np.linalg.svd(H[start:end])
        Rc = np.matmul(np.transpose(VT, (0, 2, 1)), np.transpose(U, (0, 2, 1)))
        bad = np.linalg.det(Rc) < 0.0
        if np.any(bad):
            VT2 = VT.copy()
            VT2[bad, -1, :] *= -1.0
            Rc = np.matmul(np.transpose(VT2, (0, 2, 1)), np.transpose(U, (0, 2, 1)))
        R[start:end] = Rc
        S_all[start:end] = S
    return R.astype(np.float32), S_all.astype(np.float32)


def region_ids_from_reference(X0: np.ndarray, D: float, R_bending: float) -> np.ndarray:
    """Sampling-only clamp/bending/free region labels.

    Uses the legacy tube coordinate convention first. If it is degenerate, falls back to
    axial-coordinate tertiles. Region labels are never model inputs.
    """
    X0 = np.asarray(X0, dtype=np.float64)
    z = X0[:, 2]
    tol = 1.0e-6 * float(D)
    rid = np.where(z <= tol, 0, np.where(z <= np.pi * float(R_bending) + tol, 1, 2)).astype(np.int8)
    if all(np.any(rid == k) for k in (0, 1, 2)):
        return rid
    center = X0.mean(axis=0)
    Y = X0 - center
    _, _, vt = np.linalg.svd(Y, full_matrices=False)
    s = Y @ vt[0]
    q1, q2 = np.quantile(s, [0.25, 0.75])
    return np.where(s <= q1, 0, np.where(s <= q2, 1, 2)).astype(np.int8)


def corot_edge_features_for_centers(
    X0: np.ndarray,
    U: np.ndarray,
    Q0: np.ndarray,
    R_all: np.ndarray,
    ptr: np.ndarray,
    idx: np.ndarray,
    t: int,
    centers: np.ndarray,
    D_outer: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build padded neighbor kinematics [B,K,12] and mask [B,K].

    r0  = Q0_i^T (X0_j-X0_i)
    rt  = Qt_i^T (Xt_j-Xt_i), Qt_i = R_i^t Q0_i
    d_t = rt-r0
    dd  = d_(t+1)-d_t
    All lengths are divided by D_outer.
    """
    centers = np.asarray(centers, dtype=np.int64).reshape(-1)
    B = len(centers)
    counts = (ptr[centers + 1] - ptr[centers]).astype(np.int64)
    K = int(counts.max()) if B else 0
    feat = np.zeros((B, K, 12), dtype=np.float32)
    mask = np.zeros((B, K), dtype=np.bool_)
    Xt = X0 + U[t]
    X1 = X0 + U[t + 1]
    invD = 1.0 / max(float(D_outer), EPS)
    for b, c in enumerate(centers.tolist()):
        nb = idx[int(ptr[c]): int(ptr[c + 1])].astype(np.int64, copy=False)
        q0 = Q0[c].astype(np.float64)
        qt = (R_all[t, c].astype(np.float64) @ q0)
        q1 = (R_all[t + 1, c].astype(np.float64) @ q0)
        rel0 = X0[nb].astype(np.float64) - X0[c].astype(np.float64)
        relt = Xt[nb].astype(np.float64) - Xt[c].astype(np.float64)
        rel1 = X1[nb].astype(np.float64) - X1[c].astype(np.float64)
        # Row-vector equivalent of Q.T @ v_global: v_global(row) @ Q.
        r0 = rel0 @ q0
        rt = relt @ qt
        r1 = rel1 @ q1
        d0 = rt - r0
        dd = (r1 - r0) - d0
        r0 *= invD; d0 *= invD; dd *= invD
        k = len(nb)
        feat[b, :k, 0:3] = r0.astype(np.float32)
        feat[b, :k, 3:6] = d0.astype(np.float32)
        feat[b, :k, 6:9] = dd.astype(np.float32)
        feat[b, :k, 9] = np.linalg.norm(r0, axis=1).astype(np.float32)
        feat[b, :k, 10] = np.linalg.norm(d0, axis=1).astype(np.float32)
        feat[b, :k, 11] = np.linalg.norm(dd, axis=1).astype(np.float32)
        mask[b, :k] = True
    return feat, mask


def rigid_objectivity_self_test(seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    n = 24
    x = rng.normal(size=(n, 3))
    x[:, 2] *= 0.05
    # Every center sees all other points: sufficient for the mathematical self-test.
    ptr = np.arange(0, n * (n - 1) + 1, n - 1, dtype=np.int64)
    idx = np.concatenate([np.delete(np.arange(n, dtype=np.int64), i) for i in range(n)])
    # Orthonormal reference frame shared for this synthetic patch.
    Q0 = np.repeat(np.eye(3)[None, :, :], n, axis=0).astype(np.float32)
    a = np.array([0.3, -0.4, 0.7], dtype=np.float64); a /= np.linalg.norm(a)
    ang = 0.8
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]], dtype=np.float64)
    Rg = np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
    tr = np.array([2.0, -3.0, 1.5])
    Xt = x @ Rg.T + tr
    U = np.stack([np.zeros_like(x), Xt - x], axis=0).astype(np.float32)
    R0 = np.repeat(np.eye(3, dtype=np.float32)[None, :, :], n, axis=0)
    R1, _ = compute_kabsch_rotations_numpy(x, Xt, ptr, idx)
    Rall = np.stack([R0, R1], axis=0)
    feat, mask = corot_edge_features_for_centers(x, U, Q0, Rall, ptr, idx, 0, np.arange(n), 1.0)
    max_d = float(np.abs(feat[:, :, 3:9][mask]).max())
    return {"max_abs_corot_deformation_under_rigid_motion": max_d, "passed": bool(max_d < 2e-5)}
