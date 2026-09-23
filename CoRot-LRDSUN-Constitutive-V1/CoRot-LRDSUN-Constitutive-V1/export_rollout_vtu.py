#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch

from corot_lrdsun.data import PreparedSample
from corot_lrdsun.io import load_manifest, split_records
from corot_lrdsun.model import CoRotLRDSUN
from corot_lrdsun.normalization import load_stats
from corot_lrdsun.physics import mises_from_s4_numpy
from corot_lrdsun.runtime import predict_batch


# --------------------------------------------------------------------------------------
# Minimal dependency-free VTU writer (VTK XML, inline base64 binary blocks)
# --------------------------------------------------------------------------------------

_VTK_DTYPES = {
    "Float32": np.dtype("<f4"),
    "Float64": np.dtype("<f8"),
    "Int32": np.dtype("<i4"),
    "UInt8": np.dtype("u1"),
}


def _vtk_binary_payload(arr: np.ndarray, vtk_type: str) -> str:
    dt = _VTK_DTYPES[vtk_type]
    a = np.asarray(arr, dtype=dt, order="C")
    raw = a.tobytes(order="C")
    header = np.asarray([len(raw)], dtype=np.dtype("<u4")).tobytes()
    return base64.b64encode(header + raw).decode("ascii")


def _data_array_xml(
    name: str,
    arr: np.ndarray,
    vtk_type: str = "Float32",
    ncomp: int | None = None,
    indent: str = "        ",
) -> str:
    a = np.asarray(arr)
    if a.ndim == 1:
        inferred = 1
    elif a.ndim == 2:
        inferred = int(a.shape[1])
    else:
        raise ValueError(f"Point/cell array {name!r} must be 1D or 2D, got {a.shape}")
    nc = int(ncomp if ncomp is not None else inferred)
    comp_attr = f' NumberOfComponents="{nc}"' if nc != 1 else ""
    payload = _vtk_binary_payload(a, vtk_type)
    return (
        f'{indent}<DataArray type="{vtk_type}" Name="{name}"{comp_attr} '
        f'format="binary">{payload}</DataArray>\n'
    )


def _build_cells(cells4: np.ndarray, cell_num_nodes: np.ndarray):
    cells4 = np.asarray(cells4, dtype=np.int32)
    counts = np.asarray(cell_num_nodes, dtype=np.int32).reshape(-1)
    if cells4.ndim != 2 or cells4.shape[1] < 4:
        raise ValueError(f"tube_cells must be [E,4], got {cells4.shape}")
    if len(counts) != len(cells4):
        raise ValueError("tube_cell_num_nodes length does not match tube_cells")

    conn_parts = []
    types = np.empty(len(cells4), dtype=np.uint8)
    offsets = np.empty(len(cells4), dtype=np.int32)
    off = 0
    for i, n in enumerate(counts.tolist()):
        if n == 3:
            conn_parts.append(cells4[i, :3])
            types[i] = 5   # VTK_TRIANGLE
            off += 3
        elif n == 4:
            conn_parts.append(cells4[i, :4])
            types[i] = 9   # VTK_QUAD
            off += 4
        else:
            raise ValueError(f"Unsupported shell element with {n} nodes at row {i}")
        offsets[i] = off
    connectivity = np.concatenate(conn_parts).astype(np.int32, copy=False)
    return connectivity, offsets, types


def write_vtu(
    path: Path,
    points: np.ndarray,
    cells4: np.ndarray,
    cell_num_nodes: np.ndarray,
    point_data: Dict[str, np.ndarray],
    point_int_data: Dict[str, np.ndarray] | None = None,
    cell_int_data: Dict[str, np.ndarray] | None = None,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    points = np.asarray(points, dtype=np.float32)
    npt = points.shape[0]
    connectivity, offsets, types = _build_cells(cells4, cell_num_nodes)
    ncell = len(offsets)

    lines = []
    lines.append('<?xml version="1.0"?>\n')
    lines.append(
        '<VTKFile type="UnstructuredGrid" version="0.1" '
        'byte_order="LittleEndian" header_type="UInt32">\n'
    )
    lines.append('  <UnstructuredGrid>\n')
    lines.append(f'    <Piece NumberOfPoints="{npt}" NumberOfCells="{ncell}">\n')

    lines.append('      <PointData>\n')
    for name, arr in point_data.items():
        a = np.asarray(arr)
        if a.shape[0] != npt:
            raise ValueError(f"PointData {name}: first dimension {a.shape[0]} != N={npt}")
        lines.append(_data_array_xml(name, a, "Float32", indent="        "))
    if point_int_data:
        for name, arr in point_int_data.items():
            a = np.asarray(arr)
            if a.shape[0] != npt:
                raise ValueError(f"PointData {name}: first dimension {a.shape[0]} != N={npt}")
            lines.append(_data_array_xml(name, a, "Int32", indent="        "))
    lines.append('      </PointData>\n')

    lines.append('      <CellData>\n')
    if cell_int_data:
        for name, arr in cell_int_data.items():
            a = np.asarray(arr)
            if a.shape[0] != ncell:
                raise ValueError(f"CellData {name}: first dimension {a.shape[0]} != E={ncell}")
            lines.append(_data_array_xml(name, a, "Int32", indent="        "))
    lines.append('      </CellData>\n')

    lines.append('      <Points>\n')
    lines.append(_data_array_xml("Points", points, "Float32", ncomp=3, indent="        "))
    lines.append('      </Points>\n')

    lines.append('      <Cells>\n')
    lines.append(_data_array_xml("connectivity", connectivity, "Int32", indent="        "))
    lines.append(_data_array_xml("offsets", offsets, "Int32", indent="        "))
    lines.append(_data_array_xml("types", types, "UInt8", indent="        "))
    lines.append('      </Cells>\n')

    lines.append('    </Piece>\n')
    lines.append('  </UnstructuredGrid>\n')
    lines.append('</VTKFile>\n')

    path.write_text("".join(lines), encoding="ascii")


def write_pvd(path: Path, entries: Iterable[Tuple[float, str]]):
    path = Path(path)
    lines = [
        '<?xml version="1.0"?>\n',
        '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">\n',
        '  <Collection>\n',
    ]
    for timestep, relfile in entries:
        lines.append(
            f'    <DataSet timestep="{float(timestep):.10g}" group="" part="0" file="{relfile}"/>\n'
        )
    lines += ['  </Collection>\n', '</VTKFile>\n']
    path.write_text("".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------------------
# Field construction
# --------------------------------------------------------------------------------------

_SURFACES = ((0, "Outer_SPOS"), (1, "Inner_SNEG"))
_S_NAMES = ("S_aa", "S_tt", "S_rr", "S_at")
_PE_NAMES = ("PE_aa", "PE_tt", "PE_rr", "PE_at")
_LE_NAMES = ("LE_aa", "LE_tt", "LE_rr", "LE_at")


def add_triplet(
    out: Dict[str, np.ndarray],
    prefix: str,
    gt: np.ndarray,
    pred: np.ndarray,
    unit_suffix: str = "",
):
    out[f"{prefix}_GT{unit_suffix}"] = np.asarray(gt, dtype=np.float32)
    out[f"{prefix}_Pred{unit_suffix}"] = np.asarray(pred, dtype=np.float32)
    out[f"{prefix}_AbsError{unit_suffix}"] = np.abs(
        np.asarray(pred, dtype=np.float32) - np.asarray(gt, dtype=np.float32)
    )


def make_point_data(
    s: PreparedSample,
    frame: int,
    pred_state: np.ndarray,
    pred_le: np.ndarray,
    fields_mode: str,
    gate_prob: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}

    # Geometry is always the FEM/GT deformation, because this constitutive model is driven by GT U.
    out["U_GT_mm"] = np.asarray(s.U[frame], dtype=np.float32)

    gt_states = (
        np.asarray(s.state_outer[frame], dtype=np.float32),
        np.asarray(s.state_inner[frame], dtype=np.float32),
    )
    gt_les = (
        np.asarray(s.LE_outer[frame], dtype=np.float32),
        np.asarray(s.LE_inner[frame], dtype=np.float32),
    )

    for surface, surf_name in _SURFACES:
        gt = gt_states[surface]
        pr = np.asarray(pred_state[surface], dtype=np.float32)
        gt_le = gt_les[surface]
        pr_le = np.asarray(pred_le[surface], dtype=np.float32)

        gt_vm = mises_from_s4_numpy(gt[:, :4]).reshape(-1)
        pr_vm = mises_from_s4_numpy(pr[:, :4]).reshape(-1)
        add_triplet(out, f"{surf_name}_S_Mises", gt_vm, pr_vm, "_MPa")
        add_triplet(out, f"{surf_name}_PEEQ", gt[:, 8], pr[:, 8])

        if fields_mode == "all":
            for j, nm in enumerate(_S_NAMES):
                add_triplet(out, f"{surf_name}_{nm}", gt[:, j], pr[:, j], "_MPa")
            for j, nm in enumerate(_PE_NAMES):
                add_triplet(out, f"{surf_name}_{nm}", gt[:, 4 + j], pr[:, 4 + j])
            for j, nm in enumerate(_LE_NAMES):
                add_triplet(out, f"{surf_name}_{nm}", gt_le[:, j], pr_le[:, j])

        if gate_prob is not None:
            out[f"{surf_name}_PlasticGateProb"] = np.asarray(
                gate_prob[surface], dtype=np.float32
            ).reshape(-1)

    return out


# --------------------------------------------------------------------------------------
# Autoregressive rollout + direct VTU export
# --------------------------------------------------------------------------------------


def parse_sample_ids(s: str) -> set[int] | None:
    s = str(s).strip()
    if not s:
        return None
    return {int(x.strip()) for x in s.split(",") if x.strip()}


def load_topology(raw_path: Path):
    with np.load(raw_path, allow_pickle=False) as z:
        cells = np.asarray(z["tube_cells"], dtype=np.int32)
        counts = np.asarray(z["tube_cell_num_nodes"], dtype=np.int8)
        node_labels = np.asarray(z["tube_node_label"], dtype=np.int32)
        elem_labels = np.asarray(z["tube_element_label"], dtype=np.int32)
        if "angle_deg" in z:
            angle_deg = np.asarray(z["angle_deg"], dtype=np.float64)
        else:
            angle_deg = np.arange(int(np.asarray(z["num_frames"]).reshape(-1)[0]), dtype=np.float64)
    return cells, counts, node_labels, elem_labels, angle_deg


def should_write(frame: int, last_frame: int, stride: int) -> bool:
    return frame == 0 or frame == last_frame or frame % int(stride) == 0


def main():
    ap = argparse.ArgumentParser(
        description="CoRot-LRDSUN autoregressive rollout -> VTU/PVD for ParaView"
    )
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--split", choices=("val", "test"), default="test")
    ap.add_argument("--sample-ids", type=str, default="", help="e.g. 119 or 0,16,119; empty = all in split")
    ap.add_argument("--batch-size", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Inference device. auto uses CUDA when available.",
    )
    ap.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable BF16 autocast on CUDA.",
    )
    ap.add_argument("--frame-stride", type=int, default=1, help="1 writes all 181 frames")
    ap.add_argument("--fields", choices=("core", "all"), default="all", help="core=S_Mises+PEEQ; all adds S4+PE4+LE4")
    ap.add_argument("--soft-gate", action="store_true")
    ap.add_argument("--gate-threshold", type=float, default=0.5)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    # --------------------------------------------------------
    # Device selection
    # --------------------------------------------------------
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested but CUDA is not available"
            )
        device = torch.device("cuda:0")

    elif args.device == "cpu":
        device = torch.device("cpu")

    else:
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()

    torch.set_num_threads(max(1, int(args.threads)))

    try:
        torch.set_num_interop_threads(
            max(1, min(4, int(args.threads)))
        )
    except RuntimeError:
        pass

    amp_enabled = (
        device.type == "cuda"
        and not args.no_amp
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.checkpoint, map_location=device)
    stats = ck.get("stats") or load_stats(args.cache_dir / "stats.json")
    cfg = ck.get("config", {})
    model = CoRotLRDSUN(
        int(cfg.get("hidden_dim", 192)),
        float(cfg.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ck["model"])
    model.eval()

    manifest = load_manifest(args.cache_dir)
    recs = split_records(manifest, args.split)
    wanted = parse_sample_ids(args.sample_ids)
    if wanted is not None:
        recs = [r for r in recs if int(r["sample_id"]) in wanted]
        missing = sorted(wanted - {int(r["sample_id"]) for r in recs})
        if missing:
            raise RuntimeError(f"Requested sample IDs not found in split={args.split}: {missing}")

    if not recs:
        raise RuntimeError("No samples selected")

    print("=" * 100, flush=True)
    print("CoRot-LRDSUN rollout -> VTU", flush=True)
    print(f"device       : {device}", flush=True)
    print(f"AMP/BF16     : {amp_enabled}", flush=True)
    print(f"threads      : {args.threads}", flush=True)

    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        print(
            f"GPU          : {torch.cuda.get_device_name(device)} "
            f"({p.total_memory / 1024**3:.2f} GiB)",
            flush=True,
        )
    print(f"split        : {args.split}", flush=True)
    print(f"samples      : {[int(r['sample_id']) for r in recs]}", flush=True)
    print(f"batch_size   : {args.batch_size}", flush=True)
    print(f"frame_stride : {args.frame_stride}", flush=True)
    print(f"fields       : {args.fields}", flush=True)
    print(f"output       : {args.output_dir}", flush=True)
    print("Error fields : absolute error = |Pred - GT|", flush=True)
    print("Surfaces     : Outer_SPOS and Inner_SNEG", flush=True)
    print("=" * 100, flush=True)

    with torch.inference_mode():
        for ir, rec in enumerate(recs, 1):
            sid = int(rec["sample_id"])
            s = PreparedSample(rec)
            cells, counts, node_labels, elem_labels, angle_deg = load_topology(Path(rec["raw_path"]))

            if len(node_labels) != s.N:
                raise RuntimeError(f"sample={sid}: topology N={len(node_labels)} != cache N={s.N}")

            sample_dir = args.output_dir / f"sample_{sid:04d}"
            sample_dir.mkdir(parents=True, exist_ok=True)
            pvd_entries = []

            # Frame 0 is the known initial state; prediction = GT by construction.
            pred = np.stack([
                np.asarray(s.state_outer[0], dtype=np.float32).copy(),
                np.asarray(s.state_inner[0], dtype=np.float32).copy(),
            ], axis=0)
            pred_le = np.stack([
                np.asarray(s.LE_outer[0], dtype=np.float32).copy(),
                np.asarray(s.LE_inner[0], dtype=np.float32).copy(),
            ], axis=0)
            gate_frame = np.zeros((2, s.N), dtype=np.float32)

            def export_frame(frame: int, ps: np.ndarray, pl: np.ndarray, gp: np.ndarray):
                if not should_write(frame, s.T - 1, args.frame_stride):
                    return
                fn = f"frame_{frame:03d}.vtu"
                out_path = sample_dir / fn
                if out_path.exists() and not args.overwrite:
                    print(f"  skip existing {out_path}", flush=True)
                else:
                    points = np.asarray(s.X0, dtype=np.float32) + np.asarray(s.U[frame], dtype=np.float32)
                    pd = make_point_data(s, frame, ps, pl, args.fields, gp)
                    write_vtu(
                        out_path,
                        points,
                        cells,
                        counts,
                        pd,
                        point_int_data={
                            "NodeLabel": node_labels,
                            "RegionID": np.asarray(s.region_id, dtype=np.int32),
                        },
                        cell_int_data={"ElementLabel": elem_labels},
                    )
                timestep = float(angle_deg[frame]) if frame < len(angle_deg) else float(frame)
                pvd_entries.append((timestep, fn))

            export_frame(0, pred, pred_le, gate_frame)

            for t in range(s.T - 1):
                next_pred = np.empty_like(pred)
                next_le = np.empty((2, s.N, 4), dtype=np.float32)
                next_gate = np.empty((2, s.N), dtype=np.float32)

                for surface in (0, 1):
                    for a in range(0, s.N, args.batch_size):
                        bnd = min(a + args.batch_size, s.N)
                        centers = np.arange(a, bnd, dtype=np.int64)
                        surfaces = np.full(len(centers), surface, dtype=np.int64)
                        override = pred[surface, centers]

                        batch = s.make_batch(
                            t,
                            centers,
                            surfaces,
                            state_override=override,
                        )
                        pstate, ple, gate, _, _ = predict_batch(
                            model,
                            batch,
                            stats,
                            device,
                            amp=amp_enabled,
                            hard_gate=not args.soft_gate,
                            gate_threshold=args.gate_threshold,
                        )
                        next_pred[surface, centers] = pstate.cpu().numpy()
                        next_le[surface, centers] = ple.cpu().numpy()
                        next_gate[surface, centers] = gate.cpu().numpy().reshape(-1)

                pred = next_pred
                pred_le = next_le
                gate_frame = next_gate
                frame = t + 1
                export_frame(frame, pred, pred_le, gate_frame)

                if frame % 10 == 0 or frame == s.T - 1:
                    vm_err = []
                    for surf in (0, 1):
                        gt_state = np.asarray(
                            s.state_outer[frame] if surf == 0 else s.state_inner[frame],
                            dtype=np.float32,
                        )
                        gt_vm = mises_from_s4_numpy(gt_state[:, :4]).reshape(-1)
                        pr_vm = mises_from_s4_numpy(pred[surf, :, :4]).reshape(-1)
                        vm_err.append(float(np.sqrt(np.mean((pr_vm - gt_vm) ** 2))))
                    print(
                        f"sample={sid:04d} frame={frame:03d}/{s.T-1} "
                        f"MisesRMSE outer/inner={vm_err[0]:.4f}/{vm_err[1]:.4f} MPa",
                        flush=True,
                    )

            pvd_path = sample_dir / f"sample_{sid:04d}_rollout.pvd"
            write_pvd(pvd_path, pvd_entries)
            print(
                f"[{ir}/{len(recs)}] DONE sample={sid:04d} -> {pvd_path}",
                flush=True,
            )

    print("=" * 100, flush=True)
    print("VTU EXPORT COMPLETE", flush=True)
    print(f"Open in ParaView: <output>/sample_xxxx/sample_xxxx_rollout.pvd", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
