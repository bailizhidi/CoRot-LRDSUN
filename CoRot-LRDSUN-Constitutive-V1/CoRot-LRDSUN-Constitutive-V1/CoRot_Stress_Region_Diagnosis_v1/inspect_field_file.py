#!/usr/bin/env python3
import argparse, sys, os
from pathlib import Path
import numpy as np

def load_vtu(path):
    try:
        import pyvista as pv
        mesh = pv.read(path)
        return {
            'backend': 'pyvista',
            'points': np.asarray(mesh.points),
            'point_data': {k: np.asarray(mesh.point_data[k]) for k in mesh.point_data.keys()},
        }
    except Exception as e_pv:
        try:
            import meshio
            m = meshio.read(path)
            return {
                'backend': 'meshio',
                'points': np.asarray(m.points),
                'point_data': {k: np.asarray(v) for k,v in m.point_data.items()},
            }
        except Exception as e_mi:
            raise RuntimeError(
                'Cannot read VTU. Install pyvista or meshio in this environment.\n'
                f'pyvista error: {e_pv}\nmeshio error: {e_mi}'
            )

def main():
    ap = argparse.ArgumentParser(description='Inspect VTU/NPZ fields before stress-region diagnosis.')
    ap.add_argument('path')
    args = ap.parse_args()
    p = Path(args.path)
    if not p.exists():
        raise FileNotFoundError(p)
    if p.suffix.lower() == '.npz':
        z = np.load(p, allow_pickle=True)
        print(f'FILE={p}')
        print('TYPE=npz')
        for k in z.files:
            a = np.asarray(z[k])
            print(f'{k:60s} shape={a.shape} dtype={a.dtype}')
        return
    if p.suffix.lower() == '.vtu':
        d = load_vtu(str(p))
        print(f'FILE={p}')
        print('TYPE=vtu')
        print('BACKEND=', d['backend'])
        print('points', d['points'].shape, d['points'].dtype)
        print('POINT DATA:')
        for k,a in d['point_data'].items():
            print(f'{k:60s} shape={a.shape} dtype={a.dtype}')
        return
    raise ValueError('Only .vtu and .npz are supported')

if __name__ == '__main__':
    main()
