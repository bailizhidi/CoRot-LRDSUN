from __future__ import annotations

import math
import numpy as np

from .physics import mises_from_s4_numpy


class MetricAccumulator:
    def __init__(self):
        self.sse_S = np.zeros(4, np.float64); self.sae_S = np.zeros(4, np.float64); self.max_S = np.zeros(4, np.float64); self.nS = 0
        self.sse_PE = np.zeros(4, np.float64); self.sae_PE = np.zeros(4, np.float64); self.max_PE = np.zeros(4, np.float64); self.nPE = 0
        self.sse_LE = np.zeros(4, np.float64); self.sae_LE = np.zeros(4, np.float64); self.max_LE = np.zeros(4, np.float64); self.nLE = 0
        self.sse_P = 0.0; self.sae_P = 0.0; self.max_P = 0.0; self.nP = 0
        self.sse_VM = 0.0; self.sae_VM = 0.0; self.max_VM = 0.0; self.nVM = 0
        self.tp = self.fp = self.fn = self.tn = 0

    def update(self, pred_state, true_state, pred_le, true_le, gate_prob=None, true_delta_peeq=None, plastic_threshold=1e-10):
        p = np.asarray(pred_state); y = np.asarray(true_state)
        dS = p[:, :4] - y[:, :4]; dPE = p[:, 4:8] - y[:, 4:8]; dP = p[:, 8:9] - y[:, 8:9]; dLE = np.asarray(pred_le) - np.asarray(true_le)
        pvm = mises_from_s4_numpy(p[:, :4]); yvm = mises_from_s4_numpy(y[:, :4]); dvm = pvm-yvm
        self.sse_S += (dS*dS).sum(0); self.sae_S += np.abs(dS).sum(0); self.max_S = np.maximum(self.max_S, np.max(np.abs(dS), axis=0)); self.nS += dS.shape[0]
        self.sse_PE += (dPE*dPE).sum(0); self.sae_PE += np.abs(dPE).sum(0); self.max_PE = np.maximum(self.max_PE, np.max(np.abs(dPE), axis=0)); self.nPE += dPE.shape[0]
        self.sse_LE += (dLE*dLE).sum(0); self.sae_LE += np.abs(dLE).sum(0); self.max_LE = np.maximum(self.max_LE, np.max(np.abs(dLE), axis=0)); self.nLE += dLE.shape[0]
        self.sse_P += float((dP*dP).sum()); self.sae_P += float(np.abs(dP).sum()); self.max_P = max(self.max_P, float(np.max(np.abs(dP))) if dP.size else 0.0); self.nP += dP.size
        self.sse_VM += float((dvm*dvm).sum()); self.sae_VM += float(np.abs(dvm).sum()); self.max_VM = max(self.max_VM, float(np.max(np.abs(dvm))) if dvm.size else 0.0); self.nVM += dvm.size
        if gate_prob is not None and true_delta_peeq is not None:
            pa = np.asarray(gate_prob).reshape(-1) >= 0.5
            ya = np.asarray(true_delta_peeq).reshape(-1) > float(plastic_threshold)
            self.tp += int(np.sum(pa & ya)); self.fp += int(np.sum(pa & ~ya)); self.fn += int(np.sum(~pa & ya)); self.tn += int(np.sum(~pa & ~ya))

    def as_dict(self):
        rmseS=np.sqrt(self.sse_S/max(self.nS,1)); maeS=self.sae_S/max(self.nS,1)
        rmsePE=np.sqrt(self.sse_PE/max(self.nPE,1)); maePE=self.sae_PE/max(self.nPE,1)
        rmseLE=np.sqrt(self.sse_LE/max(self.nLE,1)); maeLE=self.sae_LE/max(self.nLE,1)
        precision=self.tp/max(self.tp+self.fp,1); recall=self.tp/max(self.tp+self.fn,1); f1=2*precision*recall/max(precision+recall,1e-12)
        return {
            "rmse_S_MPa": float(np.sqrt(np.mean(self.sse_S/max(self.nS,1)))),
            "rmse_S_components_MPa": rmseS.tolist(), "mae_S_components_MPa": maeS.tolist(), "max_abs_S_components_MPa": self.max_S.tolist(),
            "rmse_PE": float(np.sqrt(np.mean(self.sse_PE/max(self.nPE,1)))),
            "rmse_PE_components": rmsePE.tolist(), "mae_PE_components": maePE.tolist(), "max_abs_PE_components": self.max_PE.tolist(),
            "rmse_PEEQ": math.sqrt(self.sse_P/max(self.nP,1)), "mae_PEEQ": self.sae_P/max(self.nP,1), "max_abs_PEEQ": self.max_P,
            "rmse_LE": float(np.sqrt(np.mean(self.sse_LE/max(self.nLE,1)))),
            "rmse_LE_components": rmseLE.tolist(), "mae_LE_components": maeLE.tolist(), "max_abs_LE_components": self.max_LE.tolist(),
            "rmse_S_mises_MPa": math.sqrt(self.sse_VM/max(self.nVM,1)), "mae_S_mises_MPa": self.sae_VM/max(self.nVM,1), "max_abs_S_mises_MPa": self.max_VM,
            "plastic_gate_precision": precision, "plastic_gate_recall": recall, "plastic_gate_f1": f1,
            "plastic_gate_counts": {"tp":self.tp,"fp":self.fp,"fn":self.fn,"tn":self.tn},
        }
