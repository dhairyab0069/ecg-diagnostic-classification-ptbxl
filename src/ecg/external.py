"""External test set: the Georgia 12-lead ECG database (PhysioNet/CinC Challenge 2021).

Georgia labels are SNOMED CT codes. Each code is mapped to the PTB-XL statement it matches and
takes that statement's superclass, so a record is labelled by the same rules as PTB-XL:
- codes whose PTB-XL equivalent is only a form or rhythm statement (T-wave abnormal, Q wave
  abnormal, ST depression/elevation, AF, ...) carry no diagnostic label;
- Georgia has no "normal ECG" code, so NORM means sinus rhythm and no other code at all;
- records with no diagnostic label are dropped, as in load_labels.
"""
import numpy as np
from scipy.io import loadmat
from scipy.signal import resample_poly

from ecg import CLASSES

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]  # same order as PTB-XL
SINUS_RHYTHM = "426783006"

# SNOMED CT code -> superclass, with the PTB-XL statement it corresponds to
SNOMED_TO_CLASS = {
    "164865005": "MI",     # myocardial infarction
    "428750005": "STTC",   # nonspecific ST-T abnormality -> NST_
    "164930006": "STTC",   # ST interval abnormal -> NST_
    "55930002": "STTC",    # ST changes -> NST_
    "425623009": "STTC",   # lateral ischaemia -> ISCAL
    "425419005": "STTC",   # inferior ischaemia -> ISCIN
    "426434006": "STTC",   # anterior ischemia -> ISCAN
    "413444003": "STTC",   # acute myocardial ischemia -> ISC_
    "111975006": "STTC",   # prolonged QT interval -> LNGQT
    "59118001": "CD",      # right bundle branch block -> CRBBB
    "713427006": "CD",     # complete right bundle branch block -> CRBBB
    "713426002": "CD",     # incomplete right bundle branch block -> IRBBB
    "164909002": "CD",     # left bundle branch block -> CLBBB
    "251120003": "CD",     # incomplete left bundle branch block -> ILBBB
    "6374002": "CD",       # bundle branch block (unspecified)
    "270492004": "CD",     # 1st degree AV block -> 1AVB
    "195042002": "CD",     # 2nd degree AV block -> 2AVB
    "27885002": "CD",      # complete heart block -> 3AVB
    "233917008": "CD",     # AV block (unspecified)
    "445118002": "CD",     # left anterior fascicular block -> LAFB
    "445211001": "CD",     # left posterior fascicular block -> LPFB
    "698252002": "CD",     # nonspecific intraventricular conduction disorder -> IVCD
    "74390002": "CD",      # Wolff-Parkinson-White pattern -> WPW
    "164873001": "HYP",    # left ventricular hypertrophy -> LVH
    "89792004": "HYP",     # right ventricular hypertrophy -> RVH
    "266249003": "HYP",    # ventricular hypertrophy
    "67741000119109": "HYP",  # left atrial enlargement -> LAO/LAE
    "253352002": "HYP",    # left atrial abnormality -> LAO/LAE
    "253339007": "HYP",    # right atrial abnormality -> RAO/RAE
    "195126007": "HYP",    # atrial hypertrophy
}


def parse_header(text):
    """Return (sampling rate, number of samples, lead names, gains per mV, Dx codes) from a .hea file."""
    lines = [line for line in text.splitlines() if line.strip()]
    _, n_sig, fs, n = lines[0].split()[:4]
    signals = [line.split() for line in lines[1:1 + int(n_sig)]]
    gains = [float(s[2].split("(")[0].split("/")[0]) for s in signals]
    leads = [s[-1] for s in signals]
    dx = []
    for line in lines:
        if line.startswith("# Dx:"):
            dx = [c.strip() for c in line[len("# Dx:"):].split(",") if c.strip()]
    return int(float(fs)), int(n), leads, gains, dx


def labels_from_dx(dx):
    """Multi-hot vector over CLASSES, or None if the record has no diagnostic label."""
    classes = {SNOMED_TO_CLASS[c] for c in dx if c in SNOMED_TO_CLASS}
    if set(dx) == {SINUS_RHYTHM}:
        classes.add("NORM")
    if not classes:
        return None
    return np.array([c in classes for c in CLASSES], dtype=np.float32)


def load_record(mat_path, gains):
    """Signal in mV, shape (T, 12)."""
    return (loadmat(mat_path)["val"] / np.asarray(gains)[:, None]).T.astype(np.float32)


def to_100hz(x, fs):
    """Resample (T, 12) to 100 Hz with an anti-aliasing filter."""
    if fs == 100:
        return x
    g = np.gcd(100, fs)
    return resample_poly(x, 100 // g, fs // g, axis=0).astype(np.float32)
