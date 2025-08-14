#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import os
import numpy as np
import librosa
from scipy.signal import correlate

# ---------- Répertoires ----------
CLEAN_DIR    = Path("audio/clean")
NOISY_DIR    = Path("audio/noisy")
ENHANCED_DIR = Path("audio/enhanced")

# Sorties
OUT_README = Path("README.md")
WRITE_GHPAGES = True
OUT_PAGES = Path("docs/index.md")  # si WRITE_GHPAGES=True

# Audio
TARGET_SR = 16000       # None = sr natif, sinon unifie (conseillé: 16000)
MAX_ALIGN_SEC = 0.25    # fenêtre de recherche de décalage ±250 ms
EXTS = {".wav", ".mp3", ".ogg"}

# ---------- Utils ----------
def posix_rel(p: Path, start: Path) -> str:
    return Path(os.path.relpath(p, start)).as_posix()

def load_mono(path: Path, sr=TARGET_SR):
    y, s = librosa.load(str(path), sr=sr, mono=True)
    return y, s

def align_by_xcorr(ref, sig, sr, max_shift_sec=0.25):
    if len(ref) == 0 or len(sig) == 0: return ref, sig
    k = int(max_shift_sec * sr) if sr else None
    if k:
        w = min(len(ref), len(sig), max(4096, 4*k))
        r, g = ref[:w], sig[:w]
        corr = correlate(g, r, mode="full")
        lags = np.arange(-len(r)+1, len(g))
        mask = (lags >= -k) & (lags <= k)
        lag = 0 if not np.any(mask) else lags[mask][np.argmax(corr[mask])]
    else:
        corr = correlate(sig, ref, mode="full")
        lags = np.arange(-len(ref)+1, len(sig))
        lag = lags[np.argmax(corr)]
    if lag > 0:   sig = sig[lag:]
    elif lag < 0: sig = np.pad(sig, (abs(lag), 0))
    L = min(len(ref), len(sig))
    return ref[:L], sig[:L]

def snr_db(clean, est, eps=1e-12):
    """SNR classique (sensible au gain) : 10log10(||s||^2 / ||s-ŝ||^2)."""
    s = np.asarray(clean); shat = np.asarray(est)
    if len(s) < 2 or len(shat) < 2: return np.nan
    e = s - shat
    num, den = np.sum(s**2), np.sum(e**2)
    if den <= eps: return float("inf")
    if num <= eps: return np.nan
    return 10*np.log10(num/den)

def si_snr_db(clean, est, eps=1e-12):
    """Scale-Invariant SNR, robuste au gain et au DC offset (comme dans la lit.)"""
    s = clean - np.mean(clean)
    sh = est - np.mean(est)
    if len(s) < 2 or len(sh) < 2: return np.nan
    s_energy = np.sum(s**2) + eps
    proj = (np.dot(sh, s) / s_energy) * s      # projection de sh sur s
    e_noise = sh - proj
    return 10*np.log10((np.sum(proj**2)+eps) / (np.sum(e_noise**2)+eps))

def audio_tag(rel_path: str):
    ext = Path(rel_path).suffix.lower()
    mime = {"wav":"audio/wav","mp3":"audio/mpeg","ogg":"audio/ogg"}.get(ext[1:], "audio/wav")
    return f'<audio controls preload="none"><source src="{rel_path}" type="{mime}"></audio>'

def list_names(folder: Path):
    return {p.name for p in folder.iterdir() if p.is_file() and p.suffix.lower() in EXTS}

# ---------- Main ----------
def main():
    root = Path.cwd()
    for d in (CLEAN_DIR, NOISY_DIR, ENHANCED_DIR):
        if not (root/d).is_dir():
            raise SystemExit(f"[ERREUR] Dossier manquant: {root/d}")

    # fichiers communs
    common = sorted(list_names(root/CLEAN_DIR) & list_names(root/NOISY_DIR) & list_names(root/ENHANCED_DIR))
    if not common:
        raise SystemExit("[ERREUR] Aucun triplet commun trouvé (clean/noisy/enhanced).")

    lines = []
    lines.append("# Comparaison audio (SNR & SI-SNR)\n")
    lines.append("> Les lecteurs audio fonctionnent dans GitHub/VS Code (cliquez ▶️). SNR = 10log10(||s||^2/||s-ŝ||^2), SI-SNR = invariant au gain.\n")

    for name in common:
        cp = root/CLEAN_DIR/name
        np_ = root/NOISY_DIR/name
        ep = root/ENHANCED_DIR/name

        # charge (sr unifié)
        clean, sr = load_mono(cp, sr=TARGET_SR)
        noisy, _  = load_mono(np_, sr=sr)
        enh, _    = load_mono(ep,  sr=sr)

        # aligne
        c_ref, n_al = align_by_xcorr(clean, noisy, sr, MAX_ALIGN_SEC)
        _,     e_al = align_by_xcorr(clean, enh,  sr, MAX_ALIGN_SEC)

        # métriques
        m_snr_n  = snr_db(c_ref, n_al)
        m_sisnr_n= si_snr_db(c_ref, n_al)
        m_snr_e  = snr_db(c_ref, e_al)
        m_sisnr_e= si_snr_db(c_ref, e_al)

        # tableau
        lines.append(f"## {name}\n")
        headers = ["Référence (Clean)", "Bruitée (Noisy)", "Rehaussée (Enhanced)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")

        rel_c = posix_rel(cp, root)
        rel_n = posix_rel(np_, root)
        rel_e = posix_rel(ep,  root)

        row_players = [audio_tag(rel_c), audio_tag(rel_n), audio_tag(rel_e)]
        def fmt(x): 
            return "∞ dB" if np.isinf(x) else ("NA" if np.isnan(x) else f"{x:.2f} dB")
        row_snr =   ["**SNR : ∞ dB**", f"**SNR : {fmt(m_snr_n)}**",   f"**SNR : {fmt(m_snr_e)}**"]
        row_sisnr = ["**SI-SNR : ∞ dB**", f"**SI-SNR : {fmt(m_sisnr_n)}**", f"**SI-SNR : {fmt(m_sisnr_e)}**"]

        lines.append("| " + " | ".join(row_players) + " |")
        lines.append("| " + " | ".join(row_snr) + " |")
        lines.append("| " + " | ".join(row_sisnr) + " |")
        lines.append("")

    OUT_README.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] README généré → {OUT_README.resolve()}")

    if WRITE_GHPAGES:
        OUT_PAGES.parent.mkdir(parents=True, exist_ok=True)
        OUT_PAGES.write_text("\n".join(lines), encoding="utf-8")
        print(f"[OK] Page GitHub Pages → {OUT_PAGES.resolve()} (active-la dans Settings → Pages)")

if __name__ == "__main__":
    main()
