#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import os
import numpy as np
import librosa
from scipy.signal import correlate

# ========= PARAMÈTRES (adapte si besoin) =========
CLEAN_DIR    = Path("audio/clean")
NOISY_DIR    = Path("audio/noisy")
ENHANCED_DIR = Path("audio/enhanced")

# Extensions acceptées
EXTS = {".wav", ".mp3", ".ogg"}

# Sortie Markdown
OUT_MD = Path("README.md")

# Audio
TARGET_SR = 16000      # force un SR commun (met None si tu veux garder le SR natif)
MAX_ALIGN_SEC = 0.25   # recherche de décalage ±250 ms
# ==================================================


def posix_relpath(p: Path, start: Path) -> str:
    """Chemin relatif avec / (compatible GitHub/VS Code)."""
    return Path(os.path.relpath(p, start)).as_posix()


def load_audio(path: Path, sr=TARGET_SR):
    y, s = librosa.load(str(path), sr=sr, mono=True)  # mono
    return y, s


def align_by_xcorr(ref, sig, sr, max_shift_sec=0.25):
    """Aligne sig -> ref par corrélation croisée, borne le décalage."""
    if len(ref) == 0 or len(sig) == 0:
        return ref, sig
    max_shift = int(max_shift_sec * sr) if sr else None

    if max_shift:  # lag borné
        w = min(len(ref), len(sig), max(4096, 4*max_shift))
        ref_c = ref[:w]
        sig_c = sig[:w]
        corr = correlate(sig_c, ref_c, mode="full")
        lags = np.arange(-len(ref_c)+1, len(sig_c))
        mask = (lags >= -max_shift) & (lags <= max_shift)
        lag = 0 if not np.any(mask) else lags[mask][np.argmax(corr[mask])]
    else:          # lag libre
        corr = correlate(sig, ref, mode="full")
        lags = np.arange(-len(ref)+1, len(sig))
        lag = lags[np.argmax(corr)]

    if lag > 0:
        sig = sig[lag:]
    elif lag < 0:
        sig = np.pad(sig, (abs(lag), 0))
    L = min(len(ref), len(sig))
    return ref[:L], sig[:L]


def snr_db(clean, test, eps=1e-12):
    """SNR = 10 log10 (||clean||^2 / ||clean-test||^2)."""
    c = np.asarray(clean); t = np.asarray(test)
    if len(c) < 2 or len(t) < 2:
        return np.nan
    noise = c - t
    num = np.sum(c**2)
    den = np.sum(noise**2)
    if den <= eps:
        return float("inf")
    if num <= eps:
        return np.nan
    return 10.0 * np.log10(num / den)


def list_names(folder: Path):
    return {p.name for p in folder.glob("*") if p.is_file() and p.suffix.lower() in EXTS}


def audio_tag(path_rel: str, mime: str = None):
    ext = Path(path_rel).suffix.lower()
    if mime is None:
        mime = {"wav": "audio/wav", "mp3": "audio/mpeg", "ogg": "audio/ogg"}.get(ext[1:], "audio/wav")
    return f'<audio controls preload="none"><source src="{path_rel}" type="{mime}"></audio>'


def main():
    root = Path.cwd()
    clean_dir, noisy_dir, enhanced_dir = root/CLEAN_DIR, root/NOISY_DIR, root/ENHANCED_DIR

    # Vérifs dossiers
    for d, name in [(clean_dir, "clean"), (noisy_dir, "noisy"), (enhanced_dir, "enhanced")]:
        if not d.is_dir():
            raise SystemExit(f"[ERREUR] Dossier {name} introuvable : {d}")

    # Intersection des noms présents dans les 3 dossiers
    common = sorted(list_names(clean_dir) & list_names(noisy_dir) & list_names(enhanced_dir))
    if not common:
        raise SystemExit("[ERREUR] Aucun fichier commun trouvé entre clean/noisy/enhanced.")

    lines = []
    lines.append("## Comparaison audio avec SNR\n")

    for name in common:
        clean_p = clean_dir / name
        noisy_p = noisy_dir / name
        enh_p   = enhanced_dir / name

        # Charge
        clean, sr = load_audio(clean_p, sr=TARGET_SR)
        noisy, _  = load_audio(noisy_p, sr=sr)
        enh, _    = load_audio(enh_p,   sr=sr)

        # Aligne
        cref, n_aligned = align_by_xcorr(clean, noisy, sr, MAX_ALIGN_SEC)
        _,    e_aligned = align_by_xcorr(clean, enh,   sr, MAX_ALIGN_SEC)

        # SNR
        snr_noisy = snr_db(cref, n_aligned)
        snr_enh   = snr_db(cref, e_aligned)

        # Markdown (un bloc par fichier)
        lines.append(f"### {name}\n")
        headers = ["Référence (Clean)", "Bruitée (Noisy)", "Rehaussée (Enhanced)"]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("|" + "|".join(["---"]*len(headers)) + "|")

        clean_rel = posix_relpath(clean_p, root)
        noisy_rel = posix_relpath(noisy_p, root)
        enh_rel   = posix_relpath(enh_p,   root)

        row_players = [
            audio_tag(clean_rel),
            audio_tag(noisy_rel),
            audio_tag(enh_rel)
        ]
        row_snrs = [
            "**SNR : &infin; dB**",
            f"**SNR : {('NA' if np.isnan(snr_noisy) else ('&infin; dB' if np.isinf(snr_noisy) else f'{snr_noisy:.2f} dB'))}**",
            f"**SNR : {('NA' if np.isnan(snr_enh)   else ('&infin; dB' if np.isinf(snr_enh)   else f'{snr_enh:.2f} dB'))}**",
        ]

        lines.append("| " + " | ".join(row_players) + " |")
        lines.append("| " + " | ".join(row_snrs) + " |")
        lines.append("")  # ligne vide

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] README généré -> {OUT_MD.resolve()}")


if __name__ == "__main__":
    main()
