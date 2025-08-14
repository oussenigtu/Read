#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from pathlib import Path
import numpy as np
import librosa
from scipy.signal import correlate

# ========= PARAMÈTRES À ADAPTER =========
# Dossiers relatifs à la racine du repo
CLEAN_DIR     = Path("audio/clean")
NOISY_DIR     = Path("audio/noisy")
ENH_PATTERNS  = [Path("audio/enhanced1"),
                 Path("audio/enhanced2"),
                 Path("audio/enhanced3"),
                 Path("audio/enhanced4"),
                 Path("audio/enhanced5")]  # adapte/ajoute si besoin

# Extensions acceptées (mets .mp3/.ogg si tu veux)
EXTS = (".wav", ".mp3", ".ogg")

# Fichier de sortie
OUT_MD = Path("README.md")

# Paramètres audio
TARGET_SR = None   # None = ne pas resampler; sinon mets 16000, 48000, etc.
MAX_ALIGN_SEC = 0.25  # fenêtre de recherche de décalage ±250 ms
# =======================================


def posix_relpath(p: Path, start: Path) -> str:
    """Chemin relatif avec / (compatible GitHub/VS Code)."""
    return Path(os.path.relpath(p, start)).as_posix()


def load_audio(path: Path, sr=TARGET_SR):
    """Charge mono à sr (ou sr natif si None). Retourne (y, sr)."""
    y, s = librosa.load(str(path), sr=sr, mono=True)
    return y, s


def align_by_xcorr(ref, sig, sr, max_shift_sec=0.25):
    """Aligne `sig` sur `ref` via corrélation croisée. Limite le décalage max."""
    if len(ref) == 0 or len(sig) == 0:
        return ref, sig
    max_shift = int(max_shift_sec * sr) if sr else None

    # on tronque pour limiter la taille de la xcorr si demandé
    if max_shift:
        # fabrique des fenêtres centrales pour chercher le lag
        w = min(len(ref), len(sig), max(4096, 4*max_shift))
        ref_c = ref[:w]
        sig_c = sig[:w]
        corr = correlate(sig_c, ref_c, mode="full")
        lags = np.arange(-len(ref_c)+1, len(sig_c))
        # borne le lag recherché
        mask = (lags >= -max_shift) & (lags <= max_shift)
        if not np.any(mask):
            lag = 0
        else:
            lag = lags[mask][np.argmax(corr[mask])]
    else:
        corr = correlate(sig, ref, mode="full")
        lags = np.arange(-len(ref)+1, len(sig))
        lag = lags[np.argmax(corr)]

    # applique le décalage (sig -> ref)
    if lag > 0:
        sig = sig[lag:]
    elif lag < 0:
        sig = np.pad(sig, (abs(lag), 0))
    # coupe à la même longueur
    L = min(len(ref), len(sig))
    return ref[:L], sig[:L]


def snr_db(clean, test, eps=1e-12):
    """SNR = 10 log10 (||clean||^2 / ||clean - test||^2). ∞ si identique."""
    c = np.asarray(clean)
    t = np.asarray(test)
    if len(c) < 2 or len(t) < 2:
        return np.nan
    noise = c - t
    num = np.sum(c**2)
    den = np.sum(noise**2)
    if den <= eps:
        return float("inf")
    if num <= eps:
        return np.nan  # clean quasi nul
    return 10.0 * np.log10(num / den)


def find_files(base: Path):
    """Retourne la liste triée des fichiers (nom) présents dans CLEAN_DIR avec extensions EXTS."""
    files = sorted([p.name for p in base.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
    return files


def format_snr(v):
    if np.isnan(v):
        return "NA"
    if np.isinf(v):
        return "&infin; dB"
    return f"{v:.2f} dB"


def audio_tag(path_rel: str, mime: str = None):
    """Balise <audio> HTML minimaliste pour GitHub/VS Code."""
    if mime is None:
        ext = Path(path_rel).suffix.lower()
        if ext == ".wav":
            mime = "audio/wav"
        elif ext == ".mp3":
            mime = "audio/mpeg"
        elif ext == ".ogg":
            mime = "audio/ogg"
        else:
            mime = "audio/wav"
    return f'<audio controls preload="none"><source src="{path_rel}" type="{mime}"></audio>'


def main():
    repo_root = Path.cwd()
    clean_dir = repo_root / CLEAN_DIR
    noisy_dir = repo_root / NOISY_DIR
    enh_dirs  = [repo_root / p for p in ENH_PATTERNS]

    # Vérifs basiques
    if not clean_dir.is_dir():
        raise SystemExit(f"[ERREUR] Dossier clean introuvable : {clean_dir}")
    if not noisy_dir.is_dir():
        raise SystemExit(f"[ERREUR] Dossier noisy introuvable : {noisy_dir}")
    for d in enh_dirs:
        if not d.is_dir():
            print(f"[AVERTISSEMENT] Dossier enhanced absent : {d}")

    names = find_files(clean_dir)
    if not names:
        raise SystemExit(f"[ERREUR] Aucun fichier audio trouvé dans {clean_dir}")

    lines = []
    lines.append("## Comparaison audio avec SNR\n")

    for name in names:
        clean_p = clean_dir / name
        noisy_p = noisy_dir / name

        if not noisy_p.is_file():
            print(f"[WARN] Manque noisy pour {name}, on saute.")
            continue

        # 5 enhanced max (ignore ceux manquants)
        enhanced_ps = [d / name for d in enh_dirs if (d / name).is_file()]
        if not enhanced_ps:
            print(f"[WARN] Pas d'enhanced pour {name}.")
        # charge clean + noisy
        clean, sr = load_audio(clean_p, sr=TARGET_SR)
        noisy, _  = load_audio(noisy_p, sr=sr)

        # aligne noisy
        cref, n_aligned = align_by_xcorr(clean, noisy, sr, MAX_ALIGN_SEC)
        snr_noisy = snr_db(cref, n_aligned)

        # prépare cellules audio + SNR
        row_players = []
        row_snr     = []

        # CLEAN
        clean_rel = posix_relpath(clean_p, repo_root)
        row_players.append(audio_tag(clean_rel))
        row_snr.append("**SNR : &infin; dB**")

        # NOISY
        noisy_rel = posix_relpath(noisy_p, repo_root)
        row_players.append(audio_tag(noisy_rel))
        row_snr.append(f"**SNR : {format_snr(snr_noisy)}**")

        # ENHANCED
        for ep in enhanced_ps[:5]:
            enh, _ = load_audio(ep, sr=sr)
            cref2, e_aligned = align_by_xcorr(clean, enh, sr, MAX_ALIGN_SEC)
            snr_enh = snr_db(cref2, e_aligned)
            enh_rel = posix_relpath(ep, repo_root)
            row_players.append(audio_tag(enh_rel))
            row_snr.append(f"**SNR : {format_snr(snr_enh)}**")

        # Complète jusqu'à 5 colonnes enhanced (si manquants)
        while len(row_players) < 2 + 5:  # 1 clean + 1 noisy + 5 enhanced
            row_players.append("—")
            row_snr.append("")

        # Titre + tableau
        lines.append(f"### {name}\n")
        header = ["Référence (Clean)", "Bruitée (Noisy)",
                  "Rehaussée #1", "Rehaussée #2", "Rehaussée #3", "Rehaussée #4", "Rehaussée #5"]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"]*len(header)) + "|")
        lines.append("| " + " | ".join(row_players) + " |")
        lines.append("| " + " | ".join(row_snr) + " |")
        lines.append("")  # ligne vide

    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] README généré -> {OUT_MD.resolve()}")


if __name__ == "__main__":
    main()
