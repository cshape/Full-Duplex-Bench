"""
Combine input.wav and output.wav into a stereo conversation file.
Left channel = user input, Right channel = model output.
Also creates a mono mix version.

Usage:
  python combine_audio.py --root_dir ../dataset/v1.0/pause_handling_10_inworld
"""

import argparse
import os
from glob import glob

import numpy as np
import soundfile as sf


def combine_folder(folder):
    input_path = os.path.join(folder, "input.wav")
    output_path = os.path.join(folder, "output.wav")

    if not os.path.exists(input_path) or not os.path.exists(output_path):
        return

    inp, sr_in = sf.read(input_path, dtype="float32")
    out, sr_out = sf.read(output_path, dtype="float32")

    # Make mono
    if inp.ndim > 1:
        inp = inp.mean(axis=1)
    if out.ndim > 1:
        out = out.mean(axis=1)

    # Resample both to 48kHz for compatibility
    target_sr = 48000
    if sr_in != target_sr:
        from scipy.signal import resample as scipy_resample
        inp = scipy_resample(inp, int(len(inp) * target_sr / sr_in)).astype(np.float32)
    if sr_out != target_sr:
        from scipy.signal import resample as scipy_resample
        out = scipy_resample(out, int(len(out) * target_sr / sr_out)).astype(np.float32)
    sr_in = target_sr

    # Pad to same length
    max_len = max(len(inp), len(out))
    inp_padded = np.zeros(max_len, dtype=np.float32)
    out_padded = np.zeros(max_len, dtype=np.float32)
    inp_padded[:len(inp)] = inp
    out_padded[:len(out)] = out

    # Stereo: left=user, right=model
    stereo = np.stack([inp_padded, out_padded], axis=1)
    stereo_path = os.path.join(folder, "conversation_stereo.wav")
    sf.write(stereo_path, stereo, sr_in)

    # Mono mix (both overlaid)
    mono = (inp_padded + out_padded) / 2.0
    mono = np.clip(mono, -1.0, 1.0)
    mono_path = os.path.join(folder, "conversation_mono.wav")
    sf.write(mono_path, mono, sr_in)

    print(f"  {os.path.basename(folder)}: stereo={max_len/sr_in:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", required=True)
    args = parser.parse_args()

    folders = sorted(
        d for d in glob(os.path.join(args.root_dir, "*"))
        if os.path.isdir(d)
    )

    for folder in folders:
        combine_folder(folder)

    print(f"\nDone. Created conversation_stereo.wav and conversation_mono.wav in each folder.")


if __name__ == "__main__":
    main()
