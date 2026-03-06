"""
ASR transcription using AssemblyAI API (no GPU required).
Drop-in replacement for asr.py — produces identical output.json format.

Requires: pip install assemblyai
Set ASSEMBLYAI_API_KEY in environment or .env file.
"""

import os
import json
import argparse
from glob import glob

from dotenv import load_dotenv
load_dotenv()  # loads from .env in current directory

import assemblyai as aai
from tqdm import tqdm


def get_time_aligned_transcription(data_path, task):
    api_key = os.environ.get("ASSEMBLYAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set ASSEMBLYAI_API_KEY in your environment")

    aai.settings.api_key = api_key
    transcriber = aai.Transcriber()

    audio_paths = sorted(glob(f"{data_path}/*/output.wav"))
    if not audio_paths:
        print(f"No output.wav files found in {data_path}/*/")
        return

    for audio_path in tqdm(audio_paths):
        print(audio_path)

        offset = 0.0
        file_to_transcribe = audio_path

        if task == "user_interruption":
            import soundfile as sf
            import tempfile

            meta_path = audio_path.replace("output.wav", "interrupt.json")
            with open(meta_path, "r") as f:
                interrupt_meta = json.load(f)

            _, end_interrupt = interrupt_meta[0]["timestamp"]
            offset = end_interrupt

            waveform, sr = sf.read(audio_path)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            start_idx = int(end_interrupt * sr)
            waveform = waveform[start_idx:]

            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            sf.write(tmp.name, waveform, sr)
            file_to_transcribe = tmp.name

        transcript = transcriber.transcribe(file_to_transcribe)

        if file_to_transcribe != audio_path:
            os.unlink(file_to_transcribe)

        if transcript.status == aai.TranscriptStatus.error:
            print(f"  Error: {transcript.error}")
            continue

        chunks = []
        text = ""
        if transcript.words:
            for w in transcript.words:
                start_time = w.start / 1000.0 + offset  # ms -> seconds
                end_time = w.end / 1000.0 + offset
                word = w.text

                text += word + " "
                chunks.append({
                    "text": word,
                    "timestamp": [start_time, end_time],
                })

        output_dict = {
            "text": text.strip(),
            "chunks": chunks,
        }

        result_path = audio_path.replace("output.wav", "output.json")
        os.makedirs(os.path.dirname(result_path), exist_ok=True)
        with open(result_path, "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Transcribe using AssemblyAI (no GPU needed)"
    )
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument(
        "--task", type=str, default="full",
        choices=["full", "user_interruption"],
    )
    args = parser.parse_args()

    get_time_aligned_transcription(args.root_dir, args.task)
