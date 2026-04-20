import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexsstv.config import Config
from nexsstv.fec.fountain import RedundancyPlanner, StripeFramer
from nexsstv.fec.robust import ChannelCodec
from nexsstv.image.progressive import ImageProcessor
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble


def _append_payload(audio_signal, modem, coded_bits, bits_per_symbol):
    audio_signal.append(modem.modulate_symbol(None, is_pilot=True))
    symbols_sent = 0
    for j in range(0, len(coded_bits), bits_per_symbol):
        if symbols_sent >= Config.SYMBOLS_PER_FRAME:
            break
        chunk = coded_bits[j : j + bits_per_symbol]
        if len(chunk) < bits_per_symbol:
            chunk = np.concatenate([chunk, np.zeros(bits_per_symbol - len(chunk), dtype=np.uint8)])
        audio_signal.append(modem.modulate_symbol(modem.dqpsk_map(chunk)))
        symbols_sent += 1

    while symbols_sent < Config.SYMBOLS_PER_FRAME:
        audio_signal.append(modem.modulate_symbol(np.ones(modem.n_subcarriers, dtype=np.complex128)))
        symbols_sent += 1


def main():
    parser = argparse.ArgumentParser(description="NexSSTV NG mode encoder")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_wav", help="Path to output WAV file")
    parser.add_argument("--mode", choices=["normal", "ultra", "classic"], default="normal")
    parser.add_argument("--quality", type=int, default=Config.DEFAULT_QUALITY, help="WebP quality")
    parser.add_argument("--max-seconds", type=float, default=Config.MAX_TX_SECONDS, help="Transmission budget")
    parser.add_argument("--redundancy", type=float, default=Config.REDUNDANCY_FACTOR, help="Stripe redundancy factor")
    args = parser.parse_args()

    params = Config.get_mode_params(args.mode)
    modem = Modem(Config.FS, params["f_min"], params["f_max"], params["n_subcarriers"], Config.CP_RATIO)
    codec = ChannelCodec(rate=Config.CODING_RATE)
    preamble = Preamble(samples_per_bit=20)
    sync_burst = preamble.generate_signal(Config.FS, params["f_min"], params["f_max"])

    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    frame_capacity = bits_per_symbol * Config.SYMBOLS_PER_FRAME
    max_raw_bits = codec.max_input_bits_for_capacity(frame_capacity)
    max_frame_bytes = max(24, min(Config.MAX_STRIPE_BYTES, max_raw_bits // 8 - 16))

    print(f"Encoding {Config.TARGET_RES[0]}x{Config.TARGET_RES[1]} image into NG-SSTV ({args.mode})...")
    stripes = ImageProcessor.encode_image(args.input_image, quality=args.quality, max_bytes=max_frame_bytes)

    schedule = RedundancyPlanner.build_schedule(Config.NUM_STRIPES, redundancy_factor=args.redundancy)
    est_seconds, _ = Config.estimate_seconds(args.mode, frame_count=len(schedule))
    while est_seconds > args.max_seconds and len(schedule) > Config.NUM_STRIPES:
        schedule.pop()
        est_seconds, _ = Config.estimate_seconds(args.mode, frame_count=len(schedule))

    audio_signal = [np.zeros(Config.LEAD_IN_SAMPLES, dtype=np.float32)]
    print(f"Planned {len(schedule)} frames (~{est_seconds:.1f}s, budget {args.max_seconds:.1f}s)")

    for frame_idx, (stripe_id, repeat_idx) in enumerate(schedule):
        packet = StripeFramer.pack_stripe(
            stripe_id,
            stripes[stripe_id],
            repeat_idx=repeat_idx,
            total_stripes=Config.NUM_STRIPES,
        )
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        if len(bits) > max_raw_bits:
            raise RuntimeError(
                f"Stripe {stripe_id} too large ({len(bits)} > {max_raw_bits} bits). Try lower --quality."
            )
        padded = np.concatenate([bits, np.zeros(max_raw_bits - len(bits), dtype=np.uint8)])
        coded = codec.encode(padded)
        coded = codec.interleave(coded, depth=Config.INTERLEAVER_DEPTH)
        coded = coded[:frame_capacity]

        audio_signal.append(sync_burst)
        _append_payload(audio_signal, modem, coded, bits_per_symbol)

        if (frame_idx + 1) % 20 == 0 or frame_idx == len(schedule) - 1:
            print(f"   Modulated {frame_idx + 1}/{len(schedule)} frames...")

    full_audio = np.concatenate(audio_signal).astype(np.float32)
    # Gentle soft limiter for speaker-safe playback.
    full_audio = np.tanh(1.3 * full_audio)
    full_audio = 0.95 * full_audio / (np.max(np.abs(full_audio)) + 1e-9)

    wavfile.write(args.output_wav, Config.FS, (full_audio * 32767).astype(np.int16))
    duration = len(full_audio) / Config.FS
    print(f"Done: {args.output_wav} ({duration:.1f}s)")


if __name__ == "__main__":
    main()
