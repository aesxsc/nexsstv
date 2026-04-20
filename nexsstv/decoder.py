import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexsstv.config import Config
from nexsstv.fec.fountain import StripeFramer
from nexsstv.fec.robust import ChannelCodec
from nexsstv.image.progressive import ImageProcessor
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble


def _bandpass(audio, fs, f_min, f_max):
    nyq = 0.5 * fs
    lo = max(10.0 / nyq, (f_min - 180.0) / nyq)
    hi = min(0.995, (f_max + 180.0) / nyq)
    if hi <= lo:
        return audio
    sos = butter(4, [lo, hi], btype="bandpass", output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def _find_sync_in_window(audio, template, start, stop):
    start = max(0, int(start))
    stop = min(len(audio), int(stop))
    if stop - start < len(template):
        return None, 0.0
    idx, conf = Preamble.detect(audio[start:stop], template)
    return start + idx, conf


def _decode_frame(audio, frame_start, modem, codec, symbol_len, frame_capacity):
    ptr = int(frame_start)
    modem.demodulate_symbol(audio[ptr : ptr + symbol_len], is_pilot=True)
    ptr += symbol_len

    rx_bits = []
    for _ in range(Config.SYMBOLS_PER_FRAME):
        diff = modem.demodulate_symbol(audio[ptr : ptr + symbol_len])
        ptr += symbol_len
        rx_bits.extend(modem.dqpsk_demap(diff))

    coded_bits = np.array(rx_bits[:frame_capacity], dtype=np.uint8)
    deint = codec.deinterleave(coded_bits, depth=Config.INTERLEAVER_DEPTH)
    decoded_bits = codec.viterbi_decode(deint)
    usable = (len(decoded_bits) // 8) * 8
    if usable <= 0:
        return None

    byte_data = np.packbits(decoded_bits[:usable]).tobytes()
    return StripeFramer.unpack_stripe(byte_data)


def main():
    parser = argparse.ArgumentParser(description="NexSSTV NG mode decoder")
    parser.add_argument("input_audio", help="Path to input audio")
    parser.add_argument("output_image", help="Path to output image")
    parser.add_argument("--mode", choices=["normal", "ultra", "classic"], default="normal")
    parser.add_argument("--sync-threshold", type=float, default=0.30, help="Preamble detection threshold")
    parser.add_argument("--prefilter", action="store_true", help="Enable band-pass prefilter for noisy acoustic recordings")
    args = parser.parse_args()

    fs, audio = wavfile.read(args.input_audio)
    if audio.ndim > 1:
        audio = np.mean(audio.astype(np.float32), axis=1)
    else:
        audio = audio.astype(np.float32)

    if np.max(np.abs(audio)) > 1.2:
        audio /= 32767.0

    params = Config.get_mode_params(args.mode)
    if args.prefilter:
        audio = _bandpass(audio, fs, params["f_min"], params["f_max"])
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    modem = Modem(Config.FS, params["f_min"], params["f_max"], params["n_subcarriers"], Config.CP_RATIO)
    codec = ChannelCodec(rate=Config.CODING_RATE)
    preamble = Preamble(samples_per_bit=20)
    sync_template = preamble.generate_signal(Config.FS, params["f_min"], params["f_max"])

    symbol_len = modem.fft_size + int(modem.fft_size * modem.cp_ratio)
    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    frame_capacity = bits_per_symbol * Config.SYMBOLS_PER_FRAME
    frame_data_len = symbol_len * (1 + Config.SYMBOLS_PER_FRAME)
    frame_len = len(sync_template) + frame_data_len

    print(f"Decoding NG-SSTV stream ({args.mode})...")

    # 1) Acquire first frame with a single broad search.
    first_sync, first_conf = _find_sync_in_window(audio, sync_template, 0, min(len(audio), fs * 3))
    if first_sync is None or first_conf < args.sync_threshold:
        print("Failed to find frame sync. Try --prefilter or lower --sync-threshold.")
        return

    # 2) Decode on a deterministic stride, with a small local correction window.
    jitter = int(symbol_len // 2)
    received_stripes = {}
    seen_packets = 0
    bad_packets = 0

    expected_sync = first_sync
    max_frames = max(1, (len(audio) - first_sync) // max(1, frame_len)) + 2

    for _ in range(max_frames):
        if expected_sync + len(sync_template) + frame_data_len > len(audio):
            break

        local_sync, local_conf = _find_sync_in_window(
            audio,
            sync_template,
            expected_sync - jitter,
            expected_sync + jitter + len(sync_template),
        )

        if local_sync is None or local_conf < args.sync_threshold:
            # fallback: broader search across roughly one frame duration
            local_sync, local_conf = _find_sync_in_window(
                audio,
                sync_template,
                expected_sync - frame_len // 3,
                expected_sync + frame_len,
            )

        if local_sync is None or local_conf < args.sync_threshold:
            expected_sync += frame_len
            continue

        frame_start = local_sync + len(sync_template)
        if frame_start + frame_data_len > len(audio):
            break

        pkt = _decode_frame(audio, frame_start, modem, codec, symbol_len, frame_capacity)
        seen_packets += 1

        if pkt is None:
            bad_packets += 1
        else:
            stripe_id = pkt["stripe_id"]
            if stripe_id not in received_stripes:
                stripe_img = ImageProcessor.decode_stripe(pkt["payload"])
                if stripe_img is not None:
                    received_stripes[stripe_id] = stripe_img
                    if len(received_stripes) % 10 == 0 or len(received_stripes) == 1:
                        print(f"   Decoded {len(received_stripes)}/{Config.NUM_STRIPES} stripes...")
                else:
                    bad_packets += 1

        expected_sync = local_sync + frame_len
        if len(received_stripes) >= Config.NUM_STRIPES:
            break

    if received_stripes:
        final_img = ImageProcessor.merge_stripes(received_stripes)
        final_img.save(args.output_image)
        print(f"Success: saved {args.output_image} ({len(received_stripes)}/{Config.NUM_STRIPES} stripes)")
        print(f"Packets: good={seen_packets - bad_packets}, bad={bad_packets}")
    else:
        print("Failed to decode any stripes. Try --prefilter or lower --sync-threshold.")


if __name__ == "__main__":
    main()
