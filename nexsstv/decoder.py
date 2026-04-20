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


def main():
    parser = argparse.ArgumentParser(description="NexSSTV NG mode decoder")
    parser.add_argument("input_audio", help="Path to input audio")
    parser.add_argument("output_image", help="Path to output image")
    parser.add_argument("--mode", choices=["normal", "ultra", "classic"], default="normal")
    parser.add_argument("--sync-threshold", type=float, default=0.34, help="Preamble detection threshold")
    args = parser.parse_args()

    fs, audio = wavfile.read(args.input_audio)
    if audio.ndim > 1:
        audio = np.mean(audio.astype(np.float32), axis=1)
    else:
        audio = audio.astype(np.float32)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
    if np.max(np.abs(audio)) > 1.2:
        audio /= 32767.0

    params = Config.get_mode_params(args.mode)
    audio = _bandpass(audio, fs, params["f_min"], params["f_max"])
    audio /= np.max(np.abs(audio)) + 1e-9

    modem = Modem(Config.FS, params["f_min"], params["f_max"], params["n_subcarriers"], Config.CP_RATIO)
    codec = ChannelCodec(rate=Config.CODING_RATE)
    preamble = Preamble(samples_per_bit=20)
    sync_template = preamble.generate_signal(Config.FS, params["f_min"], params["f_max"])

    symbol_len = modem.fft_size + int(modem.fft_size * modem.cp_ratio)
    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    frame_capacity = bits_per_symbol * Config.SYMBOLS_PER_FRAME

    received_stripes = {}
    seen_packets = 0
    bad_packets = 0

    ptr = 0
    print(f"Decoding NG-SSTV stream ({args.mode})...")
    while ptr < len(audio) - len(sync_template):
        search_range = min(len(audio) - ptr, Config.FS)
        idx, confidence = Preamble.detect(audio[ptr : ptr + search_range], sync_template)

        if confidence < args.sync_threshold:
            ptr += 384
            continue

        ptr += idx + len(sync_template)
        if ptr + symbol_len > len(audio):
            break

        modem.demodulate_symbol(audio[ptr : ptr + symbol_len], is_pilot=True)
        ptr += symbol_len

        rx_bits = []
        for _ in range(Config.SYMBOLS_PER_FRAME):
            if ptr + symbol_len > len(audio):
                break
            diff = modem.demodulate_symbol(audio[ptr : ptr + symbol_len])
            ptr += symbol_len
            rx_bits.extend(modem.dqpsk_demap(diff))

        if len(rx_bits) < frame_capacity:
            break

        coded_bits = np.array(rx_bits[:frame_capacity], dtype=np.uint8)
        deint = codec.deinterleave(coded_bits, depth=Config.INTERLEAVER_DEPTH)
        decoded_bits = codec.viterbi_decode(deint)
        usable = (len(decoded_bits) // 8) * 8
        if usable <= 0:
            bad_packets += 1
            continue

        byte_data = np.packbits(decoded_bits[:usable]).tobytes()
        pkt = StripeFramer.unpack_stripe(byte_data)
        seen_packets += 1
        if pkt is None:
            bad_packets += 1
            continue

        stripe_id = pkt["stripe_id"]
        if stripe_id in received_stripes:
            continue

        stripe_img = ImageProcessor.decode_stripe(pkt["payload"])
        if stripe_img is None:
            bad_packets += 1
            continue

        received_stripes[stripe_id] = stripe_img
        if len(received_stripes) % 10 == 0 or len(received_stripes) == 1:
            print(f"   Decoded {len(received_stripes)}/{Config.NUM_STRIPES} stripes...")

        if len(received_stripes) >= Config.NUM_STRIPES:
            break

    if received_stripes:
        final_img = ImageProcessor.merge_stripes(received_stripes)
        final_img.save(args.output_image)
        print(f"Success: saved {args.output_image} ({len(received_stripes)}/{Config.NUM_STRIPES} stripes)")
        print(f"Packets: good={seen_packets - bad_packets}, bad={bad_packets}")
    else:
        print("Failed to decode any stripes. Try adjusting --sync-threshold or mode.")


if __name__ == "__main__":
    main()
