import os
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nexsstv.config import Config
from nexsstv.fec.fountain import RedundancyPlanner, StripeFramer
from nexsstv.fec.robust import ChannelCodec
from nexsstv.image.progressive import ImageProcessor
from nexsstv.modem.ofdm import Modem


def run_self_test():
    print("=== NexSSTV NG Self-Test ===")

    print("1. Testing image fit and stripe budget...")
    test_img_path = os.path.join(tempfile.gettempdir(), "nexsstv_test_input.png")
    Image.new("RGB", (1000, 500), color=(0, 255, 0)).save(test_img_path)

    params = Config.get_mode_params("normal")
    modem = Modem(Config.FS, params["f_min"], params["f_max"], params["n_subcarriers"], Config.CP_RATIO)
    codec = ChannelCodec(rate=Config.CODING_RATE)

    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    frame_capacity = bits_per_symbol * Config.SYMBOLS_PER_FRAME
    max_raw_bits = codec.max_input_bits_for_capacity(frame_capacity)
    max_frame_bytes = max(24, min(Config.MAX_STRIPE_BYTES, max_raw_bits // 8 - 16))

    stripes = ImageProcessor.encode_image(test_img_path, quality=Config.DEFAULT_QUALITY, max_bytes=max_frame_bytes)
    print(f"   Generated {len(stripes)} stripes.")

    print("2. Testing framing + FEC + modem loopback...")
    stripe_id = 0
    packet = StripeFramer.pack_stripe(stripe_id, stripes[0], repeat_idx=0, total_stripes=Config.NUM_STRIPES)
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    bits = np.concatenate([bits, np.zeros(max_raw_bits - len(bits), dtype=np.uint8)])

    coded = codec.encode(bits)
    coded = codec.interleave(coded, depth=Config.INTERLEAVER_DEPTH)

    tx = [modem.modulate_symbol(None, is_pilot=True)]
    for i in range(0, len(coded), bits_per_symbol):
        chunk = coded[i : i + bits_per_symbol]
        if len(chunk) < bits_per_symbol:
            chunk = np.concatenate([chunk, np.zeros(bits_per_symbol - len(chunk), dtype=np.uint8)])
        tx.append(modem.modulate_symbol(modem.dqpsk_map(chunk)))

    modem_rx = Modem(Config.FS, params["f_min"], params["f_max"], params["n_subcarriers"], Config.CP_RATIO)
    modem_rx.demodulate_symbol(tx[0], is_pilot=True)

    rx_bits = []
    for sig in tx[1: 1 + Config.SYMBOLS_PER_FRAME]:
        diff = modem_rx.demodulate_symbol(sig)
        rx_bits.extend(modem_rx.dqpsk_demap(diff))

    rx_bits = np.array(rx_bits[: len(coded)], dtype=np.uint8)
    rx_bits = codec.deinterleave(rx_bits, depth=Config.INTERLEAVER_DEPTH)
    decoded = codec.viterbi_decode(rx_bits)
    usable = (len(decoded) // 8) * 8
    rx_bytes = np.packbits(decoded[:usable]).tobytes()

    unpacked = StripeFramer.unpack_stripe(rx_bytes)
    if unpacked and unpacked["stripe_id"] == stripe_id and unpacked["payload"] == stripes[0]:
        print("   Stripe integrity verified.")
    else:
        print("   Stripe integrity FAILED.")

    print("3. Testing redundancy schedule...")
    schedule = RedundancyPlanner.build_schedule(Config.NUM_STRIPES, redundancy_factor=1.3)
    print(f"   Generated {len(schedule)} scheduled frames for {Config.NUM_STRIPES} stripes.")

    print("=== NG Self-Test Complete ===")


if __name__ == "__main__":
    run_self_test()
