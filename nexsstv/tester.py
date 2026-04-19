import sys
import os
import tempfile
import numpy as np
from PIL import Image

# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexsstv.config import Config
from nexsstv.image.progressive import ImageProcessor
from nexsstv.fec.fountain import StripeFramer
from nexsstv.fec.robust import ChannelCodec
from nexsstv.modem.ofdm import Modem

def run_self_test():
    print("=== NexSSTV NextGen Self-Test ===")
    
    # 1. Create a dummy image
    print("1. Testing Image Fit (800x600)...")
    test_img_path = os.path.join(tempfile.gettempdir(), "nexsstv_test_input.png")
    img = Image.new('RGB', (1000, 500), color=(0, 255, 0)) # Non-800x600 input
    img.save(test_img_path)

    params = Config.get_mode_params('normal')
    modem = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    codec = ChannelCodec(rate=Config.CODING_RATE)
    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    stripe_capacity = bits_per_symbol * Config.SYMBOLS_PER_STRIPE
    max_raw_bits = codec.max_input_bits_for_capacity(stripe_capacity)
    max_stripe_bytes = max(16, min(Config.MAX_STRIPE_BYTES, max_raw_bits // 8 - 5))

    stripes = ImageProcessor.encode_image(
        test_img_path,
        quality=Config.DEFAULT_QUALITY,
        max_bytes=max_stripe_bytes,
    )
    print(f"   Generated {len(stripes)} stripes.")
    
    # 2. Framing & Modem Loopback
    print("2. Testing Stripe Framing + FEC + Modem Loopback...")
    
    # Test first stripe
    stripe_id = 0
    raw_data = stripes[0]
    packet = StripeFramer.pack_stripe(stripe_id, raw_data)
    bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
    coded = codec.encode(bits)
    coded = codec.interleave(coded, depth=Config.INTERLEAVER_DEPTH)
     
    audio_signal = [modem.modulate_symbol(None, is_pilot=True)]
    
    for i in range(0, len(coded), bits_per_symbol):
        chunk = coded[i : i + bits_per_symbol]
        if len(chunk) < bits_per_symbol:
            chunk = np.concatenate([chunk, np.zeros(bits_per_symbol - len(chunk), dtype=np.uint8)])
        audio_signal.append(modem.modulate_symbol(modem.dqpsk_map(chunk)))
     
    # Decode loopback
    modem_rx = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    modem_rx.demodulate_symbol(audio_signal[0], is_pilot=True)
    
    rx_bits = []
    for sig in audio_signal[1:]:
        diff = modem_rx.demodulate_symbol(sig)
        rx_bits.extend(modem_rx.dqpsk_demap(diff))

    rx_bits = np.array(rx_bits[:len(coded)], dtype=np.uint8)
    rx_bits = codec.deinterleave(rx_bits, depth=Config.INTERLEAVER_DEPTH)
    decoded = codec.viterbi_decode(rx_bits)
    usable = (len(decoded) // 8) * 8
    rx_bytes = np.packbits(decoded[:usable]).tobytes()
     
    # Unpack
    sid, sdata = StripeFramer.unpack_stripe(rx_bytes)
    if sid == stripe_id and sdata == raw_data:
        print("   Stripe Integrity Verified (0 Bit Errors)!")
    else:
        print("   Stripe Integrity FAILED!")

    # 3. Simulate Graceful Degradation
    print("3. Simulating Fragment Loss (Analog-style)...")
    stripe_dict = {i: ImageProcessor.decode_stripe(stripes[i]) for i in range(Config.NUM_STRIPES)}
    
    # Simulate losing stripes 10 to 20
    for i in range(10, 20):
        stripe_dict[i] = None
    
    final_img = ImageProcessor.merge_stripes(stripe_dict)
    output_path = os.path.join(tempfile.gettempdir(), "nexsstv_test_output.png")
    final_img.save(output_path)
    print(f"   Merged image with horizontal gap saved to {output_path}")
    
    print("\n=== NextGen Self-Test Complete ===")

if __name__ == "__main__":
    run_self_test()
