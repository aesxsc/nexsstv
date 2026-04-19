import sys
import os
import numpy as np
from PIL import Image

# Add the 'src' directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nexsstv.config import Config
from nexsstv.image.progressive import ImageProcessor
from nexsstv.fec.fountain import StripeFramer
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble

def run_self_test():
    print("=== NexSSTV Self-Test V4 (Analog-Digital Hybrid) ===")
    
    # 1. Create a dummy image
    print("1. Testing Image Fit (800x600)...")
    test_img_path = "test_input.png"
    img = Image.new('RGB', (1000, 500), color=(0, 255, 0)) # Non-800x600 input
    img.save(test_img_path)
    
    stripes = ImageProcessor.encode_image(test_img_path, quality=30)
    print(f"   Generated {len(stripes)} stripes.")
    
    # 2. Framing & Modem Loopback
    print("2. Testing Stripe Framing & Modem Loopback...")
    params = Config.get_mode_params('classic')
    modem = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    
    # Test first stripe
    stripe_id = 0
    raw_data = stripes[0]
    packet = StripeFramer.pack_stripe(stripe_id, raw_data)
    
    # Bits
    bits = []
    for byte in packet:
        bits.extend([int(b) for b in bin(byte)[2:].zfill(8)])
    
    bits_per_symbol = params['n_subcarriers'] * Config.BITS_PER_SYMBOL
    audio_signal = [modem.modulate_symbol(None, is_pilot=True)]
    
    for i in range(0, len(bits), bits_per_symbol):
        chunk = bits[i : i + bits_per_symbol]
        if len(chunk) < bits_per_symbol: chunk += [0] * (bits_per_symbol - len(chunk))
        audio_signal.append(modem.modulate_symbol(modem.dbpsk_map(chunk)))
    
    # Decode loopback
    modem_rx = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    modem_rx.demodulate_symbol(audio_signal[0], is_pilot=True)
    
    rx_bits = []
    for sig in audio_signal[1:]:
        diff = modem_rx.demodulate_symbol(sig)
        rx_bits.extend(modem_rx.dbpsk_demap(diff))
    
    # Convert bits to bytes
    rx_bytes = bytearray()
    for k in range(0, (len(rx_bits) // 8) * 8, 8):
        rx_bytes.append(int("".join(map(str, rx_bits[k:k+8])), 2))
    
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
    final_img.save("test_output.png")
    print("   Merged image with horizontal gap saved to test_output.png")
    
    print("\n=== Self-Test V4 Complete ===")

if __name__ == "__main__":
    run_self_test()
