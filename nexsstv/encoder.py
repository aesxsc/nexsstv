import sys
import os

# Path resolution
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from scipy.io import wavfile
import argparse
from nexsstv.config import Config
from nexsstv.image.progressive import ImageProcessor
from nexsstv.fec.fountain import StripeFramer
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble

def main():
    parser = argparse.ArgumentParser(description="NexSSTV V10 Flash Encoder")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_wav", help="Path to output WAV file")
    parser.add_argument("--mode", choices=["classic", "ultra"], default="classic")
    parser.add_argument("--quality", type=int, default=25, help="WebP quality (default 25)")
    args = parser.parse_args()

    # 1. Processing
    print(f"Processing 800x600 image (V10 Fast Engine)...")
    stripes = ImageProcessor.encode_image(args.input_image, quality=args.quality)
    
    # 2. Setup
    params = Config.get_mode_params(args.mode)
    modem = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    preamble = Preamble(samples_per_bit=16) 
    sync_burst = preamble.generate_signal(Config.FS, f_center=(params['f_min'] + params['f_max']) / 2)
    
    audio_signal = [np.zeros(2048)] # Lead-in
    bits_per_symbol = params['n_subcarriers'] * Config.BITS_PER_SYMBOL
    
    print(f"Modulating 75 stripes (Target: ~30s)...")
    for i, stripe_data in enumerate(stripes):
        # Frame the stripe
        packet = StripeFramer.pack_stripe(i, stripe_data)
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
            
        # 1. Sync Burst
        audio_signal.append(sync_burst)
        
        # 2. Phase Pilot
        audio_signal.append(modem.modulate_symbol(None, is_pilot=True))
            
        # 3. Payload
        symbols_sent = 0
        for j in range(0, len(bits), bits_per_symbol):
            if symbols_sent >= Config.SYMBOLS_PER_STRIPE: break
            chunk = bits[j : j + bits_per_symbol]
            if len(chunk) < bits_per_symbol:
                chunk = np.concatenate([chunk, np.zeros(bits_per_symbol - len(chunk), dtype=np.uint8)])
            
            audio_signal.append(modem.modulate_symbol(modem.dqpsk_map(chunk)))
            symbols_sent += 1
            
        # 4. Padding (Minimal to keep stream synchronous)
        while symbols_sent < Config.SYMBOLS_PER_STRIPE:
            audio_signal.append(modem.modulate_symbol(np.zeros(params['n_subcarriers'], dtype=np.complex128)))
            symbols_sent += 1
            
        if (i+1) % 15 == 0:
            print(f"   Modulated {i+1}/75 stripes...")

    full_audio = np.concatenate(audio_signal)
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-9)
    wavfile.write(args.output_wav, Config.FS, (full_audio * 32767).astype(np.int16))
    
    duration = len(full_audio) / Config.FS
    print(f"Done! {args.output_wav} is {duration:.1f}s long.")

if __name__ == "__main__":
    main()
