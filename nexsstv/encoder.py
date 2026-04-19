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
from nexsstv.fec.robust import ChannelCodec
from nexsstv.modem.ofdm import Modem
from nexsstv.sync.preamble import Preamble

def main():
    parser = argparse.ArgumentParser(description="NexSSTV NextGen Encoder")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("output_wav", help="Path to output WAV file")
    parser.add_argument("--mode", choices=["normal", "ultra", "classic"], default="normal")
    parser.add_argument("--quality", type=int, default=Config.DEFAULT_QUALITY, help="WebP quality")
    args = parser.parse_args()

    # 1. Processing
    print("Processing 800x600 image (NextGen robust mode)...")
    stripes = ImageProcessor.encode_image(
        args.input_image,
        quality=args.quality,
        max_bytes=Config.MAX_STRIPE_BYTES,
    )
    
    # 2. Setup
    params = Config.get_mode_params(args.mode)
    modem = Modem(Config.FS, params['f_min'], params['f_max'], params['n_subcarriers'], Config.CP_RATIO)
    codec = ChannelCodec()
    preamble = Preamble(samples_per_bit=16) 
    sync_burst = preamble.generate_signal(Config.FS, f_center=(params['f_min'] + params['f_max']) / 2)
    
    audio_signal = [np.zeros(2048)] # Lead-in
    bits_per_symbol = modem.n_subcarriers * Config.BITS_PER_SYMBOL
    stripe_capacity = bits_per_symbol * Config.SYMBOLS_PER_STRIPE
    max_raw_bits = (stripe_capacity // 2) - (codec.CONSTRAINT - 1)
    
    print(f"Modulating {Config.NUM_STRIPES} stripes (target <= 60s)...")
    for i, stripe_data in enumerate(stripes):
        # Frame the stripe
        packet = StripeFramer.pack_stripe(i, stripe_data)
        bits = np.unpackbits(np.frombuffer(packet, dtype=np.uint8))
        if len(bits) > max_raw_bits:
            raise RuntimeError(
                f"Stripe {i} exceeds radio budget ({len(bits)} > {max_raw_bits} raw bits). "
                f"Try lower --quality."
            )
        bits = np.concatenate([bits, np.zeros(max_raw_bits - len(bits), dtype=np.uint8)])
        coded = codec.encode(bits)
        coded = codec.interleave(coded, depth=Config.INTERLEAVER_DEPTH)
        coded = coded[:stripe_capacity]
             
        # 1. Sync Burst
        audio_signal.append(sync_burst)
        
        # 2. Phase Pilot
        audio_signal.append(modem.modulate_symbol(None, is_pilot=True))
            
        # 3. Payload
        symbols_sent = 0
        for j in range(0, len(coded), bits_per_symbol):
            if symbols_sent >= Config.SYMBOLS_PER_STRIPE: break
            chunk = coded[j : j + bits_per_symbol]
            if len(chunk) < bits_per_symbol:
                chunk = np.concatenate([chunk, np.zeros(bits_per_symbol - len(chunk), dtype=np.uint8)])
            
            audio_signal.append(modem.modulate_symbol(modem.dqpsk_map(chunk)))
            symbols_sent += 1
            
        # 4. Padding (Minimal to keep stream synchronous)
        while symbols_sent < Config.SYMBOLS_PER_STRIPE:
            # Use deterministic DQPSK (0,0) symbols (1+0j) for predictable decoder behavior.
            audio_signal.append(modem.modulate_symbol(np.ones(modem.n_subcarriers, dtype=np.complex128)))
            symbols_sent += 1
            
        if (i+1) % 15 == 0:
            print(f"   Modulated {i+1}/{Config.NUM_STRIPES} stripes...")

    full_audio = np.concatenate(audio_signal)
    full_audio = full_audio / (np.max(np.abs(full_audio)) + 1e-9)
    wavfile.write(args.output_wav, Config.FS, (full_audio * 32767).astype(np.int16))
    
    duration = len(full_audio) / Config.FS
    print(f"Done! {args.output_wav} is {duration:.1f}s long.")

if __name__ == "__main__":
    main()
