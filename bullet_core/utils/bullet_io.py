
import json
import struct
import numpy as np
import os

def fnv1a64(string):
    hash_val = 0xcbf29ce484222325
    prime = 0x1099511628211
    for char in string:
        hash_val ^= ord(char)
        hash_val *= prime
        hash_val &= 0xFFFFFFFFFFFFFFFF
    return hash_val

class BulletWriter:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.tokenizer_data = b""
        self.weights_data = b""
        self.tensors = []
        
    def set_header(self, config):
        self.header = config
        self.header["bullet_version"] = "1.0"
        
    def set_tokenizer(self, vocab):
        # vocab is a dict of token -> id or list of tokens
        # We need to sort by ID
        if isinstance(vocab, dict):
            sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
            tokens = [x[0] for x in sorted_vocab]
        else:
            tokens = vocab
            
        data = bytearray()
        data.extend(b"BULK")
        data.extend(struct.pack("<I", len(tokens)))
        
        for token in tokens:
            # Handle special tokens or byte conversion
            if isinstance(token, str):
                token_bytes = token.encode('utf-8')
            else:
                token_bytes = bytes(token)
                
            data.extend(struct.pack("<H", len(token_bytes)))
            data.extend(token_bytes)
            
        self.tokenizer_data = data
        
    def quantize_bq4(self, data):
        # data: flat float32 array
        # Returns: bytearray of BQ4 blocks
        
        # Pad to multiple of 32
        pad_len = (32 - (len(data) % 32)) % 32
        if pad_len > 0:
            data = np.pad(data, (0, pad_len), 'constant')
            
        num_blocks = len(data) // 32
        data = data.reshape(num_blocks, 32)
        
        # Symmetric Quantization (Signed 4-bit)
        # Range [-8, 7]
        # scale = max_abs / 7.0
        
        max_abs = np.max(np.abs(data), axis=1)
        scales = max_abs / 7.0
        scales[scales == 0] = 1.0 # Avoid div by zero
        
        # Zero point is unused in symmetric, set to 0
        zeros = np.zeros(num_blocks, dtype=np.int8)
        
        # Quantize
        # q = x / scale
        # We broadcast scale: (num_blocks, 1)
        q = np.round(data / scales[:, None])
        q = np.clip(q, -8, 7).astype(np.int8)
        
        # Pack
        output = bytearray()
        
        for i in range(num_blocks):
            s = scales[i]
            z = 0 # Unused
            
            # Scale f16
            # Use struct 'e' format for float16 (Python 3.6+)
            output.extend(struct.pack("<e", s))
            
            # Zero i8 (Always 0)
            output.extend(struct.pack("<b", z))
            
            # Pad
            output.extend(b"\x00")
            
            # Pack 4-bit values
            # q is int8 [-8, 7]. Cast to uint8 for bitwise ops.
            # -1 (0xFF) -> 0xF
            block_q = q[i].astype(np.uint8)
            
            packed = np.zeros(16, dtype=np.uint8)
            for j in range(16):
                low = block_q[2*j] & 0x0F
                high = block_q[2*j+1] & 0x0F
                packed[j] = (high << 4) | low
                
            if i == 0:
                print(f"DEBUG Quant: s={s}, z={z}, q={q[i]}")
            
            output.extend(packed.tobytes())
            
        return output

    def add_tensor(self, name, data, quantize=True):
        # data is numpy array
        name_hash = fnv1a64(name)
        rank = len(data.shape)
        shape = data.shape
        
        if quantize and data.dtype == np.float32 and data.size % 32 == 0:
            # Use BQ4
            quant_type = 0 # BQ4
            raw_bytes = self.quantize_bq4(data.flatten())
        else:
            # Fallback to FP16
            if data.dtype != np.float16:
                data = data.astype(np.float16)
            quant_type = 3 # FP16
            raw_bytes = data.tobytes()
            
        compressed_size = len(raw_bytes)
        
        tensor_meta = bytearray()
        tensor_meta.extend(struct.pack("<Q", name_hash))
        tensor_meta.extend(struct.pack("<B", rank))
        for dim in shape:
            tensor_meta.extend(struct.pack("<H", dim))
        tensor_meta.extend(struct.pack("<B", quant_type))
        tensor_meta.extend(struct.pack("<I", compressed_size))
        
        self.tensors.append((tensor_meta, raw_bytes))
        
    def write(self):
        with open(self.path, "wb") as f:
            # 1. Calculate Offsets
            # Header is JSON + 4 nulls
            # We need to estimate header size or reserve space?
            # The spec says "Offset: 0".
            
            # Let's build weights block first to know size?
            # No, offsets are for Tokenizer and Weights start.
            
            # Construct Weights Block
            weights_block = bytearray()
            weights_block.extend(b"BWT0")
            weights_block.extend(struct.pack("<I", len(self.tensors)))
            
            for meta, data in self.tensors:
                weights_block.extend(meta)
                weights_block.extend(data)
                
            # Calculate offsets
            # Header JSON
            # We need to put placeholders for offsets in header, dump it, check size, update offsets?
            # Or just put 0, calculate, then rewrite.
            
            self.header["file_offsets"] = {
                "tokenizer_start": 0, # Placeholder
                "weights_start": 0    # Placeholder
            }
            
            header_json = json.dumps(self.header, indent=2).encode('utf-8')
            header_len = len(header_json) + 4 # + 4 nulls
            
            # Align to 64 bytes? Spec says "All blocks MUST start on 64-byte alignment"
            # Tokenizer starts at tokenizer_start
            
            tokenizer_start = header_len
            # Align tokenizer_start to 4096 (page aligned) as per spec "Tokenizer MUST start on 4KB boundary"
            def align(val, alignment):
                return (val + alignment - 1) // alignment * alignment
                
            tokenizer_start = align(tokenizer_start, 4096)
            
            weights_start = tokenizer_start + len(self.tokenizer_data)
            # Align weights to 64 bytes
            weights_start = align(weights_start, 64)
            
            # Update Header
            self.header["file_offsets"]["tokenizer_start"] = tokenizer_start
            self.header["file_offsets"]["weights_start"] = weights_start
            
            header_json = json.dumps(self.header, indent=2).encode('utf-8')
            
            # Write Header
            f.write(header_json)
            f.write(b"\x00\x00\x00\x00")
            
            # Pad to tokenizer_start
            current_pos = f.tell()
            if current_pos < tokenizer_start:
                f.write(b"\x00" * (tokenizer_start - current_pos))
                
            # Write Tokenizer
            f.write(self.tokenizer_data)
            
            # Pad to weights_start
            current_pos = f.tell()
            if current_pos < weights_start:
                f.write(b"\x00" * (weights_start - current_pos))
                
            # Write Weights
            f.write(weights_block)
            
            # Write Footer
            f.write(b"BULLET_END")
            f.write(b"END!")

class BulletReader:
    def __init__(self, path):
        self.path = path
        self.header = {}
        self.vocab = []
        self.tensors = {}
        
    def load(self):
        with open(self.path, "rb") as f:
            # Read Header
            # Read until 4 null bytes
            buffer = bytearray()
            while True:
                chunk = f.read(1)
                if not chunk: break
                buffer.extend(chunk)
                if buffer.endswith(b"\x00\x00\x00\x00"):
                    break
            
            header_json = buffer[:-4].decode('utf-8')
            self.header = json.loads(header_json)
            
            # Read Tokenizer
            tok_start = self.header["file_offsets"]["tokenizer_start"]
            f.seek(tok_start)
            magic = f.read(4)
            if magic != b"BULK":
                raise ValueError("Invalid tokenizer magic")
            
            vocab_size = struct.unpack("<I", f.read(4))[0]
            for _ in range(vocab_size):
                token_len = struct.unpack("<H", f.read(2))[0]
                token_bytes = f.read(token_len)
                self.vocab.append(token_bytes.decode('utf-8', errors='ignore'))
                
            # Read Weights
            w_start = self.header["file_offsets"]["weights_start"]
            f.seek(w_start)
            magic = f.read(4)
            if magic != b"BWT0":
                raise ValueError("Invalid weights magic")
            
            num_tensors = struct.unpack("<I", f.read(4))[0]
            
            for _ in range(num_tensors):
                name_hash = struct.unpack("<Q", f.read(8))[0]
                rank = struct.unpack("<B", f.read(1))[0]
                shape = []
                for _ in range(rank):
                    shape.append(struct.unpack("<H", f.read(2))[0])
                
                quant_type = struct.unpack("<B", f.read(1))[0]
                compressed_size = struct.unpack("<I", f.read(4))[0]
                data_bytes = f.read(compressed_size)
                
                # Decompress/Cast
                if quant_type == 3: # FP16
                    data = np.frombuffer(data_bytes, dtype=np.float16).astype(np.float32)
                    data = data.reshape(shape)
                    self.tensors[name_hash] = data
                elif quant_type == 0: # BQ4
                    # Dequantize BQ4 manually (Symmetric)
                    # Block size 20 bytes -> 32 floats
                    num_blocks = len(data_bytes) // 20
                    data = np.zeros(num_blocks * 32, dtype=np.float32)
                    
                    for i in range(num_blocks):
                        block = data_bytes[i*20 : (i+1)*20]
                        
                        # Correctly interpret bytes as float16 using struct
                        s = struct.unpack("<e", block[0:2])[0]
                        
                        # z is unused (always 0)
                        
                        # 16 bytes of packed data
                        packed = np.frombuffer(block[4:], dtype=np.uint8)
                        
                        # Unpack
                        # byte = (high << 4) | low
                        low = packed & 0x0F
                        high = (packed >> 4) & 0x0F
                        
                        # Sign Extend 4-bit to signed int using two's complement
                        # Convert each nibble: if bit 3 is set (val >= 8), it's negative
                        # Use vectorized operation for efficiency
                        def sign_extend_4bit(nibbles):
                            # nibbles is uint8 array with values 0-15
                            # Convert to signed: if >= 8, subtract 16
                            signed = nibbles.astype(np.int8)
                            signed = np.where(signed >= 8, signed - 16, signed)
                            return signed.astype(np.float32)
                        
                        low = sign_extend_4bit(low)
                        high = sign_extend_4bit(high)
                        
                        # Interleave: low, high, low, high...
                        q = np.empty(32, dtype=np.float32)
                        q[0::2] = low
                        q[1::2] = high
                        
                        # Dequantize: x = q * scale
                        data[i*32 : (i+1)*32] = q * s
                        
                    data = data.reshape(shape)
                    self.tensors[name_hash] = data
                else:
                    print(f"Warning: Unsupported quant type {quant_type}")
