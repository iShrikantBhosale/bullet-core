"""
BQ4 Model Exporter
Converts trained Bullet model (.pkl) to BQ4 quantized format (.bq4)
"""
import pickle
import struct
import numpy as np
from pathlib import Path

class BQ4Exporter:
    def __init__(self, block_size=32):
        self.block_size = block_size
        
    def quantize_block(self, weights):
        """Quantize a block of FP32 weights to BQ4 format"""
        # Compute scale
        max_abs = np.max(np.abs(weights))
        if max_abs < 1e-8:
            max_abs = 1e-8
        
        scale = max_abs / 7.0
        inv_scale = 1.0 / scale
        
        # Quantize
        q = np.round(weights * inv_scale).astype(np.int8)
        q = np.clip(q, -8, 7)
        
        # Pack two int4 per byte
        packed = []
        for i in range(0, len(q), 2):
            lo = int(q[i]) & 0x0F
            hi = int(q[i+1]) & 0x0F if i+1 < len(q) else 0
            byte_val = (hi << 4) | lo
            packed.append(byte_val)
        
        return bytes(packed), scale
    
    def quantize_tensor(self, tensor):
        """Quantize entire tensor to BQ4"""
        # Flatten to 1D
        flat = tensor.flatten()
        
        # Pad to multiple of block_size
        remainder = len(flat) % self.block_size
        if remainder != 0:
            padding = self.block_size - remainder
            flat = np.pad(flat, (0, padding), mode='constant')
        
        # Split into blocks
        num_blocks = len(flat) // self.block_size
        blocks = []
        
        for i in range(num_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_weights = flat[start:end]
            
            packed_data, scale = self.quantize_block(block_weights)
            blocks.append((packed_data, scale))
        
        return blocks
    
    def export(self, model_path, output_path):
        """Export .pkl model to .bq4 format"""
        print(f"Loading model from {model_path}...")
        with open(model_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        # Handle different checkpoint formats
        if 'model_state' in checkpoint:
            model_state = checkpoint['model_state']
        elif 'model' in checkpoint:
            model_state = checkpoint['model']
        elif 'model_state_dict' in checkpoint:
            model_state = checkpoint['model_state_dict']
        else:
            # Assume checkpoint IS the model state
            model_state = checkpoint
        
        # Filter weight tensors (2D or higher dimensional)
        weight_tensors = {}
        for name, param in model_state.items():
            if hasattr(param, 'ndim') and param.ndim >= 2:
                weight_tensors[name] = param
            elif hasattr(param, 'shape') and len(param.shape) >= 2:
                weight_tensors[name] = param
        
        if not weight_tensors:
            print("Warning: No 2D+ tensors found. Trying all parameters...")
            weight_tensors = {k: v for k, v in model_state.items() 
                            if hasattr(v, 'shape') or hasattr(v, 'ndim')}
        
        print(f"Found {len(weight_tensors)} weight tensors to quantize")
        
        # Quantize all tensors
        quantized = {}
        for name, tensor in weight_tensors.items():
            print(f"  Quantizing {name}: {tensor.shape}")
            blocks = self.quantize_tensor(tensor)
            quantized[name] = {
                'shape': tensor.shape,
                'blocks': blocks,
                'block_size': self.block_size
            }
        
        # Write BQ4 file
        print(f"\nWriting to {output_path}...")
        self._write_bq4_file(quantized, output_path)
        
        # Report stats
        original_size = sum(t.nbytes for t in weight_tensors.values())
        output_size = Path(output_path).stat().st_size
        compression = original_size / output_size
        
        print(f"\n✅ Export complete!")
        print(f"   Original size: {original_size/1024:.1f} KB")
        print(f"   BQ4 size: {output_size/1024:.1f} KB")
        print(f"   Compression: {compression:.2f}x")
    
    def _write_bq4_file(self, quantized, output_path):
        """Write BQ4 file format"""
        with open(output_path, 'wb') as f:
            # Header
            f.write(b'BQ4F')  # magic
            f.write(struct.pack('I', 1))  # version
            f.write(struct.pack('I', len(quantized)))  # num_tensors
            
            # Tensor directory (placeholder, will update offsets later)
            dir_start = f.tell()
            tensor_entries = []
            
            for name, data in quantized.items():
                # Name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('I', len(name_bytes)))
                f.write(name_bytes)
                
                # Dims
                shape = data['shape']
                f.write(struct.pack('I', len(shape)))
                for dim in shape:
                    f.write(struct.pack('I', dim))
                
                # Block info
                f.write(struct.pack('I', data['block_size']))
                f.write(struct.pack('I', len(data['blocks'])))
                
                # Offset (placeholder)
                offset_pos = f.tell()
                f.write(struct.pack('Q', 0))
                
                tensor_entries.append((name, offset_pos))
            
            # Write quantized blocks
            for name, offset_pos in tensor_entries:
                data = quantized[name]
                
                # Record actual offset
                actual_offset = f.tell()
                
                # Write blocks
                for packed_data, scale in data['blocks']:
                    f.write(packed_data)
                    f.write(struct.pack('f', scale))
                
                # Update offset in directory
                current_pos = f.tell()
                f.seek(offset_pos)
                f.write(struct.pack('Q', actual_offset))
                f.seek(current_pos)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python export_bq4.py <model.pkl> <output.bq4>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    output_path = sys.argv[2]
    
    exporter = BQ4Exporter(block_size=32)
    exporter.export(model_path, output_path)
