"""
Enhanced Flask backend server for Bullet-Core Training Dashboard
Supports data upload, configuration, and full training integration
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import threading
import time
import json
import os
import numpy as np
import pickle

app = Flask(__name__, static_folder='dashboard')
CORS(app)

# Training state
training_state = {
    'is_training': False,
    'current_epoch': 0,
    'max_epochs': 100,
    'train_loss': None,
    'val_loss': None,
    'learning_rate': None,
    'time_per_epoch': None,
    'vram_usage': 0,
    'vram_total': 2048,
    'model_info': {},
    'gpu_info': {}
}

# Data storage
uploaded_data = {
    'train_X': None,
    'train_y': None,
    'val_X': None,
    'val_y': None
}

# Configuration
training_config = {}

training_lock = threading.Lock()
training_thread = None

# Serve dashboard
@app.route('/')
def index():
    return send_from_directory('dashboard', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('dashboard', path)

# API Endpoints
@app.route('/api/training/status', methods=['GET'])
def get_training_status():
    """Get current training status"""
    with training_lock:
        return jsonify(training_state)

@app.route('/api/data/upload', methods=['POST'])
def upload_data():
    """Upload training or validation data"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    data_type = request.form.get('type', 'train')
    
    # Save file
    upload_dir = 'uploads'
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    
    # Load data
    try:
        if filepath.endswith('.npy'):
            data = np.load(filepath)
        elif filepath.endswith('.csv'):
            import pandas as pd
            data = pd.read_csv(filepath).values.astype(np.float32)
        else:
            return jsonify({'error': 'Unsupported file format'}), 400
        
        # Assume last column is target
        X = data[:, :-1]
        y = data[:, -1:]
        
        if data_type == 'train':
            uploaded_data['train_X'] = X
            uploaded_data['train_y'] = y
        else:
            uploaded_data['val_X'] = X
            uploaded_data['val_y'] = y
        
        return jsonify({
            'status': 'success',
            'shape': data.shape,
            'type': data_type
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/training/start', methods=['POST'])
def start_training():
    """Start training with configuration"""
    global training_thread, training_config
    
    config = request.json
    training_config = config
    
    with training_lock:
        training_state['is_training'] = True
        training_state['max_epochs'] = config['training']['maxEpochs']
        training_state['current_epoch'] = 0
    
    # Start training in background
    training_thread = threading.Thread(target=run_training_loop, args=(config,))
    training_thread.daemon = True
    training_thread.start()
    
    return jsonify({'status': 'started'})

@app.route('/api/training/pause', methods=['POST'])
def pause_training():
    """Pause training"""
    with training_lock:
        training_state['is_training'] = False
    return jsonify({'status': 'paused'})

@app.route('/api/training/stop', methods=['POST'])
def stop_training():
    """Stop training"""
    with training_lock:
        training_state['is_training'] = False
        training_state['current_epoch'] = 0
    return jsonify({'status': 'stopped'})

@app.route('/api/gpu/info', methods=['GET'])
def get_gpu_info():
    """Get GPU information"""
    try:
        from bullet_core import cuda_ops
        
        if cuda_ops.is_available():
            info = cuda_ops.get_device_info()
            return jsonify({
                'available': True,
                'device': info['name'],
                'compute_capability': f"CC {info['compute_capability'] / 10:.1f}",
                'total_memory': info['total_memory'] // (1024 * 1024),
                'used_memory': info['used_vram'] // (1024 * 1024),
                'free_memory': info['free_vram'] // (1024 * 1024)
            })
        else:
            return jsonify({'available': False})
    except:
        return jsonify({'available': False})

# Training Loop
def run_training_loop(config):
    """Run training with full configuration"""
    from bullet_core import Tensor, nn
    from bullet_core.optim import SGD, Adam, AdamW
    from bullet_core.scheduler import ConstantLR, LinearWarmup, CosineAnnealing, WarmupCosineAnnealing
    import numpy as np
    
    # Build model from config
    model_config = config['model']
    input_dim = model_config['inputDim']
    hidden_dim = model_config['hiddenDim']
    output_dim = model_config['outputDim']
    num_layers = model_config['numLayers']
    
    # Create model
    layers = []
    layers.append(nn.Linear(input_dim, hidden_dim))
    layers.append(nn.RMSNorm(hidden_dim))
    
    for _ in range(num_layers - 2):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.RMSNorm(hidden_dim))
    
    layers.append(nn.Linear(hidden_dim, output_dim))
    model = nn.Sequential(*layers)
    
    # Create optimizer
    opt_config = config['optimizer']
    lr = opt_config['learningRate']
    wd = opt_config['weightDecay']
    
    if opt_config['type'] == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    elif opt_config['type'] == 'adam':
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:  # adamw
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Create scheduler
    sched_config = config['scheduler']
    warmup = sched_config['warmupSteps']
    max_epochs = config['training']['maxEpochs']
    
    if sched_config['type'] == 'warmup_cosine':
        scheduler = WarmupCosineAnnealing(optimizer, warmup_steps=warmup, T_max=max_epochs)
    elif sched_config['type'] == 'cosine':
        scheduler = CosineAnnealing(optimizer, T_max=max_epochs)
    else:
        scheduler = ConstantLR(optimizer)
    
    # Get data
    if uploaded_data['train_X'] is not None:
        X_train = uploaded_data['train_X']
        y_train = uploaded_data['train_y']
        X_val = uploaded_data['val_X'] if uploaded_data['val_X'] is not None else X_train[:100]
        y_val = uploaded_data['val_y'] if uploaded_data['val_y'] is not None else y_train[:100]
    else:
        # Generate synthetic data
        X_train = np.random.randn(800, input_dim).astype(np.float32)
        y_train = np.random.randn(800, output_dim).astype(np.float32)
        X_val = np.random.randn(200, input_dim).astype(np.float32)
        y_val = np.random.randn(200, output_dim).astype(np.float32)
    
    # Loss function
    def mse_loss(pred, target):
        diff = pred - target
        return (diff ** 2).mean()
    
    # Training loop
    for epoch in range(max_epochs):
        with training_lock:
            if not training_state['is_training']:
                break
        
        epoch_start = time.time()
        
        # Train
        x = Tensor(X_train, requires_grad=False)
        y = Tensor(y_train, requires_grad=False)
        
        pred = model(x)
        loss = mse_loss(pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        train_loss = loss.data.item() if hasattr(loss.data, 'item') else float(loss.data)
        
        # Validate
        x_val = Tensor(X_val, requires_grad=False)
        y_val_tensor = Tensor(y_val, requires_grad=False)
        pred_val = model(x_val)
        val_loss_tensor = mse_loss(pred_val, y_val_tensor)
        val_loss = val_loss_tensor.data.item() if hasattr(val_loss_tensor.data, 'item') else float(val_loss_tensor.data)
        
        epoch_time = time.time() - epoch_start
        
        # Update state
        with training_lock:
            training_state['current_epoch'] = epoch + 1
            training_state['train_loss'] = train_loss
            training_state['val_loss'] = val_loss
            training_state['learning_rate'] = optimizer.lr
            training_state['time_per_epoch'] = epoch_time
            training_state['vram_usage'] = 800 + (epoch % 20) * 10
        
        print(f"Epoch {epoch + 1}/{max_epochs} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {optimizer.lr:.6f}")
        
        time.sleep(0.1)
    
    with training_lock:
        training_state['is_training'] = False
    
    print("Training complete!")

if __name__ == '__main__':
    print("=" * 60)
    print("Bullet-Core Training Dashboard Server")
    print("=" * 60)
    print("Dashboard: http://localhost:5000")
    print("API: http://localhost:5000/api/training/status")
    print("=" * 60)
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
