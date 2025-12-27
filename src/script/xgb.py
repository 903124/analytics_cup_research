import numpy as np
import pandas as pd

def normalize_and_rotate_coordinates(segment_df, n_init_frames=3):
    """
    Normalize coordinates to origin (0,0,z) and rotate XY plane so initial 
    velocity vector points to 0 degrees (positive X-axis).
    
    Parameters:
    -----------
    segment_df : pd.DataFrame
        DataFrame containing 'ball_x', 'ball_y', 'ball_z' columns
    n_init_frames : int
        Number of initial frames to use for calculating initial velocity direction
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with additional columns:
        - ball_x_norm: normalized and rotated X coordinate
        - ball_y_norm: normalized and rotated Y coordinate  
        - ball_z_norm: unchanged Z coordinate (same as ball_z)
        - rotation_angle: the angle (in radians) used for rotation
    """
    segment_df = segment_df.copy()
    
    # Extract coordinates
    x = segment_df['ball_x'].values
    y = segment_df['ball_y'].values
    z = segment_df['ball_z'].values
    
    # Get initial position
    x0, y0, z0 = x[0], y[0], z[0]
    
    # Calculate initial velocity direction (average over first n frames)
    n_init = min(n_init_frames, len(x) - 1)
    initial_dx = x[n_init] - x[0]
    initial_dy = y[n_init] - y[0]
    
    # Calculate initial velocity angle
    initial_angle = np.arctan2(initial_dy, initial_dx)
    
    # Step 1: Translate to origin
    x_translated = x - x0
    y_translated = y - y0
    z_translated = z  # Z remains unchanged in absolute terms
    
    # Step 2: Rotate XY plane so initial velocity points to 0° (positive X-axis)
    # Rotation matrix for counterclockwise rotation by -initial_angle
    cos_theta = np.cos(-initial_angle)
    sin_theta = np.sin(-initial_angle)
    
    x_norm = x_translated * cos_theta - y_translated * sin_theta
    y_norm = x_translated * sin_theta + y_translated * cos_theta
    z_norm = z_translated
    
    # Add normalized coordinates to dataframe
    segment_df['ball_x_norm'] = x_norm
    segment_df['ball_y_norm'] = y_norm
    segment_df['ball_z_norm'] = z_norm
    segment_df['rotation_angle'] = initial_angle
    
    return segment_df


def normalize_all_segments(tracking_df, segment_id_col='segment_id'):
    """
    Apply normalization and rotation to all segments in a tracking dataframe.
    
    Parameters:
    -----------
    tracking_df : pd.DataFrame
        DataFrame containing tracking data with segment IDs
    segment_id_col : str
        Name of the column containing segment IDs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized coordinates for all segments
    """
    normalized_segments = []
    
    for seg_id in tracking_df[segment_id_col].unique():
        segment = tracking_df[tracking_df[segment_id_col] == seg_id].copy()
        segment = segment.sort_values('frame_id').reset_index(drop=True)
        
        normalized_segment = normalize_and_rotate_coordinates(segment)
        normalized_segments.append(normalized_segment)
    
    return pd.concat(normalized_segments, ignore_index=True)


# Example usage with your tracking_df_cleaned:
# tracking_df_normalized = normalize_all_segments(tracking_df_cleaned)

# Verify normalization for a sample segment
def verify_normalization(segment_df):
    """
    Print verification info for normalized coordinates.
    """
    print("=== Normalization Verification ===")
    print(f"\nOriginal coordinates:")
    print(f"  Start: ({segment_df['ball_x'].iloc[0]:.2f}, {segment_df['ball_y'].iloc[0]:.2f}, {segment_df['ball_z'].iloc[0]:.2f})")
    print(f"  End:   ({segment_df['ball_x'].iloc[-1]:.2f}, {segment_df['ball_y'].iloc[-1]:.2f}, {segment_df['ball_z'].iloc[-1]:.2f})")
    
    print(f"\nNormalized coordinates:")
    print(f"  Start: ({segment_df['ball_x_norm'].iloc[0]:.2f}, {segment_df['ball_y_norm'].iloc[0]:.2f}, {segment_df['ball_z_norm'].iloc[0]:.2f})")
    print(f"  End:   ({segment_df['ball_x_norm'].iloc[-1]:.2f}, {segment_df['ball_y_norm'].iloc[-1]:.2f}, {segment_df['ball_z_norm'].iloc[-1]:.2f})")
    
    # Calculate initial velocity direction in normalized space
    n_init = min(3, len(segment_df) - 1)
    dx_norm = segment_df['ball_x_norm'].iloc[n_init] - segment_df['ball_x_norm'].iloc[0]
    dy_norm = segment_df['ball_y_norm'].iloc[n_init] - segment_df['ball_y_norm'].iloc[0]
    angle_norm = np.arctan2(dy_norm, dx_norm) * 180 / np.pi
    
    print(f"\nInitial velocity angle in normalized space: {angle_norm:.2f}° (should be ~0°)")
    print(f"Rotation angle applied: {segment_df['rotation_angle'].iloc[0] * 180 / np.pi:.2f}°")

tracking_df_normalized = normalize_all_segments(tracking_df_cleaned)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb

# =============================================================================
# Configuration
# =============================================================================
N_INITIAL_FRAMES = 5  # Number of frames to use for initial velocity
N_FOLDS = 5
RANDOM_STATE = 42

# =============================================================================
# Helper Functions
# =============================================================================

def compute_velocity(coords, frame_ids, fps=10):
    """Compute velocity from coordinate sequence."""
    if len(coords) < 2:
        return np.array([0.0, 0.0, 0.0])
    
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    dz = np.diff(coords[:, 2])
    dt = np.diff(frame_ids) / fps
    
    # Handle zero time differences
    dt = np.where(dt == 0, 1e-6, dt)
    
    vx = dx / dt
    vy = dy / dt
    vz = dz / dt
    
    # Return mean velocity
    return np.array([np.mean(vx), np.mean(vy), np.mean(vz)])


def transform_to_normalized_coordinates(x, y, z, vx, vy):
    """
    Transform coordinates so that:
    - Initial position is at (0, 0)
    - Initial velocity points to 0 degrees on xy plane
    """
    # Translate to origin
    x_norm = x - x[0]
    y_norm = y - y[0]
    z_norm = z - z[0]
    
    # Calculate rotation angle to align velocity with x-axis
    initial_velocity_angle = np.arctan2(vy, vx)
    
    # Rotate all points
    cos_theta = np.cos(-initial_velocity_angle)
    sin_theta = np.sin(-initial_velocity_angle)
    
    x_rot = x_norm * cos_theta - y_norm * sin_theta
    y_rot = x_norm * sin_theta + y_norm * cos_theta
    
    return x_rot, y_rot, z_norm, initial_velocity_angle


# =============================================================================
# Feature Engineering
# =============================================================================

def create_training_samples(tracking_df, n_initial_frames=5):
    """
    Create training samples from tracking data.
    Each sample uses initial frames to predict target position.
    """
    samples = []
    
    for segment_id in tracking_df['segment_id'].unique():
        segment = tracking_df[tracking_df['segment_id'] == segment_id].copy()
        segment = segment.sort_values('frame_id').reset_index(drop=True)
        
        # Need at least initial frames + 1 target frame
        if len(segment) < n_initial_frames + 1:
            continue
        
        # Remove rows with missing coordinates
        segment = segment[
            segment['ball_x'].notna() & 
            segment['ball_y'].notna() & 
            segment['ball_z'].notna()
        ]
        
        if len(segment) < n_initial_frames + 1:
            continue
        
        # Extract initial frames
        initial_segment = segment.iloc[:n_initial_frames]
        
        # Get coordinates
        x_coords = initial_segment['ball_x'].values
        y_coords = initial_segment['ball_y'].values
        z_coords = initial_segment['ball_z'].values
        frame_ids = initial_segment['frame_id'].values
        
        # Compute initial velocity
        coords_3d = np.column_stack([x_coords, y_coords, z_coords])
        velocity = compute_velocity(coords_3d, frame_ids)
        
        vx_initial, vy_initial, vz_initial = velocity
        velocity_magnitude = np.sqrt(vx_initial**2 + vy_initial**2)
        
        # Skip if velocity is too small (stationary ball)
        if velocity_magnitude < 0.1:
            continue
        
        # For each remaining frame in the segment, create a sample
        for target_idx in range(n_initial_frames, len(segment)):
            target_row = segment.iloc[target_idx]
            
            # Get all coordinates for transformation (only x and y)
            all_x = np.append(x_coords, target_row['ball_x'])
            all_y = np.append(y_coords, target_row['ball_y'])
            all_z = np.append(z_coords, target_row['ball_z'])
            
            # Transform coordinates (only x and y)
            x_norm, y_norm, _, angle = transform_to_normalized_coordinates(
                all_x, all_y, all_z, vx_initial, vy_initial
            )
            
            # Target is the last transformed point
            target_x_norm = x_norm[-1]
            target_y_norm = y_norm[-1]
            target_z = all_z[-1]  # Keep original z value
            
            # Compute time to target
            time_to_target = (target_row['frame_id'] - frame_ids[0]) / 10.0  # 10 fps
            
            # Compute normalized velocities after transformation
            # After rotation, initial velocity should point along x-axis
            vx_norm = velocity_magnitude
            vy_norm = 0.0  # By construction
            vz_norm = vz_initial  # Keep original vz
            
            sample = {
                'segment_id': segment_id,
                'target_frame_idx': target_idx,
                # Initial position (always 0,0 for x,y after normalization, original z0)
                'x0': 0.0,
                'y0': 0.0,
                'z0': z_coords[0],  # Keep original z
                # Initial velocity (normalized)
                'vx0': vx_norm,
                'vy0': vy_norm,
                'vz0': vz_norm,
                'v_magnitude': velocity_magnitude,
                # Time to target
                'time_to_target': time_to_target,
                # Original transformation info (for inverse transform)
                'original_x0': x_coords[0],
                'original_y0': y_coords[0],
                'original_z0': z_coords[0],
                'rotation_angle': angle,
                # Target (normalized x,y, original z)
                'target_x_norm': target_x_norm,
                'target_y_norm': target_y_norm,
                'target_z_norm': target_z,  # Original z value, not transformed
            }
            
            samples.append(sample)
    
    return pd.DataFrame(samples)


# =============================================================================
# Create Training Data
# =============================================================================

print("Creating training samples from tracking data...")
training_data = create_training_samples(tracking_df_cleaned, n_initial_frames=N_INITIAL_FRAMES)

print(f"\nTraining data shape: {training_data.shape}")
print(f"Number of unique segments: {training_data['segment_id'].nunique()}")
print(f"\nSample of training data:")
print(training_data.head())

print(f"\nTarget statistics:")
print(training_data[['target_x_norm', 'target_y_norm', 'target_z_norm']].describe())

# =============================================================================
# Prepare Features and Targets
# =============================================================================

feature_cols = [
    'vx0', 'vy0', 'vz0', 'v_magnitude', 'time_to_target'
]

X = training_data[feature_cols].values
y_x = training_data['target_x_norm'].values
y_y = training_data['target_y_norm'].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target X shape: {y_x.shape}")
print(f"Target Y shape: {y_y.shape}")

# =============================================================================
# 5-Fold Cross Validation with XGBoost
# =============================================================================

kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

# Storage for out-of-fold predictions
oof_predictions_x = np.zeros(len(X))
oof_predictions_y = np.zeros(len(X))

# Storage for models
models_x = []
models_y = []

# Storage for metrics
fold_metrics = []

print("\n" + "="*80)
print("Training XGBoost Models with 5-Fold Cross Validation")
print("="*80)

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
    print(f"\n--- Fold {fold_idx + 1}/{N_FOLDS} ---")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
    y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]
    
    # XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': RANDOM_STATE,
        'n_jobs': -1
    }
    
    # Train model for X coordinate
    print(f"Training X-coordinate model...")
    model_x = xgb.XGBRegressor(**params)
    model_x.fit(
        X_train, y_x_train,
        eval_set=[(X_val, y_x_val)],
        verbose=False
    )
    models_x.append(model_x)
    
    # Train model for Y coordinate
    print(f"Training Y-coordinate model...")
    model_y = xgb.XGBRegressor(**params)
    model_y.fit(
        X_train, y_y_train,
        eval_set=[(X_val, y_y_val)],
        verbose=False
    )
    models_y.append(model_y)
    
    # Make predictions on validation set
    pred_x_val = model_x.predict(X_val)
    pred_y_val = model_y.predict(X_val)
    
    # Store out-of-fold predictions
    oof_predictions_x[val_idx] = pred_x_val
    oof_predictions_y[val_idx] = pred_y_val
    
    # Calculate metrics
    mse_x = mean_squared_error(y_x_val, pred_x_val)
    mae_x = mean_absolute_error(y_x_val, pred_x_val)
    mse_y = mean_squared_error(y_y_val, pred_y_val)
    mae_y = mean_absolute_error(y_y_val, pred_y_val)
    
    euclidean_error = np.sqrt((y_x_val - pred_x_val)**2 + (y_y_val - pred_y_val)**2)
    mean_euclidean_error = np.mean(euclidean_error)
    
    fold_metrics.append({
        'fold': fold_idx + 1,
        'mse_x': mse_x,
        'mae_x': mae_x,
        'rmse_x': np.sqrt(mse_x),
        'mse_y': mse_y,
        'mae_y': mae_y,
        'rmse_y': np.sqrt(mse_y),
        'mean_euclidean_error': mean_euclidean_error
    })
    
    print(f"  X-coordinate - RMSE: {np.sqrt(mse_x):.4f}, MAE: {mae_x:.4f}")
    print(f"  Y-coordinate - RMSE: {np.sqrt(mse_y):.4f}, MAE: {mae_y:.4f}")
    print(f"  Mean Euclidean Error: {mean_euclidean_error:.4f} meters")

# =============================================================================
# Overall Performance
# =============================================================================

print("\n" + "="*80)
print("Overall Out-of-Fold Performance")
print("="*80)

oof_mse_x = mean_squared_error(y_x, oof_predictions_x)
oof_mae_x = mean_absolute_error(y_x, oof_predictions_x)
oof_mse_y = mean_squared_error(y_y, oof_predictions_y)
oof_mae_y = mean_absolute_error(y_y, oof_predictions_y)

oof_euclidean_error = np.sqrt((y_x - oof_predictions_x)**2 + (y_y - oof_predictions_y)**2)
oof_mean_euclidean_error = np.mean(oof_euclidean_error)

print(f"\nX-coordinate:")
print(f"  RMSE: {np.sqrt(oof_mse_x):.4f} meters")
print(f"  MAE: {oof_mae_x:.4f} meters")

print(f"\nY-coordinate:")
print(f"  RMSE: {np.sqrt(oof_mse_y):.4f} meters")
print(f"  MAE: {oof_mae_y:.4f} meters")

print(f"\nMean Euclidean Error: {oof_mean_euclidean_error:.4f} meters")

# =============================================================================
# Store Results
# =============================================================================

results_df = training_data.copy()
results_df['pred_x_norm'] = oof_predictions_x
results_df['pred_y_norm'] = oof_predictions_y
results_df['error_x'] = y_x - oof_predictions_x
results_df['error_y'] = y_y - oof_predictions_y
results_df['euclidean_error'] = oof_euclidean_error

print(f"\nResults dataframe shape: {results_df.shape}")
print(f"\nSample predictions:")
print(results_df[['target_x_norm', 'pred_x_norm', 'target_y_norm', 'pred_y_norm', 'euclidean_error']].head(10))

# =============================================================================
# Feature Importance
# =============================================================================

print("\n" + "="*80)
print("Feature Importance (averaged across folds)")
print("="*80)

# Average feature importance across all folds
importance_x = np.mean([model.feature_importances_ for model in models_x], axis=0)
importance_y = np.mean([model.feature_importances_ for model in models_y], axis=0)

importance_df = pd.DataFrame({
    'Feature': feature_cols,
    'Importance_X': importance_x,
    'Importance_Y': importance_y
})
importance_df = importance_df.sort_values('Importance_X', ascending=False)

print("\nX-coordinate model:")
print(importance_df[['Feature', 'Importance_X']].to_string(index=False))

print("\nY-coordinate model:")
print(importance_df[['Feature', 'Importance_Y']].to_string(index=False))

# =============================================================================
# Visualization
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Predicted vs Actual X
ax = axes[0, 0]
ax.scatter(y_x, oof_predictions_x, alpha=0.3, s=10)
ax.plot([y_x.min(), y_x.max()], [y_x.min(), y_x.max()], 'r--', lw=2)
ax.set_xlabel('Actual X (normalized)', fontsize=12)
ax.set_ylabel('Predicted X (normalized)', fontsize=12)
ax.set_title(f'X Coordinate Prediction\nRMSE: {np.sqrt(oof_mse_x):.4f}m', fontsize=14)
ax.grid(True, alpha=0.3)

# 2. Predicted vs Actual Y
ax = axes[0, 1]
ax.scatter(y_y, oof_predictions_y, alpha=0.3, s=10)
ax.plot([y_y.min(), y_y.max()], [y_y.min(), y_y.max()], 'r--', lw=2)
ax.set_xlabel('Actual Y (normalized)', fontsize=12)
ax.set_ylabel('Predicted Y (normalized)', fontsize=12)
ax.set_title(f'Y Coordinate Prediction\nRMSE: {np.sqrt(oof_mse_y):.4f}m', fontsize=14)
ax.grid(True, alpha=0.3)

# 3. Error Distribution
ax = axes[1, 0]
ax.hist(oof_euclidean_error, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(oof_mean_euclidean_error, color='r', linestyle='--', linewidth=2, label=f'Mean: {oof_mean_euclidean_error:.2f}m')
ax.set_xlabel('Euclidean Error (meters)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Prediction Errors', fontsize=14)
ax.legend()
ax.grid(True, alpha=0.3)

# 4. Error vs Time to Target
ax = axes[1, 1]
ax.scatter(training_data['time_to_target'], oof_euclidean_error, alpha=0.3, s=10)
ax.set_xlabel('Time to Target (seconds)', fontsize=12)
ax.set_ylabel('Euclidean Error (meters)', fontsize=12)
ax.set_title('Error vs Time to Target', fontsize=14)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# Fold-wise Performance Summary
# =============================================================================

print("\n" + "="*80)
print("Fold-wise Performance Summary")
print("="*80)
metrics_df = pd.DataFrame(fold_metrics)
print(metrics_df.to_string(index=False))

print("\n" + "="*80)
print("Training Complete!")
print("="*80)
print(f"Total samples: {len(results_df)}")
print(f"Models trained: {len(models_x)} X-models, {len(models_y)} Y-models")
print(f"Out-of-fold predictions saved in 'results_df'")

