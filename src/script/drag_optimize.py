import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
from scipy.optimize import minimize
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
import multiprocessing
import math

# =============================================================================
# CONFIGURATION
# =============================================================================
N_INITIAL_FRAMES = 5
N_FOLDS = 5
RANDOM_STATE = 42
M_TO_FT = 3.28084

# =============================================================================
# PART 1: XGBOOST TRAINING (from Document 1)
# =============================================================================

def compute_velocity(coords, frame_ids, fps=10):
    """Compute velocity from coordinate sequence."""
    if len(coords) < 2:
        return np.array([0.0, 0.0, 0.0])
    
    dx = np.diff(coords[:, 0])
    dy = np.diff(coords[:, 1])
    dz = np.diff(coords[:, 2])
    dt = np.diff(frame_ids) / fps
    dt = np.where(dt == 0, 1e-6, dt)
    
    vx = dx / dt
    vy = dy / dt
    vz = dz / dt
    
    return np.array([np.mean(vx), np.mean(vy), np.mean(vz)])


def transform_to_normalized_coordinates(x, y, z, vx, vy):
    """Transform coordinates to normalized frame."""
    x_norm = x - x[0]
    y_norm = y - y[0]
    z_norm = z - z[0]
    
    initial_velocity_angle = np.arctan2(vy, vx)
    cos_theta = np.cos(-initial_velocity_angle)
    sin_theta = np.sin(-initial_velocity_angle)
    
    x_rot = x_norm * cos_theta - y_norm * sin_theta
    y_rot = x_norm * sin_theta + y_norm * cos_theta
    
    return x_rot, y_rot, z_norm, initial_velocity_angle


def create_training_samples(tracking_df, n_initial_frames=5):
    """Create training samples from tracking data."""
    samples = []
    
    for segment_id in tracking_df['segment_id'].unique():
        segment = tracking_df[tracking_df['segment_id'] == segment_id].copy()
        segment = segment.sort_values('frame_id').reset_index(drop=True)
        
        if len(segment) < n_initial_frames + 1:
            continue
        
        segment = segment[
            segment['ball_x'].notna() & 
            segment['ball_y'].notna() & 
            segment['ball_z'].notna()
        ]
        
        if len(segment) < n_initial_frames + 1:
            continue
        
        initial_segment = segment.iloc[:n_initial_frames]
        x_coords = initial_segment['ball_x'].values
        y_coords = initial_segment['ball_y'].values
        z_coords = initial_segment['ball_z'].values
        frame_ids = initial_segment['frame_id'].values
        
        coords_3d = np.column_stack([x_coords, y_coords, z_coords])
        velocity = compute_velocity(coords_3d, frame_ids)
        
        vx_initial, vy_initial, vz_initial = velocity
        velocity_magnitude = np.sqrt(vx_initial**2 + vy_initial**2)
        
        if velocity_magnitude < 0.1:
            continue
        
        for target_idx in range(n_initial_frames, len(segment)):
            target_row = segment.iloc[target_idx]
            
            all_x = np.append(x_coords, target_row['ball_x'])
            all_y = np.append(y_coords, target_row['ball_y'])
            all_z = np.append(z_coords, target_row['ball_z'])
            
            x_norm, y_norm, _, angle = transform_to_normalized_coordinates(
                all_x, all_y, all_z, vx_initial, vy_initial
            )
            
            target_x_norm = x_norm[-1]
            target_y_norm = y_norm[-1]
            target_z = all_z[-1]
            
            time_to_target = (target_row['frame_id'] - frame_ids[0]) / 10.0
            
            vx_norm = velocity_magnitude
            vy_norm = 0.0
            vz_norm = vz_initial
            
            sample = {
                'segment_id': segment_id,
                'target_frame_idx': target_idx,
                'x0': 0.0,
                'y0': 0.0,
                'z0': z_coords[0],
                'vx0': vx_norm,
                'vy0': vy_norm,
                'vz0': vz_norm,
                'v_magnitude': velocity_magnitude,
                'time_to_target': time_to_target,
                'original_x0': x_coords[0],
                'original_y0': y_coords[0],
                'original_z0': z_coords[0],
                'rotation_angle': angle,
                'target_x_norm': target_x_norm,
                'target_y_norm': target_y_norm,
                'target_z_norm': target_z,
                'num_frames': target_idx - 0,
            }
            
            samples.append(sample)
    
    return pd.DataFrame(samples)


def train_xgboost_models(training_data):
    """Train XGBoost models with cross-validation."""
    feature_cols = ['vx0', 'vy0', 'vz0', 'v_magnitude', 'time_to_target']
    
    X = training_data[feature_cols].values
    y_x = training_data['target_x_norm'].values
    y_y = training_data['target_y_norm'].values
    
    kfold = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    
    models_x = []
    models_y = []
    
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
    
    print("\nTraining XGBoost Models...")
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_x_train, y_x_val = y_x[train_idx], y_x[val_idx]
        y_y_train, y_y_val = y_y[train_idx], y_y[val_idx]
        
        model_x = xgb.XGBRegressor(**params)
        model_x.fit(X_train, y_x_train, eval_set=[(X_val, y_x_val)], verbose=False)
        models_x.append(model_x)
        
        model_y = xgb.XGBRegressor(**params)
        model_y.fit(X_train, y_y_train, eval_set=[(X_val, y_y_val)], verbose=False)
        models_y.append(model_y)
        
        print(f"Fold {fold_idx + 1}/{N_FOLDS} complete")
    
    return models_x, models_y, feature_cols


# =============================================================================
# PART 2: PHYSICS SIMULATION (from Document 2)
# =============================================================================

class SoccerPhysics:
    """Physics engine for soccer ball trajectory simulation."""
    def __init__(self):
        self.mass_oz = 15.0
        self.circumference_in = 27.0
        self.g = 32.174  # ft/s^2
        
        self.Temp_F = 78
        self.elev_ft = 0
        self.relative_humidity = 50
        self.barometric_pressure = 29.92
        
        # Calculate air density
        self.Temp_C = (5/9) * (self.Temp_F - 32)
        beta = 0.0001217
        SVP = 4.5841 * math.exp((18.687 - self.Temp_C / 234.5) * self.Temp_C / (257.14 + self.Temp_C))
        barometric_pressure_mmHg = self.barometric_pressure * 1000 / 39.37
        rho_kg_m3 = 1.2929 * (273 / (self.Temp_C + 273) * (barometric_pressure_mmHg * math.exp(-beta * self.elev_ft) - 0.3783 * self.relative_humidity * SVP / 100) / 760)
        self.rho_lb_ft3 = rho_kg_m3 * 0.06261
        
        self.base_c0 = 0.07182 * self.rho_lb_ft3 * (5.125 / self.mass_oz) * (self.circumference_in / 9.125)**2

    def get_derivatives(self, t, state, drag_mult, spins):
        """Calculate derivatives for RK4 integration."""
        x, y, z, vx, vy, vz = state
        backspin, sidespin = spins
        
        v = math.sqrt(vx*vx + vy*vy + vz*vz)
        if v == 0:
            return np.array([vx, vy, vz, 0, 0, 0])
        
        omega_side = sidespin * (2 * math.pi) / 60
        omega_back = backspin * (2 * math.pi) / 60
        
        wx = -omega_back
        wy = 0
        wz = omega_side
        omega = math.sqrt(wx*wx + wy*wy + wz*wz)
        
        if omega == 0:
            S = 0
            Cl = 0
            Cd = 0.4105 * drag_mult
        else:
            romega = (self.circumference_in / 2 / math.pi) * omega / 12
            tau = 25
            S = (romega / v) * math.exp(-t / (tau * 146.7 / v))
            Cd = (0.4105 * (1 + 0.2017 * S * S)) * drag_mult
            Cl = 1 / (2.32 + 0.4 / S)
        
        const = self.base_c0 * (Cl / omega) * v if omega > 0 else 0
        
        if omega > 0:
            aMagx = const * (wy*vz - wz*vy)
            aMagy = const * (wz*vx - wx*vz)
            aMagz = const * (wx*vy - wy*vx)
        else:
            aMagx, aMagy, aMagz = 0, 0, 0
        
        drag_const = -self.base_c0 * Cd * v
        ax = drag_const * vx + aMagx
        ay = drag_const * vy + aMagy
        az = drag_const * vz + aMagz - self.g
        
        return np.array([vx, vy, vz, ax, ay, az])

    def simulate(self, initial_state, params, times):
        """Run RK4 simulation."""
        drag_mult, sidespin, backspin = params
        state = np.array(initial_state, dtype=float)
        
        dt = 0.01
        max_t = times[-1] + dt
        n_steps = int(max_t / dt) + 5
        
        traj_t = np.zeros(n_steps)
        traj_pos = np.zeros((n_steps, 3))
        traj_t[0] = 0.0
        traj_pos[0] = state[:3]
        
        current_t = 0.0
        step_idx = 0
        
        while current_t < max_t and step_idx < n_steps - 1:
            k1 = self.get_derivatives(current_t, state, drag_mult, [backspin, sidespin])
            k2 = self.get_derivatives(current_t + dt/2, state + k1*dt/2, drag_mult, [backspin, sidespin])
            k3 = self.get_derivatives(current_t + dt/2, state + k2*dt/2, drag_mult, [backspin, sidespin])
            k4 = self.get_derivatives(current_t + dt, state + k3*dt, drag_mult, [backspin, sidespin])
            
            state = state + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
            current_t += dt
            step_idx += 1
            
            traj_t[step_idx] = current_t
            traj_pos[step_idx] = state[:3]
            
            if state[2] < 0:
                break
        
        traj_t = traj_t[:step_idx+1]
        traj_pos = traj_pos[:step_idx+1]
        
        try:
            interp_x = interp1d(traj_t, traj_pos[:, 0], kind='cubic', fill_value="extrapolate")
            interp_y = interp1d(traj_t, traj_pos[:, 1], kind='cubic', fill_value="extrapolate")
            interp_z = interp1d(traj_t, traj_pos[:, 2], kind='cubic', fill_value="extrapolate")
            
            sim_pos = np.column_stack((interp_x(times), interp_y(times), interp_z(times)))
            return sim_pos
        except:
            return np.zeros((len(times), 3))


# =============================================================================
# PART 3: HYBRID PREDICTION PIPELINE WITH SHARED DRAG
# =============================================================================

def derive_initial_conditions_xgb(segment, models_x, models_y, feature_cols, n_initial=5):
    """Use XGBoost to predict trajectory and derive initial velocities."""
    segment = segment.sort_values('frame_id').reset_index(drop=True)
    
    if len(segment) < n_initial:
        return None
    
    initial_segment = segment.iloc[:n_initial]
    x_coords = initial_segment['ball_x'].values
    y_coords = initial_segment['ball_y'].values
    z_coords = initial_segment['ball_z'].values
    frame_ids = initial_segment['frame_id'].values
    
    coords_3d = np.column_stack([x_coords, y_coords, z_coords])
    velocity = compute_velocity(coords_3d, frame_ids)
    vx_initial, vy_initial, vz_initial = velocity
    velocity_magnitude = np.sqrt(vx_initial**2 + vy_initial**2)
    
    if velocity_magnitude < 0.1:
        return None
    
    all_x = x_coords
    all_y = y_coords
    all_z = z_coords
    
    x_norm, y_norm, z_norm, angle = transform_to_normalized_coordinates(
        all_x, all_y, all_z, vx_initial, vy_initial
    )
    
    total_time = (segment['frame_id'].iloc[-1] - frame_ids[0]) / 10.0
    num_frames = len(segment)
    
    features = np.array([[
        velocity_magnitude,
        0.0,
        vz_initial,
        velocity_magnitude,
        total_time
    ]])
    
    pred_x = np.mean([model.predict(features)[0] for model in models_x])
    pred_y = np.mean([model.predict(features)[0] for model in models_y])
    
    vx_xgb = pred_x / total_time if total_time > 0 else velocity_magnitude
    vy_xgb = pred_y / total_time if total_time > 0 else 0.0
    
    z_initial = segment['ball_z'].iloc[0]
    z_final = segment['ball_z'].iloc[-1]
    
    g_m = 9.81
    vz_derived = (z_final - z_initial + 0.5 * g_m * total_time**2) / total_time if total_time > 0 else vz_initial
    
    return {
        'segment_id': segment['segment_id'].iloc[0],
        'x0': x_coords[0],
        'y0': y_coords[0],
        'z0': z_coords[0],
        'vx_xgb': vx_xgb,
        'vy_xgb': vy_xgb,
        'vz_derived': vz_derived,
        'vx_direct': vx_initial,
        'vy_direct': vy_initial,
        'vz_direct': vz_initial,
        'total_time': total_time,
        'num_frames': num_frames,
        'rotation_angle': angle,
        'segment_data': segment
    }


def prepare_segment_data(init_cond):
    """Prepare segment data for optimization."""
    if init_cond is None:
        return None
    
    segment = init_cond['segment_data']
    coords = segment[['ball_x', 'ball_y', 'ball_z']].values
    
    x_norm, y_norm, z_norm, _ = transform_to_normalized_coordinates(
        coords[:, 0], coords[:, 1], coords[:, 2],
        init_cond['vx_direct'], init_cond['vy_direct']
    )
    
    real_pos_m = np.column_stack([x_norm, y_norm, coords[:, 2]])
    real_pos_ft = real_pos_m * M_TO_FT
    real_pos_mapped = np.column_stack([real_pos_ft[:, 1], real_pos_ft[:, 0], real_pos_ft[:, 2]])
    
    times = np.arange(len(segment)) * 0.1
    
    vx_ft = init_cond['vx_xgb'] * M_TO_FT
    vy_ft = init_cond['vy_xgb'] * M_TO_FT
    vz_ft = init_cond['vz_derived'] * M_TO_FT
    
    initial_state = [
        real_pos_mapped[0, 0],
        real_pos_mapped[0, 1],
        real_pos_mapped[0, 2],
        vy_ft,
        vx_ft,
        vz_ft
    ]
    
    return {
        'segment_id': init_cond['segment_id'],
        'real_pos': real_pos_mapped,
        'times': times,
        'initial_state': initial_state,
        'vx_xgb': init_cond['vx_xgb'],
        'vy_xgb': init_cond['vy_xgb'],
        'vz_derived': init_cond['vz_derived']
    }


def optimize_spins_fixed_drag(segment_data, drag_mult, physics_engine):
    """Optimize spin rates for a single segment with fixed drag."""
    def objective(spins):
        params = [drag_mult, spins[0], spins[1]]  # drag is fixed, optimize spins
        sim_pos = physics_engine.simulate(segment_data['initial_state'], params, segment_data['times'])
        diff = sim_pos - segment_data['real_pos']
        weights = np.array([1.0, 1.0, 2.0])
        weighted_sq_error = (diff**2) * weights
        return np.mean(weighted_sq_error)
    
    x0 = [0.0, 100.0]  # [sidespin, backspin]
    bnds = ((-1500, 1500), (-1000, 3000))
    
    try:
        res = minimize(objective, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter': 50})
        final_path = physics_engine.simulate(
            segment_data['initial_state'], 
            [drag_mult, res.x[0], res.x[1]], 
            segment_data['times']
        )
        
        return {
            'segment_id': segment_data['segment_id'],
            'success': True,
            'sidespin_rpm': res.x[0],
            'backspin_rpm': res.x[1],
            'rmse': np.sqrt(res.fun),
            'real_path': segment_data['real_pos'],
            'sim_path': final_path,
            'times': segment_data['times'],
            'vx_xgb': segment_data['vx_xgb'],
            'vy_xgb': segment_data['vy_xgb'],
            'vz_derived': segment_data['vz_derived']
        }
    except Exception as e:
        return {'segment_id': segment_data['segment_id'], 'success': False, 'error': str(e)}


def evaluate_drag_value(drag_mult, all_segment_data, physics_engine):
    """Evaluate a single drag value across all segments (parallelizable)."""
    results = Parallel(n_jobs=-1, verbose=0)(
        delayed(optimize_spins_fixed_drag)(seg_data, drag_mult, physics_engine)
        for seg_data in all_segment_data
    )
    
    successful_results = [r for r in results if r.get('success', False)]
    if not successful_results:
        return 1e6
    
    total_error = sum(r['rmse']**2 for r in successful_results)
    avg_error = total_error / len(successful_results)
    return avg_error


def optimize_shared_drag(all_segment_data, physics_engine):
    """Optimize shared drag coefficient across all segments using grid search."""
    print("\n[4/5] Optimizing shared drag coefficient with parallel grid search...")
    
    # Grid search over drag values
    drag_candidates = np.linspace(0.1,2, 20)
    
    print(f"   Evaluating {len(drag_candidates)} drag values in parallel...")
    errors = Parallel(n_jobs=-1, verbose=10)(
        delayed(evaluate_drag_value)(drag, all_segment_data, physics_engine)
        for drag in drag_candidates
    )
    
    # Find best drag value
    best_idx = np.argmin(errors)
    optimal_drag = drag_candidates[best_idx]
    best_error = errors[best_idx]
    
    print(f"\n✓ Optimal shared drag coefficient: {optimal_drag:.4f} (RMSE: {np.sqrt(best_error):.4f} ft)")
    
    # Optionally refine with local optimization around best value
    print("   Refining with local optimization...")
    def objective(drag_val):
        return evaluate_drag_value(drag_val[0], all_segment_data, physics_engine)
    
    x0 = [optimal_drag]
    bnds = [(max(0.1, optimal_drag - 0.2), min(2, optimal_drag + 0.2))]
    
    res = minimize(objective, x0, method='L-BFGS-B', bounds=bnds, options={'maxiter': 5})
    refined_drag = res.x[0]
    
    print(f"✓ Refined drag coefficient: {refined_drag:.4f}")
    return refined_drag


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main(tracking_df):
    """Main pipeline with shared drag optimization."""
    print("="*80)
    print("SHARED DRAG SOCCER BALL TRAJECTORY PREDICTOR")
    print("="*80)
    
    # Step 1: Train XGBoost
    print("\n[1/5] Creating training samples...")
    training_data = create_training_samples(tracking_df, n_initial_frames=N_INITIAL_FRAMES)
    print(f"   Created {len(training_data)} samples from {training_data['segment_id'].nunique()} segments")
    
    print("\n[2/5] Training XGBoost models...")
    models_x, models_y, feature_cols = train_xgboost_models(training_data)
    print("   Training complete!")
    
    # Step 2: Derive initial conditions
    print("\n[3/5] Deriving initial conditions using XGBoost...")
    physics_engine = SoccerPhysics()
    
    unique_segments = tracking_df['segment_id'].unique()
    initial_conditions = []
    
    for seg_id in unique_segments:
        segment = tracking_df[tracking_df['segment_id'] == seg_id]
        init_cond = derive_initial_conditions_xgb(segment, models_x, models_y, feature_cols)
        if init_cond is not None:
            initial_conditions.append(init_cond)
    
    print(f"   Derived initial conditions for {len(initial_conditions)} segments")
    
    # Step 3: Prepare all segment data
    all_segment_data = [prepare_segment_data(ic) for ic in initial_conditions]
    all_segment_data = [sd for sd in all_segment_data if sd is not None]
    
    # Step 4: Optimize shared drag
    optimal_drag = optimize_shared_drag(all_segment_data, physics_engine)
    
    # Step 5: Final optimization with shared drag
    print(f"\n[5/5] Running final optimization with shared drag ({multiprocessing.cpu_count()} cores)...")
    results = Parallel(n_jobs=-1, verbose=5)(
        delayed(optimize_spins_fixed_drag)(seg_data, optimal_drag, physics_engine)
        for seg_data in all_segment_data
    )
    
    # Aggregate results
    df_results = pd.DataFrame([r for r in results if r and r.get('success', False)])
    df_results['drag_mult'] = optimal_drag
    
    if not df_results.empty:
        print("\n" + "="*80)
        print("RESULTS SUMMARY")
        print("="*80)
        print(f"\nShared Drag Coefficient: {optimal_drag:.4f}")
        print(f"\nSpin Statistics:")
        print(df_results[['segment_id', 'sidespin_rpm', 'backspin_rpm', 'rmse']].describe())
        
        # Visualize best fit
        best_idx = df_results['rmse'].idxmin()
        best_fit = df_results.loc[best_idx]
        
        fig = plt.figure(figsize=(15, 5))
        
        # Top view
        ax1 = fig.add_subplot(131)
        ax1.plot(best_fit['real_path'][:, 1], best_fit['real_path'][:, 0], 'ko-', label='Real', markersize=4)
        ax1.plot(best_fit['sim_path'][:, 1], best_fit['sim_path'][:, 0], 'r--', label='Physics Sim', linewidth=2)
        ax1.set_title(f"Top View (RMSE: {best_fit['rmse']:.3f} ft)\nSidespin: {best_fit['sidespin_rpm']:.1f} rpm")
        ax1.set_xlabel('Range (ft)')
        ax1.set_ylabel('Deviation (ft)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Side view
        ax2 = fig.add_subplot(132)
        ax2.plot(best_fit['real_path'][:, 1], best_fit['real_path'][:, 2], 'ko-', label='Real', markersize=4)
        ax2.plot(best_fit['sim_path'][:, 1], best_fit['sim_path'][:, 2], 'r--', label='Physics Sim', linewidth=2)
        ax2.set_title(f"Side View\nBackspin: {best_fit['backspin_rpm']:.1f} rpm")
        ax2.set_xlabel('Range (ft)')
        ax2.set_ylabel('Height (ft)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3D view
        ax3 = fig.add_subplot(133, projection='3d')
        ax3.plot(best_fit['real_path'][:, 0], best_fit['real_path'][:, 1], 
                 best_fit['real_path'][:, 2], 'ko-', label='Real', markersize=3)
        ax3.plot(best_fit['sim_path'][:, 0], best_fit['sim_path'][:, 1], 
                 best_fit['sim_path'][:, 2], 'r--', label='Physics Sim', linewidth=2)
        ax3.set_title(f"3D Trajectory\nShared Drag: {optimal_drag:.4f}")
        ax3.set_xlabel('X (ft)')
        ax3.set_ylabel('Y (ft)')
        ax3.set_zlabel('Z (ft)')
        ax3.legend()
        
        plt.tight_layout()
        plt.savefig('shared_drag_prediction.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"\nVisualization saved as 'shared_drag_prediction.png'")
        
    else:
        print("\n⚠ No successful optimizations found.")
    
    return df_results, optimal_drag, models_x, models_y


# Example usage:
results, optimal_drag, models_x, models_y = main(tracking_df_cleaned)