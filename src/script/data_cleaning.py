import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kloppy import skillcorner
# import torch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
pd.options.display.max_columns = None

# %%
match_ids = [
    1886347, 1899585, 1925299, 1953632, 1996435,
    2006229, 2011166, 2013725, 2015213, 2017461,
]

all_events = []
match_records = {}
all_tracking_segments = []

for match_id in match_ids:
    tracking_url = (
        f"https://media.githubusercontent.com/media/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_tracking_extrapolated.jsonl"
    )
    meta_url = (
        f"https://raw.githubusercontent.com/SkillCorner/opendata/master/"
        f"data/matches/{match_id}/{match_id}_match.json"
    )

    dataset = skillcorner.load(
        meta_data=meta_url,
        raw_data=tracking_url,
        coordinates="skillcorner",
        sample_rate=1,
    )

    event_path = f"../opendata/data/matches/{match_id}/{match_id}_dynamic_events.csv"
    event_df = pd.read_csv(event_path)

    records_by_frame = {}
    frame_records = []
    for frame in dataset.records:
        records_by_frame[frame.frame_id] = frame
        coords = frame.ball_coordinates
        frame_records.append(
            {
                "frame_id": frame.frame_id,
                "ball_x": coords.x if coords is not None else np.nan,
                "ball_y": coords.y if coords is not None else np.nan,
                "ball_z": getattr(coords, "z", np.nan) if coords is not None else np.nan,
            }
        )

    match_records[match_id] = records_by_frame

    frames_df = (
        pd.DataFrame(frame_records)
        .sort_values("frame_id")
        .reset_index(drop=True)
    )

    _event_ordered = event_df.copy()
    _event_ordered["_event_idx"] = _event_ordered.index
    _event_sorted = _event_ordered.sort_values("frame_start").reset_index(drop=True)

    merged_df = pd.merge_asof(
        _event_sorted,
        frames_df,
        left_on="frame_start",
        right_on="frame_id",
        direction="nearest",
    )
    merged_df["frame_diff"] = (merged_df["frame_start"] - merged_df["frame_id"]).abs()

    events_with_frames = (
        merged_df.sort_values("_event_idx")
        .drop(columns="_event_idx")
        .reset_index(drop=True)
    )
    events_with_frames["match_id"] = match_id
    
    passes_this_match = events_with_frames[events_with_frames["pass_outcome"].notna()].copy()
    passes_this_match = passes_this_match.sort_values("frame_start").reset_index(drop=True)
    
    for i in range(len(passes_this_match) - 1):
        pass_end_frame = passes_this_match.iloc[i]["frame_start"]
        next_pass_start_frame = passes_this_match.iloc[i + 1]["frame_start"]
        
        if next_pass_start_frame - pass_end_frame > 50:
            continue
        
        segment_frames = frames_df[
            (frames_df["frame_id"] >= pass_end_frame) & 
            (frames_df["frame_id"] <= next_pass_start_frame)
        ].copy()
        
        if len(segment_frames) > 0:
            segment_frames["match_id"] = match_id
            segment_frames["pass_event_id"] = passes_this_match.iloc[i]["event_id"]
            segment_frames["next_event_id"] = passes_this_match.iloc[i + 1]["event_id"]
            segment_frames["segment_id"] = f"{match_id}_{i}"
            all_tracking_segments.append(segment_frames)
    
    all_events.append(events_with_frames)

events_with_frames = pd.concat(all_events, ignore_index=True)
tracking_df = pd.concat(all_tracking_segments, ignore_index=True)

print(f"Total events with frames: {len(events_with_frames)}")
print(f"Total tracking frames between pass events: {len(tracking_df)}")

# Filter for high passes
segment_max_z = tracking_df.groupby('segment_id')['ball_z'].max().reset_index()
segment_max_z.columns = ['segment_id', 'max_z']

high_segments = segment_max_z[segment_max_z['max_z'] > 1.8]['segment_id']
tracking_df_filtered = tracking_df[tracking_df['segment_id'].isin(high_segments)].copy()

print(f"\n=== FILTERING RESULTS ===")
print(f"Original segments: {tracking_df['segment_id'].nunique()}")
print(f"High pass segments (max z > 1.8m): {len(high_segments)}")
print(f"Original tracking frames: {len(tracking_df)}")
print(f"Filtered tracking frames: {len(tracking_df_filtered)}")

def clean_high_pass_segment(segment_df, direction_change_threshold=45, cone_angle=30, 
                            max_ending_height=1.8, max_start_height=3.0, 
                            max_frame_distance=2.0):
    segment_df = segment_df.sort_values('frame_id').reset_index(drop=True)
    segment_df = segment_df[segment_df['ball_z'].notna()].reset_index(drop=True)
    
    if len(segment_df) == 0:
        return None, False, "no_data"
    
    start_height = segment_df['ball_z'].iloc[0]
    if start_height > max_start_height:
        return None, False, "start_too_high"
    
    x = segment_df['ball_x'].values
    y = segment_df['ball_y'].values
    z = segment_df['ball_z'].values
    
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    distances = np.sqrt(dx**2 + dy**2 + dz**2)
    
    if np.any(distances > max_frame_distance):
        return None, False, "frame_gap_too_large"
    
    landing_idx = None
    reached_high = False
    parabola_detected = False
    
    if len(z) >= 5:
        dz_dt = np.diff(z)
        ddz = np.diff(dz_dt)
        
        for i in range(len(z)):
            if z[i] > 1.8:
                reached_high = True
                break
        
        if reached_high:
            search_start = min(len(ddz), np.argmax(z) + 2)
            
            for i in range(search_start, len(ddz) - 1):
                if ddz[i] < 0 and ddz[i + 1] > 0:
                    landing_idx = i + 2
                    parabola_detected = True
                    break
    
    end_idx = len(segment_df)
    
    if landing_idx is not None:
        end_idx = min(end_idx, landing_idx + 1)
    elif reached_high:
        for i, z_val in enumerate(z):
            if z_val > 1.8:
                reached_high = True
            elif reached_high and z_val < 0.3:
                end_idx = i + 1
                break
    
    segment_df = segment_df.iloc[:end_idx].reset_index(drop=True)
    
    if len(segment_df) < 3:
        return (segment_df if len(segment_df) > 0 else None), parabola_detected, "too_short"
    
    x = segment_df['ball_x'].values
    y = segment_df['ball_y'].values
    
    dx = np.diff(x)
    dy = np.diff(y)
    
    angles = np.arctan2(dy, dx) * 180 / np.pi
    angle_changes = np.diff(angles)
    
    angle_changes = np.where(angle_changes > 180, angle_changes - 360, angle_changes)
    angle_changes = np.where(angle_changes < -180, angle_changes + 360, angle_changes)
    
    angle_changes_abs = np.abs(angle_changes)
    
    check_start = min(3, len(angle_changes_abs))
    sudden_changes = np.where(angle_changes_abs[check_start:] > direction_change_threshold)[0]
    
    if len(sudden_changes) > 0:
        truncate_idx = sudden_changes[0] + check_start + 2
        segment_df = segment_df.iloc[:truncate_idx].reset_index(drop=True)
    
    if len(segment_df) < 3:
        return (segment_df if len(segment_df) > 0 else None), parabola_detected, "too_short"
    
    final_height = segment_df['ball_z'].iloc[-1]
    if final_height > max_ending_height:
        return None, parabola_detected, "end_too_high"
    
    return (segment_df if len(segment_df) > 0 else None), parabola_detected, "cleaned"

# Clean segments
cleaned_segments = []
cleaning_stats = {
    'original_segments': len(high_segments),
    'cleaned_segments': 0,
    'removed_segments': 0,
    'removed_by_start_height': 0,
    'removed_by_frame_gap': 0,
    'removed_by_cone_filter': 0,
    'removed_by_end_height': 0,
    'removed_too_short': 0,
    'initial_frames_removed': 0,
    'ending_frames_removed': 0,
    'direction_change_truncations': 0,
    'parabola_detected': 0,
    'parabola_landing_truncations': 0,
    'original_frames': len(tracking_df_filtered),
    'cleaned_frames': 0
}

for seg_id in high_segments:
    segment = tracking_df_filtered[tracking_df_filtered['segment_id'] == seg_id].copy()
    original_len = len(segment)
    
    cleaned, parabola_detected, filter_reason = clean_high_pass_segment(
        segment, 
        direction_change_threshold=45, 
        cone_angle=30,
        max_start_height=3.0,
        max_frame_distance=2.0
    )
    
    if cleaned is not None and len(cleaned) > 0:
        cleaned_segments.append(cleaned)
        cleaning_stats['cleaned_segments'] += 1
        cleaning_stats['cleaned_frames'] += len(cleaned)
        
        if parabola_detected:
            cleaning_stats['parabola_detected'] += 1
            if len(cleaned) < original_len:
                cleaning_stats['parabola_landing_truncations'] += 1
        
        frames_removed = original_len - len(cleaned)
        if frames_removed > 0:
            if cleaned['frame_id'].iloc[0] > segment['frame_id'].iloc[0]:
                initial_removed = len(segment[segment['frame_id'] < cleaned['frame_id'].iloc[0]])
                cleaning_stats['initial_frames_removed'] += initial_removed
            if cleaned['frame_id'].iloc[-1] < segment['frame_id'].iloc[-1]:
                ending_removed = len(segment[segment['frame_id'] > cleaned['frame_id'].iloc[-1]])
                cleaning_stats['ending_frames_removed'] += ending_removed
                cleaning_stats['direction_change_truncations'] += 1
    else:
        cleaning_stats['removed_segments'] += 1
        if filter_reason == "start_too_high":
            cleaning_stats['removed_by_start_height'] += 1
        elif filter_reason == "frame_gap_too_large":
            cleaning_stats['removed_by_frame_gap'] += 1
        elif filter_reason == "cone_filter":
            cleaning_stats['removed_by_cone_filter'] += 1
        elif filter_reason == "end_too_high":
            cleaning_stats['removed_by_end_height'] += 1
        elif filter_reason == "too_short":
            cleaning_stats['removed_too_short'] += 1

tracking_df_cleaned = pd.concat(cleaned_segments, ignore_index=True) if cleaned_segments else pd.DataFrame()

# ============================================================================
# PLOT 1: Segment Count Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
categories = ['Original\nHigh Passes', 'After\nCleaning', 'Removed']
values = [cleaning_stats['original_segments'], 
          cleaning_stats['cleaned_segments'],
          cleaning_stats['removed_segments']]
colors_bar = ['blue', 'green', 'red']
bars = ax.bar(categories, values, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Segments', fontsize=12)
ax.set_title('Segment Cleaning Results', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 2, str(val), 
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plot1_segment_count.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 2: Frame Removal Breakdown
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
frame_categories = ['Original\nFrames', 'Cleaned\nFrames', 'Initial\nRemoved', 'Ending\nRemoved']
frame_values = [cleaning_stats['original_frames'],
                cleaning_stats['cleaned_frames'],
                cleaning_stats['initial_frames_removed'],
                cleaning_stats['ending_frames_removed']]
colors_frame = ['blue', 'green', 'orange', 'purple']
bars = ax.bar(frame_categories, frame_values, color=colors_frame, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Frames', fontsize=12)
ax.set_title('Frame Cleaning Breakdown', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, frame_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 50, str(val), 
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plot2_frame_breakdown.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 3: Parabola Detection Statistics
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
parabola_categories = ['Cleaned\nSegments', 'Parabola\nDetected', 'Parabola\nLanding\nTruncation']
parabola_values = [
    cleaning_stats['cleaned_segments'],
    cleaning_stats['parabola_detected'],
    cleaning_stats['parabola_landing_truncations']
]
colors_parabola = ['green', 'cyan', 'magenta']
bars = ax.bar(parabola_categories, parabola_values, color=colors_parabola, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Number of Segments', fontsize=12)
ax.set_title('Parabola Detection Statistics', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, parabola_values):
    ax.text(bar.get_x() + bar.get_width()/2, val + 1, str(val), 
            ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('plot3_parabola_stats.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 4: Cone Visualization (ROTATED to 0 degrees)
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
if len(cleaned_segments) > 0:
    n_samples = min(20, len(cleaned_segments))
    sample_indices = np.random.choice(len(cleaned_segments), n_samples, replace=False)
    
    colors_samples = plt.cm.viridis(np.linspace(0, 1, n_samples))
    
    for idx, seg_idx in enumerate(sample_indices):
        seg = cleaned_segments[seg_idx]
        seg = seg.sort_values('frame_id').reset_index(drop=True)
        
        x = seg['ball_x'].values
        y = seg['ball_y'].values
        
        # Shift to start at origin
        x_shifted = x - x[0]
        y_shifted = y - y[0]
        
        # Calculate initial direction (using first few frames)
        n_init = min(3, len(x) - 1)
        init_dx = x[n_init] - x[0]
        init_dy = y[n_init] - y[0]
        initial_angle = np.arctan2(init_dy, init_dx)
        
        # Rotate so initial velocity points to 0 degrees (positive x-axis)
        cos_theta = np.cos(-initial_angle)
        sin_theta = np.sin(-initial_angle)
        
        x_rotated = x_shifted * cos_theta - y_shifted * sin_theta
        y_rotated = x_shifted * sin_theta + y_shifted * cos_theta
        
        ax.plot(x_rotated, y_rotated, 'o-', color=colors_samples[idx], 
                linewidth=2, markersize=5, alpha=0.7, label=f'Pass {idx+1}')
        
        ax.scatter(0, 0, color=colors_samples[idx], s=100, marker='o', 
                  edgecolors='black', linewidths=2, zorder=5)
        
        ax.scatter(x_rotated[-1], y_rotated[-1], color=colors_samples[idx], 
                  s=100, marker='s', edgecolors='black', linewidths=2, zorder=5)
    
    # Draw cone at 0 degrees
    cone_angle = 30
    max_dist = 30
    
    # Initial velocity arrow pointing to 0 degrees
    ax.arrow(0, 0, 5, 0, head_width=1, head_length=0.5, fc='black', 
            ec='black', linewidth=2, alpha=0.7, zorder=3, label='Initial Velocity')
    
    # Cone boundaries
    angle1 = np.radians(cone_angle)
    angle2 = np.radians(-cone_angle)
    
    cone_x1 = [0, max_dist * np.cos(angle1)]
    cone_y1 = [0, max_dist * np.sin(angle1)]
    cone_x2 = [0, max_dist * np.cos(angle2)]
    cone_y2 = [0, max_dist * np.sin(angle2)]
    
    ax.plot(cone_x1, cone_y1, 'r--', linewidth=2, alpha=0.5, label='±30° Cone')
    ax.plot(cone_x2, cone_y2, 'r--', linewidth=2, alpha=0.5)
    
    # Fill cone
    theta = np.linspace(-cone_angle, cone_angle, 50)
    theta_rad = np.radians(theta)
    r = max_dist
    x_cone = r * np.cos(theta_rad)
    y_cone = r * np.sin(theta_rad)
    x_cone = np.concatenate([[0], x_cone, [0]])
    y_cone = np.concatenate([[0], y_cone, [0]])
    ax.fill(x_cone, y_cone, color='green', alpha=0.1)
    
    ax.set_xlabel('X (meters, rotated to initial direction)', fontsize=12)
    ax.set_ylabel('Y (meters, rotated to initial direction)', fontsize=12)
    ax.set_title('Pass Trajectories with ±30° Cone Filter\n(All trajectories rotated so initial velocity points to 0°)', 
                fontsize=14, fontweight='bold')
    # ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
    
plt.tight_layout()
plt.savefig('plot4_cone_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 5: Height Profile with 2nd Derivative
# ============================================================================
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()

if len(cleaned_segments) > 0:
    sample_seg_id = np.random.choice([s['segment_id'].iloc[0] for s in cleaned_segments])
    original_seg = tracking_df_filtered[tracking_df_filtered['segment_id'] == sample_seg_id].copy()
    original_seg = original_seg.sort_values('frame_id').reset_index(drop=True)
    
    cleaned_seg = tracking_df_cleaned[tracking_df_cleaned['segment_id'] == sample_seg_id]
    
    if len(cleaned_seg) > 0:
        cleaned_seg = cleaned_seg.sort_values('frame_id').reset_index(drop=True)
        
        ax1.plot(original_seg['frame_id'], original_seg['ball_z'], 
                'o-', color='blue', alpha=0.5, linewidth=2, markersize=6, label='Original Height')
        
        ax1.plot(cleaned_seg['frame_id'], cleaned_seg['ball_z'], 
                's-', color='green', linewidth=2, markersize=6, label='Cleaned Height')
        
        z = cleaned_seg['ball_z'].values
        if len(z) >= 5:
            dz = np.diff(z)
            ddz = np.diff(dz)
            frame_ids_ddz = cleaned_seg['frame_id'].values[2:]
            
            ax2.plot(frame_ids_ddz, ddz, 'v-', color='red', linewidth=1.5, 
                    markersize=4, alpha=0.7, label='2nd Derivative (d²z)')
            ax2.axhline(0, color='red', linestyle='--', linewidth=1, alpha=0.3)
            
            for i in range(len(ddz) - 1):
                if ddz[i] < 0 and ddz[i + 1] > 0:
                    landing_frame = frame_ids_ddz[i + 1]
                    ax1.axvline(landing_frame, color='magenta', linestyle=':', 
                               linewidth=2.5, alpha=0.8, label='Landing (d²z sign change)')
                    break
        
        ax1.axhline(0.3, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='z = 0.3m')
        ax1.axhline(1.8, color='red', linestyle='--', linewidth=2, alpha=0.7, label='z = 1.8m')
        
        ax1.set_xlabel('Frame ID', fontsize=12)
        ax1.set_ylabel('Ball Height (z, meters)', fontsize=12, color='blue')
        ax2.set_ylabel('2nd Derivative (d²z/dt²)', fontsize=12, color='red')
        ax1.set_title(f'Height Profile with 2nd Derivative\nSegment: {sample_seg_id}', 
                     fontsize=14, fontweight='bold')
        
        ax1.tick_params(axis='y', labelcolor='blue')
        ax2.tick_params(axis='y', labelcolor='red')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
        ax1.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot5_height_profile.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 6: XY Trajectory Comparison
# ============================================================================
fig, ax = plt.subplots(figsize=(10, 8))
if len(cleaned_segments) > 0:
    for seg_id in [s['segment_id'].iloc[0] for s in cleaned_segments[:3]]:
        original_seg = tracking_df_filtered[tracking_df_filtered['segment_id'] == seg_id].copy()
        original_seg = original_seg.sort_values('frame_id').reset_index(drop=True)
        cleaned_seg = tracking_df_cleaned[tracking_df_cleaned['segment_id'] == seg_id]
        
        if len(cleaned_seg) > 0 and len(original_seg) >= len(cleaned_seg):
            cleaned_seg = cleaned_seg.sort_values('frame_id').reset_index(drop=True)
            
            ax.plot(original_seg['ball_x'], original_seg['ball_y'], 
                    'o-', color='blue', alpha=0.3, linewidth=1.5, markersize=4)
            
            ax.plot(cleaned_seg['ball_x'], cleaned_seg['ball_y'], 
                    's-', color='green', linewidth=2, markersize=5)
            
            ax.scatter(cleaned_seg['ball_x'].iloc[0], cleaned_seg['ball_y'].iloc[0],
                      color='yellow', s=100, marker='o', edgecolors='black', 
                      linewidths=2, zorder=5)
            
            ax.scatter(cleaned_seg['ball_x'].iloc[-1], cleaned_seg['ball_y'].iloc[-1],
                      color='red', s=100, marker='X', edgecolors='black', 
                      linewidths=2, zorder=5)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Y (meters)', fontsize=12)
    ax.set_title('XY Trajectory Samples', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')

plt.tight_layout()
plt.savefig('plot6_xy_trajectory.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# PLOT 7: Distribution of Segment Lengths
# ============================================================================
fig, ax = plt.subplots(figsize=(8, 6))
if len(tracking_df_cleaned) > 0:
    segment_lengths_cleaned = tracking_df_cleaned.groupby('segment_id').size()
    ax.hist(segment_lengths_cleaned, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(segment_lengths_cleaned.mean(), color='red', linestyle='--', 
               linewidth=2, label=f'Mean: {segment_lengths_cleaned.mean():.1f}')
    ax.set_xlabel('Segment Length (frames)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Cleaned Segment Lengths', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot7_segment_lengths.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n✓ All plots saved as separate images!")
print(f"✓ Cleaned data has {len(tracking_df_cleaned)} frames across {tracking_df_cleaned['segment_id'].nunique()} segments.")

passes = events_with_frames[events_with_frames['pass_outcome'].notna()]

# Ensure correct ordering
passes = passes.sort_values(['match_id', 'event_id']).reset_index(drop=True)

# Extract numeric part of event_id
passes['event_num'] = passes['event_id'].str.split('_').str[1].astype(int)
passes['event_prefix'] = passes['event_id'].str.split('_').str[0]

# Create next event_id
passes['next_event_id'] = (
    passes['event_prefix'] + '_' + (passes['event_num'] + 1).astype(str)
)

# Merge to get target coordinates (including z)
passes_with_target = passes.merge(
    passes[['match_id', 'event_id', 'ball_x', 'ball_y', 'ball_z']],
    left_on=['match_id', 'next_event_id'],
    right_on=['match_id', 'event_id'],
    how='left',
    suffixes=('_start', '_end')
)

# Rename for clarity
passes_with_target = passes_with_target.rename(columns={
    'ball_x_start': 'pass_start_x',
    'ball_y_start': 'pass_start_y',
    'ball_z_start': 'pass_start_z',
    'ball_x_end': 'pass_target_x',
    'ball_y_end': 'pass_target_y',
    'ball_z_end': 'pass_target_z'
})

# Keep valid passes with 3D coordinates
valid_passes = passes_with_target[
    passes_with_target['pass_start_x'].notna() &
    passes_with_target['pass_target_x'].notna() &
    passes_with_target['pass_start_z'].notna() &
    passes_with_target['pass_target_z'].notna()
].copy()

print(f"Total passes with valid 3D coordinates: {len(valid_passes)}")

# Sample passes for visualization
sample_size = min(50, len(valid_passes))
sample_passes = valid_passes.sample(n=sample_size, random_state=42)

# -------------------------------------------------
# Pitch dimensions (centered at 0,0)
# -------------------------------------------------

pitch_length = 105  # meters
pitch_width = 68

half_length = pitch_length / 2
half_width = pitch_width / 2

# -------------------------------------------------
# Create 3D figure
# -------------------------------------------------

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Draw pitch outline on ground (z=0)
pitch_x = [-half_length, half_length, half_length, -half_length, -half_length]
pitch_y = [-half_width, -half_width, half_width, half_width, -half_width]
pitch_z = [0, 0, 0, 0, 0]
ax.plot(pitch_x, pitch_y, pitch_z, color='white', linewidth=2)

# Halfway line
ax.plot([0, 0], [-half_width, half_width], [0, 0], color='white', linewidth=2)

# Penalty areas
penalty_length = 16.5
penalty_width = 40.3

# Left penalty area
ax.plot(
    [-half_length, -half_length + penalty_length, -half_length + penalty_length, -half_length, -half_length],
    [-penalty_width / 2, -penalty_width / 2, penalty_width / 2, penalty_width / 2, -penalty_width / 2],
    [0, 0, 0, 0, 0],
    color='white',
    linewidth=1.5
)

# Right penalty area
ax.plot(
    [half_length, half_length - penalty_length, half_length - penalty_length, half_length, half_length],
    [-penalty_width / 2, -penalty_width / 2, penalty_width / 2, penalty_width / 2, -penalty_width / 2],
    [0, 0, 0, 0, 0],
    color='white',
    linewidth=1.5
)

# -------------------------------------------------
# Plot 3D passes with trajectories from tracking_df
# -------------------------------------------------

for _, row in sample_passes.iterrows():
    start_x = row['pass_start_x']
    start_y = row['pass_start_y']
    start_z = row['pass_start_z']
    end_x = row['pass_target_x']
    end_y = row['pass_target_y']
    end_z = row['pass_target_z']

    if row['pass_outcome'] == 'successful':
        color = 'cyan'
    else:
        color = 'red'

    # Try to get full trajectory from tracking_df if available
    pass_event_id = row['event_id_start']
    segment = tracking_df_cleaned[tracking_df_cleaned['pass_event_id'] == pass_event_id]
    
    if len(segment) > 1:
        # Plot full trajectory
        valid_segment = segment[segment['ball_x'].notna() & segment['ball_y'].notna() & segment['ball_z'].notna()]
        if len(valid_segment) > 1:
            ax.plot(
                valid_segment['ball_x'].values,
                valid_segment['ball_y'].values,
                valid_segment['ball_z'].values,
                color=color,
                linewidth=2,
                alpha=0.6
            )
    else:
        # Fall back to straight line
        ax.plot(
            [start_x, end_x],
            [start_y, end_y],
            [start_z, end_z],
            color=color,
            linewidth=2,
            alpha=0.6
        )

    # Plot start and end points
    ax.scatter(start_x, start_y, start_z, color='yellow', s=50, marker='o', edgecolors='black', linewidths=1)
    ax.scatter(end_x, end_y, end_z, color=color, s=50, marker='s', edgecolors='black', linewidths=1)

# -------------------------------------------------
# Axis & styling
# -------------------------------------------------

ax.set_xlim(-half_length - 5, half_length + 5)
ax.set_ylim(-half_width - 5, half_width + 5)
ax.set_zlim(0, 10)  # Height up to 10 meters

ax.set_xlabel('X (meters)', fontsize=12, labelpad=10)
ax.set_ylabel('Y (meters)', fontsize=12, labelpad=10)
ax.set_zlabel('Z (meters)', fontsize=12, labelpad=10)

# Set viewing angle
ax.view_init(elev=20, azim=45)

# Set background color
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.xaxis.pane.set_edgecolor('gray')
ax.yaxis.pane.set_edgecolor('gray')
ax.zaxis.pane.set_edgecolor('gray')
ax.grid(True, alpha=0.3)

# Title
successful_count = (sample_passes['pass_outcome'] == 'successful').sum()
ax.set_title(
    f'3D Pass Visualization\n'
    f'Successful: {successful_count}, '
    f'Unsuccessful: {sample_size - successful_count}',
    fontsize=16,
    fontweight='bold',
    pad=20
)

plt.tight_layout()
plt.show()

# -------------------------------------------------
# Summary statistics
# -------------------------------------------------

avg_distance = np.sqrt(
    (valid_passes['pass_target_x'] - valid_passes['pass_start_x'])**2 +
    (valid_passes['pass_target_y'] - valid_passes['pass_start_y'])**2
).mean()

print("\n=== Pass Statistics ===")
print(f"Total valid passes: {len(valid_passes)}")
print(f"Successful passes: {(valid_passes['pass_outcome'] == 'successful').sum()}")
print(f"Unsuccessful passes: {(valid_passes['pass_outcome'] != 'successful').sum()}")
print(f"Average pass distance: {avg_distance:.2f} meters")

