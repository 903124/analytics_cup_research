import os
import math
import copy
import warnings
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import RAdam
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')

# ============================================================================
# SOCCER-SPECIFIC CONFIGURATION
# ============================================================================

class SoccerConfig:
    # Data config
    MAX_PLAYERS = 22  # Same as NFL
    INPUT_FRAMES = 10  # Frames before segment to use as input
    OUTPUT_FRAMES = None  # Will be variable per segment (up to max)
    MAX_OUTPUT_FRAMES = 40  # Maximum segment length we'll predict
    
    # Feature dimensions
    DYNAMIC_FEATURES = 8  # x, y, sin(dir)*s, cos(dir)*s, dx_ball_land, dy_ball_land, dx_ball_curr, dy_ball_curr
    STATIC_FEATURES = 7   # offense/defense(1) + num_pred_frames(1) + num_input_frames(1) + ball_start_xy(2) + ball_land_xy(2)
    
    # Model config
    DYNAMIC_OUT_DIM = 64  # Per-feature after depthwise conv
    DYNAMIC_TOTAL_DIM = 512  # 8 features * 64 = 512
    STATIC_OUT_DIM = 64
    PLAYER_HIDDEN_DIM = 256
    
    # Transformer config
    NUM_TRANSFORMER_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Decoder config
    DECODER_HIDDEN = 1536
    DECODER_CHANNELS = 32
    
    # Training config
    N_FOLDS = 3
    N_REPEATS = 1
    EPOCHS = 150
    EVAL_EVERY = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EMA_DECAY = 0.9995
    
    # Augmentation config
    ROTATION_PROB = 0.5
    FLIP_PROB = 0.5

# Create config instance FIRST
soccer_config = SoccerConfig()

# ============================================================================
# SOCCER DATA EXTRACTION
# ============================================================================

def extract_player_data_for_frame(frame, match_id):
    """
    Extract player positions from a single frame.
    Returns DataFrame with columns: match_id, frame_id, player_id, team_id, jersey_no, x, y
    """
    players_list = []
    
    for player, data in frame.players_data.items():
        if data.coordinates is not None:
            players_list.append({
                'match_id': match_id,
                'frame_id': frame.frame_id,
                'player_id': player.player_id,
                'team_id': player.team.team_id if player.team else None,
                'jersey_no': player.jersey_no,
                'x': data.coordinates.x,
                'y': data.coordinates.y
            })
    
    return pd.DataFrame(players_list)


def build_soccer_tracking_data_updated(match_records, tracking_df_cleaned):
    """
    Build comprehensive tracking data with player positions and ball ownership.
    
    Args:
        match_records: Dict mapping match_id to frame records
        tracking_df_cleaned: DataFrame with cleaned ball trajectory segments
    
    Returns:
        player_tracking_df: DataFrame with all player positions
        ball_ownership_df: DataFrame with ball ownership per frame
    """
    all_player_frames = []
    all_ball_ownership = []
    
    for seg_id in tqdm(tracking_df_cleaned['segment_id'].unique(), desc="Extracting player data"):
        segment = tracking_df_cleaned[tracking_df_cleaned['segment_id'] == seg_id]
        match_id = segment['match_id'].iloc[0]
        
        # Get frames for this segment (including some frames before)
        min_frame = segment['frame_id'].min()
        max_frame = segment['frame_id'].max()
        
        # Get INPUT_FRAMES before segment start
        start_frame = max(0, min_frame - soccer_config.INPUT_FRAMES)
        
        # Extract player data for all frames
        frame_dict = match_records[match_id]
        for frame_id in range(start_frame, max_frame + 1):
            if frame_id in frame_dict:
                frame_record = frame_dict[frame_id]
                player_df = extract_player_data_for_frame(frame_record, match_id)
                player_df['segment_id'] = seg_id
                all_player_frames.append(player_df)
                
                # Extract ball ownership if available
                ball_owning_team = getattr(frame_record, 'ball_owning_team', None)
                if ball_owning_team is not None:
                    all_ball_ownership.append({
                        'match_id': match_id,
                        'frame_id': frame_id,
                        'segment_id': seg_id,
                        'ball_owning_team': ball_owning_team.team_id if hasattr(ball_owning_team, 'team_id') else ball_owning_team
                    })
    
    player_tracking_df = pd.concat(all_player_frames, ignore_index=True)
    ball_ownership_df = pd.DataFrame(all_ball_ownership)
    
    return player_tracking_df, ball_ownership_df


# ============================================================================
# SOCCER FEATURE ENGINEERING
# ============================================================================

class SoccerFeatureExtractor:
    """
    Extract dynamic and static features for soccer players.
    
    Dynamic Features (8-dim per frame, INPUT_FRAMES frames):
        - x, y coordinates
        - sin(dir) * s, cos(dir) * s - velocity direction weighted by speed
        - dx_ball_land, dy_ball_land - relative to ball landing position
        - dx_ball_curr, dy_ball_curr - relative to current ball position
    
    Static Features (7-dim):
        - offense/defense (1 if offense - same team as ball owner, 0 if defense)
        - Number of prediction frames (normalized)
        - Number of input frames used (normalized)
        - Ball start coordinates (2)
        - Ball landing coordinates (2)
    """
    
    def __init__(self, config: SoccerConfig):
        self.config = config
    
    def extract_dynamic_features(
        self,
        player_frames: pd.DataFrame,
        ball_frames: pd.DataFrame,
        ball_land_x: float,
        ball_land_y: float,
        num_input_frames: int = None
    ) -> np.ndarray:
        """
        Extract dynamic features for a single player.
        
        Args:
            player_frames: DataFrame with player tracking data (sorted by frame_id)
            ball_frames: DataFrame with ball positions for corresponding frames
            ball_land_x, ball_land_y: Ball landing coordinates
            num_input_frames: Number of frames to use
        
        Returns:
            np.ndarray of shape (8, INPUT_FRAMES)
        """
        if num_input_frames is None:
            num_input_frames = self.config.INPUT_FRAMES
        
        num_input_frames = max(1, num_input_frames)
        
        # Get last N frames
        frames = player_frames.tail(num_input_frames).copy()
        n_frames = len(frames)
        
        if n_frames == 0:
            frames = player_frames.tail(1).copy()
            n_frames = len(frames)
        
        # Initialize output array
        features = np.zeros((8, self.config.INPUT_FRAMES), dtype=np.float32)
        
        # Extract positions
        x = frames['x'].values.astype(np.float32)
        y = frames['y'].values.astype(np.float32)
        
        # Compute velocity
        if n_frames > 1:
            dx = np.diff(x)
            dy = np.diff(y)
            # Pad first frame with 0 velocity
            dx = np.concatenate([[0], dx])
            dy = np.concatenate([[0], dy])
        else:
            dx = np.zeros(n_frames, dtype=np.float32)
            dy = np.zeros(n_frames, dtype=np.float32)
        
        # Compute speed and direction
        speed = np.sqrt(dx**2 + dy**2)
        # Avoid division by zero
        speed_safe = np.where(speed > 1e-6, speed, 1.0)
        sin_dir_s = (dy / speed_safe) * speed
        cos_dir_s = (dx / speed_safe) * speed
        # Set to 0 where speed was too small
        sin_dir_s = np.where(speed > 1e-6, sin_dir_s, 0.0)
        cos_dir_s = np.where(speed > 1e-6, cos_dir_s, 0.0)
        
        # Get ball positions for corresponding frames
        ball_x = np.zeros(n_frames, dtype=np.float32)
        ball_y = np.zeros(n_frames, dtype=np.float32)
        
        for i, frame_id in enumerate(frames['frame_id'].values):
            ball_frame = ball_frames[ball_frames['frame_id'] == frame_id]
            if len(ball_frame) > 0:
                ball_x[i] = ball_frame['ball_x'].iloc[0]
                ball_y[i] = ball_frame['ball_y'].iloc[0]
        
        # Compute relative positions
        dx_ball_land = x - ball_land_x
        dy_ball_land = y - ball_land_y
        dx_ball_curr = x - ball_x
        dy_ball_curr = y - ball_y
        
        # Stack features: (8, n_frames)
        frame_features = np.stack([
            x, y,
            sin_dir_s, cos_dir_s,
            dx_ball_land, dy_ball_land,
            dx_ball_curr, dy_ball_curr
        ], axis=0)
        
        # Pad to INPUT_FRAMES
        if n_frames < self.config.INPUT_FRAMES:
            pad_width = self.config.INPUT_FRAMES - n_frames
            pad_values = frame_features[:, :1]
            padding = np.repeat(pad_values, pad_width, axis=1)
            features = np.concatenate([padding, frame_features], axis=1)
        else:
            features = frame_features[:, -self.config.INPUT_FRAMES:]
        
        return features
    
    def extract_static_features(
        self,
        player_team_id: int,
        ball_owning_team_id: int,
        num_output_frames: int,
        num_input_frames: int,
        ball_start_x: float,
        ball_start_y: float,
        ball_land_x: float,
        ball_land_y: float
    ) -> np.ndarray:
        """
        Extract static features for a single player.
        
        Args:
            player_team_id: Team ID of the player
            ball_owning_team_id: Team ID that owns the ball (from frame.ball_owning_team)
            num_output_frames: Number of frames to predict
            num_input_frames: Number of input frames used
            ball_start_x, ball_start_y: Ball start position
            ball_land_x, ball_land_y: Ball landing position
        
        Returns:
            np.ndarray of shape (7,)
        """
        features = np.zeros(7, dtype=np.float32)
        
        # Offense/Defense: 1 if player's team owns the ball (offense), 0 if defense
        features[0] = 1.0 if player_team_id == ball_owning_team_id else 0.0
        
        # Number of prediction frames (normalized)
        features[1] = num_output_frames / self.config.MAX_OUTPUT_FRAMES
        
        # Number of input frames used
        features[2] = num_input_frames / self.config.INPUT_FRAMES
        
        # Ball start coordinates (assuming field ~105m x 68m)
        features[3] = ball_start_x / 105.0
        features[4] = ball_start_y / 68.0
        
        # Ball landing coordinates
        features[5] = ball_land_x / 105.0
        features[6] = ball_land_y / 68.0
        
        return features


# ============================================================================
# SOCCER DATASET CLASS
# ============================================================================

class SoccerSegmentDataset(Dataset):
    """
    Dataset for soccer player trajectory prediction during ball segments.
    Optimized for faster data loading and processing.
    """
    
    def __init__(
        self,
        player_tracking_df: pd.DataFrame,
        ball_tracking_df: pd.DataFrame,
        config: SoccerConfig,
        training: bool = True
    ):
        self.config = config
        self.training = training
        self.extractor = SoccerFeatureExtractor(config)
        
        # OPTIMIZATION: Pre-group data by segment_id for O(1) access
        print("Pre-processing data for faster access...")
        self.segment_ball_data = ball_tracking_df.groupby('segment_id')
        self.segment_player_data = player_tracking_df.groupby('segment_id')
        
        # OPTIMIZATION: Pre-compute segment metadata
        self.segments = self._prepare_segments_optimized()
        print(f"Prepared {len(self.segments)} segments (training={training})")
    
    def _prepare_segments_optimized(self) -> List[Dict]:
        """Optimized segment preparation with pre-grouped data"""
        segments = []
        
        # Get all segment IDs that exist in both datasets
        ball_segments = set(self.segment_ball_data.groups.keys())
        player_segments = set(self.segment_player_data.groups.keys())
        valid_segments = ball_segments & player_segments
        
        for seg_id in tqdm(valid_segments, desc="Preparing segments"):
            # OPTIMIZATION: O(1) access instead of filtering
            seg_ball = self.segment_ball_data.get_group(seg_id).sort_values('frame_id')
            seg_players = self.segment_player_data.get_group(seg_id)
            
            # Pre-compute segment bounds
            min_frame = seg_ball['frame_id'].min()
            max_frame = seg_ball['frame_id'].max()
            
            # Pre-compute ball trajectory info
            ball_start_x = seg_ball['ball_x'].iloc[0]
            ball_start_y = seg_ball['ball_y'].iloc[0]
            ball_land_x = seg_ball['ball_x'].iloc[-1]
            ball_land_y = seg_ball['ball_y'].iloc[-1]
            num_output_frames = len(seg_ball)
            match_id = seg_ball['match_id'].iloc[0]
            
            # Pre-compute input frame range
            input_start_frame = min_frame - self.config.INPUT_FRAMES
            
            # OPTIMIZATION: Pre-filter and group players by availability in input frames
            input_players_data = seg_players[seg_players['frame_id'] < min_frame]
            if len(input_players_data) == 0:
                continue
                
            input_players = input_players_data['player_id'].unique()
            
            # OPTIMIZATION: Pre-group player data by player_id for O(1) access
            player_groups = {}
            for player_id in input_players:
                player_data = seg_players[seg_players['player_id'] == player_id]
                # Pre-sort by frame_id
                player_data_sorted = player_data.sort_values('frame_id')
                player_groups[player_id] = player_data_sorted
            
            # Get ball owning team
            ball_owning_team = getattr(seg_ball, 'ball_owning_team', None)
            if ball_owning_team is not None:
                ball_owning_team = seg_ball['ball_owning_team'].iloc[0]
            else:
                ball_owning_team = 0  # fallback
            
            segments.append({
                'segment_id': seg_id,
                'match_id': match_id,
                'player_ids': input_players,
                'player_groups': player_groups,  # OPTIMIZATION: Pre-grouped player data
                'ball_data': seg_ball,  # Pre-sorted
                'input_start_frame': input_start_frame,
                'segment_start_frame': min_frame,
                'segment_end_frame': max_frame,
                'ball_start_x': ball_start_x,
                'ball_start_y': ball_start_y,
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                'num_output_frames': num_output_frames,
                'ball_owning_team': ball_owning_team
            })
        
        return segments
    
    def __len__(self):
        return len(self.segments)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        segment = self.segments[idx]
        
        n_players = len(segment['player_ids'])
        
        # Initialize tensors
        dynamic_features = np.zeros(
            (self.config.MAX_PLAYERS, 8, self.config.INPUT_FRAMES),
            dtype=np.float32
        )
        
        static_features = np.zeros(
            (self.config.MAX_PLAYERS, 7),  # 7 features
            dtype=np.float32
        )
        
        targets = np.zeros(
            (self.config.MAX_PLAYERS, 2, self.config.MAX_OUTPUT_FRAMES),
            dtype=np.float32
        )
        
        player_mask = np.zeros(self.config.MAX_PLAYERS, dtype=np.float32)
        final_positions = np.zeros((self.config.MAX_PLAYERS, 2), dtype=np.float32)
        
        for i, player_id in enumerate(segment['player_ids'][:self.config.MAX_PLAYERS]):
            # OPTIMIZATION: O(1) access to pre-grouped player data
            player_data = segment['player_groups'][player_id]
            
            # OPTIMIZATION: Pre-split input and output frames
            player_input = player_data[player_data['frame_id'] < segment['segment_start_frame']]
            player_output = player_data[
                (player_data['frame_id'] >= segment['segment_start_frame']) &
                (player_data['frame_id'] <= segment['segment_end_frame'])
            ]
            
            if len(player_input) == 0:
                continue
            
            # Get player info (already sorted)
            team_id = player_input['team_id'].iloc[0]
            final_x, final_y = player_input['x'].iloc[-1], player_input['y'].iloc[-1]
            
            # Extract dynamic features
            dynamic_features[i] = self.extractor.extract_dynamic_features(
                player_input,  # Already sorted
                segment['ball_data'],  # Already sorted
                segment['ball_land_x'],
                segment['ball_land_y'],
                num_input_frames=self.config.INPUT_FRAMES
            )
            
            # Extract static features
            static_features[i] = self.extractor.extract_static_features(
                player_team_id=team_id,
                ball_owning_team_id=segment['ball_owning_team'],
                num_output_frames=segment['num_output_frames'],
                num_input_frames=self.config.INPUT_FRAMES,
                ball_start_x=segment['ball_start_x'],
                ball_start_y=segment['ball_start_y'],
                ball_land_x=segment['ball_land_x'],
                ball_land_y=segment['ball_land_y']
            )
            
            # Extract targets
            if len(player_output) > 0:
                output_x = player_output['x'].values[:self.config.MAX_OUTPUT_FRAMES]
                output_y = player_output['y'].values[:self.config.MAX_OUTPUT_FRAMES]
                n_out = len(output_x)
                
                targets[i, 0, :n_out] = output_x - final_x
                targets[i, 1, :n_out] = output_y - final_y
                
                # Pad with last displacement if needed
                if n_out < self.config.MAX_OUTPUT_FRAMES:
                    if n_out > 0:
                        targets[i, 0, n_out:] = targets[i, 0, n_out-1]
                        targets[i, 1, n_out:] = targets[i, 1, n_out-1]
            
            player_mask[i] = 1.0
            final_positions[i] = [final_x, final_y]
        
        return {
            'dynamic_features': torch.from_numpy(dynamic_features),
            'static_features': torch.from_numpy(static_features),
            'targets': torch.from_numpy(targets),
            'player_mask': torch.from_numpy(player_mask),
            'final_positions': torch.from_numpy(final_positions),
            'segment_id': segment['segment_id'],
            'num_output_frames': segment['num_output_frames']
        }
    
import os
import math
import copy
import warnings
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import RAdam
from sklearn.model_selection import GroupKFold
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

pd.options.display.max_columns = None
warnings.filterwarnings('ignore')

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ============================================================================
# CONFIGURATION
# ============================================================================

class SoccerConfig:
    # Data config
    MAX_PLAYERS = 22  # Same as NFL
    INPUT_FRAMES = 10  # Frames before segment to use as input
    OUTPUT_FRAMES = None  # Will be variable per segment (up to max)
    MAX_OUTPUT_FRAMES = 40  # Maximum segment length we'll predict
    
    # Feature dimensions
    DYNAMIC_FEATURES = 8  # x, y, sin(dir)*s, cos(dir)*s, dx_ball_land, dy_ball_land, dx_ball_curr, dy_ball_curr
    STATIC_FEATURES = 7   # offense/defense(1) + num_pred_frames(1) + num_input_frames(1) + ball_start_xy(2) + ball_land_xy(2)
    
    # Model config
    DYNAMIC_OUT_DIM = 64  # Per-feature after depthwise conv
    DYNAMIC_TOTAL_DIM = 512  # 8 features * 64 = 512
    STATIC_OUT_DIM = 64
    PLAYER_HIDDEN_DIM = 256
    
    # Transformer config
    NUM_TRANSFORMER_LAYERS = 3
    NUM_HEADS = 8
    DROPOUT = 0.1
    
    # Decoder config
    DECODER_HIDDEN = 1536
    DECODER_CHANNELS = 32
    
    # Training config
    N_FOLDS = 3
    N_REPEATS = 1
    EPOCHS = 150
    EVAL_EVERY = 5
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-3
    EMA_DECAY = 0.9995
    
    # Augmentation config
    ROTATION_PROB = 0.5
    FLIP_PROB = 0.5
    
    # Augmentation config
    ROTATION_PROB = 0.5
    MAX_EARLY_FRAMES = 20  # Up to 20 frames earlier for prediction
    FLIP_PROB = 0.5
    
    # Player roles (4 categories)
    PLAYER_ROLES = ['Defensive Coverage', 'Other Route Runner', 'Passer',
       'Targeted Receiver']
    
    # Outlier plays to remove
    OUTLIER_PLAYS = {
        (2023091100, 3167),  # too long
        (2023122100, 1450),  # too long
        (2023091001, 3216),  # no passer
        (2023112606, 4180),  # no passer
        (2023121009, 3594),  # no passer
    }

config = SoccerConfig()

class SoccerTrajectoryModel(nn.Module):
    """Main model for soccer player trajectory prediction."""
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.dynamic_encoder = SoccerDynamicFeatureEncoder(config)
        self.static_encoder = SoccerStaticFeatureEncoder(config)
        
        # Merge layer: 512 + 64 = 576 (same as before since we removed 1 and added 1 feature)
        self.merge = nn.Sequential(
            nn.Linear(config.DYNAMIC_TOTAL_DIM + config.STATIC_OUT_DIM, config.PLAYER_HIDDEN_DIM),
            nn.LayerNorm(config.PLAYER_HIDDEN_DIM),
            nn.SiLU()
        )
        
        # Inter-player interaction (reuse from NFL model)
        self.interaction = InterPlayerInteraction(config)
        
        # Decoder
        self.decoder = SoccerTrajectoryDecoder(config)
    
    def forward(self, dynamic_features, static_features, player_mask):
        batch_size = dynamic_features.size(0)
        n_players = dynamic_features.size(1)
        
        # Reshape for encoding
        dynamic_flat = dynamic_features.view(batch_size * n_players, 8, self.config.INPUT_FRAMES)
        static_flat = static_features.view(batch_size * n_players, -1)
        
        # Encode
        dynamic_encoded = self.dynamic_encoder(dynamic_flat).squeeze(-1)  # (B*P, 512)
        static_encoded = self.static_encoder(static_flat)  # (B*P, 64)
        
        # Merge
        merged = torch.cat([dynamic_encoded, static_encoded], dim=1)  # (B*P, 576)
        player_features = self.merge(merged)  # (B*P, 256)
        
        # Reshape for transformer
        player_features = player_features.view(batch_size, n_players, -1)
        
        # Inter-player interaction
        player_features = self.interaction(player_features, mask=player_mask.bool())
        
        # Reshape for decoder
        player_features = player_features.view(batch_size * n_players, -1)
        
        # Decode
        mean, var, aux_var = self.decoder(player_features)
        
        # Reshape outputs
        mean = mean.view(batch_size, n_players, 2, self.config.MAX_OUTPUT_FRAMES)
        var = var.view(batch_size, n_players, 2, self.config.MAX_OUTPUT_FRAMES)
        aux_var = aux_var.view(batch_size, n_players, 4, self.config.MAX_OUTPUT_FRAMES)
        
        return mean, var, aux_var


# %%
# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Remove outlier plays from dataframe using vectorized operations"""
    if df.empty:
        return df
    # Create tuple column for fast lookup
    play_keys = list(zip(df['game_id'], df['play_id']))
    mask = [key not in config.OUTLIER_PLAYS for key in play_keys]
    return df[mask].reset_index(drop=True)


def normalize_play_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Rotate plays where play_direction == 'left' by 180 degrees"""
    df = df.copy()
    left_mask = df['play_direction'] == 'left'
    
    if left_mask.any():
        # Rotate coordinates (flip x around field center, flip y around field center)
        df.loc[left_mask, 'x'] = 120 - df.loc[left_mask, 'x']
        df.loc[left_mask, 'y'] = 53.3 - df.loc[left_mask, 'y']
        
        # Rotate angles by 180 degrees
        for angle_col in ['dir', 'o']:
            if angle_col in df.columns:
                df.loc[left_mask, angle_col] = (df.loc[left_mask, angle_col] + 180) % 360
        
        # Flip ball landing coordinates
        if 'ball_land_x' in df.columns:
            df.loc[left_mask, 'ball_land_x'] = 120 - df.loc[left_mask, 'ball_land_x']
        if 'ball_land_y' in df.columns:
            df.loc[left_mask, 'ball_land_y'] = 53.3 - df.loc[left_mask, 'ball_land_y']
    
    return df


def load_all_data(weeks: range = range(1, 19)) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load all weekly data files and combine"""
    input_dfs = []
    output_dfs = []
    
    for week in weeks:
        week_str = f"{week:02d}"
        input_file = f"train/input_2023_w{week_str}.csv"
        output_file = f"train/output_2023_w{week_str}.csv"
        
        try:
            input_df = pd.read_csv(input_file)
            output_df = pd.read_csv(output_file)
            input_dfs.append(input_df)
            output_dfs.append(output_df)
            print(f"Week {week:2d}: input={len(input_df):,} rows, output={len(output_df):,} rows")
        except FileNotFoundError as e:
            print(f"Week {week:2d}: File not found")
    
    train_data = pd.concat(input_dfs, ignore_index=True) if input_dfs else pd.DataFrame()
    output_data = pd.concat(output_dfs, ignore_index=True) if output_dfs else pd.DataFrame()
    
    print(f"\nTotal before filtering: input={len(train_data):,} rows, output={len(output_data):,} rows")
    
    # Filter outliers
    train_data = filter_outliers(train_data)
    output_data = filter_outliers(output_data)
    
    # Normalize play direction
    train_data = normalize_play_direction(train_data)
    output_data_with_dir = output_data.merge(
        train_data[['game_id', 'play_id', 'nfl_id', 'play_direction']].drop_duplicates(),
        on=['game_id', 'play_id', 'nfl_id'],
        how='left'
    )
    output_data_normalized = normalize_play_direction(output_data_with_dir)
    output_data = output_data_normalized[['game_id', 'play_id', 'nfl_id', 'frame_id', 'x', 'y']]
    
    print(f"Total after filtering: input={len(train_data):,} rows, output={len(output_data):,} rows")
    
    return train_data, output_data

# Load data (just week 1 for initial testing)
print("Loading week 1 data for testing...")
# train_data, output_data = load_all_data(weeks=range(1, 2))

# %%
# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureExtractor:
    """
    Extract dynamic and static features for the model.
    
    Dynamic Features (10-dim per frame, 20 frames):
        - x, y coordinates
        - sin(o), cos(o) - orientation
        - sin(dir) * s, cos(dir) * s - velocity direction weighted by speed
        - x - ball_land_x, y - ball_land_y - relative to ball landing
        - x - receiver_x, y - receiver_y - relative to receiver (target)
    
    Static Features (12-dim):
        - One-hot player_role (4 dims)
        - Number of prediction frames (1)
        - Number of input frames used (1)
        - Passer's final-frame coordinates (2)
        - Ball landing coordinates (2)
        - Player's final-frame coordinates (2)
    """
    
    def __init__(self, config: SoccerConfig):
        self.config = config
        self.role_to_idx = {role: i for i, role in enumerate(config.PLAYER_ROLES)}
    
    def _deg_to_rad(self, degrees: np.ndarray) -> np.ndarray:
        """Convert degrees to radians"""
        return np.deg2rad(degrees)
    
    def extract_dynamic_features(
        self,
        player_frames: pd.DataFrame,
        ball_land_x: float,
        ball_land_y: float,
        receiver_x: float,
        receiver_y: float,
        num_input_frames: int = None
    ) -> np.ndarray:
        """
        Extract dynamic features for a single player.
        
        Args:
            player_frames: DataFrame with player tracking data (sorted by frame_id)
            ball_land_x, ball_land_y: Ball landing coordinates
            receiver_x, receiver_y: Receiver final position
            num_input_frames: Number of frames to use (for earlier frame augmentation)
        
        Returns:
            np.ndarray of shape (10, INPUT_FRAMES)
        """
        if num_input_frames is None:
            num_input_frames = self.config.INPUT_FRAMES
        
        # Ensure at least 1 frame
        num_input_frames = max(1, num_input_frames)
        
        # Get last N frames (or pad if fewer available)
        frames = player_frames.tail(num_input_frames).copy()
        n_frames = len(frames)
        
        # Handle empty frames case
        if n_frames == 0:
            frames = player_frames.tail(1).copy()
            n_frames = len(frames)
        
        # Initialize output array
        features = np.zeros((10, self.config.INPUT_FRAMES), dtype=np.float32)
        
        # Extract raw values
        x = frames['x'].values.astype(np.float32)
        y = frames['y'].values.astype(np.float32)
        s = frames['s'].values.astype(np.float32)  # speed
        o = self._deg_to_rad(frames['o'].values.astype(np.float32))  # orientation in radians
        dir_rad = self._deg_to_rad(frames['dir'].values.astype(np.float32))  # direction in radians
        
        # Compute features
        sin_o = np.sin(o)
        cos_o = np.cos(o)
        sin_dir_s = np.sin(dir_rad) * s
        cos_dir_s = np.cos(dir_rad) * s
        dx_ball = x - ball_land_x
        dy_ball = y - ball_land_y
        dx_recv = x - receiver_x
        dy_recv = y - receiver_y
        
        # Stack features: (10, n_frames)
        frame_features = np.stack([
            x, y,
            sin_o, cos_o,
            sin_dir_s, cos_dir_s,
            dx_ball, dy_ball,
            dx_recv, dy_recv
        ], axis=0)
        
        # Pad to INPUT_FRAMES (pad at the beginning with first frame values)
        if n_frames < self.config.INPUT_FRAMES:
            pad_width = self.config.INPUT_FRAMES - n_frames
            # Pad with the first available value (earliest frame)
            pad_values = frame_features[:, :1]  # Shape (10, 1)
            padding = np.repeat(pad_values, pad_width, axis=1)
            features = np.concatenate([padding, frame_features], axis=1)
        else:
            features = frame_features[:, -self.config.INPUT_FRAMES:]
        
        return features
    
    def extract_static_features(
        self,
        player_role: str,
        num_output_frames: int,
        num_input_frames: int,
        passer_x: float,
        passer_y: float,
        ball_land_x: float,
        ball_land_y: float,
        player_final_x: float,
        player_final_y: float
    ) -> np.ndarray:
        """
        Extract static features for a single player.
        
        Returns:
            np.ndarray of shape (12,)
        """
        features = np.zeros(12, dtype=np.float32)
        
        # One-hot encoding of player role (4 dims)
        role_idx = self.role_to_idx.get(player_role, 0)
        features[role_idx] = 1.0
        
        # Number of prediction frames (normalized)
        features[4] = num_output_frames / self.config.OUTPUT_FRAMES
        
        # Number of input frames used (for earlier frame augmentation)
        features[5] = num_input_frames / self.config.INPUT_FRAMES
        
        # Passer's final-frame coordinates (normalized)
        features[6] = passer_x / 120.0
        features[7] = passer_y / 53.3
        
        # Ball landing coordinates (normalized)
        features[8] = ball_land_x / 120.0
        features[9] = ball_land_y / 53.3
        
        # Player's final-frame coordinates (normalized)
        features[10] = player_final_x / 120.0
        features[11] = player_final_y / 53.3
        
        return features


# Test feature extractor
extractor = SoccerFeatureExtractor(config)
print("Feature extractor initialized.")


# %%
# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class DataAugmentation:
    """
    Data augmentation techniques:
    1. Rotation augmentation (50% chance, uniform 0-360°)
    2. Predicting from earlier frames (up to 20 frames earlier)
    3. Vertical flip (flip along X-axis)
    """
    
    def __init__(self, config: SoccerConfig):
        self.config = config
    
    def rotate_play(
        self,
        dynamic_features: np.ndarray,  # (players, 10, frames)
        static_features: np.ndarray,   # (players, 12)
        targets: np.ndarray,           # (players, 2, output_frames)
        center_x: float,
        center_y: float,
        angle_deg: float = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Rotate the entire play around the mean player position.
        
        Args:
            dynamic_features: (players, 10, frames) - features
            static_features: (players, 12) - static features
            targets: (players, 2, output_frames) - target displacements
            center_x, center_y: Center of rotation
            angle_deg: Rotation angle in degrees (random if None)
        """
        if angle_deg is None:
            angle_deg = np.random.uniform(0, 360)
        
        angle_rad = np.deg2rad(angle_deg)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        
        # Rotation matrix
        def rotate_point(x, y):
            x_centered = x - center_x
            y_centered = y - center_y
            x_rot = x_centered * cos_a - y_centered * sin_a + center_x
            y_rot = x_centered * sin_a + y_centered * cos_a + center_y
            return x_rot, y_rot
        
        # Rotate dynamic features
        dynamic_out = dynamic_features.copy()
        # x, y at indices 0, 1
        x_vals = dynamic_out[:, 0, :]
        y_vals = dynamic_out[:, 1, :]
        x_rot, y_rot = rotate_point(x_vals, y_vals)
        dynamic_out[:, 0, :] = x_rot
        dynamic_out[:, 1, :] = y_rot
        
        # Rotate orientation: indices 2, 3 are sin(o), cos(o)
        old_sin_o = dynamic_out[:, 2, :]
        old_cos_o = dynamic_out[:, 3, :]
        # Rotate angle by adding rotation angle
        dynamic_out[:, 2, :] = old_sin_o * cos_a + old_cos_o * sin_a
        dynamic_out[:, 3, :] = old_cos_o * cos_a - old_sin_o * sin_a
        
        # Rotate velocity direction: indices 4, 5 are sin(dir)*s, cos(dir)*s
        old_sin_dir_s = dynamic_out[:, 4, :]
        old_cos_dir_s = dynamic_out[:, 5, :]
        dynamic_out[:, 4, :] = old_sin_dir_s * cos_a + old_cos_dir_s * sin_a
        dynamic_out[:, 5, :] = old_cos_dir_s * cos_a - old_sin_dir_s * sin_a
        
        # Relative positions (indices 6-9) need to be rotated too
        # dx_ball, dy_ball at 6, 7
        dx_ball = dynamic_out[:, 6, :]
        dy_ball = dynamic_out[:, 7, :]
        dynamic_out[:, 6, :] = dx_ball * cos_a - dy_ball * sin_a
        dynamic_out[:, 7, :] = dx_ball * sin_a + dy_ball * cos_a
        
        # dx_recv, dy_recv at 8, 9
        dx_recv = dynamic_out[:, 8, :]
        dy_recv = dynamic_out[:, 9, :]
        dynamic_out[:, 8, :] = dx_recv * cos_a - dy_recv * sin_a
        dynamic_out[:, 9, :] = dx_recv * sin_a + dy_recv * cos_a
        
        # Rotate static features (coordinates)
        static_out = static_features.copy()
        # Passer coords at 6, 7
        px, py = static_out[:, 6] * 120, static_out[:, 7] * 53.3
        px_rot, py_rot = rotate_point(px, py)
        static_out[:, 6] = px_rot / 120.0
        static_out[:, 7] = py_rot / 53.3
        
        # Ball land coords at 8, 9
        bx, by = static_out[:, 8] * 120, static_out[:, 9] * 53.3
        bx_rot, by_rot = rotate_point(bx, by)
        static_out[:, 8] = bx_rot / 120.0
        static_out[:, 9] = by_rot / 53.3
        
        # Player final coords at 10, 11
        fx, fy = static_out[:, 10] * 120, static_out[:, 11] * 53.3
        fx_rot, fy_rot = rotate_point(fx, fy)
        static_out[:, 10] = fx_rot / 120.0
        static_out[:, 11] = fy_rot / 53.3
        
        # Rotate targets (displacements)
        targets_out = targets.copy()
        dx = targets_out[:, 0, :]
        dy = targets_out[:, 1, :]
        targets_out[:, 0, :] = dx * cos_a - dy * sin_a
        targets_out[:, 1, :] = dx * sin_a + dy * cos_a
        
        return dynamic_out, static_out, targets_out
    
    def vertical_flip(
        self,
        dynamic_features: np.ndarray,
        static_features: np.ndarray,
        targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Flip the play along the X-axis (vertical flip).
        y -> 53.3 - y, and negate y-components of direction/velocity
        """
        dynamic_out = dynamic_features.copy()
        static_out = static_features.copy()
        targets_out = targets.copy()
        
        # Flip y coordinates (index 1)
        dynamic_out[:, 1, :] = 53.3 - dynamic_out[:, 1, :]
        
        # Negate y-components
        dynamic_out[:, 2, :] = -dynamic_out[:, 2, :]  # sin(o) -> -sin(o)
        dynamic_out[:, 4, :] = -dynamic_out[:, 4, :]  # sin(dir)*s
        dynamic_out[:, 7, :] = -dynamic_out[:, 7, :]  # dy_ball
        dynamic_out[:, 9, :] = -dynamic_out[:, 9, :]  # dy_recv
        
        # Flip static y coordinates
        static_out[:, 7] = 1.0 - static_out[:, 7]   # passer_y
        static_out[:, 9] = 1.0 - static_out[:, 9]   # ball_land_y
        static_out[:, 11] = 1.0 - static_out[:, 11] # player_final_y
        
        # Negate target dy
        targets_out[:, 1, :] = -targets_out[:, 1, :]
        
        return dynamic_out, static_out, targets_out
    
    def apply_earlier_frame_augmentation(
        self,
        dynamic_features: np.ndarray,  # (players, 10, INPUT_FRAMES)
        targets: np.ndarray,           # (players, 2, OUTPUT_FRAMES)
        num_earlier_frames: int
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        Shift the prediction start point earlier by num_earlier_frames.
        This means we use fewer input frames and need to predict more output frames.
        
        Returns:
            dynamic_features: Modified dynamic features
            targets: Extended targets
            actual_output_frames: New number of output frames
        """
        if num_earlier_frames <= 0:
            return dynamic_features, targets, targets.shape[2]
        
        # Remove last N input frames
        dynamic_out = dynamic_features.copy()
        # Shift features: the last frame becomes earlier
        # We need to pad the end with the new "last frame"
        if num_earlier_frames < self.config.INPUT_FRAMES:
            # Use frames up to (INPUT_FRAMES - num_earlier_frames)
            effective_frames = self.config.INPUT_FRAMES - num_earlier_frames
            # Pad from the front to maintain INPUT_FRAMES
            dynamic_out = np.concatenate([
                np.repeat(dynamic_features[:, :, :1], num_earlier_frames, axis=2),
                dynamic_features[:, :, :effective_frames]
            ], axis=2)
        
        # Extend targets with the "transition" frames
        # In reality, we'd need the actual positions for those frames
        # For simplicity, we'll just note this extends the prediction window
        actual_output_frames = targets.shape[2] + num_earlier_frames
        
        return dynamic_out, targets, actual_output_frames


augmenter = DataAugmentation(config)
print("Data augmentation initialized.")

# %%
# ============================================================================
# DATASET CLASS
# ============================================================================

"""
Modified NFLPlayDataset with Random Frame Offset Augmentation
==============================================================
Changes the frame offset randomly for each epoch/sample during training.
"""

class NFLPlayDataset(Dataset):
    """
    Dataset for NFL play trajectory prediction.
    Each sample is a single play with up to 22 players.
    
    AUGMENTATION: Random frame offset applied per sample during training.
    This means each epoch will see different temporal windows of the same play.
    """
    
    def __init__(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame,
        config: SoccerConfig,
        training: bool = True,
        fixed_offset: int = None  # For validation/test: use fixed offset
    ):
        self.config = config
        self.training = training
        self.fixed_offset = fixed_offset  # None for training (random), set for val/test
        self.extractor = FeatureExtractor(config)
        self.augmenter = DataAugmentation(config)
        
        # Group by play
        self.plays = self._prepare_plays(input_data, output_data)
        print(f"Prepared {len(self.plays)} plays (training={training})")
    
    def _prepare_plays(
        self,
        input_data: pd.DataFrame,
        output_data: pd.DataFrame
    ) -> List[Dict]:
        """Prepare play-level data structures"""
        plays = []
        
        # Get unique plays
        play_keys = input_data[['game_id', 'play_id']].drop_duplicates()
        
        for _, row in tqdm(play_keys.iterrows(), total=len(play_keys), desc="Preparing plays"):
            game_id, play_id = row['game_id'], row['play_id']
            
            # Get all players in this play
            play_input = input_data[
                (input_data['game_id'] == game_id) & 
                (input_data['play_id'] == play_id)
            ].copy()
            
            play_output = output_data[
                (output_data['game_id'] == game_id) & 
                (output_data['play_id'] == play_id)
            ].copy()
            
            if play_input.empty:
                continue
            
            # Get play-level info
            ball_land_x = play_input['ball_land_x'].iloc[0]
            ball_land_y = play_input['ball_land_y'].iloc[0]
            num_output_frames = int(play_input['num_frames_output'].iloc[0])
            
            # Find passer and receiver
            passer_data = play_input[play_input['player_role'] == 'Passer']
            receiver_data = play_input[play_input['player_role'] == 'Targeted Receiver']
            
            if passer_data.empty:
                continue  # Skip plays without passer
            
            passer_final = passer_data.sort_values('frame_id').iloc[-1]
            passer_x, passer_y = passer_final['x'], passer_final['y']
            
            # Get receiver final position (use ball landing if no receiver)
            if not receiver_data.empty:
                receiver_final = receiver_data.sort_values('frame_id').iloc[-1]
                receiver_x, receiver_y = receiver_final['x'], receiver_final['y']
            else:
                receiver_x, receiver_y = ball_land_x, ball_land_y
            
            # Get unique players
            player_ids = play_input['nfl_id'].unique()
            
            plays.append({
                'game_id': game_id,
                'play_id': play_id,
                'player_ids': player_ids,
                'input_data': play_input,
                'output_data': play_output,
                'ball_land_x': ball_land_x,
                'ball_land_y': ball_land_y,
                'passer_x': passer_x,
                'passer_y': passer_y,
                'receiver_x': receiver_x,
                'receiver_y': receiver_y,
                'num_output_frames': num_output_frames
            })
        
        return plays
    
    def __len__(self):
        return len(self.plays)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        play = self.plays[idx]
        
        # Initialize tensors for all players
        n_players = len(play['player_ids'])
        
        # Dynamic features: (max_players, 10, INPUT_FRAMES)
        dynamic_features = np.zeros(
            (self.config.MAX_PLAYERS, 10, self.config.INPUT_FRAMES),
            dtype=np.float32
        )
        
        # Static features: (max_players, 12)
        static_features = np.zeros(
            (self.config.MAX_PLAYERS, 12),
            dtype=np.float32
        )
        
        # Targets: (max_players, 2, OUTPUT_FRAMES) - displacement from final input frame
        targets = np.zeros(
            (self.config.MAX_PLAYERS, 2, self.config.OUTPUT_FRAMES),
            dtype=np.float32
        )
        
        # Player mask: which players are valid
        player_mask = np.zeros(self.config.MAX_PLAYERS, dtype=np.float32)
        
        # Final positions (for converting displacement to absolute)
        final_positions = np.zeros((self.config.MAX_PLAYERS, 2), dtype=np.float32)
        
        # ============================================================
        # KEY CHANGE: Determine frame offset for this sample
        # ============================================================
        if self.training and self.fixed_offset is None:
            # Random offset for each sample during training
            # Can shift back up to MAX_EARLY_FRAMES
            max_offset = min(self.config.MAX_EARLY_FRAMES, self.config.INPUT_FRAMES - 1)
            num_earlier_frames = np.random.randint(0, max_offset + 1)
        else:
            # Fixed offset for validation/test (default: 0, no shift)
            num_earlier_frames = self.fixed_offset if self.fixed_offset is not None else 0
        
        # Calculate effective number of input frames to use
        effective_input_frames = self.config.INPUT_FRAMES - num_earlier_frames
        
        for i, nfl_id in enumerate(play['player_ids'][:self.config.MAX_PLAYERS]):
            player_input = play['input_data'][
                play['input_data']['nfl_id'] == nfl_id
            ].sort_values('frame_id')
            
            player_output = play['output_data'][
                play['output_data']['nfl_id'] == nfl_id
            ].sort_values('frame_id')
            
            if player_input.empty:
                continue
            
            # Get player info
            player_role = player_input['player_role'].iloc[0]
            
            # ============================================================
            # Apply frame offset: use frames up to (last - num_earlier_frames)
            # ============================================================
            if num_earlier_frames > 0 and len(player_input) > num_earlier_frames:
                # Remove the last N frames from input
                player_input_shifted = player_input.iloc[:-num_earlier_frames]
                
                # The "final frame" is now earlier
                if len(player_input_shifted) > 0:
                    final_frame = player_input_shifted.iloc[-1]
                else:
                    # Fallback if we removed too much
                    final_frame = player_input.iloc[-1]
                    player_input_shifted = player_input
            else:
                # No shift or not enough frames
                player_input_shifted = player_input
                final_frame = player_input.iloc[-1]
            
            final_x, final_y = final_frame['x'], final_frame['y']
            
            # Extract dynamic features with effective frame count
            dynamic_features[i] = self.extractor.extract_dynamic_features(
                player_input_shifted,
                play['ball_land_x'],
                play['ball_land_y'],
                play['receiver_x'],
                play['receiver_y'],
                num_input_frames=effective_input_frames
            )
            
            # Extract static features
            static_features[i] = self.extractor.extract_static_features(
                player_role=player_role,
                num_output_frames=play['num_output_frames'] + num_earlier_frames,  # Extended prediction
                num_input_frames=effective_input_frames,
                passer_x=play['passer_x'],
                passer_y=play['passer_y'],
                ball_land_x=play['ball_land_x'],
                ball_land_y=play['ball_land_y'],
                player_final_x=final_x,
                player_final_y=final_y
            )
            
            # Extract targets (displacement from final input position)
            if not player_output.empty:
                # ============================================================
                # Adjust target frames: if we shifted earlier, we need to predict
                # the frames that were originally in the input + all output frames
                # ============================================================
                if num_earlier_frames > 0:
                    # Get the frames that were removed from input
                    original_input_last_frames = player_input.iloc[-num_earlier_frames:]
                    
                    # Combine: removed input frames + output frames
                    combined_x = np.concatenate([
                        original_input_last_frames['x'].values,
                        player_output['x'].values
                    ])
                    combined_y = np.concatenate([
                        original_input_last_frames['y'].values,
                        player_output['y'].values
                    ])
                    
                    # Take up to OUTPUT_FRAMES
                    output_x = combined_x[:self.config.OUTPUT_FRAMES]
                    output_y = combined_y[:self.config.OUTPUT_FRAMES]
                else:
                    # Normal case: just use output frames
                    output_x = player_output['x'].values[:self.config.OUTPUT_FRAMES]
                    output_y = player_output['y'].values[:self.config.OUTPUT_FRAMES]
                
                n_out = len(output_x)
                
                # Target is displacement from final input position
                targets[i, 0, :n_out] = output_x - final_x
                targets[i, 1, :n_out] = output_y - final_y
                
                # Pad with last displacement if needed
                if n_out < self.config.OUTPUT_FRAMES:
                    if n_out > 0:
                        targets[i, 0, n_out:] = targets[i, 0, n_out-1]
                        targets[i, 1, n_out:] = targets[i, 1, n_out-1]
            
            player_mask[i] = 1.0
            final_positions[i] = [final_x, final_y]
        
        # Apply spatial augmentations during training
        if self.training:
            # Calculate center for rotation
            valid_mask = player_mask > 0
            if valid_mask.sum() > 0:
                center_x = dynamic_features[valid_mask, 0, -1].mean()
                center_y = dynamic_features[valid_mask, 1, -1].mean()
            else:
                center_x, center_y = 60.0, 26.65
            
            # Rotation augmentation (50% chance)
            if np.random.random() < self.config.ROTATION_PROB:
                dynamic_features, static_features, targets = self.augmenter.rotate_play(
                    dynamic_features, static_features, targets,
                    center_x, center_y
                )
            
            # Vertical flip (50% chance)
            if np.random.random() < self.config.FLIP_PROB:
                dynamic_features, static_features, targets = self.augmenter.vertical_flip(
                    dynamic_features, static_features, targets
                )
        
        return {
            'dynamic_features': torch.from_numpy(dynamic_features),  # (MAX_PLAYERS, 10, INPUT_FRAMES)
            'static_features': torch.from_numpy(static_features),    # (MAX_PLAYERS, 12)
            'targets': torch.from_numpy(targets),                    # (MAX_PLAYERS, 2, OUTPUT_FRAMES)
            'player_mask': torch.from_numpy(player_mask),            # (MAX_PLAYERS,)
            'final_positions': torch.from_numpy(final_positions),    # (MAX_PLAYERS, 2)
            'game_id': play['game_id'],
            'play_id': play['play_id'],
            'frame_offset': num_earlier_frames  # For debugging/analysis
        }


# # Create dataset for testing
# print("\nCreating dataset...")
# dataset = NFLPlayDataset(train_data, output_data, config, training=True)
# print(f"Dataset size: {len(dataset)}")

# # Test one sample
# sample = dataset[0]
# print(f"\nSample shapes:")
# for key, value in sample.items():
#     if isinstance(value, torch.Tensor):
#         print(f"  {key}: {value.shape}")

# # %%
# ============================================================================
# MODEL COMPONENTS
# ============================================================================

class DepthwiseConv1dBlock(nn.Module):
    """
    Depthwise separable Conv1d block.
    Each feature channel is processed independently.
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            groups=in_channels, padding=0  # No padding to emphasize last frame
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        # x: (batch, channels, frames)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x
class SoccerDynamicFeatureEncoder(nn.Module):
    """
    Encode dynamic features for soccer with 10 input frames.
    
    Input: (batch, 10 features, 10 frames)
    Output: (batch, 640, 1 frame)
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # Use fewer conv layers to handle shorter sequences
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=1),  # 10 -> 10
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),  # 10 -> 10
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),  # 10 -> 10
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.AdaptiveAvgPool1d(1)  # Pool to single frame
            )
            for _ in range(config.DYNAMIC_FEATURES)
        ])
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 10, 10)
        Returns:
            (batch * players, 640, 1)
        """
        feature_outputs = []
        for i, encoder in enumerate(self.feature_encoders):
            feat = x[:, i:i+1, :]  # (B, 1, 10)
            encoded = encoder(feat)  # (B, 64, 1)
            feature_outputs.append(encoded)
        
        return torch.cat(feature_outputs, dim=1)  # (B, 640, 1)

class DynamicFeatureEncoder(nn.Module):
    """
    Encode dynamic features using depthwise Conv1d.
    
    Input: (batch, 10 features, 20 frames)
    Output: (batch, 640, 1 frame)
    
    Each of the 10 features is processed by 7 depthwise Conv1d layers
    to produce 64-dim output, then concatenated to 640-dim.
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # 7 depthwise conv layers: 20 -> 18 -> 16 -> 14 -> 12 -> 10 -> 8 -> 6
        # Then we take only the last frame: 6 -> 1
        # Actually: 20 -> 18 -> 16 -> 14 -> 12 -> 10 -> 8 (7 layers of kernel=3)
        # We want to end up with 1 frame, so:
        # After 7 conv with kernel=3: 20 - 7*2 = 6 frames, then take last
        
        # Process each feature independently then project to 64 dims
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=0),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
            )
            for _ in range(config.DYNAMIC_FEATURES)
        ])
        
        # After 7 conv layers with kernel=3: 20 - 7*2 = 6 frames remaining
        # We take only the last frame
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 10, 20)
        Returns:
            (batch * players, 640, 1)
        """
        # Process each feature independently
        feature_outputs = []
        for i, encoder in enumerate(self.feature_encoders):
            feat = x[:, i:i+1, :]  # (B, 1, 20)
            encoded = encoder(feat)  # (B, 64, 6)
            # Take only the last frame
            encoded = encoded[:, :, -1:]  # (B, 64, 1)
            feature_outputs.append(encoded)
        
        # Concatenate all features
        out = torch.cat(feature_outputs, dim=1)  # (B, 640, 1)
        return out


class StaticFeatureEncoder(nn.Module):
    """
    Encode static features using Conv1d.
    
    Input: (batch * players, 12)
    Output: (batch * players, 64)
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.STATIC_FEATURES, 64),
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 12)
        Returns:
            (batch * players, 64)
        """
        return self.encoder(x)
class SoccerDynamicFeatureEncoder(nn.Module):
    """
    Encode dynamic features for soccer (8 features instead of 10).
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # Process each of 8 features independently
        self.feature_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=3, padding=0),
                nn.BatchNorm1d(16),
                nn.SiLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=0),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Conv1d(32, 64, kernel_size=3, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Conv1d(64, 64, kernel_size=2, padding=0),
                nn.BatchNorm1d(64),
                nn.SiLU(),
            )
            for _ in range(config.DYNAMIC_FEATURES)  # 8 features
        ])
        
        # After 4 conv layers: 10 - 4*2 = 2 frames remaining
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 8, 10)
        Returns:
            (batch * players, 512, 1)
        """
        feature_outputs = []
        for i, encoder in enumerate(self.feature_encoders):
            feat = x[:, i:i+1, :]  # (B, 1, 10)
            encoded = encoder(feat)  # (B, 64, 2)
            # Take only the last frame
            encoded = encoded[:, :, -1:]  # (B, 64, 1)
            feature_outputs.append(encoded)
        
        # Concatenate all features
        out = torch.cat(feature_outputs, dim=1)  # (B, 512, 1)
        return out


class SoccerStaticFeatureEncoder(nn.Module):
    """Encode static features (7 dims instead of 8)."""
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(config.STATIC_FEATURES, 64),  # 7 -> 64
            nn.BatchNorm1d(64),
            nn.SiLU()
        )
    
    def forward(self, x):
        return self.encoder(x)


class SoccerTrajectoryDecoder(nn.Module):
    """
    Decoder for variable-length trajectories.
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # 256 -> (32, MAX_OUTPUT_FRAMES)
        hidden_size = 32 * config.MAX_OUTPUT_FRAMES
        
        self.expand = nn.Sequential(
            nn.Linear(config.PLAYER_HIDDEN_DIM, hidden_size),
            nn.SiLU()
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        
        # Output: 2 (xy) + 2 (var) + 4 (aux var)
        self.output_head = nn.Conv1d(32, 8, kernel_size=1)
    
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.expand(x)  # (B, 32*MAX_OUTPUT_FRAMES)
        x = x.view(batch_size, 32, self.config.MAX_OUTPUT_FRAMES)
        x = self.conv_layers(x)
        out = self.output_head(x)  # (B, 8, MAX_OUTPUT_FRAMES)
        
        mean = out[:, :2, :]
        var_raw = out[:, 2:4, :]
        aux = out[:, 4:, :]
        
        var = F.softplus(var_raw) + 1e-3
        aux_var = F.softplus(aux) + 1e-3
        
        return mean, var, aux_var

class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, dim: int, hidden_dim: int = None, dropout: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w3(F.silu(self.w1(x)) * self.w2(x)))


class TransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with SwiGLU FFN.
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = SwiGLU(d_model, d_model * 4, dropout)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, players, d_model)
            mask: (batch, players) - True for valid players
        Returns:
            (batch, players, d_model)
        """
        # Create key_padding_mask (True means ignore/pad)
        key_padding_mask = None
        if mask is not None:
            # mask is (batch, players) with True for valid players
            # key_padding_mask needs True for positions to IGNORE
            key_padding_mask = ~mask  # Invert: True for invalid players
        
        # Self-attention
        x2 = self.norm1(x)
        x2, _ = self.self_attn(x2, x2, x2, key_padding_mask=key_padding_mask)
        x = x + self.dropout(x2)
        
        # FFN
        x2 = self.norm2(x)
        x2 = self.ffn(x2)
        x = x + x2
        
        return x


class InterPlayerInteraction(nn.Module):
    """
    Transformer encoder for inter-player interaction.
    
    Input: (batch, players, 256)
    Output: (batch, players, 256)
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                config.PLAYER_HIDDEN_DIM,
                config.NUM_HEADS,
                config.DROPOUT
            )
            for _ in range(config.NUM_TRANSFORMER_LAYERS)
        ])
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch, players, 256)
            mask: (batch, players) - True for valid players
        Returns:
            (batch, players, 256)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return x


print("Model components defined.")

# %%
# ============================================================================
# DECODER AND MAIN MODEL
# ============================================================================

class TrajectoryDecoder(nn.Module):
    """
    Decode player features to trajectory predictions.
    
    Input: (batch * players, 256)
    Output: 
        - trajectory: (batch * players, 2, 48) - mean xy for each frame
        - variance: (batch * players, 2, 48) - variance for GaussianNLL
        - auxiliary: (batch * players, 4, 48) - velocity and acceleration variances
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # 256 -> 1536 -> reshape to (32, 48)
        self.expand = nn.Sequential(
            nn.Linear(config.PLAYER_HIDDEN_DIM, config.DECODER_HIDDEN),
            nn.SiLU()
        )
        
        # Reshape: 1536 -> (32, 48)
        # 1536 = 32 * 48
        
        # Conv1d refinement layers
        self.conv_layers = nn.Sequential(
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.SiLU(),
        )
        
        # Output heads
        # Main output: 2 (xy mean) + 2 (xy variance)
        # Auxiliary: 2 (velocity variance) + 2 (acceleration variance)
        self.output_head = nn.Conv1d(32, 2 + 2 + 4, kernel_size=1)
    
    def forward(self, x):
        """
        Args:
            x: (batch * players, 256)
        Returns:
            mean: (batch * players, 2, 48)
            var: (batch * players, 2, 48)
            aux: (batch * players, 4, 48)
        """
        batch_size = x.size(0)
        
        # Expand
        x = self.expand(x)  # (B, 1536)
        
        # Reshape to (B, 32, 48)
        x = x.view(batch_size, 32, self.config.OUTPUT_FRAMES)
        
        # Conv refinement
        x = self.conv_layers(x)  # (B, 32, 48)
        
        # Output
        out = self.output_head(x)  # (B, 8, 48)
        
        # Split outputs
        mean = out[:, :2, :]  # (B, 2, 48) - xy displacement
        var_raw = out[:, 2:4, :]  # (B, 2, 48) - xy variance
        aux = out[:, 4:, :]  # (B, 4, 48) - auxiliary (velocity/accel variance)
        
        # Constrain variance to positive: softplus(var) + 1e-3
        var = F.softplus(var_raw) + 1e-3
        aux_var = F.softplus(aux) + 1e-3
        
        return mean, var, aux_var


class NFLTrajectoryModel(nn.Module):
    """
    Main model for NFL player trajectory prediction.
    
    Architecture:
        1. Dynamic Feature Encoder: (10, 20) -> (640, 1)
        2. Static Feature Encoder: (12,) -> (64,)
        3. Merge: (640 + 64) -> 256
        4. Inter-player Interaction: Transformer x 3
        5. Decoder: 256 -> (2, 48) + auxiliary
    """
    
    def __init__(self, config: SoccerConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.dynamic_encoder = SoccerDynamicFeatureEncoder(config)
        self.static_encoder = StaticFeatureEncoder(config)
        
        # Merge layer
        self.merge = nn.Sequential(
            nn.Linear(config.DYNAMIC_TOTAL_DIM + config.STATIC_OUT_DIM, config.PLAYER_HIDDEN_DIM),
            nn.LayerNorm(config.PLAYER_HIDDEN_DIM),
            nn.SiLU()
        )
        
        # Inter-player interaction
        self.interaction = InterPlayerInteraction(config)
        
        # Decoder
        self.decoder = TrajectoryDecoder(config)
    
    def forward(
        self,
        dynamic_features: torch.Tensor,
        static_features: torch.Tensor,
        player_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            dynamic_features: (batch, players, 10, 20)
            static_features: (batch, players, 12)
            player_mask: (batch, players) - 1 for valid players
        
        Returns:
            mean: (batch, players, 2, 48) - predicted xy displacement
            var: (batch, players, 2, 48) - variance
            aux_var: (batch, players, 4, 48) - auxiliary variance
        """
        batch_size = dynamic_features.size(0)
        n_players = dynamic_features.size(1)
        
        # Reshape for encoding: (batch * players, ...)
        dynamic_flat = dynamic_features.view(batch_size * n_players, 10, self.config.INPUT_FRAMES)
        static_flat = static_features.view(batch_size * n_players, -1)
        
        # Encode features
        dynamic_encoded = self.dynamic_encoder(dynamic_flat)  # (B*P, 640, 1)
        dynamic_encoded = dynamic_encoded.squeeze(-1)  # (B*P, 640)
        
        static_encoded = self.static_encoder(static_flat)  # (B*P, 64)
        
        # Merge
        merged = torch.cat([dynamic_encoded, static_encoded], dim=1)  # (B*P, 704)
        player_features = self.merge(merged)  # (B*P, 256)
        
        # Reshape for transformer: (batch, players, 256)
        player_features = player_features.view(batch_size, n_players, -1)
        
        # Inter-player interaction
        player_features = self.interaction(
            player_features, 
            mask=player_mask.bool()
        )  # (B, P, 256)
        
        # Reshape for decoder: (batch * players, 256)
        player_features = player_features.view(batch_size * n_players, -1)
        
        # Decode
        mean, var, aux_var = self.decoder(player_features)
        
        # Reshape outputs: (batch, players, ...)
        mean = mean.view(batch_size, n_players, 2, self.config.OUTPUT_FRAMES)
        var = var.view(batch_size, n_players, 2, self.config.OUTPUT_FRAMES)
        aux_var = aux_var.view(batch_size, n_players, 4, self.config.OUTPUT_FRAMES)
        
        return mean, var, aux_var


# Test model
print("Testing updated soccer model...")
model = SoccerTrajectoryModel(soccer_config).to(device)  # Use SoccerTrajectoryModel

# Count parameters
n_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {n_params:,}")

# Test forward pass - UPDATED for 7 static features instead of 12
test_dynamic = torch.randn(2, soccer_config.MAX_PLAYERS, 8, soccer_config.INPUT_FRAMES).to(device)  # 8 dynamic features
test_static = torch.randn(2, soccer_config.MAX_PLAYERS, 7).to(device)  # 7 static features (updated!)
test_mask = torch.ones(2, soccer_config.MAX_PLAYERS).to(device)
test_mask[:, 15:] = 0  # Only 15 players valid

with torch.no_grad():
    mean, var, aux_var = model(test_dynamic, test_static, test_mask)

print(f"Output shapes:")
print(f"  mean: {mean.shape}")
print(f"  var: {var.shape}")
print(f"  aux_var: {aux_var.shape}")

# %%
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class TrajectoryLoss(nn.Module):
    """
    Combined loss for trajectory prediction.
    
    Main loss: GaussianNLLLoss for trajectory prediction
    Auxiliary loss: GaussianNLLLoss for velocity and acceleration
    """
    
    def __init__(self, config: SoccerConfig, aux_weight: float = 0.1):
        super().__init__()
        self.config = config
        self.aux_weight = aux_weight
        self.gaussian_nll = nn.GaussianNLLLoss(reduction='none')
    
    def compute_derivatives(self, trajectory: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute velocity (1st difference) and acceleration (2nd difference).
        
        Args:
            trajectory: (batch, players, 2, frames)
        
        Returns:
            velocity: (batch, players, 2, frames-1)
            acceleration: (batch, players, 2, frames-2)
        """
        velocity = trajectory[..., 1:] - trajectory[..., :-1]
        acceleration = velocity[..., 1:] - velocity[..., :-1]
        return velocity, acceleration
    
    def forward(
        self,
        pred_mean: torch.Tensor,
        pred_var: torch.Tensor,
        pred_aux_var: torch.Tensor,
        target: torch.Tensor,
        player_mask: torch.Tensor,
        num_valid_frames: int = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            pred_mean: (batch, players, 2, 48) - predicted displacement
            pred_var: (batch, players, 2, 48) - variance
            pred_aux_var: (batch, players, 4, 48) - aux variance (vel_var, accel_var)
            target: (batch, players, 2, 48) - target displacement
            player_mask: (batch, players) - 1 for valid players
            num_valid_frames: number of valid output frames (default: all)
        
        Returns:
            Dictionary with loss components
        """
        if num_valid_frames is None:
            num_valid_frames = self.config.MAX_OUTPUT_FRAMES
        
        # Expand player mask for broadcasting
        mask = player_mask.unsqueeze(-1).unsqueeze(-1)  # (B, P, 1, 1)
        
        # Main trajectory loss (GaussianNLL)
        main_loss = self.gaussian_nll(
            pred_mean[..., :num_valid_frames],
            target[..., :num_valid_frames],
            pred_var[..., :num_valid_frames]
        )

        main_loss = (main_loss * mask).sum() / (mask.sum() * 2 * num_valid_frames + 1e-8)
        
        # Compute velocity and acceleration from predictions and targets
        pred_vel, pred_accel = self.compute_derivatives(pred_mean)
        target_vel, target_accel = self.compute_derivatives(target)
        
        # Velocity loss
        vel_var = pred_aux_var[:, :, :2, :-1]  # (B, P, 2, 47)
        vel_frames = min(num_valid_frames - 1, vel_var.size(-1))
        vel_loss = self.gaussian_nll(
            pred_vel[..., :vel_frames],
            target_vel[..., :vel_frames],
            vel_var[..., :vel_frames]
        )
        vel_loss = (vel_loss * mask).sum() / (mask.sum() * 2 * vel_frames + 1e-8)
        
        # Acceleration loss
        accel_var = pred_aux_var[:, :, 2:, :-2]  # (B, P, 2, 46)
        accel_frames = min(num_valid_frames - 2, accel_var.size(-1))
        if accel_frames > 0:
            accel_loss = self.gaussian_nll(
                pred_accel[..., :accel_frames],
                target_accel[..., :accel_frames],
                accel_var[..., :accel_frames]
            )
            accel_loss = (accel_loss * mask).sum() / (mask.sum() * 2 * accel_frames + 1e-8)
        else:
            accel_loss = torch.tensor(0.0, device=pred_mean.device)
        
        # Combined auxiliary loss
        aux_loss = vel_loss + accel_loss
        
        # Total loss
        total_loss = main_loss + self.aux_weight * aux_loss
        
        return {
            'total': total_loss,
            'main': main_loss,
            'velocity': vel_loss,
            'acceleration': accel_loss,
            'auxiliary': aux_loss
        }


# ============================================================================
# EXPONENTIAL MOVING AVERAGE
# ============================================================================

class EMA:
    """
    Exponential Moving Average for model parameters.
    """
    
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """Restore original parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# Test loss function
loss_fn = TrajectoryLoss(config)
print("Loss function initialized.")

# Test EMA
ema = EMA(model)
print(f"EMA initialized with decay={config.EMA_DECAY}")

# %%
# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for DataLoader."""
    return {
        'dynamic_features': torch.stack([b['dynamic_features'] for b in batch]),
        'static_features': torch.stack([b['static_features'] for b in batch]),
        'targets': torch.stack([b['targets'] for b in batch]),
        'player_mask': torch.stack([b['player_mask'] for b in batch]),
        'start_positions': torch.stack([b['start_positions'] for b in batch]),
        'final_positions': torch.stack([b['final_positions'] for b in batch]),
        'segment_ids': [b['segment_id'] for b in batch]
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: TrajectoryLoss,
    ema: EMA,
    device: torch.device,
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    total_main_loss = 0.0
    total_aux_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        pred_mean, pred_var, pred_aux_var = model(dynamic, static, mask)
        
        # Compute loss
        losses = loss_fn(pred_mean, pred_var, pred_aux_var, targets, mask)
        
        # Backward pass
        losses['total'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update EMA
        ema.update()
        
        # Accumulate losses
        total_loss += losses['total'].item()
        total_main_loss += losses['main'].item()
        total_aux_loss += losses['auxiliary'].item()
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'main': f"{losses['main'].item():.4f}",
            'aux': f"{losses['auxiliary'].item():.4f}"
        })
    
    return {
        'loss': total_loss / n_batches,
        'main_loss': total_main_loss / n_batches,
        'aux_loss': total_aux_loss / n_batches
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: TrajectoryLoss,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate model on validation set."""
    model.eval()
    
    total_loss = 0.0
    total_main_loss = 0.0
    total_mse = 0.0  # Changed from total_mae
    n_batches = 0
    
    for batch in tqdm(val_loader, desc="Evaluating"):
        # Move to device
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        
        # Forward pass
        pred_mean, pred_var, pred_aux_var = model(dynamic, static, mask)
        
        # Compute loss
        losses = loss_fn(pred_mean, pred_var, pred_aux_var, targets, mask)
        
        # Compute MSE (Mean Squared Error) for trajectory
        # pred_mean is displacement, targets is displacement
        squared_error = (pred_mean - targets) ** 2
        mse = (squared_error * mask.unsqueeze(-1).unsqueeze(-1)).sum() / (mask.sum() * 2 * config.MAX_OUTPUT_FRAMES + 1e-8)
        
        total_loss += losses['total'].item()
        total_main_loss += losses['main'].item()
        total_mse += mse.item()
        n_batches += 1
    
    # Compute RMSE from average MSE
    avg_mse = total_mse / n_batches
    rmse = avg_mse ** 0.5
    
    return {
        'loss': total_loss / n_batches,
        'main_loss': total_main_loss / n_batches,
        'rmse': rmse,
        'mse': avg_mse  # Also return MSE for reference
    }

print("Training utilities defined with RMSE evaluation.")

# %%
# %%
# ============================================================================
# LOAD WEIGHTS AND COLLECT CV PREDICTIONS
# ============================================================================

def collect_cv_results_from_checkpoints(
    dataset: NFLPlayDataset,
    config: SoccerConfig,
    device: torch.device,
    checkpoint_dir: str = "checkpoints",
    checkpoint_pattern: str = "model_repeat*_fold*.pt"
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Load saved model checkpoints and collect CV predictions without retraining.
    
    Args:
        dataset: Dataset containing all data
        config: Model configuration
        device: Device for inference
        checkpoint_dir: Directory containing checkpoint files
        checkpoint_pattern: Pattern to match checkpoint files (glob pattern)
    
    Returns:
        Dictionary with fold names as keys and (pred_xy, actual_xy) tuples
    """
    import glob
    import re
    
    cv_predictions = {}
    
    # Find all checkpoint files
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, checkpoint_pattern))
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching pattern: {checkpoint_pattern}")
        return cv_predictions
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    # Get game_ids for creating fold splits
    game_ids = np.array([play['game_id'] for play in dataset.plays])
    
    for checkpoint_path in sorted(checkpoint_files):
        # Extract fold info from filename
        filename = os.path.basename(checkpoint_path)
        
        # Parse: model_repeat1_fold2_valloss0.1234.pt
        match = re.search(r'repeat(\d+)_fold(\d+)', filename)
        if not match:
            print(f"Skipping {filename} - couldn't parse fold info")
            continue
        
        repeat_num = int(match.group(1))
        fold_num = int(match.group(2))
        fold_name = f"repeat{repeat_num}_fold{fold_num}"
        
        print(f"\nProcessing {fold_name}...")
        print(f"  Loading checkpoint: {filename}")
        
        # Initialize model
        model = NFLTrajectoryModel(config).to(device)
        
        # Load checkpoint
        try:
            model, ema, epoch, val_loss = load_model(
                model=model,
                checkpoint_path=checkpoint_path,
                device=device,
                load_optimizer=False
            )
        except Exception as e:
            print(f"  Error loading checkpoint: {e}")
            continue
        
        # Recreate the same fold split
        # Note: This assumes the same split logic as training
        n_folds = config.N_FOLDS
        
        # Use GroupKFold with the same seed
        np.random.seed(42 + repeat_num - 1)
        unique_games = np.unique(game_ids)
        shuffled_games = unique_games.copy()
        np.random.shuffle(shuffled_games)
        
        kfold = GroupKFold(n_splits=n_folds)
        
        # Get the specific fold split
        for current_fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)), groups=game_ids), 1):
            if current_fold == fold_num:
                # Create validation subset
                val_subset = torch.utils.data.Subset(dataset, val_idx)
                val_loader = DataLoader(
                    val_subset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                    pin_memory=True
                )
                
                print(f"  Validation samples: {len(val_idx)}")
                
                # Apply EMA weights and collect predictions
                ema.apply_shadow()
                pred_xy, actual_xy, _ = collect_cv_predictions(model, val_loader, device)
                ema.restore()
                
                cv_predictions[fold_name] = (pred_xy, actual_xy)
                print(f"  Collected {len(pred_xy)} valid predictions")
                
                break
    
    return cv_predictions


def evaluate_single_checkpoint(
    checkpoint_path: str,
    dataset: NFLPlayDataset,
    config: SoccerConfig,
    device: torch.device,
    val_indices: List[int] = None
) -> Dict[str, float]:
    """
    Evaluate a single checkpoint on specified validation indices.
    
    Args:
        checkpoint_path: Path to checkpoint file
        dataset: Full dataset
        config: Model configuration
        device: Device for inference
        val_indices: Validation indices (if None, uses all data)
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\nEvaluating checkpoint: {os.path.basename(checkpoint_path)}")
    
    # Initialize model
    model = NFLTrajectoryModel(config).to(device)
    
    # Load checkpoint
    model, ema, epoch, val_loss = load_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        load_optimizer=False
    )
    
    # Create dataloader
    if val_indices is not None:
        val_subset = torch.utils.data.Subset(dataset, val_indices)
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    else:
        val_loader = DataLoader(
            dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=True
        )
    
    # Evaluate with EMA weights
    loss_fn = TrajectoryLoss(config)
    ema.apply_shadow()
    metrics = evaluate(model, val_loader, loss_fn, device)
    ema.restore()
    
    # Collect predictions
    ema.apply_shadow()
    pred_xy, actual_xy, _ = collect_cv_predictions(model, val_loader, device)
    ema.restore()
    
    # Calculate additional metrics
    mae_x = np.mean(np.abs(pred_xy[:, 0] - actual_xy[:, 0]))
    mae_y = np.mean(np.abs(pred_xy[:, 1] - actual_xy[:, 1]))
    overall_mae = (mae_x + mae_y) / 2
    
    print(f"Checkpoint Epoch: {epoch}")
    print(f"Val Loss: {metrics['loss']:.4f}")
    print(f"Val RMSE: {metrics['rmse']:.4f}")
    print(f"Overall MAE: {overall_mae:.4f} (X: {mae_x:.4f}, Y: {mae_y:.4f})")
    
    return {
        'epoch': epoch,
        'loss': metrics['loss'],
        'rmse': metrics['rmse'],
        'mae': overall_mae,
        'mae_x': mae_x,
        'mae_y': mae_y,
        'pred_xy': pred_xy,
        'actual_xy': actual_xy
    }


print("Checkpoint loading and evaluation functions defined.")

# %%
# ============================================================================
# CV PREDICTION COLLECTION AND PLOTTING
# ============================================================================

@torch.no_grad()
def collect_cv_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect predictions and actual values from validation set.
    
    Returns:
        pred_xy: (N, 2) array of predicted (x, y) positions
        actual_xy: (N, 2) array of actual (x, y) positions
        mask: (N,) boolean array indicating valid samples
    """
    model.eval()
    
    all_pred_x = []
    all_pred_y = []
    all_actual_x = []
    all_actual_y = []
    all_masks = []
    
    for batch in tqdm(val_loader, desc="Collecting predictions"):
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        
        # Forward pass
        pred_mean, _, _ = model(dynamic, static, mask)  # (B, P, 2, 48)
        
        # Convert displacement to absolute positions
        final_pos_expanded = final_pos.unsqueeze(-1)  # (B, P, 2, 1)
        pred_abs = pred_mean + final_pos_expanded  # (B, P, 2, 48)
        target_abs = targets + final_pos_expanded  # (B, P, 2, 48)
        
        # Take the last frame prediction (frame 48)
        pred_last = pred_abs[:, :, :, -1]  # (B, P, 2)
        target_last = target_abs[:, :, :, -1]  # (B, P, 2)
        
        # Flatten and collect
        B, P = mask.shape
        mask_flat = mask.cpu().numpy().flatten()  # (B*P,)
        pred_x_flat = pred_last[:, :, 0].cpu().numpy().flatten()  # (B*P,)
        pred_y_flat = pred_last[:, :, 1].cpu().numpy().flatten()  # (B*P,)
        actual_x_flat = target_last[:, :, 0].cpu().numpy().flatten()  # (B*P,)
        actual_y_flat = target_last[:, :, 1].cpu().numpy().flatten()  # (B*P,)
        
        all_masks.append(mask_flat)
        all_pred_x.append(pred_x_flat)
        all_pred_y.append(pred_y_flat)
        all_actual_x.append(actual_x_flat)
        all_actual_y.append(actual_y_flat)
    
    # Concatenate all batches
    mask_all = np.concatenate(all_masks)
    pred_x_all = np.concatenate(all_pred_x)
    pred_y_all = np.concatenate(all_pred_y)
    actual_x_all = np.concatenate(all_actual_x)
    actual_y_all = np.concatenate(all_actual_y)
    
    # Filter by mask
    valid_mask = mask_all > 0.5
    pred_xy = np.stack([pred_x_all[valid_mask], pred_y_all[valid_mask]], axis=1)
    actual_xy = np.stack([actual_x_all[valid_mask], actual_y_all[valid_mask]], axis=1)
    
    return pred_xy, actual_xy, valid_mask


def plot_cv_results(
    cv_predictions: Dict[str, Tuple[np.ndarray, np.ndarray]],
    save_path: str = "cv_results_plot.png"
):
    """
    Plot CV results: predicted vs actual XY positions.
    
    Args:
        cv_predictions: Dictionary with fold names as keys and (pred_xy, actual_xy) tuples as values
        save_path: Path to save the plot
    """
    n_folds = len(cv_predictions)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, n_folds, figsize=(5*n_folds, 10))
    if n_folds == 1:
        axes = axes.reshape(2, 1)
    
    fold_names = sorted(cv_predictions.keys())
    
    for idx, fold_name in enumerate(fold_names):
        pred_xy, actual_xy = cv_predictions[fold_name]
        
        # X coordinate subplot
        ax_x = axes[0, idx]
        ax_x.scatter(actual_xy[:, 0], pred_xy[:, 0], alpha=0.3, s=1)
        
        # Add diagonal line (perfect prediction)
        min_val = min(actual_xy[:, 0].min(), pred_xy[:, 0].min())
        max_val = max(actual_xy[:, 0].max(), pred_xy[:, 0].max())
        ax_x.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate metrics
        mae_x = np.mean(np.abs(pred_xy[:, 0] - actual_xy[:, 0]))
        rmse_x = np.sqrt(np.mean((pred_xy[:, 0] - actual_xy[:, 0])**2))
        
        ax_x.set_xlabel('Actual X', fontsize=12)
        ax_x.set_ylabel('Predicted X', fontsize=12)
        ax_x.set_title(f'{fold_name}\nMAE: {mae_x:.3f}, RMSE: {rmse_x:.3f}', fontsize=12)
        ax_x.legend()
        ax_x.grid(True, alpha=0.3)
        
        # Y coordinate subplot
        ax_y = axes[1, idx]
        ax_y.scatter(actual_xy[:, 1], pred_xy[:, 1], alpha=0.3, s=1)
        
        # Add diagonal line
        min_val = min(actual_xy[:, 1].min(), pred_xy[:, 1].min())
        max_val = max(actual_xy[:, 1].max(), pred_xy[:, 1].max())
        ax_y.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect')
        
        # Calculate metrics
        mae_y = np.mean(np.abs(pred_xy[:, 1] - actual_xy[:, 1]))
        rmse_y = np.sqrt(np.mean((pred_xy[:, 1] - actual_xy[:, 1])**2))
        
        ax_y.set_xlabel('Actual Y', fontsize=12)
        ax_y.set_ylabel('Predicted Y', fontsize=12)
        ax_y.set_title(f'{fold_name}\nMAE: {mae_y:.3f}, RMSE: {rmse_y:.3f}', fontsize=12)
        ax_y.legend()
        ax_y.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {save_path}")
    plt.show()
    
    # Print overall statistics
    print("\n" + "="*50)
    print("Overall CV Statistics")
    print("="*50)
    
    all_pred_x = []
    all_pred_y = []
    all_actual_x = []
    all_actual_y = []
    
    for fold_name in fold_names:
        pred_xy, actual_xy = cv_predictions[fold_name]
        all_pred_x.append(pred_xy[:, 0])
        all_pred_y.append(pred_xy[:, 1])
        all_actual_x.append(actual_xy[:, 0])
        all_actual_y.append(actual_xy[:, 1])
    
    all_pred_x = np.concatenate(all_pred_x)
    all_pred_y = np.concatenate(all_pred_y)
    all_actual_x = np.concatenate(all_actual_x)
    all_actual_y = np.concatenate(all_actual_y)
    
    overall_mae_x = np.mean(np.abs(all_pred_x - all_actual_x))
    overall_mae_y = np.mean(np.abs(all_pred_y - all_actual_y))
    overall_rmse_x = np.sqrt(np.mean((all_pred_x - all_actual_x)**2))
    overall_rmse_y = np.sqrt(np.mean((all_pred_y - all_actual_y)**2))
    overall_mae = (overall_mae_x + overall_mae_y) / 2
    overall_rmse = np.sqrt((overall_rmse_x**2 + overall_rmse_y**2) / 2)
    
    print(f"X - MAE: {overall_mae_x:.3f}, RMSE: {overall_rmse_x:.3f}")
    print(f"Y - MAE: {overall_mae_y:.3f}, RMSE: {overall_rmse_y:.3f}")
    print(f"Overall - MAE: {overall_mae:.3f}, RMSE: {overall_rmse:.3f}")


print("CV prediction collection and plotting functions defined.")

# %%
# ============================================================================
# MODEL SAVING UTILITIES
# ============================================================================

def save_model(
    model: nn.Module,
    ema: EMA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    save_dir: str = "checkpoints",
    filename: str = None
):
    """
    Save model checkpoint with EMA weights.
    
    Args:
        model: The model to save
        ema: EMA object containing shadow weights
        optimizer: Optimizer state
        epoch: Current epoch
        val_loss: Validation loss
        save_dir: Directory to save checkpoint
        filename: Checkpoint filename (optional)
    """
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Generate filename if not provided
    if filename is None:
        filename = f"model_epoch{epoch}_valloss{val_loss:.4f}.pt"
    
    # Full path
    save_path = os.path.join(save_dir, filename)
    
    # Prepare checkpoint
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'ema_shadow': ema.shadow,
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
        'config': {
            'max_players': config.MAX_PLAYERS,
            'input_frames': config.INPUT_FRAMES,
            'output_frames': config.MAX_OUTPUT_FRAMES,
            'dynamic_features': config.DYNAMIC_FEATURES,
            'static_features': config.STATIC_FEATURES,
        }
    }
    
    # Save checkpoint
    torch.save(checkpoint, save_path)


def load_model(
    model: nn.Module,
    checkpoint_path: str,
    device: torch.device,
    load_optimizer: bool = False,
    optimizer: torch.optim.Optimizer = None
) -> Tuple[nn.Module, Optional[EMA], int, float]:
    """
    Load model checkpoint.
    
    Args:
        model: Model instance to load weights into
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer instance (required if load_optimizer=True)
    
    Returns:
        model: Model with loaded weights
        ema: EMA object with loaded shadow weights
        epoch: Epoch number from checkpoint
        val_loss: Validation loss from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model weights
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Recreate EMA with shadow weights
    ema = EMA(model)
    ema.shadow = checkpoint['ema_shadow']
    
    # Load optimizer if requested
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    val_loss = checkpoint['val_loss']
    
    print(f"Loaded checkpoint from {checkpoint_path}")
    print(f"  Epoch: {epoch}, Val Loss: {val_loss:.4f}")
    
    return model, ema, epoch, val_loss



print("Model saving/loading utilities defined.")
# Build player tracking data from cleaned segments
print("Building soccer tracking data with ball ownership...")
player_tracking_df, ball_ownership_df = build_soccer_tracking_data_updated(match_records, tracking_df_cleaned)

# # Merge ball ownership into ball tracking data
tracking_df_cleaned = tracking_df_cleaned.merge(
    ball_ownership_df[['match_id', 'frame_id', 'ball_owning_team']],
    on=['match_id', 'frame_id'],
    how='left'
)

print(f"Player tracking data: {len(player_tracking_df)} rows")
print(f"Ball ownership data: {len(ball_ownership_df)} rows")
print(f"Segments with ball ownership: {tracking_df_cleaned['ball_owning_team'].notna().sum()}")

# Create dataset with updated feature extraction
print("\nCreating updated soccer dataset...")
soccer_dataset = SoccerSegmentDataset(
    player_tracking_df=player_tracking_df,
    ball_tracking_df=tracking_df_cleaned,
    config=soccer_config,
    training=True
)
# ============================================================================
# COMPLETE SOCCER CV TRAINING AND PREDICTION PIPELINE
# ============================================================================


def train_soccer_cv(
    player_tracking_df: pd.DataFrame,
    ball_tracking_df: pd.DataFrame,
    config: SoccerConfig,
    device: torch.device,
    save_dir: str = "soccer_checkpoints"
):
    """
    Complete CV training pipeline for soccer trajectory prediction.
    
    Args:
        player_tracking_df: DataFrame with player tracking data
        ball_tracking_df: DataFrame with ball tracking data  
        config: SoccerConfig instance
        device: Training device
        save_dir: Directory to save checkpoints
    
    Returns:
        cv_results: Dictionary with CV results and predictions
    """
    from sklearn.model_selection import GroupKFold
    from torch.optim import RAdam
    import torch.utils.data as data_utils
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Create full dataset
    full_dataset = SoccerSegmentDataset(
        player_tracking_df=player_tracking_df,
        ball_tracking_df=ball_tracking_df,
        config=config,
        training=True
    )
    
    print(f"Total segments: {len(full_dataset)}")
    
    # Setup cross-validation
    segment_ids = np.array([seg['segment_id'] for seg in full_dataset.segments])
    kfold = GroupKFold(n_splits=config.N_FOLDS)
    
    cv_results = {
        'fold': [],
        'models': [],
        'train_losses': [],
        'val_losses': [],
        'val_rmses': [],
        'predictions': [],  # Store predictions for each fold
        'segment_ids': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)), groups=segment_ids), 1):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{config.N_FOLDS}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_subset = data_utils.Subset(full_dataset, train_idx)
        val_subset = data_utils.Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        print(f"Train segments: {len(train_idx)}, Val segments: {len(val_idx)}")
        
        # Initialize model for this fold
        model = SoccerTrajectoryModel(config).to(device)
        ema = EMA(model, decay=config.EMA_DECAY)
        optimizer = RAdam(model.parameters(), lr=config.LEARNING_RATE)
        loss_fn = TrajectoryLoss(config)
        
        if fold == 1:
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Training loop
        best_val_loss = float('inf')
        fold_history = {'train_loss': [], 'val_loss': [], 'val_rmse': []}
        
        with tqdm(range(1, config.EPOCHS + 1), desc=f"Fold {fold}") as pbar:
            for epoch in pbar:
                # Train one epoch
                train_metrics = train_one_epoch(
                    model, train_loader, optimizer, loss_fn, ema, device, epoch
                )
                
                # Only evaluate every 6th epoch
                if epoch % 6 == 0:
                    # Evaluate with EMA
                    ema.apply_shadow()
                    val_metrics = evaluate(model, val_loader, loss_fn, device)
                    ema.restore()
                    
                    # Record metrics
                    fold_history['train_loss'].append(train_metrics['loss'])
                    fold_history['val_loss'].append(val_metrics['loss'])
                    fold_history['val_rmse'].append(val_metrics['rmse'])
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'train': f"{train_metrics['loss']:.4f}",
                        'val': f"{val_metrics['loss']:.4f}",
                        'rmse': f"{val_metrics['rmse']:.4f}"
                    })
                    
                    # Save best model
                    if val_metrics['loss'] < best_val_loss:
                        best_val_loss = val_metrics['loss']
                        best_model_state = {
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'ema_shadow': ema.shadow,
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_metrics['loss'],
                            'val_rmse': val_metrics['rmse'],
                            'fold': fold
                        }
                else:
                    # Still record training loss for non-evaluation epochs
                    fold_history['train_loss'].append(train_metrics['loss'])
                    # Add placeholder values for validation metrics to keep lists consistent
                    fold_history['val_loss'].append(float('nan'))
                    fold_history['val_rmse'].append(float('nan'))
                    
                    # Update progress bar with just training loss
                    pbar.set_postfix({
                        'train': f"{train_metrics['loss']:.4f}",
                        'val': "skip",
                        'rmse': "skip"
                    })
        
        # Save best model for this fold
        checkpoint_path = os.path.join(save_dir, f'soccer_model_fold{fold}.pt')
        torch.save(best_model_state, checkpoint_path)
        print(f"Saved fold {fold} model to {checkpoint_path}")
        
        # Load best model and collect predictions
        model.load_state_dict(best_model_state['model_state_dict'])
        ema.shadow = best_model_state['ema_shadow']
        
        ema.apply_shadow()
        fold_pred_xy, fold_actual_xy, fold_mask = collect_soccer_predictions(
            model, val_loader, device
        )
        ema.restore()
        
        # Store results
        cv_results['fold'].append(fold)
        cv_results['models'].append((model, ema))
        cv_results['train_losses'].append(fold_history['train_loss'])
        cv_results['val_losses'].append(fold_history['val_loss'])
        cv_results['val_rmses'].append(fold_history['val_rmse'])
        cv_results['predictions'].append({
            'pred_xy': fold_pred_xy,
            'actual_xy': fold_actual_xy,
            'mask': fold_mask,
            'segment_ids': [full_dataset.segments[i]['segment_id'] for i in val_idx]
        })
        cv_results['segment_ids'].append([full_dataset.segments[i]['segment_id'] for i in val_idx])
        
        print(f"Fold {fold} - Best Val Loss: {best_val_loss:.4f}, RMSE: {best_model_state['val_rmse']:.4f}")
    
    # Print CV summary
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")
    
    # Filter out NaN values when finding best validation metrics
    avg_best_val_loss = np.mean([min([x for x in losses if not np.isnan(x)]) for losses in cv_results['val_losses']])
    avg_best_val_rmse = np.mean([min([x for x in rmses if not np.isnan(x)]) for rmses in cv_results['val_rmses']])
    
    print(f"Average Best Val Loss: {avg_best_val_loss:.4f}")
    print(f"Average Best Val RMSE: {avg_best_val_rmse:.4f}")
    
    for fold in range(config.N_FOLDS):
        # Filter out NaN values when finding best validation metrics for each fold
        best_loss = min([x for x in cv_results['val_losses'][fold] if not np.isnan(x)])
        best_rmse = min([x for x in cv_results['val_rmses'][fold] if not np.isnan(x)])
        print(f"  Fold {fold+1}: Loss={best_loss:.4f}, RMSE={best_rmse:.4f}")
    
    return cv_results


@torch.no_grad()
def collect_soccer_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect predictions from soccer model.
    
    Returns:
        pred_xy: (N, 2) array of predicted final positions
        actual_xy: (N, 2) array of actual final positions  
        mask: (N,) boolean array indicating valid samples
    """
    model.eval()
    
    all_pred_x = []
    all_pred_y = []
    all_actual_x = []
    all_actual_y = []
    all_masks = []
    
    for batch in val_loader:
        dynamic = batch['dynamic_features'].to(device)
        static = batch['static_features'].to(device)
        targets = batch['targets'].to(device)
        mask = batch['player_mask'].to(device)
        final_pos = batch['final_positions'].to(device)
        
        # Forward pass
        pred_mean, _, _ = model(dynamic, static, mask)
        
        # Convert displacement to absolute positions
        final_pos_expanded = final_pos.unsqueeze(-1)
        pred_abs = pred_mean + final_pos_expanded
        target_abs = targets + final_pos_expanded
        
        # Take final frame prediction
        pred_last = pred_abs[:, :, :, -1]  # (B, P, 2)
        target_last = target_abs[:, :, :, -1]  # (B, P, 2)
        
        # Flatten
        B, P = mask.shape
        mask_flat = mask.cpu().numpy().flatten()
        pred_x_flat = pred_last[:, :, 0].cpu().numpy().flatten()
        pred_y_flat = pred_last[:, :, 1].cpu().numpy().flatten()
        actual_x_flat = target_last[:, :, 0].cpu().numpy().flatten()
        actual_y_flat = target_last[:, :, 1].cpu().numpy().flatten()
        
        all_masks.append(mask_flat)
        all_pred_x.append(pred_x_flat)
        all_pred_y.append(pred_y_flat)
        all_actual_x.append(actual_x_flat)
        all_actual_y.append(actual_y_flat)
    
    # Concatenate
    mask_all = np.concatenate(all_masks)
    pred_x_all = np.concatenate(all_pred_x)
    pred_y_all = np.concatenate(all_pred_y)
    actual_x_all = np.concatenate(all_actual_x)
    actual_y_all = np.concatenate(all_actual_y)
    
    # Filter valid samples
    valid_mask = mask_all > 0.5
    pred_xy = np.stack([pred_x_all[valid_mask], pred_y_all[valid_mask]], axis=1)
    actual_xy = np.stack([actual_x_all[valid_mask], actual_y_all[valid_mask]], axis=1)
    
    return pred_xy, actual_xy, valid_mask


def plot_soccer_field(ax):
    """Plot soccer field boundaries on given axis."""
    ax.plot([-52.5, 52.5, 52.5, -52.5, -52.5], [-34, -34, 34, 34, -34], 'k-', linewidth=2)
    ax.plot([0, 0], [-34, 34], 'k-', linewidth=1)  # Halfway line
    ax.set_xlim(-57.5, 57.5)
    ax.set_ylim(-39, 39)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

# Assuming you have the prepared data from earlier cells:
player_tracking_df, ball_tracking_df, match_records

if __name__ == "__main__":
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize config
    config = SoccerConfig()
    
    # Run CV training
    print("Starting soccer CV training...")
    cv_results = train_soccer_cv(
        player_tracking_df=player_tracking_df,
        ball_tracking_df=tracking_df_cleaned,
        config=config,
        device=device,
        save_dir="soccer_checkpoints"
    )
    
    # Plot CV results
    plot_soccer_cv_results(cv_results, save_path="soccer_cv_summary.png")
    
    # Plot trajectory comparisons for random segments
    plot_soccer_trajectory_comparison(
        cv_results=cv_results,
        player_tracking_df=player_tracking_df,
        ball_tracking_df=tracking_df_cleaned,
        match_records=match_records,
        n_examples=3
    )
    
    print("\nCV training and evaluation complete!")
    print("Check the generated plots for detailed results.")

def plot_soccer_trajectory_comparison_single_fold(
    cv_results: Dict,
    player_tracking_df: pd.DataFrame,
    ball_tracking_df: pd.DataFrame,
    match_records: Dict,
    config,
    fold_idx: int = 0,  # Specify which fold to use
    n_examples: int = 3,
    figsize: Tuple[int, int] = (16, 12)
):
    """
    Plot predicted vs actual trajectories for segments from a specific fold.
    
    Args:
        cv_results: Results from CV training
        player_tracking_df: Player tracking data
        ball_tracking_df: Ball tracking data
        match_records: Match frame records
        config: SoccerConfig instance
        fold_idx: Which fold to sample segments from
        n_examples: Number of random examples to plot
        figsize: Figure size
    """
    # Get data from specific fold
    fold_data = cv_results['predictions'][fold_idx]
    segment_ids = fold_data['segment_ids']
    pred_xy = fold_data['pred_xy']
    actual_xy = fold_data['actual_xy']
    mask = fold_data['mask']
    
    # Analyze segment prediction structure for this fold
    segment_pred_counts = {}
    segment_pred_indices = {}
    
    global_idx = 0
    
    for seg_id in segment_ids:
        # Count valid predictions for this segment
        n_valid = 0
        start_idx = global_idx
        
        # Find how many valid predictions this segment has
        while n_valid < config.MAX_PLAYERS and global_idx < len(mask) and mask[global_idx]:
            n_valid += 1
            global_idx += 1
        
        # Skip any remaining masked predictions for this segment
        while global_idx < len(mask) and not mask[global_idx] and n_valid < config.MAX_PLAYERS:
            global_idx += 1
            n_valid += 1
        
        end_idx = global_idx
        
        segment_pred_counts[seg_id] = n_valid
        segment_pred_indices[seg_id] = (start_idx, end_idx)
    
    # Select random segments from this fold
    if len(segment_ids) < n_examples:
        selected_segments = segment_ids
    else:
        selected_segments = np.random.choice(segment_ids, n_examples, replace=False)
    
    fig, axes = plt.subplots(n_examples, 1, figsize=figsize)
    if n_examples == 1:
        axes = [axes]
    
    for i, segment_id in enumerate(selected_segments):
        ax = axes[i]
        plot_soccer_field(ax)
        
        # Get segment data
        segment_ball = ball_tracking_df[ball_tracking_df['segment_id'] == segment_id].sort_values('frame_id')
        segment_players = player_tracking_df[player_tracking_df['segment_id'] == segment_id]
        
        if len(segment_ball) == 0:
            continue
        
        match_id = segment_ball['match_id'].iloc[0]
        
        # Get prediction indices for this segment
        start_idx, end_idx = segment_pred_indices[segment_id]
        seg_pred_xy = pred_xy[start_idx:end_idx]
        seg_actual_xy = actual_xy[start_idx:end_idx]
        seg_mask = mask[start_idx:end_idx]
        
        # Only use valid predictions
        valid_mask = seg_mask.astype(bool)
        seg_pred_xy = seg_pred_xy[valid_mask]
        seg_actual_xy = seg_actual_xy[valid_mask]
        
        # Plot ball trajectory
        ax.plot(segment_ball['ball_x'], segment_ball['ball_y'], 
                'ro-', linewidth=3, markersize=6, alpha=0.8, label='Ball Trajectory')
        
        # Plot actual and predicted player trajectories
        player_colors = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_idx = 0
        
        pred_player_idx = 0  # Index into the valid predictions array
        
        for player_id in segment_players['player_id'].unique():
            if pred_player_idx >= len(seg_pred_xy):
                break
                
            player_data = segment_players[segment_players['player_id'] == player_id].sort_values('frame_id')
            
            if len(player_data) < 2:
                continue
            
            # Assign color
            if player_id not in player_colors:
                player_colors[player_id] = color_cycle[color_idx % len(color_cycle)]
                color_idx += 1
            
            color = player_colors[player_id]
            
            # Plot actual trajectory
            ax.plot(player_data['x'], player_data['y'], 
                    color=color, linewidth=3, alpha=0.8, label=f'Player {player_id} (Actual)')
            
            # Mark actual start and end positions
            ax.scatter(player_data['x'].iloc[0], player_data['y'].iloc[0], 
                      color=color, s=100, marker='o', alpha=0.8, edgecolor='black')
            ax.scatter(player_data['x'].iloc[-1], player_data['y'].iloc[-1], 
                      color=color, s=120, marker='s', alpha=1.0, edgecolor='black', linewidth=2)
            
            # Plot predicted trajectory as a line from start to predicted final position
            start_pos = np.array([player_data['x'].iloc[0], player_data['y'].iloc[0]])
            pred_final_pos = seg_pred_xy[pred_player_idx]
            actual_final_pos = seg_actual_xy[pred_player_idx]
            
            # Predicted trajectory (dashed line)
            ax.plot([start_pos[0], pred_final_pos[0]], [start_pos[1], pred_final_pos[1]], 
                    color=color, linewidth=2, linestyle='--', alpha=0.7, 
                    label=f'Player {player_id} (Predicted)')
            
            # Mark predicted final position
            ax.scatter(pred_final_pos[0], pred_final_pos[1], 
                      color=color, s=120, marker='*', alpha=1.0, edgecolor='black', linewidth=2)
            
            # Draw line showing prediction error
            ax.plot([pred_final_pos[0], actual_final_pos[0]], [pred_final_pos[1], actual_final_pos[1]], 
                    'k:', linewidth=2, alpha=0.6)
            
            pred_player_idx += 1
        
        ax.set_title(f'Predicted vs Actual Trajectories - Fold {fold_idx}, Segment {segment_id}', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Add legend elements for markers
        legend_elements = [
            plt.Line2D([0], [0], color='red', marker='o', linestyle='-', markersize=8, label='Ball'),
            plt.Line2D([0], [0], marker='o', color='gray', markersize=8, linestyle='None', label='Start Position'),
            plt.Line2D([0], [0], marker='s', color='gray', markersize=8, linestyle='None', label='Actual Final'),
            plt.Line2D([0], [0], marker='*', color='gray', markersize=8, linestyle='None', label='Predicted Final'),
            plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2, label='Prediction Error')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(f'soccer_trajectory_comparison_fold{fold_idx}.png', dpi=150, bbox_inches='tight')
    print(f"Trajectory comparison plot saved to soccer_trajectory_comparison_fold{fold_idx}.png")
    plt.show()

plot_soccer_trajectory_comparison_single_fold(
    cv_results, player_tracking_df, tracking_df_cleaned, match_records, config,
    fold_idx=0,  # Use fold 0
    n_examples=1, 
    figsize=(16, 12)
)