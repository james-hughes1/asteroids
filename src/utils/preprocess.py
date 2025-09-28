import cv2
import numpy as np
from collections import deque

def preprocess_frame(frame, network_size=(84, 84), rgb=True, return_uint8=True):
    """
    Preprocess a single frame for the network.

    Args:
        frame (np.array): RGB frame from environment (HxWx3)
        network_size (tuple): (height, width) to resize for network
        rgb (bool): Keep RGB if True; else convert to grayscale
        return_uint8 (bool): Store as uint8 to save memory

    Returns:
        np.array: Preprocessed frame
    """
    if not rgb:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, network_size, interpolation=cv2.INTER_AREA)
    
    if return_uint8:
        return resized.astype(np.uint8)
    else:
        return resized.astype(np.float32) / 255.0


def stack_frames(stacked_frames, frame, is_new_episode, num_frames=5, network_size=(84,84), rgb=True):
    """
    Stack frames for temporal information.
    
    Returns:
        stacked_state (np.array): shape (num_frames*channels, H, W)
        stacked_frames (deque): updated deque
    """
    processed_frame = preprocess_frame(frame, network_size=network_size, rgb=rgb, return_uint8=True)
    
    if is_new_episode:
        stacked_frames = deque([processed_frame]*num_frames, maxlen=num_frames)
    else:
        stacked_frames.append(processed_frame)
    
    # Stack along channel dimension: (num_frames*channels, H, W)
    if processed_frame.ndim == 2:
        # grayscale
        stacked_state = np.stack(stacked_frames, axis=0)
    else:
        # RGB: shape (H,W,3) -> (3,H,W)
        stacked_state = np.stack([f.transpose(2,0,1) for f in stacked_frames], axis=0)
    
    return stacked_state, stacked_frames
