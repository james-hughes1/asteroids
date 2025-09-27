# utils/preprocess.py

import cv2
import numpy as np
from collections import deque

def preprocess_frame(frame, network_size=(84, 84), grayscale=True):
    """
    Preprocess a single frame for the network.
    
    Args:
        frame (np.array): RGB frame from environment (HxWx3)
        network_size (tuple): (height, width) to resize for network
        grayscale (bool): Convert frame to grayscale if True
    
    Returns:
        np.array: Preprocessed frame (network_size), values normalized [0,1]
    """
    if grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(frame, network_size, interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return normalized

def stack_frames(stacked_frames, frame, is_new_episode, num_frames=5, network_size=(84,84), grayscale=True):
    """
    Stack frames for temporal information.
    
    Args:
        stacked_frames (deque): deque storing last frames
        frame (np.array): current frame from environment
        is_new_episode (bool): True if starting new episode
        num_frames (int): number of frames to stack
        network_size (tuple): size to resize frames for network
        grayscale (bool): convert frames to grayscale if True
    
    Returns:
        stacked_state (np.array): stacked frames of shape (num_frames, H, W)
        stacked_frames (deque): updated deque
    """
    processed_frame = preprocess_frame(frame, network_size=network_size, grayscale=grayscale)
    
    if is_new_episode:
        # Clear deque and start with repeated frames
        stacked_frames = deque([processed_frame]*num_frames, maxlen=num_frames)
    else:
        stacked_frames.append(processed_frame)
    
    stacked_state = np.stack(stacked_frames, axis=0)
    return stacked_state, stacked_frames
