"""
Poker Model Module for V12

Shared model definitions for multiprocessing training.
This module is imported by both the main process (notebook) and worker processes.
"""

import torch
import torch.nn as nn
import numpy as np


class StatisticsPokerNet(nn.Module):
    """Feed-forward network for poker decision making based on poker statistics."""
    
    def __init__(self, state_dim=380, action_dim=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )
    
    def forward(self, x):
        return self.network(x)


def select_action_with_model(state: np.ndarray, model_state_dict: dict, 
                              epsilon: float, legal_actions: list) -> int:
    """
    Select action using the trained model (epsilon-greedy).
    
    Args:
        state: Current state observation (numpy array)
        model_state_dict: Model weights dictionary
        epsilon: Exploration rate
        legal_actions: List of legal action indices
        
    Returns:
        Selected action index
    """
    import random
    
    if random.random() < epsilon:
        return random.choice(legal_actions)
    
    # Create model and load weights
    model = StatisticsPokerNet()
    model.load_state_dict(model_state_dict)
    model.eval()
    
    with torch.no_grad():
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_tensor).squeeze(0)
        
        # Mask illegal actions with very negative values
        masked_q = torch.full_like(q_values, float('-inf'))
        for a in legal_actions:
            masked_q[a] = q_values[a]
        
        return masked_q.argmax().item()
