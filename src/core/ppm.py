"""
Process Preference Model (PPM) for evaluating reasoning steps
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class PPMConfig:
    input_dim: int = 768
    hidden_dim: int = 256
    learning_rate: float = 0.001
    batch_size: int = 32

class StepEncoder(nn.Module):
    """Encodes reasoning steps into fixed-size vectors."""
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class ProcessPreferenceModel(nn.Module):
    def __init__(self, config: Optional[PPMConfig] = None):
        super().__init__()
        self.config = config or PPMConfig()
        
        # Step encoder
        self.step_encoder = StepEncoder(
            self.config.input_dim,
            self.config.hidden_dim
        )
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim // 2),
            nn.LayerNorm(self.config.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.hidden_dim // 2, 1)
        )
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.config.learning_rate
        )
        
    @classmethod
    def from_config_file(cls, config_path: str) -> 'ProcessPreferenceModel':
        """Create PPM instance from config file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config = PPMConfig(**config_data['ppm'])
        return cls(config)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        encoded = self.step_encoder(x)
        value = self.value_head(encoded)
        return value
    
    def evaluate_step(self, step: str, embedder: Any) -> float:
        """Evaluate a single reasoning step."""
        self.eval()
        with torch.no_grad():
            # Convert step to embedding using provided embedder
            embedding = embedder.encode(step)
            embedding_tensor = torch.FloatTensor(embedding).unsqueeze(0)
            
            # Get value prediction
            value = self(embedding_tensor)
            return value.item()
    
    def train_step(self, 
                  preferred_steps: List[str],
                  non_preferred_steps: List[str],
                  embedder: Any) -> float:
        """Train the model on a batch of preferred vs non-preferred steps."""
        self.train()
        
        # Convert steps to embeddings
        preferred_embeddings = torch.FloatTensor([
            embedder.encode(step) for step in preferred_steps
        ])
        non_preferred_embeddings = torch.FloatTensor([
            embedder.encode(step) for step in non_preferred_steps
        ])
        
        # Get value predictions
        preferred_values = self(preferred_embeddings)
        non_preferred_values = self(non_preferred_embeddings)
        
        # Compute preference loss (preferred steps should have higher values)
        loss = F.relu(non_preferred_values - preferred_values + 1.0).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str) -> None:
        """Save model state to file."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path: str) -> None:
        """Load model state from file."""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.config = checkpoint['config']
        
class PPMTrainer:
    def __init__(self, model: ProcessPreferenceModel):
        self.model = model
        
    def train(self, 
             training_data: List[Dict[str, Any]], 
             embedder: Any,
             num_epochs: int = 100,
             validation_data: Optional[List[Dict[str, Any]]] = None) -> Dict[str, List[float]]:
        """Train the PPM model on a dataset of preferred reasoning trajectories."""
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            epoch_loss = 0.0
            for batch in self._create_batches(training_data, self.model.config.batch_size):
                loss = self.model.train_step(
                    batch['preferred'],
                    batch['non_preferred'],
                    embedder
                )
                epoch_loss += loss
            train_losses.append(epoch_loss / len(training_data))
            
            # Validation
            if validation_data:
                val_loss = self._validate(validation_data, embedder)
                val_losses.append(val_loss)
                
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if validation_data else []
        }
    
    def _create_batches(self, 
                       data: List[Dict[str, Any]], 
                       batch_size: int) -> List[Dict[str, List[str]]]:
        """Create batches from training data."""
        batches = []
        for i in range(0, len(data), batch_size):
            batch_data = data[i:i + batch_size]
            batch = {
                'preferred': [d['preferred'] for d in batch_data],
                'non_preferred': [d['non_preferred'] for d in batch_data]
            }
            batches.append(batch)
        return batches
    
    def _validate(self, 
                 validation_data: List[Dict[str, Any]], 
                 embedder: Any) -> float:
        """Compute validation loss."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in self._create_batches(validation_data, self.model.config.batch_size):
                preferred_embeddings = torch.FloatTensor([
                    embedder.encode(step) for step in batch['preferred']
                ])
                non_preferred_embeddings = torch.FloatTensor([
                    embedder.encode(step) for step in batch['non_preferred']
                ])
                
                preferred_values = self.model(preferred_embeddings)
                non_preferred_values = self.model(non_preferred_embeddings)
                
                loss = F.relu(non_preferred_values - preferred_values + 1.0).mean()
                total_loss += loss.item()
                
        return total_loss / len(validation_data)
