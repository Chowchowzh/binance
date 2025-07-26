import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings, so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    
    This implementation is adapted from the PyTorch tutorial on Transformers.
    """
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    """
    A Transformer model for time series classification.
    """
    def __init__(self, num_features, num_classes, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        """
        Args:
            num_features (int): The number of input features.
            num_classes (int): The number of output classes.
            d_model (int): The number of expected features in the encoder/decoder inputs.
            nhead (int): The number of heads in the multiheadattention models.
            num_encoder_layers (int): The number of sub-encoder-layers in the encoder.
            dim_feedforward (int): The dimension of the feedforward network model.
            dropout (float): The dropout value.
        """
        super(TimeSeriesTransformer, self).__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        
        # 1. Input Embedding Layer
        # This layer projects the input features from num_features to d_model dimension.
        self.input_embedding = nn.Linear(num_features, d_model)
        
        # 2. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 3. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        # 4. Output Layer
        # This layer maps the Transformer's output to the desired number of classes.
        self.output_layer = nn.Linear(d_model, num_classes)

    def forward(self, src):
        """
        Forward pass for the TimeSeriesTransformer.

        Args:
            src (torch.Tensor): The input sequence to the model.
                               Shape: (batch_size, seq_len, num_features)

        Returns:
            torch.Tensor: The output logits from the model.
                          Shape: (batch_size, num_classes)
        """
        # Step 1: Apply input embedding
        # src shape: (batch_size, seq_len, num_features) -> (batch_size, seq_len, d_model)
        src = self.input_embedding(src) * math.sqrt(self.d_model)
        
        # Step 2: Add positional encoding
        # NOTE: PyTorch's Transformer modules expect (seq_len, batch_size, d_model) by default.
        # However, we configured TransformerEncoderLayer with batch_first=True,
        # so we can keep the batch dimension first.
        # But our PositionalEncoding is not batch_first, so we need to permute.
        src = src.permute(1, 0, 2)  # (batch_size, seq_len, d_model) -> (seq_len, batch_size, d_model)
        src = self.pos_encoder(src)
        src = src.permute(1, 0, 2)  # (seq_len, batch_size, d_model) -> (batch_size, seq_len, d_model)
        
        # Step 3: Pass through the Transformer Encoder
        # src shape: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_model)
        output = self.transformer_encoder(src)
        
        # Step 4: Use the output of the last time step for classification
        # We take the output corresponding to the last element of the sequence.
        # output shape: (batch_size, seq_len, d_model) -> (batch_size, d_model)
        output = output[:, -1, :]
        
        # Step 5: Final classification layer
        # output shape: (batch_size, d_model) -> (batch_size, num_classes)
        logits = self.output_layer(output)
        
        return logits

if __name__ == '__main__':
    # This block is for demonstrating and testing the Transformer model.
    print("--- Testing TimeSeriesTransformer ---")

    # Model configuration
    NUM_FEATURES = 50  # Example: Number of features from our feature engineering
    NUM_CLASSES = 3    # (Up, Down, Flat)
    SEQ_LENGTH = 60    # Example: 60 minutes of data
    BATCH_SIZE = 32
    
    # 1. Create an instance of the model
    model = TimeSeriesTransformer(num_features=NUM_FEATURES, num_classes=NUM_CLASSES)
    model.eval() # Set model to evaluation mode
    print("\nModel created successfully.")
    print(model)

    # 2. Create a dummy input tensor to test the forward pass
    dummy_input = torch.randn(BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES)
    print(f"\nShape of dummy input tensor: {dummy_input.shape}")
    
    # 3. Perform a forward pass
    try:
        with torch.no_grad(): # No need to calculate gradients for this test
            output_logits = model(dummy_input)
        
        print("\nForward pass successful.")
        print(f"Shape of output logits: {output_logits.shape}")
        print(f"  - Expected: ({BATCH_SIZE}, {NUM_CLASSES})")
    except Exception as e:
        print(f"\nAn error occurred during the forward pass: {e}") 