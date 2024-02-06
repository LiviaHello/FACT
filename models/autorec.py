import torch
import torch.nn as nn


class AutoRec(nn.Module):
    def __init__(self, n_items, num_users, emb_size, hidden_size, device):
        super(AutoRec, self).__init__()
        self.num_users = num_users
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.device = device

        self.encoder = nn.Linear(n_items, self.hidden_size).to(self.device)  # Encode item features
        self.decoder = nn.Linear(self.hidden_size, n_items).to(self.device)  # Decode back to item space


    def forward(self, user_ratings):
        # user_ratings is a matrix of size [batch_size, num_users]
        encoded = torch.sigmoid(self.encoder(user_ratings))
        decoded = self.decoder(encoded)
        return decoded

    def train_model(self, train_matrix, mask_matrix, optimizer, device):
        self.train()  # Set the model to training mode
        optimizer.zero_grad()  # Clear gradients

        # Forward pass
        decoded_matrix = self.forward(train_matrix).to(device)

        # Calculate loss
        loss, emb_loss = self.calc_loss(decoded_matrix, train_matrix, mask_matrix)
        total_loss = loss + emb_loss

        # Backward and optimize
        total_loss.backward()  # Compute gradients
        optimizer.step()  # Update parameters

        return total_loss

    def calc_loss(self, decoded_matrix, true_matrix, mask_matrix):
        # Implement your loss calculation here (MSE, regularization, etc.)
        pred_loss = nn.functional.mse_loss(decoded_matrix * mask_matrix, true_matrix * mask_matrix)
        lambda_reg = 0.01 
        emb_loss = lambda_reg * (1 / 2) * (self.encoder.weight.norm(2).pow(2) + self.decoder.weight.norm(2).pow(2))
        return pred_loss, emb_loss
