from model import Autoencoder

INPUT_SIZE = 18
MODEL1 = Autoencoder(input_dim=INPUT_SIZE, hidden_dims=[16, 8])
MODEL2 = Autoencoder(input_dim=18, hidden_dims=[16, 8, 4])
MODEL4 = Autoencoder(input_dim=INPUT_SIZE, hidden_dims=[32, 16, 8])
MODEL3 = Autoencoder(input_dim=INPUT_SIZE, hidden_dims=[64, 32, 16, 8])
