import jax
import jax.numpy as jnp
import equinox as eqx


class Encoder(eqx.Module):
    layers: list

    def __init__(self, in_size, hidden_size, out_size, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 9)
        self.layers = [
            eqx.nn.Linear(in_size, hidden_size, key=k1),
            eqx.nn.LayerNorm(hidden_size, key=k2),
            jax.nn.gelu,
            eqx.Dropout(0.1, key=k3),
            eqx.nn.Linear(hidden_size, out_size, key=k4),
            eqx.nn.LayerNorm(out_size, key=k5),
            jax.nn.gelu,
            eqx.Dropout(0.2, key=k6),
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x


class LSTM(eqx.Module):
    cell: eqx.nn.LSTMCell

    def __init__(self, in_size, hidden_size, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 3)

        self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=k1)
        self.h0 = eqx.nn.Linear(1, hidden_size, key=k2)
        self.c0 = eqx.nn.Linear(1, hidden_size, key=k3)
        self.h0_dropout = eqx.Dropout(0.5, key=k4)
        self.c0_dropout = eqx.Dropout(0.5, key=k5)
        self.out_dropout = eqx.Dropout(0.5, key=k6)

    def __call__(self, xs, mask):
        """
        Process sequential data through a bidirectional LSTM.

        Args:
            xs: Array of shape [seq_len, feature_dim] containing the input sequence
            mask: Boolean array of shape [seq_len] indicating which sequence positions
                are valid (True for valid positions, False for masked/padding)

        Returns:
            Array of shape [2*hidden_size] containing the concatenated forward and
            reverse final hidden states
        """

        def scan_fn(state, inputs):
            x, m = inputs
            # Only update state if mask is True
            new_state = jax.lax.cond(
                m,
                lambda: self.cell(x, state),  # Update state with input
                lambda: state,  # Keep state unchanged
            )
            return new_state, None

        init_state = (
            self.h0_dropout(self.h0(jnp.ones(1))),
            self.c0_dropout(self.c0(jnp.ones(1))),
        )

        # Forward pass
        (out_fwd, _), _ = jax.lax.scan(scan_fn, init_state, (xs, mask))

        # Backward pass - reverse both the sequence and the mask
        (out_rev, _), _ = jax.lax.scan(scan_fn, init_state, (xs[::-1], mask[::-1]))

        return self.out_dropout(jnp.concatenate([out_fwd, out_rev], axis=-1))


class Decoder(eqx.Module):
    """
    Decoder module for activity type classification.
    """

    def __init__(self, in_size, hidden_size1, hidden_size2, out_size, key):
        k1, k2, k3, k4, k5, k6 = jax.random.split(key, 9)
        self.layers = [
            # hidden 1
            eqx.nn.Linear(in_size, hidden_size1, key=k1),
            eqx.nn.LayerNorm(hidden_size1, key=k2),
            jax.nn.gelu,
            eqx.Dropout(0.5, key=k3),

            # hidden 2
            eqx.nn.Linear(hidden_size1, hidden_size2, key=k4),
            eqx.nn.LayerNorm(hidden_size2, key=k5),
            jax.nn.gelu,
            eqx.Dropout(0.4, key=k6),

            # out
            eqx.nn.Linear(hidden_size2, out_size, key=k4),
            eqx.nn.LayerNorm(out_size, key=k5),
            jax.nn.softmax,
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        
        return x


class ActivityTypeClassifier(eqx.Module):
    """
    Classifier for detecting activity types from GPS and other sensor data.

    Uses an encoder-LSTM architecture to process sequential movement data and
    classify the activity type (e.g., walking, running, cycling).

    Attributes:
        encoder: Encoder module that transforms input features
        lstm: Bidirectional LSTM module that processes sequential data
    """

    encoder: Encoder
    lstm: LSTM

    def __init__(self, key):
        k1, k2 = jax.random.split(key, 2)
        self.encoder = Encoder(7, 16, 32, key=k1)
        self.lstm = LSTM(32, 128, key=k2)
        self.decoder = Decoder(256, 128, 32, 8, key=k2)

    def __call__(
        self,
        offsets: jax.Array,
        distances: jax.Array,
        elevations: jax.Array,
        times: jax.Array,
        mask: jax.Array,
    ):
        """
        Process movement data to extract features for activity classification.

        Args:
            offsets: Array of shape [seq_len, 2] containing x and y offsets in meters
            distances: Array of shape [seq_len, 1] containing distances in meters
            elevations: Array of shape [seq_len, 2] containing elevation gain and loss in meters
            times: Array of shape [seq_len, 2] containing elapsed time and moving time
            mask: Boolean array of shape [seq_len] indicating which sequence positions are valid
                  (True for valid positions, False for masked/padding positions)
        """
        offsets = offsets / 150
        distances = distances / 1000 * 4
        elevations = elevations / 20 * 4
        times = times / 40

        xs = jnp.concatenate([offsets, distances, elevations, times], axis=-1)
        xs = jax.vmap(self.encoder)(xs)

        x = self.lstm(xs, mask)
        out = self.decoder(x)
        return out
