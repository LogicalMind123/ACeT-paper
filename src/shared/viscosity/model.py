# ------------------------------------------------------------------------------
# 1) DenseKANRBF Layer Definition
# ------------------------------------------------------------------------------
class DenseKANRBF(layers.Layer):
    def __init__(self, units,
                 grid_size=5,
                 grid_range=(-1.0,1.0),
                 basis_function='rbf',
                 mlp_units=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.grid_size = grid_size
        self.grid_range = grid_range
        self.basis_function = basis_function
        self.mlp_units = mlp_units or []

    def build(self, input_shape):
        in_f = int(input_shape[-1])
        low, high = self.grid_range
        centers_1d = tf.linspace(low, high, self.grid_size)
        centers_1d = tf.cast(centers_1d, dtype=self.dtype)
        centers    = tf.tile(centers_1d[None, :], [in_f, 1])
        self.centers = self.add_weight(
            'centers',
            shape=(in_f, self.grid_size),
            initializer=tf.keras.initializers.Constant(centers),
            trainable=False,
            dtype=self.dtype
        )
        self.basis_kernel = self.add_weight(
            'basis_kernel',
            shape=(in_f, self.grid_size, self.units),
            initializer='glorot_uniform',
            trainable=True
        )
        if self.mlp_units:
            mlp_layers = []
            for u in self.mlp_units:
                mlp_layers.append(layers.Dense(u, activation='gelu'))
            mlp_layers.append(layers.Dense(self.units))
            self.mlp = models.Sequential(mlp_layers)
        self.bias = self.add_weight('bias', shape=(self.units,), initializer='zeros')
        super().build(input_shape)

    def call(self, inputs):
        B = tf.shape(inputs)[0]
        F = inputs.shape[-1]
        x_exp = tf.reshape(inputs, [B, F, 1])
        centers = tf.reshape(self.centers, [1, F, self.grid_size])
        diff = x_exp - centers
        if self.basis_function == 'rbf':
            basis = tf.exp(-tf.square(diff))
        else:
            basis = tf.exp(-tf.abs(diff))
        weighted = tf.einsum('bfg,fgu->bfu', basis, self.basis_kernel)
        out = tf.reduce_sum(weighted, axis=1)
        if hasattr(self, 'mlp'):
            out += self.mlp(inputs)
        out += self.bias
        return out

# ------------------------------------------------------------------------------
# 2) Transformer Builder with Switchable Head 
# ------------------------------------------------------------------------------
def build_transformer_model(
    num_features,
    task,
    embed_dim=16,
    num_heads=2,
    ff_dim=32,
    num_transformer_blocks=1,
    mlp_units=[64,32],
    dropout_rate=0.3,
    l2_reg=1e-5,
    num_classes=None,
    head_type='mlp'
):
    inputs = layers.Input(shape=(num_features,))
    tokens = []
    for i in range(num_features):
        t = layers.Lambda(lambda x,i=i: x[:,i:i+1])(inputs)
        t = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))(t)
        t = layers.Lambda(lambda x: tf.expand_dims(x,1))(t)
        tokens.append(t)
    x = layers.Concatenate(axis=1)(tokens)
    positions = tf.range(0, num_features, 1)
    pos_emb = layers.Embedding(input_dim=num_features, output_dim=embed_dim)(positions)
    x = x + pos_emb
    for _ in range(num_transformer_blocks):
        x1 = layers.LayerNormalization(epsilon=1e-6)(x)
        att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x1, x1)
        x2 = layers.Add()([x, att])
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        ffn = layers.Dense(ff_dim, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x3)
        ffn = layers.Dense(embed_dim, kernel_regularizer=regularizers.l2(l2_reg))(ffn)
        x = layers.Add()([x2, ffn])
    x = layers.GlobalAveragePooling1D()(x)
    if head_type == 'mlp':
        for u in mlp_units:
            x = layers.Dense(u, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg))(x)
            #x = layers.GaussianDropout(dropout_rate)(x)
    elif head_type == 'rbf':
        out_dim = mlp_units[-1] if mlp_units else embed_dim
        x = DenseKANRBF(
            units=out_dim,
            grid_size=8,
            grid_range=(-1,1),
            basis_function='rbf',
            mlp_units=mlp_units
        )(x)

    elif head_type == 'spline':
        # Cubic spline head with fixed knots at -1,0,1
        def spline_basis(z):
            # z: (batch, num_features)
            knots = tf.constant([-1.0, 0.0, 1.0], dtype=z.dtype)        # shape (5,)
            # expand to (batch, num_features, 1), subtract gives (batch, num_features, 3)
            diff = tf.nn.relu(tf.expand_dims(z, -1) - knots)
            # cubic basis
            return tf.pow(diff, 3)                                      # (batch, num_features, 3)

        # apply Lambda to get shape (batch, num_features, 3)
        x = layers.Lambda(spline_basis)(inputs)
        # now reshape to (batch, num_features * 3) so Dense can infer its input size
        num_features = inputs.shape[-1]                                 # static integer
        x = layers.Reshape((num_features * 3,))(x)
        x = layers.Dense(64, activation='gelu',
                             kernel_regularizer=regularizers.l2(l2_reg))(x)

    elif head_type == 'kan':
           # Kolmogorov-Arnold Network head
         out_dim = mlp_units[-1] if mlp_units else embed_dim
         #out_dim = num_features
         x = DenseKAN(units=out_dim)(x)  
    
    elif head_type == 'interaction':
         # build all i<j products via Multiply() + Concatenate()
         pairs = []
         for i in range(num_features):
            for j in range(i+1, num_features):
               prod = layers.Multiply()(
                [inputs[:, i:i+1], inputs[:, j:j+1]]
            )  # shape (batch,1)
            pairs.append(prod)
         x = layers.Concatenate(name='interaction_features')(pairs)  # (batch, num_feats*(num_feats-1)/2)
         x = layers.Dense(64, activation='gelu', kernel_regularizer=regularizers.l2(l2_reg), name='interaction_dense')(x)


    else:
        raise ValueError("head_type must be 'mlp' or 'rbf' or 'spline' or 'kan' or 'interaction'")
    if task == 'regression':
        outputs = layers.Dense(1, activation='linear')(x)
    elif task == 'classification':
        outputs = layers.Dense(num_classes, activation='softmax')(x)
    else:
        raise ValueError("Task must be 'regression' or 'classification'")
    return models.Model(inputs, outputs)
