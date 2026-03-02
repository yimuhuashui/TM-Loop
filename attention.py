# -- coding: gbk --
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Layer
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

class MultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim, dropout=0.0, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.dropout = dropout

    def build(self, input_shape):
        
        self.d_model = input_shape[-1]
        
       
        if self.d_model % self.num_heads != 0:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by num_heads ({self.num_heads})")
        
       
        self.Wq = self.add_weight(name='Wq', 
                                 shape=(self.d_model, self.num_heads * self.key_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wk = self.add_weight(name='Wk', 
                                 shape=(self.d_model, self.num_heads * self.key_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wv = self.add_weight(name='Wv', 
                                 shape=(self.d_model, self.num_heads * self.key_dim),
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.Wo = self.add_weight(name='Wo', 
                                  shape=(self.num_heads * self.key_dim, self.d_model),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.dropout_layer = Dropout(self.dropout)
        super(MultiHeadAttention, self).build(input_shape)  

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        
        q = tf.matmul(inputs, self.Wq)  
        k = tf.matmul(inputs, self.Wk)
        v = tf.matmul(inputs, self.Wv)
        
        
        q = tf.reshape(q, (batch_size, seq_len, self.num_heads, self.key_dim))
        k = tf.reshape(k, (batch_size, seq_len, self.num_heads, self.key_dim))
        v = tf.reshape(v, (batch_size, seq_len, self.num_heads, self.key_dim))
        
        
        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 1, 3])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        
       
        matmul_qk = tf.matmul(q, k, transpose_b=True)  
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        attention_weights = self.dropout_layer(attention_weights)
        
        output = tf.matmul(attention_weights, v) 
        
        
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        
        
        output = tf.reshape(output, (batch_size, seq_len, self.num_heads * self.key_dim))
        
       
        output = tf.matmul(output, self.Wo)
        return output
    
    def get_config(self):
        config = super(MultiHeadAttention, self).get_config()
        config.update({
            'num_heads': self.num_heads,
            'key_dim': self.key_dim,
            'dropout': self.dropout
        })
        return config
