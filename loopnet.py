# -- coding: gbk --
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, Layer,BatchNormalization, Reshape,GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D,Concatenate,Add,Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
from attention import MultiHeadAttention
import numpy as np
from plotmodel import plot_training_history


class PositionalEncoding(Layer):
   
    def __init__(self, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
    
    def build(self, input_shape):
        seq_len = input_shape[1]
        d_model = input_shape[2]
        
        
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe = np.zeros((1, seq_len, d_model))
        pe[0, :, 0::2] = np.sin(position * div_term)
        pe[0, :, 1::2] = np.cos(position * div_term)
        
        self.pe = tf.constant(pe, dtype=tf.float32)
        super(PositionalEncoding, self).build(input_shape)
    
    def call(self, inputs):
        return inputs + self.pe
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        return config

class loopnet():
    def __init__(self, learning_rate, epochs, train_x, train_y, test_x, test_y, chromname, kernel_size, save_dir, input_shape):
        """
        Initialize loopnet model for multi-omics data with Transformer architecture
        """
        self.rate = learning_rate
        self.epochs = epochs
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y
        self.chromname = chromname
        self.kernel_size = kernel_size
        self.save_dir = save_dir
        self.input_shape = input_shape 

    def create_model(self):
      
        inputs = Input(shape=(self.input_shape[0], 1))
        
        
        x = Dense(64)(inputs)
        
        
        x = PositionalEncoding()(x)
        
        for i in range(2):  
            attn_output = MultiHeadAttention(
                num_heads=4,  
                key_dim=16,   
                dropout=0.1
            )(x)
            x = LayerNormalization()(x + attn_output)
            
          
            ffn = Dense(512, activation='relu')(x)
            ffn = Dropout(0.1)(ffn)
            ffn = Dense(1)(ffn)  
            x = LayerNormalization()(x + ffn)
        
        
        x = GlobalAveragePooling1D()(x)
        
        
        x = Dense(128, activation='relu')(x)
        outputs = Dense(2, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        
        optimizer = Adam(learning_rate=self.rate)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        model.summary()
        return model

    def train_model(self):
        
        train_x_seq = self.train_x.reshape(-1, self.input_shape[0], 1)
        test_x_seq = self.test_x.reshape(-1, self.input_shape[0], 1)
        
        
        lr_scheduler = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )  

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=10,
            min_delta=0.001,
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint = ModelCheckpoint(
            filepath=f"{self.save_dir}/{self.chromname}_best_model.h5",
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        
        model = self.create_model()
        
        
        history = model.fit(
            train_x_seq, 
            self.train_y, 
            epochs=self.epochs,
            batch_size=256,
            validation_data=(test_x_seq, self.test_y),
            callbacks=[early_stop, lr_scheduler, checkpoint],
            verbose=1,
            shuffle=True  
        )
        
        
        model.save(f"{self.save_dir}/{self.chromname}_final_model.h5")
        
      
        
        return history, model
    
    def evaluate_model(self, model, history):
       
        
        y_pred = model.predict(self.test_x)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.test_y, axis=1)
        
        
        from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
        
        print("\n" + "="*60)
        print("="*60)
        print(classification_report(y_true, y_pred_classes))
        
        
        try:
            auc_score = roc_auc_score(y_true, y_pred[:, 1])
            print(f"AUC Score: {auc_score:.4f}")
        except Exception as e:
            print(f"{e}")
        
        
        plot_training_history(history, y_true, y_pred_classes)
        
        return y_pred, y_pred_classes
  