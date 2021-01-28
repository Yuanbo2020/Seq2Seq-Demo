
# Training details
```shell
Chinese time order: yy/mm/dd  ['94-08-29', '32-05-25', '02-03-02'] 
English time order: dd/M/yyyy  ['29/Aug/1994', '25/May/2032', '02/Mar/2002']

Model: "seq2seq_model_from_Yuanbo"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
encoder_in (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
decoder_in (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
encoder_emb (Embedding)         (None, None, 16)     432         encoder_in[0][0]                 
__________________________________________________________________________________________________
decoder_emb (Embedding)         (None, None, 16)     432         decoder_in[0][0]                 
__________________________________________________________________________________________________
encoder_lstm (LSTM)             [(64, None, 32), (64 6272        encoder_emb[0][0]                
__________________________________________________________________________________________________
basic_decoder (BasicDecoder)    (BasicDecoderOutput( 7163        decoder_emb[0][0]                
                                                                 encoder_lstm[0][1]               
                                                                 encoder_lstm[0][2]               
==================================================================================================
Total params: 14,299
Trainable params: 14,299
Non-trainable params: 0
__________________________________________________________________________________________________

validation step:  0 | loss: 3.292 | input:  27-02-26 | target:  26/Feb/2027 | inference:  ///////<EOS>
validation step:  70 | loss: 1.055 | input:  82-11-07 | target:  07/Nov/1982 | inference:  03/Aug/2020<EOS>
validation step:  140 | loss: 0.720 | input:  79-07-03 | target:  03/Jul/1979 | inference:  03/Apr/2002<EOS>
...
validation step:  560 | loss: 0.092 | input:  00-10-06 | target:  06/Oct/2000 | inference:  06/Oct/2000<EOS>
validation step:  630 | loss: 0.046 | input:  34-01-22 | target:  22/Jan/2034 | inference:  22/Jan/2034<EOS>
validation step:  700 | loss: 0.029 | input:  92-02-06 | target:  06/Feb/1992 | inference:  06/Feb/1992<EOS>
...
validation step:  1330 | loss: 0.003 | input:  09-09-21 | target:  21/Sep/2009 | inference:  21/Sep/2009<EOS>
validation step:  1400 | loss: 0.002 | input:  83-09-07 | target:  07/Sep/1983 | inference:  07/Sep/1983<EOS>
validation step:  1470 | loss: 0.002 | input:  97-04-15 | target:  15/Apr/1997 | inference:  15/Apr/1997<EOS>

```

# Testing details
```shell
Model: "seq2seq_model_from_Yuanbo"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
encoder_in (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
decoder_in (InputLayer)         [(None, None)]       0                                            
__________________________________________________________________________________________________
encoder_emb (Embedding)         (None, None, 16)     432         encoder_in[0][0]                 
__________________________________________________________________________________________________
decoder_emb (Embedding)         (None, None, 16)     432         decoder_in[0][0]                 
__________________________________________________________________________________________________
encoder_lstm (LSTM)             [(64, None, 32), (64 6272        encoder_emb[0][0]                
__________________________________________________________________________________________________
basic_decoder (BasicDecoder)    (BasicDecoderOutput( 7163        decoder_emb[0][0]                
                                                                 encoder_lstm[0][1]               
                                                                 encoder_lstm[0][2]               
==================================================================================================
Total params: 14,299
Trainable params: 14,299
Non-trainable params: 0
__________________________________________________________________________________________________

Testing sample:  0 | input:  27-02-26 | target:  26/Feb/2027 | inference:  26/Feb/2027<EOS>
Testing sample:  1 | input:  32-11-23 | target:  23/Nov/2032 | inference:  23/Nov/2032<EOS>
Testing sample:  2 | input:  06-03-07 | target:  07/Mar/2006 | inference:  07/Mar/2006<EOS>
...
Testing sample:  10 | input:  10-07-11 | target:  11/Jul/2010 | inference:  11/Jul/2010<EOS>
Testing sample:  11 | input:  07-08-23 | target:  23/Aug/2007 | inference:  23/Aug/2007<EOS>
Testing sample:  12 | input:  95-04-17 | target:  17/Apr/1995 | inference:  17/Apr/1995<EOS>
...
Testing sample:  2061 | input:  86-12-14 | target:  14/Dec/1986 | inference:  14/Dec/1986<EOS>
Testing sample:  2062 | input:  18-08-18 | target:  18/Aug/2018 | inference:  18/Aug/2018<EOS>
Testing sample:  2063 | input:  90-05-03 | target:  03/May/1990 | inference:  03/May/1990<EOS>
```
