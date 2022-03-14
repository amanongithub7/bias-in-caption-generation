#import torch
import sys
import tensorflow as tf # for tokenization
import pandas as pd
from decoder import RNN_Decoder
from encoder import CNN_Encoder
from dataset import ImgCaptionDataset
from utils import clip_gradient, save_checkpoint

# constants
size = (3, 224, 224)

train_count = 127131 # 80%
valid_count = 15891 # 10%
test_count = 15892 # 10%

max_length = 50 # a maximum of 50 words per caption
vocabulary_size = 5000

BATCH_SIZE = 64

if torch.cuda.is_available():
  print('using GPU')
else:
  print('couldnt find GPU')
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# model constants
embedding_dim = 256
encoder_fc_units = 512
decoder_fc_units = 512
num_steps = train_count // BATCH_SIZE
gru_hidden_size = 500
attention_dim = 512

def main():
  print('importing csv')
  # directory of processed csv
  dataset_df = pd.read_csv('/content/drive/MyDrive/CS 269 Project/captions.csv')
  img_dir = 'dataset/images'

  # tokenizing captions
  captions_list = dataset_df['caption'].tolist()
  caption_dataset = tf.data.Dataset.from_tensor_slices(captions_list)

  print('starting tokenizing')
  # remove punctuations and make text lowercase
  def standardize(inputs):
    inputs = tf.strings.lower(inputs)
    return tf.strings.regex_replace(inputs,r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
  
  tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize= standardize,
    output_sequence_length=max_length
  )

  tokenizer.adapt(caption_dataset)

  # Create the tokenized vectors
  cap_vector = caption_dataset.map(lambda x: tokenizer(x))
  
  # making a list of all the vectorized captions
  cap_vector_list = []
  for cap in cap_vector:
    cap_vector_list.append((cap.numpy()).tolist())
  

  print('saving tokenized captions')
  # placing vectorized captions back into dataframe
  print(dataset_df.head())
  dataset_df['caption'] = cap_vector_list
  print(dataset_df.head())

  
  # Create mappings for words to indices and indicies to words.
  word_to_index = tf.keras.layers.StringLookup(
      mask_token="",
      vocabulary=tokenizer.get_vocabulary())
  index_to_word = tf.keras.layers.StringLookup(
      mask_token="",
      vocabulary=tokenizer.get_vocabulary(),
      invert=True)

  print('creating dataset class')
  dataset = ImgCaptionDataset(dataset_df, img_dir)
  print(dataset[0])
  sys.exit(0)

  # creating dataloaders
  train_data, valid_data, test_data = torch.utils.data.random_split(dataset, (train_count, valid_count, test_count))
  train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
  valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
  test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)


  encoder = CNN_Encoder(encoder_fc_units, embedding_dim)
  decoder = RNN_Decoder(embedding_dim=embedding_dim, fc_units=decoder_fc_units, 
                    vocab_size=tokenizer.vocabulary_size(), hidden_size=gru_hidden_size, attention_dim=attention_dim)

  encoder.to(device)
  decoder.to(device)

  encoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, encoder.parameters()))
  decoder_optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, decoder.parameters()))

  criterion = torch.nn.CrossEntropyLoss().to(device)

  num_epochs = 5
  print_frequency = 300

  for i in range(num_epochs):
      epoch_loss = train_one_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch=i, print_freq=print_frequency)
      print('Epoch', i, 'loss:', epoch_loss)


def train_one_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, epoch, grad_clip=None, print_freq=500):
  total_loss = 0
  for (batch, (img_tensor, target)) in enumerate(train_dataloader):
    
    img_tensor = img_tensor.to(device)
    target = target.to(device)

    loss = 0
    hidden = decoder.reset_state(batch_size=BATCH_SIZE)

    # extract features from encoder
    features = encoder(img_tensor)
    features = torch.unsqueeze(features, 1) # make features 3-dimensional so that attention torch.bmm works

    # features.shape = 64, 1, 256

    word_codes = target[:, 0]
    dec_input = torch.tensor(word_codes, dtype=torch.long)
    dec_input = dec_input[:, None]
  
    for i in range(1, target.shape[1]):
      # passing the features through the decoder
      predictions, hidden, _ = decoder(dec_input, features, hidden)
      hidden = torch.squeeze(hidden)

      loss += criterion(predictions, target[:, i])
      
      # preparing input for the next iteration
      word_codes = target[:, i]
      dec_input = torch.tensor(word_codes, dtype=torch.long)
      dec_input = dec_input[:, None]
    
    # set gradients to 0
    decoder_optimizer.zero_grad()
    if encoder_optimizer is not None:
      encoder_optimizer.zero_grad()
    
    # backprop
    loss.backward()

    # gradient clipping
    if grad_clip is not None:
      clip_gradient(decoder_optimizer, grad_clip)
      if encoder_optimizer is not None:
        clip_gradient(encoder_optimizer, grad_clip)
    
    # Update weights
    decoder_optimizer.step()
    if encoder_optimizer is not None:
        encoder_optimizer.step()

    # metrics
    total_loss += loss.item() # adding the loss for the current batch
    if batch % print_freq == 0:
      if batch != 0:
        print('Loss as of batch', batch, ':', (total_loss/batch))
      else:
        print('Loss as of batch', batch, ':', loss.item())
    

    # save the model every 500 batches
    if batch % 5 == 0:
      print('Saving model')
      filename = str(epoch) + '-' + str(batch)
      save_checkpoint('flickr30k', epoch, 0, encoder, decoder, encoder_optimizer,
                          decoder_optimizer, 0, True, filename)

    return total_loss

def generate_captions(decoder, encoder, index_to_word, image, target):
  hidden = decoder.reset_state(batch_size=1)

  # assumes that image is already a tensor, normalized etc.
  # i.e. image is supplied by a dataloader
  features = encoder(image)

  dec_input = torch.tensor(target[0])
  dec_input = torch.unsqueeze(dec_input, 0) # make it 1x1

  caption = []
  for i in range(max_length):
        predictions, hidden, _ = decoder(dec_input, features,hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        caption.append(predicted_word)

        if predicted_word == '<end>':
            return caption

        dec_input = torch.unsqueeze([predicted_id], 0)

  return caption

if __name__ == '__main__':
  main()