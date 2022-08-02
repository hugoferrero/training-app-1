
### Libraries ###

import pandas as pd #Incluir en el requirement.txt
import tensorflow as tf 
import numpy as np #Incluir en el requirement.txt
import tensorflow_recommenders as tfrs #Incluir en el requirement.txt
import tensorflow_datasets as tfds #Incluir en el requirement.txt


### Importing Dataset from GCS ###

dataset_path = 'gs://hf-exp/vpoc/tfrs/data/ratings.csv'

### Training set ###

df1 = pd.read_csv(dataset_path)

df1 = df1[['movieId','userId']] 
df1['movieId'] = df1['movieId'].apply(str)
df1['userId'] = df1['userId'].apply(str)

pelis =  tf.data.Dataset.from_tensor_slices(dict(df1))

tf.random.set_seed(42)
shuffled = pelis.shuffle(100_000, seed=42, reshuffle_each_iteration=False)
train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)
cached_train = train.shuffle(100_000).batch(2048) # Training set
cached_test = test.batch(4096).cache() # Validation set

df3 = df1[['movieId']]

u_values = df3['movieId'].unique()

movies =  tf.data.Dataset.from_tensor_slices(u_values)



###### Modeling #######

###### Model for embedding user queries #######

class UserModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    unique_user_ids = np.array(range(943)).astype(str)
   
    unique_movie_ids_1 = np.array(range(1682)).astype(str)

    # Compute embeddings for users.
    self.user_embedding = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_user_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension),
    ])

  def call(self, user_id):
    # Take the input dictionary, pass it through each input layer,
    # and concatenate the result.

    return tf.concat([self.user_embedding(user_id)], axis=1)


###### Model(NN) for training user queries ######    

class QueryModel(tf.keras.Model):
  """Model for encoding user queries."""

  def __init__(self, layer_sizes):
    """Model for encoding user queries.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    # We first use the user model for generating embeddings.
    self.embedding_model = UserModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)

###### Model for embedding user movies #######
class MovieModel(tf.keras.Model):
  
  def __init__(self):
    super().__init__()
    embedding_dimension = 32

    unique_movie_ids = np.array(range(1682)).astype(str)
   
     # Compute embeddings for movies.
    self.movie_embeddings = tf.keras.Sequential([
        tf.keras.layers.StringLookup(
            vocabulary=unique_movie_ids, mask_token=None),
        tf.keras.layers.Embedding(len(unique_movie_ids) + 1, embedding_dimension),
    ])
    

  def call(self, movie_id):

    return tf.concat([self.movie_embeddings(movie_id)], axis=1)

###### Model(NN) for training movies ######

class CandidateModel(tf.keras.Model):
  """Model for encoding movies."""

  def __init__(self, layer_sizes):
    """Model for encoding movies.

    Args:
      layer_sizes:
        A list of integers where the i-th entry represents the number of units
        the i-th layer contains.
    """
    super().__init__()

    self.embedding_model = MovieModel()

    # Then construct the layers.
    self.dense_layers = tf.keras.Sequential()

    # Use the ReLU activation for all but the last layer.
    for layer_size in layer_sizes[:-1]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

    # No activation for the last layer.
    for layer_size in layer_sizes[-1:]:
      self.dense_layers.add(tf.keras.layers.Dense(layer_size))
    
  def call(self, inputs):
    feature_embedding = self.embedding_model(inputs)
    return self.dense_layers(feature_embedding)


###### Full Model  (wrap up model) ###### 

class MovielensModel(tfrs.models.Model):

  def __init__(self, layer_sizes):
    super().__init__()
    self.query_model = QueryModel(layer_sizes)
    self.candidate_model = CandidateModel(layer_sizes)
    self.task = tfrs.tasks.Retrieval(
        metrics=tfrs.metrics.FactorizedTopK(
            candidates=movies.batch(128).map(self.candidate_model),
        ),
    )

  def compute_loss(self, features, training=False):
   
     query_embeddings = self.query_model(features["userId"])
     movie_embeddings = self.candidate_model(features["movieId"])
    # query_embeddings y movie_embeddings son los outputs de las últimas capas de la red (query_embeddings son los outputs de la torre de usuarios y 
    # movie_embeddings son los outputs de la torre de movies), son los arrays con los cuales se hace el producto escalar para deternimar similitud.
     return self.task(
        query_embeddings, movie_embeddings, compute_metrics=not training)

 ###### Training #######

model = MovielensModel([64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(
    cached_train,
    validation_data=cached_test,
    epochs=4,
    verbose=1)       

###### Evaluate ###### 
model.evaluate(cached_test, return_dict=True)

#### Saving Model ######
index = tfrs.layers.factorized_top_k.BruteForce(model.query_model)
index.index_from_dataset(
  tf.data.Dataset.zip((movies.batch(100), movies.batch(100).map(model.candidate_model)))
)

path = 'gs://hf-exp/vpoc/tfrs/model'

tf.saved_model.save(index, path)

