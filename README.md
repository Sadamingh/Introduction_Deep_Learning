# Introduction_Deep_Learning

1. Deep Learning (DL) is 
- a subset of Machine Learning 
- with **high-level features** (large and unstructured)

2. Select the fields for deep learning
- Image Science (CNN/GAN)
- NLP (RNN)
- Reinforcement Learning

3. Suppose we have a cat image, then what are the high level features?
- Two Ears
- Two Eyes
- Whiskers
- Looks fluffy

4. Suppose we have a cat image, then what are the low level features?
- Edge detection
- Noisiness
- Blob detection

5. What is true for the DL high-level features?
- learned from data
- Constructed from learned low-level features
- Usually NOT interpretable
- Only as good as your data

6. What are the techniques used in deeplearning?
- Word Embeddings: simple linear mapping
- Neural Network: stack linear functions one after the other

7. What is true for "unstructured data"?
- tabular data is not unstructured data
- images and text are unstructured data
- video and audio are unstructured data
- We usually use Deep Learning for unstructured data

8. What are the applications for images in data science?
- Classification (facial/object recognition, avoid poisonous plants, etc.)
- Medical Imaging (detecting disease, predicting outcomes of radiation, segmentation of medical images)
- Autonomous Driving (driver assistance, fully autonomous vehicles)
- Deepfakes and deepfake detection

9. Why images are special to other kinds of data?
- Images are deceptively hard
- Images are big with many features (pixels)
- Geometry matters: Pixels near each other interact in different ways

10. What is true about the convolution data?
- these are linear operations
- get the result by element-wise products
- filter is also called kernel

11. If input is dimension 16 and output is dimension 9, how many for FC?
- 4

12. Why we typically use convolution for image NN?
- less parameters
- translational equivariance
- weight sharing

13. What are the common NLP applications?
- Sentiment Analysis
- Auto-complete
- Translation
- Question answering
- Conversation

14. What's the goal of tokenization?
- break the sentences into meaningful categorical variables

15. What's the goal of word embedding?
- convert the high-dimension tokens to a low-dimentional space

16. What are the typical approaches for tokenization?
- by words: n-grams
- by characters
- by subwords: lemmatization, good if we have many UNK
- by sentences: EOS, SOS, or decion tree

17. What are the advantages of using subwords for tokenization?
- smaller dictionary
- less tokens
- better at handling unknown

18. What are the algorithms for subwords tokenization?
- BPE
- Unigram
- WordPiece

19. What are the benefits of lemmatization?
- Reduce words to their base
- Shrink dictionary size

20. What are the followings are NLP techniques?
- Handle infrequent words with UNK 
- Lower case
- Remove weird characters/numbers/punctuation
- Remove stop words

21. What are the common DL models used in NLP?
- NER
- CBOW
- 1D CNN
- Recurrent Neural Network (RNN)

22. What's true about RNN?
- Keep track of a hidden state vector of features as you move along a sequence
- Sequence length agnostic

23. What's the meaning of Gradient Descent?
- Trying to find a “good” minimum for our loss function

24. When we are fail for gradient descending?
- Too flat (or saddle point)
- “Bad” minima or local minima

25. What can be the result if we have a overly large learning rate?
- diverges

26. What can be the result if we have a overly small learning rate?
- converge too slow

27. What's the meaning of learning rate annealing?
- Start with High LR, then decrease over time

28. What's the meaning of learning rate warm-up?
- set LR to low -> high -> low, which improves stability

29. What's the meaning of learning rate Cyclical?
- get the LR bound from the maximum value and the minimum value

30. How to prevent overfitting?
- Early Stopping
- Weight Decay
- Dropout: Reduces reliance on a single node/feature
- Using mini-batch: GD on a small batch of the data
 
31. What are the benefits of mini-batch?
- avoid overfitting (regularization)
- reduce the data load to GPU
- model update more frequently (faster)

32. What are the goals for normalization?
- Puts features on a similar scale
- Potentially avoid vanishing/exploding gradients

33. Why vanishing/exploding gradients can become even worse in a deep network?
- becasue of chain rule
 
34. How can we avoid vanishing/exploding gradients?
- by Batch normalization

35. Why do we need to make loss landscape smooth?
- making the loss landscape easier to traverse
- Skip Connections

36. What are the common optimizers?
- RMSprop
- Adam
- AdamW (adam with weight decay)
- Adadelta

37. What's the goal for tweaking weight initialization?
- get better/more stable starting points

38. What are the results of changing the batch size?
- Spectrum from stochastic to one batch
- Smaller batches usually results in noisier training

39. What to do if we have a mix of numerical and categorical variables for your input?
- Use embedding

40. How to decide the architecture for a given problem?
- what's the task
- available computing problem
- How your model will be used
- How interpretable you want your model to be

41. The answer to “how many layers” or “how many nodes” is usually determined by
- What other people have had success with
- Your own experiments with different architectures
