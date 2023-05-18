# Audio Analysis Scratch

A collection of various scripts and experiments for audio feature extraction using Python, Librosa and other relevant libraries aimed at music genre classification and recommendations.

## Data Set

Training on the GTZAN dataset

https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

## To Explore

1. **Spectral contrast**: Spectral contrast is a measure of the difference in amplitude between peaks and valleys in an audio signal's frequency spectrum. It provides valuable information about the texture and timbre of sounds, as different instruments or voices produce unique spectral contrasts.
2. **Chroma features**: Chroma features are a representation of the energy distribution across different pitches within an audio signal. They capture harmonic and melodic content and are often used for tasks like chord recognition, key detection, and genre classification.
3. **Mel-frequency cepstral coefficients (MFCCs)**: MFCCs are widely used features in speech recognition systems, as well as music analysis applications. They describe the shape of power spectra on a mel-frequency scale, which more closely mimics human perception than linear frequency scales do.
4. **Constant-Q Transform (CQT)**: The CQT is a time-frequency representation that provides constant Q resolution at all frequencies, meaning that it maintains consistent spectral resolution throughout its range while preserving temporal information better than other transforms like Fourier or Wavelet-based techniques.
5. **Beat tracking/rhythm pattern analysis**: Beat tracking aims to identify the underlying pulse or tempo of music by analysing regularities and patterns within an audio signal's structure over time whereas rhythm pattern analysis focuses on identifying rhythmic elements such as individual beats, accents or syncopations that contribute to creating complex musical structures.
6. **Pitch class profile/Tonal centroid features**: Pitch class profiles represent histograms providing distributions of pitch classes within music segments by considering their relative importance across octaves.Tonal centroid features deal with extracting multidimensional spaces where perceptually similar chords occupy nearby regions providing insights into tonality,hence contributing to interpretation , evaluation & synthesis tasks in Music Information Retrieval(MIR).
7. **Statistical summary of spectral data**: mean, variance ,skewness etc.: Statistical summaries compute various statistical measures such as mean (central tendency), variance (quantifying how much variation exists across values) and skewness(indicating the degree of asymmetry in distributions) on an audio signal's spectral data. These features help characterise the signal's timbral, harmonic or dynamic attributes & are often used as input for Machine Learning algorithms.
8. **Onset detection & strength signal calculation**: Onset detection identifies moments in time when new sounds or musical events begin by detecting abrupt changes in energy levels or spectral content. The strength signal calculation quantifies how strong these changes are (typically using some transformation of amplitude information), and can be used to assess the importance of detected onsets, contributing valuable information about a track's rhythmic structure,timbre ,status etc.

## Resources: 

[GTZAN Dataset - Music Genre Classification](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
Audio Files | Mel Spectrograms | CSV with extracted features

[Music Genre Classification with Python](https://towardsdatascience.com/music-genre-classification-with-python-c714d032f0d8#aa21)  
A Guide to analyzing Audio/Music signals in Python

[Music genre classification using Librosa and Tensorflow/Keras](https://blog.paperspace.com/music-genre-classification-using-librosa-and-pytorch/)  
How to implement a music genre classifier from scratch in TensorFlow/Keras using those features calculated by the Librosa library.

[MaSC Compendium Visualization](https://github.com/chrispla/MaSC_sim_vis)  
A collection of scripts for visualizing the Arab Mashriq collection of the NYU Abu Dhabi Library and the Eisenberg collection

[Getting to Know the Mel Spectrogram](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0)  
Read this short post if you want to be like Neo and know all about the Mel Spectrogram!

[music2vec: Generating Vector Embeddings for Genre-Classification Task](https://medium.com/@rajatheb/music2vec-generating-vector-embedding-for-genre-classification-task-411187a20820)  
The aim of our project was to obtain similar vector representation for music segments. We hope to capture the structural and stylistic information of the music in this low dimensional space. using genre classification as the end task.

[FMA: A Dataset For Music Analysis](https://github.com/mdeff/fma)  
We introduce the Free Music Archive (FMA), an open and easily accessible dataset suitable for evaluating several tasks in MIR, a field concerned with browsing, searching, and organizing large music collections.

[Recommending music on Spotify with deep learning](https://sander.ai/2014/08/05/spotify-cnns.html)  
Content-based music recommendation using convolutional neural networks.

[Music Recommendation System Using Machine Learning](https://www.geeksforgeeks.org/music-recommendation-system-using-machine-learning/)  
In this article, we will try to build a very basic recommender system that can recommend songs based on which songs you hear.

[Music Genre Classification using LSTM](https://www.servomagazine.com/magazine/article/music-genre-classification-using-lstm)  
Learn to build your own model which will take in a song as an input and predict or classify that particular song in one of the basic genres. We’ll be classifying among the following groups: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, and rock.

[t-SNE clearly explained](https://towardsdatascience.com/t-sne-clearly-explained-d84c537f53a)  
An intuitive explanation of t-SNE algorithm and why it’s so useful in practice.

[Understanding K-means Clustering in Machine Learning](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1)  
K-means clustering is one of the simplest and popular unsupervised machine learning algorithms.

[Data Visualization using Python for Machine Learning and Data science](https://towardsdatascience.com/data-visualization-for-machine-learning-and-data-science-a45178970be7)  
Python has several good packages to plot the data and among them Matplotlib is the most prominent one. Seaborn is also a great package which offers a lot more appealing plot and even it uses matplotlib as its base layer.

[How I Understood: What features to consider while training audio files?](https://towardsdatascience.com/how-i-understood-what-features-to-consider-while-training-audio-files-eedfb6e9002b)  
This post is aimed at briefing through some of the most important features that may be needed to build a model for an audio classification task.
