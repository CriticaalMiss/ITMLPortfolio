# ML-Portfolio

## Table of Contents

- [Overview](#overview)
- [Project Options](#project-options)
- [Built With](#built-with)
- [Further Reading](#further-reading)
- [References](#references)

## Overview

This Portfolio is a demonstration of various machine learning techniques. 

## Project Options

### Supervised Learning Techniques
1. [Student Grade Predictor](Student-Grade-PredictorGP.ipynb) 
    - The Grade Predictor Model uses a linear regression model to predict the final grade based on different customisable categories. The linear regression model is trained using a train set within the certain portion of the dataset to a high enough accuracy to then test out the model. It uses multiple libraries, Numpy, Pandas, matplotlib, scikit-learn intro, scikit-learn tut and pickle. Linear regression works by looking for a "line of best fit" or a correlation among a set of data. It presents the result on a scatterplot graph.
2. [K-NN Predictor Diabetes](KNN-Predictor-Diabetes/KNN-Predict.ipynb)
   - Uses a dataset of diabetic patients to find correlations between certain factors. The K-NN model is used in this case to handel categorised data. We choose a K value (in this case 11) and then finds the nearest K members around a random query point. The goal is to take these "voting members" and make a prediction based on their positions and data. K in most instances must be an odd number to avoid specific cases where an even amount would lead to a "tie" in the voting process.

### Unsupervised Learning Technique
3. [K-Means](KMeans/KMeans.ipynb)
- The K-Means model uses an unsupervised learning approach and a K-Means model to find patterns among the data. The unsupervised learning works to sort uncategorised data. K-Means clusters the data (in this case 3 clusters) then compacts the data by minimising the sum of squared distances between data points and the assigned centroids within a cluster.
  
### Reinforcement Learning
4. [CartPole](CartPole/CartPole-v1.ipynb)
   - Reinforcement learning works by having an agent take an action in an evironment and depending and how the action resolves gets a reward. Reinforcement learning works by running multiple generations with this agent-reward cycle to eventually produce a "trained" agent that successfully performs the action after being rewarded and passing down information after each generation. It works in a similar function to evolution. 

### Essay
5. ["Privacy in the Age of AI: Navigating the Ethical Dimensions of Machine Learning"]((https://github.com/CriticaalMiss/ITMLPortfolio/blob/main/Privacy%20in%20the%20Age%20of%20AI%20Navigating%20the%20Ethical%20Dimensions%20of%20Machine%20Learning%20-%20Google%20Docs.pdf))


## Built With

#### Python and Jupyter notebooks

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Jupyter Notebook](https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=for-the-badge&logo=jupyter&logoColor=white)
![Anaconda](https://img.shields.io/badge/Anaconda-%2344A833.svg?style=for-the-badge&logo=anaconda&logoColor=white)
![Visual Studio Code](https://img.shields.io/badge/Visual%20Studio%20Code-0078d7.svg?style=for-the-badge&logo=visual-studio-code&logoColor=white)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)


## Further Reading

- [Introduction to AI - Course Guide](https://cgsacteduau.sharepoint.com/:w:/s/cgssharedfolders/EUlW1KFBKzJGskD936SUUCMBLgqp_OeB3nzkrVs3cELybA?e=lFQruw)
- [What is Machine Learning](https://www.mathworks.com/discovery/machine-learning.html)
- [Understanding Machine Learning & Deep Learning](https://dltlabs.medium.com/understanding-machine-learning-deep-learning-f5aa95264d61)
- [Q-Learning Algorithm](https://aleksandarhaber.com/q-learning-in-python-with-tests-in-cart-pole-openai-gym-environment-reinforcement-learning-tutorial/)
- [Epsilon-Greedy Algorithm](https://www.geeksforgeeks.org/epsilon-greedy-algorithm-in-reinforcement-learning/)


## References

- [Cart Pole Control Environment in OpenAI Gym](https://aleksandarhaber.com/cart-pole-control-environment-in-openai-gym-gymnasium-introduction-to-openai-gym/) - Aleksandar Haber
- [Markdown Syntax](https://wilsonmar.github.io/markdown-text-for-github-from-html/) - Learn your md
- [Markdown Badge List](https://github.com/Ileriayo/markdown-badges) - Style your md with a badge
- [One Direction](https://www.youtube.com/watch?v=AsmHz9JCU4M) - Coding Music?
- [Twent-One Pilots](https://www.youtube.com/watch?v=pXRviuL6vMY) - Getting better...
- [Ultimate Jupyter NB Flexes](https://noteable.io/blog/jupyter-notebook-shortcuts-boost-productivity/#:~:text=The%20shortcut%20to%20add%20a,cell%2C%20use%20the%20shortcut%20B.) - Flex correctly
