# Step-by-Step Implementation of Artificial Neural Networks (ANN) from Scratch for Binary Classification in Python

<img src="Images/ANN-Banner.png" width="1000" />
  
## 1. Objective

The objective of this project is to demonstrate the step-by-step implementation of an Artificial Neural Network (ANN) from scratch to classify images of cats.

## 2. Motivation

It is often said that “What I cannot build, I do not understand”. Thus, in order to gain a deeper understanding of Artificial Neural Networks (ANN), I embarked on this project with the aim to built an ANN, step-by-step and from scratch in NumPy, without making use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 

* Implementing an ANN from scratch is complex process, but it can be broken down to iterating over the following steps:

  * Initializing the parameters for a two-layer network and for an 𝐿L-layer neural network
  * Implementing the forward propagation module (shown in purple in the figure below)
  * Computing the loss
  * Implementing the backward propagation module (denoted in red in the figure below)
  * Updating the parameters

We shall illustrate the step-by-step implementation of these two main phases and break them down into several functionalities. We shall use a small cat vs. non-cat data set as the basis for learning and practicing how to develop, evaluate, and use artificial neural networks for image classification from scratch. 

## 3. Data

* We will be using a small "Cat vs non-Cat" dataset, which consists of the following:
  * A training set of 209 images labelled as cat (1) or non-cat (0)
  * A test set of 50 images labelled as cat and non-cat
  * Each image is of shape (64, 64, 3).
  
## 4. Development

In this section, we shall demonstrate how to develop an Artificial  Neural Network (CNN) for binary image classification from scratch, without making use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 

* The development process involves:

  * Reading and pre-processing the training and test data
  * Exploring and visualizing the training and test data:
  * Building the forward phase and backward phase of the ANN model
  * Training the built ANN model
  * Evaluating the performance of the trained ANN model.

  * Author: Mohsen Ghazel (mghazel)
  * Date: May 15th, 2021

  * Project: Step-by-Step Implementation Artificial Neural Networks (ANN) from Scratch using Numpy.

* To build your neural network, we will be implementing several utilities functions, as follows:
    * Initialize the parameters for a two-layer network and for an 𝐿L-layer neural network
    * Implement the forward propagation module
    * Compute the loss
    * Implement the backward propagation module
    * Finally, update the parameters.

This development process is illustrated in the figure below.

<img src="Images/final outline.png" width="1000" />

* First, we make the following observations:

  * For every forward function, there is a corresponding backward function.
  * This is why at every step of your forward module you will be storing some values in a cache. 
  * These cached values are useful for computing gradients.
  * In the backpropagation module, you can then use the cache to calculate the gradients.


### 4.1. Step 1: Imports and global variables:

#### 4.1.1. Standard scientific Python imports:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">import</span> time
<span style="color:#200080; font-weight:bold; ">import</span> numpy <span style="color:#200080; font-weight:bold; ">as</span> np
<span style="color:#200080; font-weight:bold; ">import</span> h5py
<span style="color:#200080; font-weight:bold; ">import</span> matplotlib<span style="color:#308080; ">.</span>pyplot <span style="color:#200080; font-weight:bold; ">as</span> plt
<span style="color:#200080; font-weight:bold; ">import</span> scipy
<span style="color:#200080; font-weight:bold; ">from</span> PIL <span style="color:#200080; font-weight:bold; ">import</span> Image
<span style="color:#200080; font-weight:bold; ">from</span> scipy <span style="color:#200080; font-weight:bold; ">import</span> ndimage

<span style="color:#595979; "># random number generators values</span>
<span style="color:#595979; "># seed for reproducing the random number generation</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> seed
<span style="color:#595979; "># random integers: I(0,M)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> randint
<span style="color:#595979; "># random standard unform: U(0,1)</span>
<span style="color:#200080; font-weight:bold; ">from</span> random <span style="color:#200080; font-weight:bold; ">import</span> random
<span style="color:#595979; "># time</span>
<span style="color:#200080; font-weight:bold; ">import</span> datetime
<span style="color:#595979; "># I/O</span>
<span style="color:#200080; font-weight:bold; ">import</span> os
<span style="color:#595979; "># sys</span>
<span style="color:#200080; font-weight:bold; ">import</span> sys

<span style="color:#44aadd; ">%</span>matplotlib inline
plt<span style="color:#308080; ">.</span>rcParams<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'figure.figsize'</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008000; ">5.0</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">4.0</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># set default size of plots</span>
plt<span style="color:#308080; ">.</span>rcParams<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'image.interpolation'</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'nearest'</span>
plt<span style="color:#308080; ">.</span>rcParams<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'image.cmap'</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'gray'</span>
</pre>


### 4.1.2. Global variables:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># Set the random state to 101</span>
<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># - This ensures repeatable results everytime you run the code. </span>
RANDOM_STATE <span style="color:#308080; ">=</span> <span style="color:#008c00; ">101</span>

<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># We set the Numpy pseudo-random generator at a fixed value:</span>
<span style="color:#595979; ">#-------------------------------------------------------------------------------</span>
<span style="color:#595979; "># - This ensures repeatable results everytime you run the code. </span>
np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>seed<span style="color:#308080; ">(</span>RANDOM_STATE<span style="color:#308080; ">)</span>

<span style="color:#595979; "># the number of visualized images</span>
NUM_VISUALIZED_IMAGES <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span>
</pre>

### 4.2. Step 2: Load and process the dataset:

* We will be using a small "Cat vs non-Cat" dataset, which consists of the following:
  * A training set of 209 images labelled as cat (1) or non-cat (0)
  * A test set of 50 images labelled as cat and non-cat
  * Each image is of shape (64, 64, 3)
  * Let's get more familiar with the dataset. Load the data by running the cell below.
  
#### 4.2.1. Implement functionality to load the dataset:



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> load_data<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Load the the dataseta nd split it into training and test subsets.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    train_dataset <span style="color:#308080; ">=</span> h5py<span style="color:#308080; ">.</span><span style="color:#400000; ">File</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'datasets/train_catvnoncat.h5'</span><span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"r"</span><span style="color:#308080; ">)</span>
    train_set_x_orig <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>train_dataset<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"train_set_x"</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># your train set features</span>
    train_set_y_orig <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>train_dataset<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"train_set_y"</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># your train set labels</span>

    test_dataset <span style="color:#308080; ">=</span> h5py<span style="color:#308080; ">.</span><span style="color:#400000; ">File</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'datasets/test_catvnoncat.h5'</span><span style="color:#308080; ">,</span> <span style="color:#1060b6; ">"r"</span><span style="color:#308080; ">)</span>
    test_set_x_orig <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>test_dataset<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"test_set_x"</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># your test set features</span>
    test_set_y_orig <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>test_dataset<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"test_set_y"</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># your test set labels</span>

    classes <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>test_dataset<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"list_classes"</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># the list of classes</span>
    
    train_set_y_orig <span style="color:#308080; ">=</span> train_set_y_orig<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> train_set_y_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    test_set_y_orig <span style="color:#308080; ">=</span> test_set_y_orig<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> test_set_y_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> train_set_x_orig<span style="color:#308080; ">,</span> train_set_y_orig<span style="color:#308080; ">,</span> test_set_x_orig<span style="color:#308080; ">,</span> test_set_y_orig<span style="color:#308080; ">,</span> classes
</pre>

#### 4.2.2. Load the training and test subsets:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># load the traininga nd test data subsets</span>
train_x_orig<span style="color:#308080; ">,</span> train_y<span style="color:#308080; ">,</span> test_x_orig<span style="color:#308080; ">,</span> test_y<span style="color:#308080; ">,</span> classes <span style="color:#308080; ">=</span> load_data<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>


#### 4.2.3. Display the number and shape of the training and test subsets:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># # Explore your dataset:</span>
<span style="color:#595979; ">#------------------------------------------------------</span>
<span style="color:#595979; "># the number of training examples</span>
m_train <span style="color:#308080; ">=</span> train_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#595979; "># the number of image pixels</span>
num_px <span style="color:#308080; ">=</span> train_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
<span style="color:#595979; "># the number of test examples</span>
m_test <span style="color:#308080; ">=</span> test_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span>
<span style="color:#595979; "># display dataset properties</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of training examples: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>m_train<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Number of testing examples: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>m_test<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Each image is of size: ("</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>num_px<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">", "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>num_px<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">", 3)"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"train_x_orig shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>train_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"train_y shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"test_x_orig shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>test_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"test_y shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>test_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>


Number of training examples<span style="color:#308080; ">:</span> <span style="color:#008c00; ">209</span>
Number of testing examples<span style="color:#308080; ">:</span> <span style="color:#008c00; ">50</span>
Each image <span style="color:#200080; font-weight:bold; ">is</span> of size<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
train_x_orig shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">209</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
train_y shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">209</span><span style="color:#308080; ">)</span>
test_x_orig shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">50</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span>
test_y shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">50</span><span style="color:#308080; ">)</span>
</pre>

#### 4.2.4. Display the targets/classes:

* We expect 2 classes:
  * 1: Cat
  * 0: Not-Cat


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Classes/labels:"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'The target labels: '</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"----------------------------------------------"</span><span style="color:#308080; ">)</span>

<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
Classes<span style="color:#44aadd; ">/</span>labels<span style="color:#308080; ">:</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
The target labels<span style="color:#308080; ">:</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
<span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span><span style="color:#44aadd; ">-</span>
</pre>

#### 4.2.5. Examine the number of images for each class of the training and testing subsets:

##### 4.2.5.1 First implement a functionality to generate the histogram of the number of training and test images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># create a histogram of the number of images in each class/digit:</span>
<span style="color:#200080; font-weight:bold; ">def</span> plot_bar<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">,</span> relative<span style="color:#308080; ">=</span><span style="color:#074726; ">False</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    width <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.35</span>
    <span style="color:#200080; font-weight:bold; ">if</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span><span style="color:#008000; ">0.5</span>
    <span style="color:#200080; font-weight:bold; ">elif</span> loc <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">:</span>
        n <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.5</span>
     
    <span style="color:#595979; "># calculate counts per type and sort, to ensure their order</span>
    unique<span style="color:#308080; ">,</span> counts <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>unique<span style="color:#308080; ">(</span>y<span style="color:#308080; ">,</span> return_counts<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    sorted_index <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>argsort<span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span>
    unique <span style="color:#308080; ">=</span> unique<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
     
    <span style="color:#200080; font-weight:bold; ">if</span> relative<span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot as a percentage</span>
        counts <span style="color:#308080; ">=</span> <span style="color:#008c00; ">100</span><span style="color:#44aadd; ">*</span>counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span><span style="color:#44aadd; ">/</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>y<span style="color:#308080; ">)</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'% Count'</span>
    <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># plot counts</span>
        counts <span style="color:#308080; ">=</span> counts<span style="color:#308080; ">[</span>sorted_index<span style="color:#308080; ">]</span>
        ylabel_text <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">'Count'</span>
         
    xtemp <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>arange<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>unique<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>bar<span style="color:#308080; ">(</span>xtemp <span style="color:#44aadd; ">+</span> n<span style="color:#44aadd; ">*</span>width<span style="color:#308080; ">,</span> counts<span style="color:#308080; ">,</span> align<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'center'</span><span style="color:#308080; ">,</span> alpha<span style="color:#308080; ">=</span><span style="color:#008000; ">.7</span><span style="color:#308080; ">,</span> width<span style="color:#308080; ">=</span>width<span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xticks<span style="color:#308080; ">(</span>xtemp<span style="color:#308080; ">,</span> unique<span style="color:#308080; ">,</span> rotation<span style="color:#308080; ">=</span><span style="color:#008c00; ">45</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Digit'</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span>ylabel_text<span style="color:#308080; ">)</span>
<span style="color:#595979; "># add title</span>
plt<span style="color:#308080; ">.</span>suptitle<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Percentage of images per digit (0-9)'</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

##### 4.2.5.2. Call the functionality to generate the histogram of the number of training and test images:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># Call the function to create the histograms of the </span>
<span style="color:#595979; "># training and test images:</span>
<span style="color:#595979; ">#----------------------------------------------------</span>
<span style="color:#595979; "># set the figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># training data histogram</span>
plot_bar<span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'left'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># test data histogram</span>
plot_bar<span style="color:#308080; ">(</span>test_y<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>test_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> loc<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'right'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">[</span>
    <span style="color:#1060b6; ">'Training dataset: ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>train_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
    <span style="color:#1060b6; ">'Test dataset: ({0} images)'</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span><span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>test_y<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>test_y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> 
<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="results/Histogram.png" width="1000" />

#### 4.2.6 Visualize some of the training and test images and their associated targets:
##### 4.2.6.1. First implement a visualization functionality to visualize the number of randomly selected images:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">"""</span>
<span style="color:#595979; "># A utility function to visualize multiple images:</span>
<span style="color:#595979; ">"""</span>
<span style="color:#200080; font-weight:bold; ">def</span> visualize_images_and_labels<span style="color:#308080; ">(</span>num_visualized_images <span style="color:#308080; ">=</span> <span style="color:#008c00; ">25</span><span style="color:#308080; ">,</span> dataset_flag <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
  <span style="color:#595979; ">"""To visualize images.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Keyword arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- num_visualized_images -- the number of visualized images (deafult 25)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- dataset_flag -- 1: training dataset, 2: test dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Return:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- None</span>
<span style="color:#595979; ">&nbsp;&nbsp;"""</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  <span style="color:#595979; "># the suplot grid shape:</span>
  <span style="color:#595979; ">#--------------------------------------------</span>
  num_rows <span style="color:#308080; ">=</span> <span style="color:#008c00; ">5</span>
  <span style="color:#595979; "># the number of columns</span>
  num_cols <span style="color:#308080; ">=</span> num_visualized_images <span style="color:#44aadd; ">//</span> num_rows
  <span style="color:#595979; "># setup the subplots axes</span>
  fig<span style="color:#308080; ">,</span> axes <span style="color:#308080; ">=</span> plt<span style="color:#308080; ">.</span>subplots<span style="color:#308080; ">(</span>nrows<span style="color:#308080; ">=</span>num_rows<span style="color:#308080; ">,</span> ncols<span style="color:#308080; ">=</span>num_cols<span style="color:#308080; ">,</span> figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">8</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">10</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
  <span style="color:#595979; "># set a seed random number generator for reproducible results</span>
  seed<span style="color:#308080; ">(</span>RANDOM_STATE<span style="color:#308080; ">)</span>
  <span style="color:#595979; "># iterate over the sub-plots</span>
  <span style="color:#200080; font-weight:bold; ">for</span> row <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_rows<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
      <span style="color:#200080; font-weight:bold; ">for</span> col <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_cols<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># get the next figure axis</span>
        ax <span style="color:#308080; ">=</span> axes<span style="color:#308080; ">[</span>row<span style="color:#308080; ">,</span> col<span style="color:#308080; ">]</span><span style="color:#308080; ">;</span>
        <span style="color:#595979; "># turn-off subplot axis</span>
        ax<span style="color:#308080; ">.</span>set_axis_off<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># if the dataset_flag = 1: Training data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">if</span> <span style="color:#308080; ">(</span> dataset_flag <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span> <span style="color:#308080; ">)</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>m_train<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the training image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>train_x_orig<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> train_y<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># dataset_flag = 2: Test data set</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span> 
          <span style="color:#595979; "># generate a random image counter</span>
          counter <span style="color:#308080; ">=</span> randint<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>m_test<span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the test image</span>
          image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>test_x_orig<span style="color:#308080; ">[</span>counter<span style="color:#308080; ">,</span><span style="color:#308080; ">:</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
          <span style="color:#595979; "># get the target associated with the image</span>
          label <span style="color:#308080; ">=</span> test_y<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> counter<span style="color:#308080; ">]</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        <span style="color:#595979; "># display the image</span>
        <span style="color:#595979; ">#--------------------------------------------</span>
        ax<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> cmap<span style="color:#308080; ">=</span>plt<span style="color:#308080; ">.</span>cm<span style="color:#308080; ">.</span>gray_r<span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        <span style="color:#595979; "># set the title showing the image label</span>
        ax<span style="color:#308080; ">.</span>set_title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'y ='</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>label<span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> size <span style="color:#308080; ">=</span> <span style="color:#008c00; ">8</span><span style="color:#308080; ">)</span>
</pre>


##### 4.2.6.2. Visualize some of the training images and their associated targets:

<img src="results/Training-25-sample-images.PNG" width="1000" />


##### 4.2.6.3. Visualize some of the test images and their associated targets:

<img src="results/Test-25-sample-images.PNG" width="1000" />

#### 4.2.7. Reshape and Normalize the training and test images:
* We need to reshape and standardize the images before feeding them to the network:

##### 4.2.7.1. Flatten the 2D images to 1D vectors:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Reshape the training and test examples </span>
<span style="color:#595979; "># The "-1" makes reshape flatten the remaining dimensions</span>
train_x_flatten <span style="color:#308080; ">=</span> train_x_orig<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>train_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>T   
<span style="color:#595979; "># The "-1" makes reshape flatten the remaining dimensions</span>
test_x_flatten <span style="color:#308080; ">=</span> test_x_orig<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>test_x_orig<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> <span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span>
</pre>

<img src="Images/imvectorkiank.png" width="1000" />

##### 4.2.7.2) Normalize the training and test images to the (0,1) range:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Standardize data to have feature values between 0 and 1.</span>
train_x <span style="color:#308080; ">=</span> train_x_flatten<span style="color:#44aadd; ">/</span><span style="color:#008000; ">255.</span>
test_x <span style="color:#308080; ">=</span> test_x_flatten<span style="color:#44aadd; ">/</span><span style="color:#008000; ">255.</span>

<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"train_x's shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>train_x<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"test_x's shape: "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>test_x<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

train_x<span style="color:#1060b6; ">'s shape: (12288, 209</span><span style="color:#ffffff; background:#dd9999; font-weight:bold; font-style:italic; ">)</span><span style="color:#1060b6; "></span>
<span style="color:#1060b6; ">test_x'</span>s shape<span style="color:#308080; ">:</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">12288</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">50</span><span style="color:#308080; ">)</span>
</pre>


### 4.3. Step 3: Build the Artificial Neural Network Model Architecture:

#### 4.3.1. The L-layer Deep Neural Network:

* The L-layer deep neural network has the following simplified network representation:

<img src="Images/L-layer-ANN-Cats-Calssification.PNG" width="1000" />


#### 4.3.2. General Methodology:

* As usual, you'll follow the Deep Learning methodology to build the model:

  * Initialize parameters / Define hyperparameters
  * Loop for num_iterations: a. Forward propagation b. Compute cost function c. Backward propagation d. Update parameters (using parameters, and grads from backprop)
  * Use trained parameters to predict labels
  * Now go ahead and implement the L-layer ANN model.
  
#### 4.3.3 Utilities functions:

##### 4.3.3.1. Functions to initialize parameters and define hyperparameters:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> initialize_parameters<span style="color:#308080; ">(</span>n_x<span style="color:#308080; ">,</span> n_h<span style="color:#308080; ">,</span> n_y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Argument:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;n_x -- size of the input layer</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;n_h -- size of the hidden layer</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;n_y -- size of the output layer</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- python dictionary containing your parameters:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;W1 -- weight matrix of shape (n_h, n_x)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b1 -- bias vector of shape (n_h, 1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;W2 -- weight matrix of shape (n_y, n_h)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b2 -- bias vector of shape (n_y, 1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>seed<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    
    W1 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>n_h<span style="color:#308080; ">,</span> n_x<span style="color:#308080; ">)</span><span style="color:#44aadd; ">*</span><span style="color:#008000; ">0.01</span>
    b1 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>n_h<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    W2 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>n_y<span style="color:#308080; ">,</span> n_h<span style="color:#308080; ">)</span><span style="color:#44aadd; ">*</span><span style="color:#008000; ">0.01</span>
    b2 <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>n_y<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>W1<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>n_h<span style="color:#308080; ">,</span> n_x<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>b1<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>n_h<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>W2<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>n_y<span style="color:#308080; ">,</span> n_h<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>b2<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>n_y<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    parameters <span style="color:#308080; ">=</span> <span style="color:#406080; ">{</span><span style="color:#1060b6; ">"W1"</span><span style="color:#308080; ">:</span> W1<span style="color:#308080; ">,</span>
                  <span style="color:#1060b6; ">"b1"</span><span style="color:#308080; ">:</span> b1<span style="color:#308080; ">,</span>
                  <span style="color:#1060b6; ">"W2"</span><span style="color:#308080; ">:</span> W2<span style="color:#308080; ">,</span>
                  <span style="color:#1060b6; ">"b2"</span><span style="color:#308080; ">:</span> b2<span style="color:#406080; ">}</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> parameters     


<span style="color:#200080; font-weight:bold; ">def</span> initialize_parameters_deep<span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;layer_dims -- python array (list) containing the dimensions of each layer in our network</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;bl -- bias vector of shape (layer_dims[l], 1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>seed<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    parameters <span style="color:#308080; ">=</span> <span style="color:#406080; ">{</span><span style="color:#406080; ">}</span>
    L <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">)</span>            <span style="color:#595979; "># number of layers in the network</span>

    <span style="color:#200080; font-weight:bold; ">for</span> l <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> L<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'W'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>randn<span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">[</span>l<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> layer_dims<span style="color:#308080; ">[</span>l<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">/</span> np<span style="color:#308080; ">.</span>sqrt<span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">[</span>l<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span> <span style="color:#595979; ">#*0.01</span>
        parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'b'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">[</span>l<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        
        <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'W'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">[</span>l<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> layer_dims<span style="color:#308080; ">[</span>l<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'b'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>layer_dims<span style="color:#308080; ">[</span>l<span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

        
    <span style="color:#200080; font-weight:bold; ">return</span> parameters
</pre>


#### 4.3.3.2. Functions to implement the Forward Propagation:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> sigmoid<span style="color:#308080; ">(</span>Z<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implements the sigmoid activation in numpy</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Z -- numpy array of any shape</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A -- output of sigmoid(z), same shape as Z</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- returns Z as well, useful during backpropagation</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    A <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#44aadd; ">+</span>np<span style="color:#308080; ">.</span>exp<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span>Z<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    cache <span style="color:#308080; ">=</span> Z
    
    <span style="color:#200080; font-weight:bold; ">return</span> A<span style="color:#308080; ">,</span> cache

<span style="color:#200080; font-weight:bold; ">def</span> relu<span style="color:#308080; ">(</span>Z<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the RELU function.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Z -- Output of the linear layer, of any shape</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A -- Post-activation parameter, of the same shape as Z</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    A <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>maximum<span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>Z<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>A<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> Z<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    
    cache <span style="color:#308080; ">=</span> Z 
    <span style="color:#200080; font-weight:bold; ">return</span> A<span style="color:#308080; ">,</span> cache



<span style="color:#200080; font-weight:bold; ">def</span> linear_forward<span style="color:#308080; ">(</span>A<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the linear part of a layer's forward propagation.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A -- activations from previous layer (or input data): (size of previous layer, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;b -- bias vector, numpy array of shape (size of the current layer, 1)</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Z -- the input of the activation function, also called pre-activation parameter </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    Z <span style="color:#308080; ">=</span> W<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>A<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> b
    
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>Z<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>W<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> A<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    cache <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>A<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> Z<span style="color:#308080; ">,</span> cache

<span style="color:#200080; font-weight:bold; ">def</span> linear_activation_forward<span style="color:#308080; ">(</span>A_prev<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b<span style="color:#308080; ">,</span> activation<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the forward propagation for the LINEAR-&gt;ACTIVATION layer</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;b -- bias vector, numpy array of shape (size of the current layer, 1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;A -- the output of the activation function, also called the post-activation value </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- a python dictionary containing "linear_cache" and "activation_cache";</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;stored for computing the backward pass efficiently</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    <span style="color:#200080; font-weight:bold; ">if</span> activation <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">"sigmoid"</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Inputs: "A_prev, W, b". Outputs: "A, activation_cache".</span>
        Z<span style="color:#308080; ">,</span> linear_cache <span style="color:#308080; ">=</span> linear_forward<span style="color:#308080; ">(</span>A_prev<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b<span style="color:#308080; ">)</span>
        A<span style="color:#308080; ">,</span> activation_cache <span style="color:#308080; ">=</span> sigmoid<span style="color:#308080; ">(</span>Z<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">elif</span> activation <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">"relu"</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># Inputs: "A_prev, W, b". Outputs: "A, activation_cache".</span>
        Z<span style="color:#308080; ">,</span> linear_cache <span style="color:#308080; ">=</span> linear_forward<span style="color:#308080; ">(</span>A_prev<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b<span style="color:#308080; ">)</span>
        A<span style="color:#308080; ">,</span> activation_cache <span style="color:#308080; ">=</span> relu<span style="color:#308080; ">(</span>Z<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>A<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span>W<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> A_prev<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    cache <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span>linear_cache<span style="color:#308080; ">,</span> activation_cache<span style="color:#308080; ">)</span>

    <span style="color:#200080; font-weight:bold; ">return</span> A<span style="color:#308080; ">,</span> cache

<span style="color:#200080; font-weight:bold; ">def</span> L_model_forward<span style="color:#308080; ">(</span>X<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement forward propagation for the [LINEAR-&gt;RELU]*(L-1)-&gt;LINEAR-&gt;SIGMOID computation</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;X -- data, numpy array of shape (input size, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- output of initialize_parameters_deep()</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;AL -- last post-activation value</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;caches -- list of caches containing:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the cache of linear_sigmoid_forward() (there is one, indexed L-1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>

    caches <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>
    A <span style="color:#308080; ">=</span> X
    L <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">2</span>                  <span style="color:#595979; "># number of layers in the neural network</span>
    
    <span style="color:#595979; "># Implement [LINEAR -&gt; RELU]*(L-1). Add "cache" to the "caches" list.</span>
    <span style="color:#200080; font-weight:bold; ">for</span> l <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> L<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        A_prev <span style="color:#308080; ">=</span> A 
        A<span style="color:#308080; ">,</span> cache <span style="color:#308080; ">=</span> linear_activation_forward<span style="color:#308080; ">(</span>A_prev<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'W'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'b'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> activation <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"relu"</span><span style="color:#308080; ">)</span>
        caches<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>cache<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Implement LINEAR -&gt; SIGMOID. Add "cache" to the "caches" list.</span>
    AL<span style="color:#308080; ">,</span> cache <span style="color:#308080; ">=</span> linear_activation_forward<span style="color:#308080; ">(</span>A<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'W'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>L<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'b'</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>L<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> activation <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"sigmoid"</span><span style="color:#308080; ">)</span>
    caches<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>cache<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>AL<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span>X<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
            
    <span style="color:#200080; font-weight:bold; ">return</span> AL<span style="color:#308080; ">,</span> caches
</pre>

##### 4.3.3.3. Function to compute the cost:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> compute_cost<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the cost function defined by equation (7).</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;AL -- probability vector corresponding to your label predictions, shape (1, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cost -- cross-entropy cost</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    m <span style="color:#308080; ">=</span> Y<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>

    <span style="color:#595979; "># Compute loss from aL and y.</span>
    cost <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008000; ">1.</span><span style="color:#44aadd; ">/</span>m<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">*</span> <span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span>np<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>Y<span style="color:#308080; ">,</span>np<span style="color:#308080; ">.</span>log<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">-</span> np<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#44aadd; ">-</span>Y<span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>log<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#44aadd; ">-</span>AL<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    cost <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>cost<span style="color:#308080; ">)</span>      <span style="color:#595979; "># To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).</span>
    <span style="color:#200080; font-weight:bold; ">assert</span><span style="color:#308080; ">(</span>cost<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> <span style="color:#308080; ">(</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> cost
</pre>

##### 4.3.3.4. Functions to implement the Backward propagation:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> relu_backward<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> cache<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the backward propagation for a single RELU unit.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dA -- post-activation gradient, of any shape</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- 'Z' where we store for computing backward propagation efficiently</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dZ -- Gradient of the cost with respect to Z</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    Z <span style="color:#308080; ">=</span> cache
    dZ <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> copy<span style="color:#308080; ">=</span><span style="color:#074726; ">True</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># just converting dz to a correct object.</span>
    
    <span style="color:#595979; "># When z &lt;= 0, you should set dz to 0 as well. </span>
    dZ<span style="color:#308080; ">[</span>Z <span style="color:#44aadd; ">&lt;=</span> <span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> Z<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> dZ

<span style="color:#200080; font-weight:bold; ">def</span> sigmoid_backward<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> cache<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the backward propagation for a single SIGMOID unit.</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dA -- post-activation gradient, of any shape</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- 'Z' where we store for computing backward propagation efficiently</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dZ -- Gradient of the cost with respect to Z</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    Z <span style="color:#308080; ">=</span> cache
    
    s <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#44aadd; ">/</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#44aadd; ">+</span>np<span style="color:#308080; ">.</span>exp<span style="color:#308080; ">(</span><span style="color:#44aadd; ">-</span>Z<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    dZ <span style="color:#308080; ">=</span> dA <span style="color:#44aadd; ">*</span> s <span style="color:#44aadd; ">*</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#44aadd; ">-</span>s<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> Z<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> dZ


<span style="color:#200080; font-weight:bold; ">def</span> linear_backward<span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">,</span> cache<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the linear portion of backward propagation for a single layer (layer l)</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dZ -- Gradient of the cost with respect to the linear output (of current layer l)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer</span>
<span style="color:#595979; "></span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dW -- Gradient of the cost with respect to W (current layer l), same shape as W</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;db -- Gradient of the cost with respect to b (current layer l), same shape as b</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    A_prev<span style="color:#308080; ">,</span> W<span style="color:#308080; ">,</span> b <span style="color:#308080; ">=</span> cache
    m <span style="color:#308080; ">=</span> A_prev<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>

    dW <span style="color:#308080; ">=</span> <span style="color:#008000; ">1.</span><span style="color:#44aadd; ">/</span>m <span style="color:#44aadd; ">*</span> np<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">,</span>A_prev<span style="color:#308080; ">.</span>T<span style="color:#308080; ">)</span>
    db <span style="color:#308080; ">=</span> <span style="color:#008000; ">1.</span><span style="color:#44aadd; ">/</span>m <span style="color:#44aadd; ">*</span> np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">,</span> axis <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> keepdims <span style="color:#308080; ">=</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>
    dA_prev <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>dot<span style="color:#308080; ">(</span>W<span style="color:#308080; ">.</span>T<span style="color:#308080; ">,</span>dZ<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>dA_prev<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> A_prev<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>dW<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> W<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">assert</span> <span style="color:#308080; ">(</span>db<span style="color:#308080; ">.</span>shape <span style="color:#44aadd; ">==</span> b<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> dA_prev<span style="color:#308080; ">,</span> dW<span style="color:#308080; ">,</span> db

<span style="color:#200080; font-weight:bold; ">def</span> linear_activation_backward<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> cache<span style="color:#308080; ">,</span> activation<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the backward propagation for the LINEAR-&gt;ACTIVATION layer.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dA -- post-activation gradient for current layer l </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;dW -- Gradient of the cost with respect to W (current layer l), same shape as W</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;db -- Gradient of the cost with respect to b (current layer l), same shape as b</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    linear_cache<span style="color:#308080; ">,</span> activation_cache <span style="color:#308080; ">=</span> cache
    
    <span style="color:#200080; font-weight:bold; ">if</span> activation <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">"relu"</span><span style="color:#308080; ">:</span>
        dZ <span style="color:#308080; ">=</span> relu_backward<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> activation_cache<span style="color:#308080; ">)</span>
        dA_prev<span style="color:#308080; ">,</span> dW<span style="color:#308080; ">,</span> db <span style="color:#308080; ">=</span> linear_backward<span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">,</span> linear_cache<span style="color:#308080; ">)</span>
        
    <span style="color:#200080; font-weight:bold; ">elif</span> activation <span style="color:#44aadd; ">==</span> <span style="color:#1060b6; ">"sigmoid"</span><span style="color:#308080; ">:</span>
        dZ <span style="color:#308080; ">=</span> sigmoid_backward<span style="color:#308080; ">(</span>dA<span style="color:#308080; ">,</span> activation_cache<span style="color:#308080; ">)</span>
        dA_prev<span style="color:#308080; ">,</span> dW<span style="color:#308080; ">,</span> db <span style="color:#308080; ">=</span> linear_backward<span style="color:#308080; ">(</span>dZ<span style="color:#308080; ">,</span> linear_cache<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> dA_prev<span style="color:#308080; ">,</span> dW<span style="color:#308080; ">,</span> db

<span style="color:#200080; font-weight:bold; ">def</span> L_model_backward<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">,</span> caches<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implement the backward propagation for the [LINEAR-&gt;RELU] * (L-1) -&gt; LINEAR -&gt; SIGMOID group</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;AL -- probability vector, output of the forward propagation (L_model_forward())</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Y -- true "label" vector (containing 0 if non-cat, 1 if cat)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;caches -- list of caches containing:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;grads -- A dictionary with the gradients</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grads["dA" + str(l)] = ... </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grads["dW" + str(l)] = ...</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;grads["db" + str(l)] = ... </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    grads <span style="color:#308080; ">=</span> <span style="color:#406080; ">{</span><span style="color:#406080; ">}</span>
    L <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>caches<span style="color:#308080; ">)</span> <span style="color:#595979; "># the number of layers</span>
    m <span style="color:#308080; ">=</span> AL<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
    Y <span style="color:#308080; ">=</span> Y<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">)</span> <span style="color:#595979; "># after this line, Y is the same shape as AL</span>
    
    <span style="color:#595979; "># Initializing the backpropagation</span>
    dAL <span style="color:#308080; ">=</span> <span style="color:#44aadd; ">-</span> <span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>divide<span style="color:#308080; ">(</span>Y<span style="color:#308080; ">,</span> AL<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">-</span> np<span style="color:#308080; ">.</span>divide<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">-</span> Y<span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span> <span style="color:#44aadd; ">-</span> AL<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Lth layer (SIGMOID -&gt; LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]</span>
    current_cache <span style="color:#308080; ">=</span> caches<span style="color:#308080; ">[</span>L<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
    grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dA"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>L<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dW"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>L<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"db"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>L<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> linear_activation_backward<span style="color:#308080; ">(</span>dAL<span style="color:#308080; ">,</span> current_cache<span style="color:#308080; ">,</span> activation <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"sigmoid"</span><span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">for</span> l <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">reversed</span><span style="color:#308080; ">(</span><span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>L<span style="color:#44aadd; ">-</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#595979; "># lth layer: (RELU -&gt; LINEAR) gradients.</span>
        current_cache <span style="color:#308080; ">=</span> caches<span style="color:#308080; ">[</span>l<span style="color:#308080; ">]</span>
        dA_prev_temp<span style="color:#308080; ">,</span> dW_temp<span style="color:#308080; ">,</span> db_temp <span style="color:#308080; ">=</span> linear_activation_backward<span style="color:#308080; ">(</span>grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dA"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">,</span> current_cache<span style="color:#308080; ">,</span> activation <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"relu"</span><span style="color:#308080; ">)</span>
        grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dA"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> dA_prev_temp
        grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dW"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> dW_temp
        grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"db"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> db_temp

    <span style="color:#200080; font-weight:bold; ">return</span> grads

<span style="color:#200080; font-weight:bold; ">def</span> update_parameters<span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">,</span> grads<span style="color:#308080; ">,</span> learning_rate<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Update parameters using gradient descent</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- python dictionary containing your parameters </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;grads -- python dictionary containing your gradients, output of L_model_backward</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- python dictionary containing your updated parameters </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters["W" + str(l)] = ... </span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;parameters["b" + str(l)] = ...</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    L <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">2</span> <span style="color:#595979; "># number of layers in the neural network</span>

    <span style="color:#595979; "># Update rule for each parameter. Use a for loop.</span>
    <span style="color:#200080; font-weight:bold; ">for</span> l <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>L<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"W"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"W"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">-</span> learning_rate <span style="color:#44aadd; ">*</span> grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"dW"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span>
        parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"b"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> parameters<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"b"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span> <span style="color:#44aadd; ">-</span> learning_rate <span style="color:#44aadd; ">*</span> grads<span style="color:#308080; ">[</span><span style="color:#1060b6; ">"db"</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>l<span style="color:#44aadd; ">+</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span>
        
    <span style="color:#200080; font-weight:bold; ">return</span> parameters
</pre>


#### 4.3.4. Define the L-layer ANN model:

* Define the layers structure:
  * L = 4: Layers
  * Input size: 12288x1
    * Layer 1: 20 units
    * Layer 2: 7 units
    * Layer 3: 5 units
  * Output layer: 1 unit



<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"> <span style="color:#595979; ">#  4-layer model </span>
layers_dims <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">12288</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">20</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">7</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
</pre>


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Build the 4-layer model:</span>
<span style="color:#200080; font-weight:bold; ">def</span> L_layer_model<span style="color:#308080; ">(</span>X<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">,</span> layers_dims<span style="color:#308080; ">,</span> learning_rate <span style="color:#308080; ">=</span> <span style="color:#008000; ">0.0075</span><span style="color:#308080; ">,</span> num_iterations <span style="color:#308080; ">=</span> <span style="color:#008c00; ">3000</span><span style="color:#308080; ">,</span> print_cost<span style="color:#308080; ">=</span><span style="color:#074726; ">False</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Implements a L-layer neural network: [LINEAR-&gt;RELU]*(L-1)-&gt;LINEAR-&gt;SIGMOID.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;X -- data, numpy array of shape (num_px * num_px * 3, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;learning_rate -- learning rate of the gradient descent update rule</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;num_iterations -- number of iterations of the optimization loop</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;print_cost -- if True, it prints the cost every 100 steps</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- parameters learnt by the model. They can then be used to predict.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>

    np<span style="color:#308080; ">.</span>random<span style="color:#308080; ">.</span>seed<span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
    <span style="color:#595979; "># keep track of cost</span>
    costs <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#308080; ">]</span>                         
    
    <span style="color:#595979; "># Parameters initialization.</span>
    parameters <span style="color:#308080; ">=</span> initialize_parameters_deep<span style="color:#308080; ">(</span>layers_dims<span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Loop (gradient descent)</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> num_iterations<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>

        <span style="color:#595979; "># Forward propagation: [LINEAR -&gt; RELU]*(L-1) -&gt; LINEAR -&gt; SIGMOID.</span>
        AL<span style="color:#308080; ">,</span> caches <span style="color:#308080; ">=</span> L_model_forward<span style="color:#308080; ">(</span>X<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span>
        
        <span style="color:#595979; "># Compute cost.</span>
        cost <span style="color:#308080; ">=</span> compute_cost<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">)</span>
    
        <span style="color:#595979; "># Backward propagation.</span>
        grads <span style="color:#308080; ">=</span> L_model_backward<span style="color:#308080; ">(</span>AL<span style="color:#308080; ">,</span> Y<span style="color:#308080; ">,</span> caches<span style="color:#308080; ">)</span>
 
        <span style="color:#595979; "># Update parameters.</span>
        parameters <span style="color:#308080; ">=</span> update_parameters<span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">,</span> grads<span style="color:#308080; ">,</span> learning_rate<span style="color:#308080; ">)</span>
                
        <span style="color:#595979; "># Print the cost every 100 iterations</span>
        <span style="color:#200080; font-weight:bold; ">if</span> print_cost <span style="color:#200080; font-weight:bold; ">and</span> i <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">or</span> i <span style="color:#44aadd; ">==</span> num_iterations <span style="color:#44aadd; ">-</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">:</span>
            <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Cost after iteration {}: {}"</span><span style="color:#308080; ">.</span>format<span style="color:#308080; ">(</span>i<span style="color:#308080; ">,</span> np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>cost<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        <span style="color:#200080; font-weight:bold; ">if</span> i <span style="color:#44aadd; ">%</span> <span style="color:#008c00; ">100</span> <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">0</span> <span style="color:#200080; font-weight:bold; ">or</span> i <span style="color:#44aadd; ">==</span> num_iterations<span style="color:#308080; ">:</span>
            costs<span style="color:#308080; ">.</span>append<span style="color:#308080; ">(</span>cost<span style="color:#308080; ">)</span>
    
    <span style="color:#200080; font-weight:bold; ">return</span> parameters<span style="color:#308080; ">,</span> costs
</pre>

#### 4.3.5. Train the defined 4-layer ANN model:

##### 4.3.5.1. Call the functionality to train the model


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;">parameters<span style="color:#308080; ">,</span> costs <span style="color:#308080; ">=</span> L_layer_model<span style="color:#308080; ">(</span>train_x<span style="color:#308080; ">,</span> train_y<span style="color:#308080; ">,</span> layers_dims<span style="color:#308080; ">,</span> num_iterations <span style="color:#308080; ">=</span> <span style="color:#008c00; ">2500</span><span style="color:#308080; ">,</span> print_cost <span style="color:#308080; ">=</span> <span style="color:#074726; ">True</span><span style="color:#308080; ">)</span>


Cost after iteration <span style="color:#008c00; ">0</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.7717493284237686</span>
Cost after iteration <span style="color:#008c00; ">100</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.6720534400822914</span>
Cost after iteration <span style="color:#008c00; ">200</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.6482632048575212</span>
Cost after iteration <span style="color:#008c00; ">300</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.6115068816101354</span>
Cost after iteration <span style="color:#008c00; ">400</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.5670473268366111</span>
Cost after iteration <span style="color:#008c00; ">500</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.54013766345478</span>
Cost after iteration <span style="color:#008c00; ">600</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.5279299569455267</span>
Cost after iteration <span style="color:#008c00; ">700</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.4654773771766851</span>
Cost after iteration <span style="color:#008c00; ">800</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.3691258524959279</span>
Cost after iteration <span style="color:#008c00; ">900</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.39174697434805344</span>
Cost after iteration <span style="color:#008c00; ">1000</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.3151869888600617</span>
Cost after iteration <span style="color:#008c00; ">1100</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.2726998441789385</span>
Cost after iteration <span style="color:#008c00; ">1200</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.23741853400268137</span>
Cost after iteration <span style="color:#008c00; ">1300</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.19960120532208647</span>
Cost after iteration <span style="color:#008c00; ">1400</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.18926300388463305</span>
Cost after iteration <span style="color:#008c00; ">1500</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.1611885466582775</span>
Cost after iteration <span style="color:#008c00; ">1600</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.14821389662363316</span>
Cost after iteration <span style="color:#008c00; ">1700</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.13777487812972944</span>
Cost after iteration <span style="color:#008c00; ">1800</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.1297401754919012</span>
Cost after iteration <span style="color:#008c00; ">1900</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.12122535068005211</span>
Cost after iteration <span style="color:#008c00; ">2000</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.1138206066863371</span>
Cost after iteration <span style="color:#008c00; ">2100</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.10783928526254133</span>
Cost after iteration <span style="color:#008c00; ">2200</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.10285466069352679</span>
Cost after iteration <span style="color:#008c00; ">2300</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.10089745445261787</span>
Cost after iteration <span style="color:#008c00; ">2400</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.09287821526472397</span>
Cost after iteration <span style="color:#008c00; ">2499</span><span style="color:#308080; ">:</span> <span style="color:#008000; ">0.088439943441702</span>
</pre>

##### 4.3.5.2. Plot the Loss function:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># Plot cost the Cost function</span>
<span style="color:#595979; ">#------------------------------------------------</span>
<span style="color:#595979; "># set the figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize<span style="color:#308080; ">=</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">10</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">6</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># plot the cost</span>
plt<span style="color:#308080; ">.</span>plot<span style="color:#308080; ">(</span>costs<span style="color:#308080; ">,</span> <span style="color:#1060b6; ">'r-'</span><span style="color:#308080; ">,</span> linewidth <span style="color:#308080; ">=</span> <span style="color:#008000; ">2.0</span><span style="color:#308080; ">,</span> label<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'Loss'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># xlabel</span>
plt<span style="color:#308080; ">.</span>xlabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'# Epochs'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># ylabel</span>
plt<span style="color:#308080; ">.</span>ylabel<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Loss'</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># legend</span>
plt<span style="color:#308080; ">.</span>legend<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># set grid on</span>
plt<span style="color:#308080; ">.</span>grid<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># show the figure</span>
plt<span style="color:#308080; ">.</span>show<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
</pre>

<img src="results/Cost-function.png" width="1000" />


### 4.4. Step 4: Evaluate the trained model:

#### 4.4.1. Functions to predict labels and visualize mis-classifications:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#200080; font-weight:bold; ">def</span> predict<span style="color:#308080; ">(</span>X<span style="color:#308080; ">,</span> y<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;This function is used to predict the results of a  L-layer neural network.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Arguments:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;X -- data set of examples you would like to label</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;parameters -- parameters of the trained model</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Returns:</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;p -- predictions for the given dataset X</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    
    m <span style="color:#308080; ">=</span> X<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
    n <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>parameters<span style="color:#308080; ">)</span> <span style="color:#44aadd; ">//</span> <span style="color:#008c00; ">2</span> <span style="color:#595979; "># number of layers in the neural network</span>
    p <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>zeros<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span>m<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    
    <span style="color:#595979; "># Forward propagation</span>
    probas<span style="color:#308080; ">,</span> caches <span style="color:#308080; ">=</span> L_model_forward<span style="color:#308080; ">(</span>X<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span>

    
    <span style="color:#595979; "># convert probas to 0/1 predictions</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span> probas<span style="color:#308080; ">.</span>shape<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        <span style="color:#200080; font-weight:bold; ">if</span> probas<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>i<span style="color:#308080; ">]</span> <span style="color:#44aadd; ">&gt;</span> <span style="color:#008000; ">0.5</span><span style="color:#308080; ">:</span>
            p<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>i<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">1</span>
        <span style="color:#200080; font-weight:bold; ">else</span><span style="color:#308080; ">:</span>
            p<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>i<span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#008c00; ">0</span>
    
    <span style="color:#595979; ">#print results</span>
    <span style="color:#595979; ">#print ("predictions: " + str(p))</span>
    <span style="color:#595979; ">#print ("true labels: " + str(y))</span>
    <span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Accuracy: "</span>  <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span><span style="color:#400000; ">sum</span><span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>p <span style="color:#44aadd; ">==</span> y<span style="color:#308080; ">)</span><span style="color:#44aadd; ">/</span>m<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
        
    <span style="color:#200080; font-weight:bold; ">return</span> p

<span style="color:#200080; font-weight:bold; ">def</span> print_mislabeled_images<span style="color:#308080; ">(</span>classes<span style="color:#308080; ">,</span> X<span style="color:#308080; ">,</span> y<span style="color:#308080; ">,</span> p<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
    <span style="color:#595979; ">"""</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;Plots images where predictions and truth were different.</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;X -- dataset</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;y -- true labels</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;p -- predictions</span>
<span style="color:#595979; ">&nbsp;&nbsp;&nbsp;&nbsp;"""</span>
    a <span style="color:#308080; ">=</span> p <span style="color:#44aadd; ">+</span> y
    mislabeled_indices <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>asarray<span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>where<span style="color:#308080; ">(</span>a <span style="color:#44aadd; ">==</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
    plt<span style="color:#308080; ">.</span>rcParams<span style="color:#308080; ">[</span><span style="color:#1060b6; ">'figure.figsize'</span><span style="color:#308080; ">]</span> <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008000; ">15.0</span><span style="color:#308080; ">,</span> <span style="color:#008000; ">40.0</span><span style="color:#308080; ">)</span> <span style="color:#595979; "># set default size of plots</span>
    num_images <span style="color:#308080; ">=</span> <span style="color:#400000; ">len</span><span style="color:#308080; ">(</span>mislabeled_indices<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">]</span><span style="color:#308080; ">)</span>
    <span style="color:#200080; font-weight:bold; ">for</span> i <span style="color:#200080; font-weight:bold; ">in</span> <span style="color:#400000; ">range</span><span style="color:#308080; ">(</span>num_images<span style="color:#308080; ">)</span><span style="color:#308080; ">:</span>
        index <span style="color:#308080; ">=</span> mislabeled_indices<span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span><span style="color:#308080; ">[</span>i<span style="color:#308080; ">]</span>
        
        plt<span style="color:#308080; ">.</span>subplot<span style="color:#308080; ">(</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span> <span style="color:#008c00; ">2</span><span style="color:#308080; ">,</span> i <span style="color:#44aadd; ">+</span> <span style="color:#008c00; ">1</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>X<span style="color:#308080; ">[</span><span style="color:#308080; ">:</span><span style="color:#308080; ">,</span>index<span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">64</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span> interpolation<span style="color:#308080; ">=</span><span style="color:#1060b6; ">'nearest'</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
        plt<span style="color:#308080; ">.</span>title<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"Prediction: "</span> <span style="color:#44aadd; ">+</span> classes<span style="color:#308080; ">[</span><span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>p<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>index<span style="color:#308080; ">]</span><span style="color:#308080; ">)</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>decode<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"utf-8"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">" </span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; "> Class: "</span> <span style="color:#44aadd; ">+</span> classes<span style="color:#308080; ">[</span>y<span style="color:#308080; ">[</span><span style="color:#008c00; ">0</span><span style="color:#308080; ">,</span>index<span style="color:#308080; ">]</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>decode<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"utf-8"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
</pre>

#### 4.4.2. Compute the accuracy for the training data:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the predict function to compute the accuracy for the training data:</span>
pred_train <span style="color:#308080; ">=</span> predict<span style="color:#308080; ">(</span>train_x<span style="color:#308080; ">,</span> train_y<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span>

Accuracy<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.9856459330143539</span>
</pre>


#### 4.4.3. Compute the accuracy for the test data:


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># call the predict function to compute the accuracy for the test data:</span>
pred_test <span style="color:#308080; ">=</span> predict<span style="color:#308080; ">(</span>test_x<span style="color:#308080; ">,</span> test_y<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span>

Accuracy<span style="color:#308080; ">:</span> <span style="color:#008000; ">0.8</span>
</pre>


#### 4.4.4. Examine some of the mis-classified test images:

* Take a look at some images the L-layer model labeled incorrectly.
* This will show a few mislabeled images.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># Call the functionality to visualzie mis-classified test images</span>
print_mislabeled_images<span style="color:#308080; ">(</span>classes<span style="color:#308080; ">,</span> test_x<span style="color:#308080; ">,</span> test_y<span style="color:#308080; ">,</span> pred_test<span style="color:#308080; ">)</span>
</pre>


<table>
  <tr>
    <td> <img src="results/Miss-classified-01.PNG" width="1000" ></td>
  </tr>
  <tr>
    <td> <img src="results/Miss-classified-02.PNG" width="1000" ></td>
  </tr>
  <tr>
    <td> <img src="results/Miss-classified-03.PNG" width="1000" ></td>
  </tr>
</table>

#### 4.4.5. Observations:

* A few types of images the model tends to do poorly on include:

  * Cat body in an unusual position
  * Cat appears against a background of a similar color
  * Unusual cat color and species
  * Camera Angle
  * Brightness of the picture
  * Scale variation (cat is very large or small in image)


### 4.5. Test the model using new images

* We use 2 new test images to test the output of your model.


<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; ">#---------------------------------------------------------</span>
<span style="color:#595979; "># Test image # 1: Actual cat</span>
<span style="color:#595979; ">#---------------------------------------------------------</span>
<span style="color:#595979; "># change this to the name of your image file </span>
my_image <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"my_cat_1.jpg"</span> 
 <span style="color:#595979; "># the true class of your image (1 -&gt; cat, 0 -&gt; non-cat)</span>
my_label_y <span style="color:#308080; ">=</span> <span style="color:#308080; ">[</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">]</span>
fname <span style="color:#308080; ">=</span> <span style="color:#1060b6; ">"images/"</span> <span style="color:#44aadd; ">+</span> my_image
image <span style="color:#308080; ">=</span> np<span style="color:#308080; ">.</span>array<span style="color:#308080; ">(</span>Image<span style="color:#308080; ">.</span><span style="color:#400000; ">open</span><span style="color:#308080; ">(</span>fname<span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>resize<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span>num_px<span style="color:#308080; ">,</span> num_px<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>\
<span style="color:#595979; "># figure size</span>
plt<span style="color:#308080; ">.</span>figure<span style="color:#308080; ">(</span>figsize <span style="color:#308080; ">=</span> <span style="color:#308080; ">(</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">,</span><span style="color:#008c00; ">5</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>imshow<span style="color:#308080; ">(</span>image<span style="color:#308080; ">)</span>
plt<span style="color:#308080; ">.</span>axis<span style="color:#308080; ">(</span><span style="color:#1060b6; ">'off'</span><span style="color:#308080; ">)</span>
image <span style="color:#308080; ">=</span> image <span style="color:#44aadd; ">/</span> <span style="color:#008000; ">255.</span>
image <span style="color:#308080; ">=</span> image<span style="color:#308080; ">.</span>reshape<span style="color:#308080; ">(</span><span style="color:#308080; ">(</span><span style="color:#008c00; ">1</span><span style="color:#308080; ">,</span> num_px <span style="color:#44aadd; ">*</span> num_px <span style="color:#44aadd; ">*</span> <span style="color:#008c00; ">3</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">.</span>T
my_predicted_image <span style="color:#308080; ">=</span> predict<span style="color:#308080; ">(</span>image<span style="color:#308080; ">,</span> my_label_y<span style="color:#308080; ">,</span> parameters<span style="color:#308080; ">)</span>
<span style="color:#200080; font-weight:bold; ">print</span> <span style="color:#308080; ">(</span><span style="color:#1060b6; ">"y = "</span> <span style="color:#44aadd; ">+</span> <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>my_predicted_image<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">", your L-layer model predicts a </span><span style="color:#0f69ff; ">\"</span><span style="color:#1060b6; ">"</span> <span style="color:#44aadd; ">+</span> classes<span style="color:#308080; ">[</span><span style="color:#400000; ">int</span><span style="color:#308080; ">(</span>np<span style="color:#308080; ">.</span>squeeze<span style="color:#308080; ">(</span>my_predicted_image<span style="color:#308080; ">)</span><span style="color:#308080; ">)</span><span style="color:#308080; ">,</span><span style="color:#308080; ">]</span><span style="color:#308080; ">.</span>decode<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"utf-8"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span>  <span style="color:#1060b6; ">"</span><span style="color:#0f69ff; ">\"</span><span style="color:#1060b6; "> picture."</span><span style="color:#308080; ">)</span>

Accuracy<span style="color:#308080; ">:</span> <span style="color:#008000; ">1.0</span>
y <span style="color:#308080; ">=</span> <span style="color:#008000; ">1.0</span><span style="color:#308080; ">,</span> your L<span style="color:#44aadd; ">-</span>layer model predicts a <span style="color:#1060b6; ">"cat"</span> picture<span style="color:#308080; ">.</span>
</pre>

<img src="results/cat.png" width="1000" />


### 4.6. Part 6: Display a final message after successful execution completion:

<pre style="color:#000020;background:#e6ffff;font-size:10px;line-height:1.5;"><span style="color:#595979; "># display a final message</span>
<span style="color:#595979; "># current time</span>
now <span style="color:#308080; ">=</span> datetime<span style="color:#308080; ">.</span>datetime<span style="color:#308080; ">.</span>now<span style="color:#308080; ">(</span><span style="color:#308080; ">)</span>
<span style="color:#595979; "># display a message</span>
<span style="color:#200080; font-weight:bold; ">print</span><span style="color:#308080; ">(</span><span style="color:#1060b6; ">'Program executed successfully on: '</span><span style="color:#44aadd; ">+</span> 
      <span style="color:#400000; ">str</span><span style="color:#308080; ">(</span>now<span style="color:#308080; ">.</span>strftime<span style="color:#308080; ">(</span><span style="color:#1060b6; ">"%Y-%m-%d %H:%M:%S"</span><span style="color:#308080; ">)</span> <span style="color:#44aadd; ">+</span> <span style="color:#1060b6; ">"...Goodbye!</span><span style="color:#0f69ff; ">\n</span><span style="color:#1060b6; ">"</span><span style="color:#308080; ">)</span><span style="color:#308080; ">)</span>

Program executed successfully on<span style="color:#308080; ">:</span> <span style="color:#008c00; ">2021</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">05</span><span style="color:#44aadd; ">-</span><span style="color:#008c00; ">15</span> <span style="color:#008c00; ">14</span><span style="color:#308080; ">:</span><span style="color:#008c00; ">25</span><span style="color:#308080; ">:</span><span style="color:#008000; ">15.</span><span style="color:#308080; ">.</span><span style="color:#308080; ">.</span>Goodbye!
</pre>

## 5. Analysis

In this project, we demonstrated the step-by-step implementation of a Artificial Neural Network (CNN) from scratch to classify images of cats: 
We did not make use of any Deep Learning frameworks such as Tensorflow, Keras, etc. 
The classification accuracy achieved by the implemented ANN are comparable to those obtained using Deep learning frameworks, such as Tensorflow or Keras. 
It should be mentioned the implemented ANN is much slower, during training and inference, than using the Tensorflow or Keras, which are optimized. 
Implementing the ANN from scratch has helped gain valuable insights and understanding of convolutional networks. 

## 6. Future Work

* We plan to explore the following related issues:

  * To generalize the implementation of the ANN model to easily handle other ANN with different layers and structures. 

## 7. References

1. Coursera. Neural Networks and Deep Learning.  https://www.coursera.org/learn/neural-networks-deep-learning Coursera. 
Deep Learning Specialization. https://www.coursera.org/specializations/deep-learning 
2. James Loy. How to build your own Neural Network from scratch in Python. https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6 
3. Jason Brownlee. How to Code a Neural Network with Backpropagation In Python (from scratch). https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/ 
4. Casper Hansen. Neural networks from scratch. https://developer.ibm.com/technologies/artificial-intelligence/articles/neural-networks-from-scratch/ 
5. Usman Malik. Creating a Neural Network from Scratch in Python. https://stackabuse.com/creating-a-neural-network-from-scratch-in-python/ 
6. eBook. Neural Networks from Scratch. https://nnfs.io/

