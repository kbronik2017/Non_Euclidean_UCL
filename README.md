[![GitHub issues](https://img.shields.io/github/issues/kbronik2017/Non_Euclidean_UCL)](https://github.com/kbronik2017/Non_Euclidean_UCL/issues)
[![GitHub forks](https://img.shields.io/github/forks/kbronik2017/Non_Euclidean_UCL)](https://github.com/kbronik2017/Non_Euclidean_UCL/network)
[![GitHub stars](https://img.shields.io/github/stars/kbronik2017/Non_Euclidean_UCL)](https://github.com/kbronik2017/Non_Euclidean_UCL/stargazers)
[![GitHub license](https://img.shields.io/github/license/kbronik2017/Non_Euclidean_UCL)](https://github.com/kbronik2017/Non_Euclidean_UCL/blob/master/LICENSE)


# Variational Deep Learning models for non Euclidean structure
<br>
 <img height="310" src="images/graph.gif"/>
</br>
<br>
 <img height="510" src="images/vae1.jpg"/>
</br>

<br>
 <img height="510" src="images/vae2.jpg"/>
</br>
Click on the following link to access further information on mathematical presentation of variational autoencoder
https://github.com/kbronik2017/Non_Euclidean_UCL/blob/main/references/Auto-Encoding.pdf

<br>
 <img height="510" src="images/math1.jpg"/>
</br>
<br>
 <img height="510" src="images/math2x.jpg"/>
</br>
Click on the following link to access further information on mathematics of analytical methods
https://github.com/kbronik2017/Non_Euclidean_UCL/blob/main/references/Squartini_2011_New_J._Phys._13_083001.pdf

# Undirected Configuration Model (UCM)


<br>
 <img height="510" src="images/vae3.jpg"/>
</br>


<br>
 <img height="510" src="images/vae33.jpg"/>
</br>

# Weighted Configuration Model (WCM)

<br>
 <img height="510" src="images/vae4.jpg"/>
</br>

<br>
 <img height="510" src="images/vae44.jpg"/>
</br>


# Reciprocal Configuration Model (RCM)


<br>
 <img height="510" src="images/vae5.jpg"/>
</br>

<br>
 <img height="510" src="images/vae55.jpg"/>
</br>

# Reonstruction Network (RCON)

<br>
 <img height="510" src="images/vaerx.jpg"/>
</br>

# Parallel running


<br>
 <img height="510" src="images/vae6.jpg"/>
</br>


# Running the GUI Program! 

First, user needs to install Anaconda https://www.anaconda.com/

Then


```sh
  - conda env create -f train_test_environment.yml
  or
  - conda create --name idp --file clone-file.txt
``` 
and 

```sh
  - conda activate idp
``` 
finally

```sh
  - python  VAE_GUI.py
``` 

After lunching the graphical user interface, user will need to provide necessary information to start training/testing as follows:  

<br>
 <img height="510" src="images/cover.jpg" />
</br>


# Testing the Program (User Quick Start Guide) 
Examples of Training, Cross-validation and Testing subjects can be found in:
https://github.com/kbronik2017/Non_Euclidean_UCL/tree/main/training_testing_examples 
(which will allow users to quickly and easily train and test the program).
The results of testing(inference) can be found in the folders:
```sh
  - prediction_image_outputs
  and
  - matrix_output
``` 


