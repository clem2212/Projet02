## Projet02
Devianne Paul (CSE), Dormoy Camille (NX), Jegard Clemence (SV)

The aim of this project is to predict the collision event of an electron in water given an initial position, direction, and speed. During its journey in the water, the electron undergoes many encounters before losing all its energy and dying. At each encounter, he may or may not create a new particle with its own new properties. Using Generative Adversarial Networks (GANs), we will model these different types of interactions in order to best predict the behavior of electrons over time and space.

### Install

This project requires *Python* with the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](https://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org)
- [pytorch](https://pytorch.org)

You will also need to have software installed to run and execute a [Jupyter Notebook](http://jupyter.org/install.html).


### Code

Our code is composed of different jupyter Notebook : 
The first part of our project is composed of cleaning the data, analysis of the data and creation of a probability table for the classification. For these tasks, all the functions that will be used are together in the Python file `implementations.py`. Then the data analysis is done in the notebook file `data_analysis.ipynb` and the classification in `classification.ipynb`.

The second part of our projet is the generation of a model with GANs, for the implementation of this part, we first have all the neccessary function in the Python file `GAN_functions.py`. Then, the study done to define the parameters of our GANs is in the notebook `Gan_study.ipynb` and the creation of the GANs model is in `Gan_models.ipynb`. 
Our 6 GAN models created are stocked in a file named `saved_model` so we don't have to regenerate them each time we want to use them.

Finally the running of our model and the visualisation of the results are in the notebook `final_test.ipynb` and the referenced used are in the text file `references.txt`.


### Report
A 4-page report explaining how the data was analyzed and processed. 
