This repository provides data and supplementary material to the paper: Deep learning of quantum entanglement from incomplete measurements, preprint:  <a href="https://arxiv.org/abs/2205.01462">arxiv.org/abs/2205.01462</a>, by Dominik Koutný, Laia Ginés, Magdalena Moczała-Dusanowska, Sven Höfling, Christian Schneider, Ana Predojević, and Miroslav Ježek.

<h3>
modelConcurrence.h5
</h3>
The convolutional deep neural network (DNN) trained to predict the concurrence independently of the quantum state and measurement.

<h3>
modelMI.h5
</h3>
The convolutional DNN trained to predict the mutual information independently of the quantum state and measurement.

<h3>
spec_net_conc_R0-R4.zip
</h3>
These files contain the measurement-specific DNNs. To predict the concurrence and mutual information with either measurement-specific or measurement-independent DNNs, go through the notebook: predict_concurrence_or_MI.ipynb, which serves as a tutorial.

<h3>
plots.ipynb
</h3>
The Jupyter notebook plots.ipynb contains code to generate all the figures present in our manuscript. 

<h3>
predict_concurrence_or_MI.ipynb
</h3>
The Jupyter notebook predict_concurrence_or_MI.ipynb contains a code for the concurrence and mutual information estimation
in two-qubit systems using the trained neural networks. The code contains also the maximum likelihood algorithm for the comparison.

<h3>
netTraining_measurement_specific_DNNs.py and netTraining_meaurement_independent_DNN.py
</h3>

These Python codes implement the training of the neural networks.

All the codes were prepared and tested with Python 3.9.6, Numpy 1.19.4, Matplotlib 3.5.0, Keras 2.4.3, and Tensorflow 2.5.0.
