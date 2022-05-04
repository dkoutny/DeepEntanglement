This repository provides data and supplementary material to the paper Deep learning of polarization 
transfer in liquid crystals with application to quantum state preparation, preprint:  <a href="https://arxiv.org/abs/2205.01462">arxiv.org/abs/2205.01462</a>, 
by Dominik Koutný, Laia Ginés, Magdalena Moczała-Dusanowska, Sven Höfling, Christian Schneider, Ana Predojević, and Miroslav Ježek.

<h3>
modelConcurrence.h5
</h3>
The convolutional DNN trained to predict the concurrence independently of the quantum state and measurement.

<h3>
modelMI.h5
</h3>
The convolutional DNN trained to predict the mutual information independently of the quantum state and measurement.

<h1>
spec_net_conc_R0-R4.zip
</h1>
These files contain the measurement-specific NNs. If you wish to predict the concurrence or the mutual information with either measurement-specific or measurement-independent NNs, please go through the predict_concurrence_or_MI.ipynb notebook which serves as a tutorial how to do so.

<h1>
plots.ipynb
</h1>
The jupyter notebook file plots.ipynb contain code to generate all the figures present in our manuscript. 


<h1>
predict_concurrence_or_MI.ipynb
</h1>
The jupyter notebook file predict_concurrence_or_MI.ipynb contain a code where you can estimate the value of concurrence or mutual information
in the system of two qubits with our trained neural networks. The code contain also the maximum likelihood algorithm for the comparion.


<h1>
netTraining_measurement_specific_DNNs.py /
netTraining_meaurement_independent_DNN.py
</h1>
These python codes serve as a tool for reproducing the results for both types of neural networks.

