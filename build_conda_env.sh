

conda create -n speml-ex3 python=3.6
conda activate speml-ex3

pip install --upgrade https://github.com/Theano/Theano/archive/master.zip
pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip

conda install scikit-learn pillow

#conda install mkl-service

wget http://vis-www.cs.umass.edu/lfw/lfw.tgz
