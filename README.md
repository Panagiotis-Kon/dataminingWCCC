# <a name="team-members"></a>Team Members
* "Panagiotis Kontopoulos" <admin1@admin.com>
* "Alexadros-Panagiotis Tsakrilis" <admin2@admin.com>

# <a name="ubuntu"></a>Ubuntu Installation

* sudo apt-get update
* sudo apt-get -y install python-dev
* sudo apt-get -y install python-setuptools
* sudo apt-get -y install libjpeg8-dev
* sudo apt-get -y install libfreetype6-dev
* sudo apt-get -y install libffi-dev libssl-dev
* sudo apt-get -y install libatlas-base-dev gfortran
* sudo easy_install pip
* sudo pip install --upgrade pip
* sudo pip install requests[security]
* sudo pip install numpy
* sudo pip install scipy
* sudo pip install pandas
* sudo pip install -U scikit-learn
* sudo pip install wordcloud
* sudo pip install pillow
* sudo pip install nose
* sudo pip install mock
* sudo pip install matplotlib
* sudo pip install nltk
* wget https://github.com/amueller/word_cloud/archive/master.zip
* unzip master.zip
* rm master.zip
* cd ./word_cloud-master
* sudo pip install -r requirements.txt
* sudo python setup.py install
* sudo pip install --upgrade gensim

#Second way if pip install fails
* sudo apt-get -y install python-numpy
* sudo apt-get -y install python-scipy
* sudo apt-get -y install python-pandas
* sudo apt-get -y install python-scikit-learn
* sudo apt-get -y install python-wordcloud
* sudo apt-get -y install python-pillow
* sudo apt-get -y install python-nose
* sudo apt-get -y install python-mock
* sudo apt-get -y install python-matplotlib
* sudo apt-get -y install python-gensim

# <a name="windows"></a>Windows Installation
* Install python
* Install Microsoft Visual C++ Compiler for Python 2.7 (in windows)
* http://www.lfd.uci.edu/~gohlke/pythonlibs/#numpy
* version of whl depends on the installation
* python -m pip install ./numpy-1.11.0+mkl-cp27-cp27m-win_amd64.whl
* python -m pip install ./scipy-0.17.0-cp27-cp27-win_amd64.whl
* python -m pip install ./pandas-0.18.1-cp27-cp27m-win_amd64.whl
* python -m pip install -U scikit-learn
* python -m pip install wordcloud
* python -m pip install pillow
* python -m pip install nose
* python -m pip install mock
* python -m pip install matplotlib
* python -m pip install ./gensim-0.12.4-cp27-none-win_amd64.whl
