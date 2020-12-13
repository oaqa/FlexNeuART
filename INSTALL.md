# Pre-requisits
FlexNeuART should work on Linux or Mac.

It requires:
1. Github
2. Java
3. Python with virtual environment support (e.g., conda)
4. Maven
5. A C and C++ compiler.


# Installation walk through

Create a virtual environment:
```
conda create -n msmarco python=3.6
conda activate msmarco
```
Optionally install the Jupyter:
```
pip install jupyter
```

Clone the repository:
```
git clone https://github.com/oaqa/FlexNeuART.git
cd FlexNeuART/
```

Optionally check out a specific branch or tag.
For example:
```
git checkout tags/repr2020-12-06
```

Install packages and build the code:
```
./install_packages.sh 
./build.sh 
```
