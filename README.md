# photon

## Ray tracer to render volumetric data

### Installation

#### Install dependencies


##### CUDA
```
$ sudo apt update
$ sudo apt install nvidia-cuda-toolkit
```

##### OpenGl
```
$ sudo apt-get update
$ sudo apt-get install libglu1-mesa-dev freeglut3-dev  
```

#### Install Python packages

#### PyOpenGL
```
pip install PyOpenGL
```

#### PyCUDA
```
wget https://files.pythonhosted.org/packages/5a/56/4682a5118a234d15aa1c8768a528aac4858c7b04d2674e18d586d3dfda04/pycuda-2021.1.tar.gz
tar xzvf pycuda-2021.1.tar.gz
cd pycuda-2021.1/
./configure.py
cat siteconf.py | sed -e "s/CUDA_ENABLE_GL = False/CUDA_ENABLE_GL = True/" > siteconf_temp.py
mv siteconf_temp.py siteconf.py
make
python setup.py install
pip install .
```