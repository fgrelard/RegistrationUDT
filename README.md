# RegistrationUDT

Description
===========

This code runs the updated distance transformation (UDT) similarity metric for variational registration. Our similarity metric is particular for multimodal workflows. In particular, it solves the problem of images with discrepant distributions. It yields precisely registered images, while still limiting spurious deformations in the object. 

The similarity metric is implemented thanks to the framework [*Fair: Flexible Algorithms for Image Registration*](https://dl.acm.org/doi/10.5555/1816330) [1].

The algorithm is implemented thanks to the [ITK Library](https://itk.org/)



Quick Build Instructions
========================
The main instructions on linux/unix-based systems are the following:

```shell
git clone https://github.com/fgrelard/RegistrationUDT.git
cd RegistrationUDT ; mkdir build ; cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

Minimum system requirements: C++11 enabled compiler, [cmake](http://cmake.org), [ITK](http://itk.org/) (>= 5.0.1), with the following compilation options enabled:
* BUILD_STATIC_LIBS
* ITK_BUILD_DEFAULT_MODULES
* Module_VariationalRegistration
* Optional: ITK_USE_FFTWD for elastic regularization

Note: Our code has been compiled and tested on Unix distributions 18.04 and 20.04.


Usage
========================
The executable is located in the `build` directory. 
It provides a self-contained description on how to use them, available with the option -h.

Examples:
```shell
./VariationalRegistrationDT2D -h
./VariationalRegistrationDT2D -F [fixed_image] -M [moving_image] -W [warped_image] -O [transformation] -t [time_step] -r [regularization] -l [levels]
```
Details:
* To use our metric (UDT), supply the option `-f 3`. 
* To use SSD with DT values, supply the option `-f 1`.
* To use SSD with regular intensity values, supply the options `-f 1 -c`.


Data
========================
Any ITK image format is accepted as input. See [here](https://itk.org/Wiki/ITK/File_Formats) for more information.

Questions
========================
Contact at florent [dot] grelard [at] gmail [dot] com

[1] Modersitzki, J.: Fair: Flexible Algorithms for Image Registration. Society for Industrial and Applied Mathematics, Philadelphia, PA, USA (2009)
