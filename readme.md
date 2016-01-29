# Imicu

Imicu is a real-time hair simulation that makes use of a [mass-spring](https://en.wikipedia.org/wiki/Effective_mass_%28spring%E2%80%93mass_system%29) based physics model to represent a single strand of hair. Each strand is composed of a number of segments (mass) joined together by an array of springs. The model is largely based on the techniques described in this [paper](http://physbam.stanford.edu/~fedkiw/papers/stanford2008-02.pdf).

### Highlights

* C/C++
* OpenGL
* [Conjugate Gradient Method](https://en.wikipedia.org/wiki/Conjugate_gradient_method) used for fast Velocity Integration
* Strand Collision Detection using [KDOP](https://en.wikipedia.org/wiki/Bounding_volume#Common_types_of_bounding_volume) Bounding Volume Hierarchy
* Object Collision Detection using a Distance Field
* GPU version written in [CUDA](https://en.wikipedia.org/wiki/CUDA)

![screencast](https://github.com/swyngaard/imicu/raw/master/demo.gif)

### Prerequisites for Building

CPU version:
* GNU Make
* GCC

GPU version:
* CUDA v6+

### Building and Executing

Execute `make` in either of the `cpu` or `gpu` directories. To run execute the `gl` binary.

### Background

Imicu is the [Zulu](https://en.wikipedia.org/wiki/Zulu_language) word for strands or threads.

