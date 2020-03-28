<p align="center">
  <a href="https://example.com/">
    <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcRa3lRWWMvH7mT5BPe6ebOVHFi8tSuz93UKFSGSOYAtCprYhDe7" alt="Logo" width=80 height=80>
  </a>

  <h3 align="center">Iniciação científica PIBIC - IAE</h3>
</p>


## Table of contents

- [Introdução](#quick-start)
- [Status](#status)
- [Bug e novas features](#bugs-and-feature-requests)
- [Agradecimentos](#thanks)
- [Licença](#copyright-and-license)


## Introdução

The direct simulation Monte Carlo (DSMC) method is a computational tool for simulating flows in which effects at the molecular scale become significant. The Boltzmann equation [2], which is appropriate for modeling these rarefied flows, is extremely difficult to solve numerically due to its high dimensionality and the complexity of the collision term. DSMC provides a particle based alternative for obtaining realistic numerical solutions. In DSMC, the movement and collision behavior of a large number of representative ?simulated particles? within the flow field are decoupled over a time step which is a small fraction of the local mean collision time. In this way, to compute all the collision between the molecules, it is necessary the use of High Performance Computing (HPC) which a few researchers have access in Brazil. One way to overcome this issue is the use of graphics processing units (GPUs) to perform these simulations. Graphics Processing Units (GPUs) are co-processors originally designed to assist in the computations required for displaying graphics on monitor. In recent years, the GPU has been demonstrated as an effective tool in the computation of scientific and engineering problems. Nowadays, GPU can be used to accelerate the computations an it has been developed to work in many popular computing languages such as C/C++, FORTRAN and even Java. In this way, the main objective of this Scientific Initiation (IC) project is the understanding of the dsmcFoam code structure and its migration from C++ to OpenCL programming language aiming the used of the dsmcFoam code with GPU.

## Status

Iniciação científica finalizada.

## Bugs and feature requests

Para correção de bugs, novas features e sugestões, enviar email para: joaocrm@id.uff.br

## Thanks

Coordinator: Rodrigo Cassineli Palharini

## Copyright and license

Code and documentation copyright 2019 the authors. Code released under the [MIT License](https://reponame/blob/master/LICENSE).

Enjoy :metal:
