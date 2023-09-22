# Orientation dependent solar cell simulation

A novel simulation approach for quasi-one-dimensional and two dimensional materials. The device simulation is divided into multiple individual simulations for each layer or ribbon, by assuming only transport along the ribbon or layer, due to their considerably higher electron mobilities.

## Table of Contents

  - [Introduction](#introduction)
  - [Features](#features)
  - [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)

## Introduction

Unlike 3D materials, for 2D and quasi-one-dimensional materials, their highly anisotropic charge transport allows the solar cell device simulation to be reduced to a 1D problem, which can be solved analytically. However, this is only valid by assuming that charge transport is predominantly along the ribbon or layer direction, and inter-ribbon and inter-layer hopping being negligible. By solving a series of 1D simulations this simulation approach allows for more complex microstructures to be modelled.  

## Features

- Orientantion dependent simulation. The orientation of the ribbons or layers can be varied between 0 and 89 degrees with respect to the layer thickness direction.
- Grain boundary passivation. Two recombination velocities (10^3 and 10^7 cm/s) are calculated to quantify the effect of grain boundary passivation on device efficiency. However, the recombination velocities can be changed to simulate other conditions.

## Getting Started

Download all files in this repository. The current simulation parameters are for a Glass/FTO/TiO2/Sb2Se3 solar cell device. 
All the parameters can be varied according to your device architecture and settings. The complex refractive index files must be .xlsx. The existing complex refractive index files can be used as a guide for the format required.
The ribbon/layer orientation and recombination velocities can be modified in lines 124 and 133 respectively.

### Prerequisites

The simulation was done using Python 3.8. All the required libraries and modules are stated at the beginning of the code.
