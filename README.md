Efficient Multi-dimensional Diracs Estimation with Linear Sample Complexity
===========================================================================================

Authors
-------

Hanjie Pan<sup>1</sup>, Thierry Blu<sup>2</sup> and Martin Vetterli<sup>1</sup><br>
<sup>1</sup>Audiovisual Communications Laboratory ([LCAV](http://lcav.epfl.ch)) at [EPFL](http://www.epfl.ch).<br>
<sup>2</sup>Image and Video Processing Laboratory ([IVP](http://www.ee.cuhk.edu.hk/~tblu)) at [CUHK](http://www.cuhk.edu.hk).

<img src="./html/LCAV_anim_200.gif">

#### Contact

[Hanjie Pan](mailto:hanjie[dot]pan[at]epfl[dot]ch)<br>
EPFL-IC-LCAV<br>
BC 322 (Bâtiment BC)<br>
Station 14<br>
1015 Lausanne

Abstract
--------

Estimating Diracs in continuous two or higher dimensions is a fundamental problem in imaging. Previous approaches extended one dimensional methods, like the ones based on finite rate of innovation (FRI) sampling, in a separable manner, e.g., along the horizontal and vertical dimensions separately in $2$D. The separate estimation leads to a sample complexity of $\mathcal{O}\left(K^D\right)$ for $K$ Diracs in $D$ dimensions, despite that the total degrees of freedom only increase linearly with respect to $D$.

We propose a new method that enforces the continuous-domain sparsity constraints simultaneously along all dimensions, leading to a reconstruction algorithm with **linear** sample complexity $\mathcal{O}(K)$, or a gain of $\mathcal{O}\left(K^{D-1}\right)$ over previous FRI-based methods. The multi-dimensional Dirac locations are subsequently determined by the intersections of hypersurfaces (e.g., curves in $2$D), which can be computed algebraically from the common roots of polynomials.

We first demonstrate the performance of the new multi-dimensional algorithm on simulated data: multi-dimensional Dirac location retrieval under noisy measurements.

Then we show results on real data: radio astronomy point source reconstruction (from LOFAR telescope measurements) and the direction of arrival estimation of acoustic signals (using Pyramic microphone arrays).


Package Description
-------------------

This repository contains all the code to reproduce the results of the paper [*Efficient Multi-dimensional Diracs Estimation with Linear Sample Complexity*](https://infoscience.epfl.ch/record/255990). It contains a Python implementation of the proposed algorithm.

We are available for questions or requests relating to either the code or the theory behind it. 

Recreate the figures
--------------------------------------
    # Efficient reconstruction of 7 Diracs in 2D from 5x5 NOISELESS ideal 
    # low pass samples (Fig. 5)
    python example_few_data_2d.py
    
    # Average reconstruction performance under different noise levels (Fig. 6)
    python avg_perf_2d_dirac_vs_noiselevel.py
    
    # 2D Diracs that have shared x or y locations (Fig. 7)
    # noiseless case (Fig. 7a)
    python example_shared_xy_locs.py
    # noisy case (Fig. 7b)
    python example_shared_xy_locs_noisy.py
    
    # 3D Dirac reconstruction example (Fig. 8)
    # noiseless case (Fig. 8a)
    python example_dirac_3d_noiseless.py
    # noisy case (Fig. 8b)
    python example_dirac_3d_noisy.py

Data used in the paper
----------------------
The randomly generated signal parameters are saved under the folder `data/`.
    
    # Dirac parameters used in measuring the average reconstruction performance
    # under different noise levels
    data/signal_diff_noise/dirac_param.npz 
    
    # Dirac parameters with shared x or y locations
    data/dirac_param_shared_xy.npz
    
    # Dirac parameters used in the 3D reconstruction example
    data/example_3d.npz


Dependencies
------------
* A working distribution of Python 3 e.g., [here](https://www.anaconda.com/distribution/)

List of standard packages needed

    numpy, scipy, matplotlib, mkl, sympy, plotly, scikit-image, numexpr

System Tested
-------------

### macOS

| Machine | MacBook Pro Retina 15-inch, Mid 2014   |
|---------|----------------------------------------|
| System  | macOS Sierra 10.12                     |
| CPU     | Intel Core i7                          |
| RAM     | 16 GB                                  |

    System Info:
    ------------
    Darwin Kernel Version 16.7.0: Thu Jun 15 17:36:27 PDT 2017; root:xnu-3789.70.16~2/RELEASE_X86_64 x86_64

    Python Packages Info (conda)
    ----------------------------
    # packages in environment at /Users/pan/anaconda:
    
### Ubuntu Linux
| Machine | lcavsrv1                               |
|---------|----------------------------------------|
| System  | Ubuntu 16.04.2 LTS                     |
| CPU     | Intel Xeon                             |
| RAM     | 62GiB                                  |

    System Info:
    ------------
    Linux 4.4.0-62-generic #83-Ubuntu SMP Wed Jan 18 14:10:15 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux
    
    Python Packages Info (conda)
    ----------------------------
    # packages in environment at /home/pan/anaconda3:
   
License
-------

Copyright (c) 2018, Hanjie Pan<br>
The source code is released under the [MIT](LICENSE.txt) license.
