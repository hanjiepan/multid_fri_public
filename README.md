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

A number of scripts were written to apply the proposed *finite rate of innovation (FRI)*-based sparse reconstruction to point source estimation in radio astronomy, including

* Realistic simulation with LOFAR antenna layout
    - Resolving two point sources that are separated by various distances at different relative orientations (`experiment_src_separation.py`)
    - Model order selection to avoid false detection (`model_order_sel.py`)
    - Refine linear mapping that connects the uniform Fourier data of the sky image to the visibility measurements (`figure_update_G.py`)
* Point source reconstruction from actual LOFAR observation
    - Böotes field, which primarily consists of point sources (`bootes_field_narrow_fov_experiment.py`)
    - "Toothbrush" cluster (RX J0603.3+4214), which contains extended structure at the center of the field-of-view (`toothbrush_cluster_narrow_fov_experiment.py`)

We are available for questions or requests relating to either the code or the theory behind it. 

Recreate the figures
--------------------------------------
NOTE: the algorithm implementation assumes `python 3` by default. However, for the data pre-processing we have to use `casacore`, which is `python 2`-only. That is why some scripts are called with `python2`.

Fig. 4: The iterative strategy to refine linear mapping:

    python2 figure_update_G.py

Fig. 5 - 7: The average source resolution performance for LEAP and CLEAN: 

    # average performance with LEAP
    python experiment_src_separation.py
    
    # average performance with CLEAN
    python2 experiment_src_separation_clean.py
    
    # one example for visiual comparision (small separation, high SNR)
    python2 visual_example_fri_vs_clean.py
     
    # another example for source resolution (relative large separation, low SNR)
    python2 low_snr_fri_vs_clean.py

Fig. 8: Model order selection base on the fitting error to avoid false detection:

    python model_order_sel.py
    
Fig. 9: Point source reconstruction with LOFAR measurements from the Böotes field:

    # 1) Single-band case (i.e., Dataset II)
    python bootes_field_narrow_fov_experiment.py
    
    # 2) Multi-band case (i.e., Dataset III)
    python bootes_field_narrow_fov_experiment.py -m
    
Fig. 10: Point source reconstruction in the presence of extended sources with LOFAR measurement from the Toothbrush cluster:

    python toothbrush_cluster_narrow_fov_experiment.py

Data used in the paper
----------------------

* The randomly generated point source parameters are saved under the folder `data/ast_src_resolve/`:

        # Simulated point source use in Fig. 4
        src_param_20170503-150943.npz

        # Simulation point source use in Fig. 6
        src_param_20170626-231925.npz
    
        # Simulation point source use in Fig. 7
        src_param_20170331-155144.npz

        # Simulation point source use in Fig. 8
        src_param_20170701-163710.npz

* The extracted visibility measurements from the MS file of LOFAR observation are svaed under the folder `data/`:
    
        # Bootes field (single-band)
        BOOTES24_SB180-189.2ch8s_SIM_72STI_146MHz_28Station_1Subband.npz
    
        # Bootes field (multi-band)
        BOOTES24_SB180-189.2ch8s_SIM_9STI_146MHz_28Station_8Subband.npz
    
        # Toothbrush cluster
        RX42_SB100-109.2ch10s_63STI_132MHz_36Station_1Subband_FoV5.npz
    
* The raw measurement set (MS) from LOFAR was used in experiments with real data as well as in simulations, where we fill in the MS file with simulated visibilities. The dataset can be downloaded from [Zenodo](https://doi.org/10.5281/zenodo.1044019) and decompressed afterwards:

    [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1044019.svg)](https://doi.org/10.5281/zenodo.1044019)

        wget https://zenodo.org/record/1044019/files/BOOTES24_SB180-189.2ch8s_SIM.ms.tar.gz
        wget https://zenodo.org/record/1044019/files/RX42_SB100-109.2ch10s.ms.tar.gz
        tar -xzvf BOOTES24_SB180-189.2ch8s_SIM.ms.tar.gz
        tar -xzvf RX42_SB100-109.2ch10s.ms.tar.gz
        
    After decompression, update the evironment variable in `setup.py`, e.g.,:
    
        os.environ['DATA_ROOT_PATH'] = 'path_to_the_ms_files'
        os.environ['PROCESSED_DATA_ROOT_PATH'] = 'path_to_the_ms_files'
        
* We have used three catalogs in the experiments, namely the [Boötes field catalog](https://academic.oup.com/mnras/article-lookup/doi/10.1093/mnras/stw1056), [TGSS ADR1 catalog](http://tgssadr.strw.leidenuniv.nl/catalogs/TGSSADR1_7sigma_catalog.fits), and the [NVSS catalog](http://www.cv.nrao.edu/nvss/NVSSlist.shtml). We have converted the original FITS tables of the catalogs to Numpy arrays. They can be downloaded from the same [Zenodo](https://doi.org/10.5281/zenodo.1044019) webpage:

        wget https://zenodo.org/record/1044019/files/skycatalog.npz
        wget https://zenodo.org/record/1044019/files/TGSSADR1_7sigma_catalog.npz
        wget https://zenodo.org/record/1044019/files/NVSS_CATALOG.npz

Dependencies
------------

* A working distribution of [Python 3.5](https://www.python.org/downloads/) and [Python 2.7](https://www.python.org/downloads/) (A few scripts run CLEAN for comparision. [casacore](https://github.com/casacore/casacore), which only have official support for Python 2, is used to exchange data between a measurement set (MS) and numpy arrays).
* [Numpy](http://www.numpy.org/), [Scipy](http://www.scipy.org/).
* We use the distribution [Anaconda](https://www.anaconda.com/download/) to simplify the setup of the environment.
* We use the [MKL](https://store.continuum.io/cshop/mkl-optimizations/) extension of Anaconda to speed things up.
* We use joblib for parallel computations.
* [matplotlib](http://matplotlib.org) for plotting the results.
* [theano](http://deeplearning.net/software/theano/) for the parallel computation with GPU.
* [casacore](https://github.com/casacore/casacore) for the data exhcange to / from MS file and its Python wrapper [python-casacore](https://github.com/casacore/python-casacore).

List of standard packages needed

    numpy, scipy, matplotlib, mkl, joblib, theano
    
The easiest way to setup proper computing environment is first to download and install Python distributions from [Anaconda](https://www.anaconda.com/download/). Then choose a correct version with

    conda install python=3.5 mkl=11.3.3 theano joblib

The CLEAN algorithm [wsclean](https://sourceforge.net/projects/wsclean/) is used for comparisons.

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
    joblib                    0.11                     py35_0  
    matplotlib                2.0.2            py35ha43f773_1 
    mkl                       11.3.3                        0  
    mkl-service               1.1.2                    py35_2  
    numpy                     1.11.2                   py35_0
    python                    3.5.4               hf91e954_15
    scipy                     0.18.1              np111py35_0  
    theano                    0.9.0                    py35_0
    
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
    joblib                    0.11                     py35_0 
    matplotlib                2.0.2               np111py35_0  
    mkl                       11.3.3                        0  
    mkl-service               1.1.2                    py35_2  
    numpy                     1.11.2                   py35_0
    python                    3.5.3                         1
    scipy                     0.18.1              np111py35_0  
    theano                    0.9.0                    py35_0
    
License
-------

Copyright (c) 2018, Hanjie Pan<br>
The source code is released under the [MIT](LICENSE.txt) license.
