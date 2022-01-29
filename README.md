# twolayerpe

 python version of two-layer baroclinic primitive equation model from
 Zou., X. A., A. Barcilon, I. M. Navon, J. S. Whitaker, and D. G. Cacuci,
 1993: An adjoint sensitivity study of blocking in a two-layer isentropic
 model. Mon. Wea. Rev., 121, 2834-2857.
 doi: http://dx.doi.org/10.1175/1520-0493(1993)121<2833:AASSOB>2.0.CO;2

 see also https://journals.ametsoc.org/view/journals/mwre/133/11/mwr3020.1.xml

 spherical and f-plane versions (spherical version depends on 
 [shtns](https://anaconda.org/conda-forge/shtns) and f-plane version 
 depends on [pyfftw](https://anaconda.org/conda-forge/pyfftw)).

 very similar to the dry version of this
 [model](https://journals.ametsoc.org/view/journals/atsc/69/4/jas-d-11-0205.1.xml#bib32)
 (only difference is discretization of vertical momentum transport term).
 See https://aip.scitation.org/doi/10.1063/1.3582356 for the original derivation
 (pdf [here](https://www.lmd.ens.fr/glapeyre/papers/jlglvzfb2011.pdf)).

 Three-layer version derived in https://doi.org/10.1002/qj.405.
