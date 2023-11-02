# General
OT exploration @ OIST October-December 2023
TWD, inverseOT, representation learning

# Tree Wasserstein distance with weight training
This is the demo code for the paper entitled [Approximating 1-Wasserstein Distance with Trees](https://openreview.net/forum?id=Ig82l87ZVU).

Note that we used the QuadTree and clustertree implementations of [Fixed Support Tree-Sliced Wasserstein Barycenter](https://github.com/yukiTakezawa/FS_TSWB).

## Requirements
Install requirements.
```
sudo pip install -r requirement.txt
```

## Run
Run example.py
```
python example.py
```

## Citation
```
@article{
yamada2022approximating,
title={Approximating 1-Wasserstein Distance with Trees},
author={Makoto Yamada and Yuki Takezawa and Ryoma Sato and Han Bao and Zornitsa Kozareva and Sujith Ravi},
journal={Transactions on Machine Learning Research},
year={2022},
url={https://openreview.net/forum?id=Ig82l87ZVU},
note={}
}
```

## Related Github projects
- [Supervised Tree-Wasserstein Distances (ICML 2021)](https://github.com/yukiTakezawa/STW)
- [Fixed Support Tree-Sliced Wasserstein Barycenter (AISTATS 2022)](https://github.com/yukiTakezawa/FS_TSWB)

## Contributors
Name : [Makoto Yamada](https://riken-yamada.github.io/profile.html) (Okinawa Institute of Science and Technology / Kyoto University) and [Yuki Takezawa](https://yukitakezawa.github.io/) (Kyoto University)

E-mail : makoto (dot) yamada (at) oist.jp

## optimal-transport -- wasserstein.py

References:
* Alex Williams' blog post on OT, <http://alexhwilliams.info/itsneuronalblog/2020/10/09/optimal-transport/>
* Peyré and Cuturi's _Computational Optimal Transport_, <https://arxiv.org/pdf/1803.00567.pdf>
* Tutorials from NeuroHackAcademy, <https://github.com/alecgt/otml-neurohackademy-2019/blob/master/lab/notebooks/lab_main.ipynb>
* InverseOT project, <https://github.com/El-Zag/Inverse-Optimal-Transport>
* Ma et al., 2020, _Learning Cost Function for Optimal Transport_, <https://arxiv.org/pdf/2002.09650.pdf>