PyTorch implementation of HFR-TINR based on  SpectFormer: Frequency and Attention is what you need in a Vision (https://badripatro.github.io/SpectFormers/).

HFR-TINR, a high-fidelity unsteady flow field reconstruction method based on transformer and INR to concurrently improve numerical-precision and grid-resolution. HFR-TINR takes spatial coordinates and temporal information as inputs, which are obtained by entropy sampling on our public 3.9 TB datasets to concentrate on the regions-of-interest areas, and integrates a multi-scale hash-encoding block to learn long-term spatiotemporal flow fields at high accuracy and memory efficiency. In our designed transformer, a temporal-aware encoder based on the cross-attention mech- anism aims to fuse both spatial and temporal information, and capture long-range temporal dependencies. The spectral block can further enhance the reconstruction quality by considering the different frequency flow details. The qualitative and quantitative experiments on both 2D and 3D datasets indicate that HFR-TINR achieves better reconstruction quality than the state-of-the-art INR-based methods.

# Requirements
- python==3.x (Let's move on to python 3 if you still use python 2)
- pytorch==2.0.0
- numpy>=1.15.4
- sentencepiece==0.1.8
- tqdm>=4.28.1

# dataset
The volume at each time step is saved as a .dat file with the little-endian format. The data is stored in row-major order, that is, 
x-axis goes first, then y-axis, finally z-axis. The low-resolution and high-resolution volumes are both simulation data.


# train
first change the data path in dataio.py, then 
`python main.py --train`
# inference
`python main.py --inf`

