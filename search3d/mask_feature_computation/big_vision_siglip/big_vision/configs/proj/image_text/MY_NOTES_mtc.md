conda create -n siglip python=3.10
conda activate siglip
#pip install jax==0.3.25
pip -q install -r /home/ayca/big_vision-474dd2ebde37268db4ea44decef14c7c1f6a0258/big_vision/requirements.txt
pip -q install --no-cache-dir -U crcmod
pip install ipykernel

#to install the gpu version
pip install --upgrade "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html