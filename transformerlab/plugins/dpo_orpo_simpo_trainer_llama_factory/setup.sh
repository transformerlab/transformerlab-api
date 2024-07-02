pip install "trl>=0.8.2"
pip install tensorboardX # for tensorboard logging
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -e .[torch,metrics]