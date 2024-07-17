pip install "trl>=0.8.2"
pip install tensorboardX # for tensorboard logging
rm -rf LLaMA-Factory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout beec77a0898a39d94f41c23920415f5b4873a23a # this is a known good version
pip install -e .[torch,metrics]