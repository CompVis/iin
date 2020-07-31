from setuptools import setup, find_packages

setup(name='iin',
      version='0.1',
      description='Code accompanying the paper '
                  'A Disentangling Invertible Interpretation Network for Explaining Latent Representations'
                  'https://arxiv.org/abs/2004.13166',
      author='Esser, Patrick and Rombach, Robin and Ommer, Bjoern ',
      packages=find_packages(),
      install_requires=[
            'torch>=1.4.0',
            'torchvision>=0.5',
            'numpy>=1.17',
            'scipy>=1.0.1'
          ],
      zip_safe=False)
