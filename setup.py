"""analyze optical imaging results"""
from setuptools import setup, find_packages

setup(
    name='oi_analyser',
    version='0.1',
    packages=find_packages(),
    package_data={'data': ['data/*']},
    entry_points={'console_scripts': ['oi_convert = oi_analyser.imaging:convert',
                                      'oi_show = oi_analyser.imaging:calculate']},
    url='',
    license='',
    author='Keji Li',
    author_email='mail@keji.li',
    install_requires=['numpy >= 1.7', 'h5py', 'tqdm', 'scipy', 'uifunc', 'matplotlib', 'numexpr'],
    description='analyze images including cortical imaging and calcium imaging'
)
