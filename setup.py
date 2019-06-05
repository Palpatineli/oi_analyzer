"""analyze optical imaging results"""
from setuptools import setup

setup(
    name='oi_analyser',
    version='0.1',
    packages=['oi_analyser'],
    package_data={'data': ['data/*']},
    entry_points={'console_scripts': ['oi_convert = oi_analyser.imaging:convert',
                                      'oi_show = oi_analyser.imaging:calculate']},
    url='',
    license='',
    author='Keji Li',
    author_email='mail@keji.li',
    install_requires=['numpy >= 1.7', 'h5py', 'tqdm', 'scipy', 'uifunc', 'matplotlib'],
    description='analyze images including cortical imaging and calcium imaging'
)
