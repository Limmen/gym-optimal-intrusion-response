from setuptools import setup

setup(name='gym_optimal_intrusion_response',
      version='1.0.0',
      install_requires=['gym', 'pyglet', 'numpy', 'torch', 'stable_baselines3',
                        'jsonpickle'],
      author='Kim Hammar',
      author_email='hammar.kim@gmail.com',
      description='A Simulated Optimal Intrusion Response Game',
      license='Creative Commons Attribution-ShareAlike 4.0 International',
      keywords='Reinforcement-Learning Cyber-Security',
      url='https://github.com/Limmen/gym-optimal-intrusion-response',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',
          'Programming Language :: Python :: 3.5',
          'Programming Language :: Python :: 3.6'
  ]
)