from setuptools import setup

package_name = 'gender_predictor'

setup(
    name=package_name,
    version='0.0.0',
    packages=[],
    py_modules=[
        'gender_predictor.script',
    ],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('lib/gender_predictor', ['gender_predictor/alex.py']),
        ('lib/gender_predictor/models', ['models/AlexlikeMSGD.model'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hirorittsu',
    maintainer_email='zq6295migly@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gender_predictor = gender_predictor.script:main',
        ],
    },
)
