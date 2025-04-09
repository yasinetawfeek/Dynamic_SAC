from setuptools import find_packages, setup

package_name = 'cm_env'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'shapely',
        'stable_baselines3'],
    zip_safe=True,
    maintainer='yasinetawfeek',
    maintainer_email='yasinetawfeek@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cm_env = cm_env.environment:main',
            'sac = cm_env.SAC:main',
        ],
    },
)
