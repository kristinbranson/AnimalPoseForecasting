from setuptools import setup, find_packages

setup(
    name='AnimalPoseForecasting',
    version='0.1.0',
    packages=find_packages(include=['flyllm', 'apf', 'jaaba_detect', 'jaaba_detect.*']),
    package_data={'jaaba_detect': ['ellipse_from_keypoints.json']},
    include_package_data=True,
)

# run `pip install -e .` to run this in editable mode
# jaaba_detect additionally needs: h5py, scipy (+ hdf5storage for load_X_video)
# Classifiers are NOT bundled: pass any exported .classifier.mat path to
# jaaba_detect_from_X / jaaba_detect_from_track / jaaba_detect.
