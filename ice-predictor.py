# Built-in modules
#import pickle
#import sys
import os
import datetime
#import itertools
from PIL import Image
#
## Basics of Python data handling and visualization
import numpy as np
np.random.seed(42)
import geopandas as gpd
#import matplotlib.pyplot as plt
#
## Imports from eo-learn and sentinelhub-py
from eolearn.core import EOTask, EOPatch, LinearWorkflow, FeatureType, OverwritePermission, \
    LoadTask, SaveTask, EOExecutor, ExtractBandsTask, MergeFeatureTask
from eolearn.io import SentinelHubInputTask
from eolearn.mask import AddMultiCloudMaskTask, AddValidDataMaskTask
#from eolearn.geometry import VectorToRaster, PointSamplingTask, ErosionTask
from eolearn.features import LinearInterpolation, SimpleFilterTask, NormalizedDifferenceIndexTask
from sentinelhub import BBox, CRS, DataCollection

#these classes are used toidentify and remove sections that are cloudy or have not data
class SentinelHubValidData:
    """
    Combine Sen2Cor's classification map with `IS_DATA` to define a `VALID_DATA_SH` mask
    The SentinelHub's cloud mask is asumed to be found in eopatch.mask['CLM']
    """
    def __call__(self, eopatch):
        return np.logical_and(eopatch.mask['IS_DATA'].astype(np.bool),
                              np.logical_not(eopatch.mask['CLM'].astype(np.bool)))

class CountValid(EOTask):
    """
    The task counts number of valid observations in time-series and stores the results in the timeless mask.
    """
    def __init__(self, count_what, feature_name):
        self.what = count_what
        self.name = feature_name

    def execute(self, eopatch):
        eopatch.add_feature(FeatureType.MASK_TIMELESS, self.name, np.count_nonzero(eopatch.mask[self.what],axis=0))

        return eopatch

#define the necessairy EOTasks which will be used in workflow do download satellite images and their corresponding sea ice masks
#task for downloading saetlite data
band_names = ['B03', 'B04', 'B08']#false color bands

#TASK TO RETRIEVE IMAGES FROM SENTINELHUB
add_data = SentinelHubInputTask(
    bands_feature=(FeatureType.DATA, 'BANDS'),#location where the images will be stored in the EOPatch
    bands = band_names,#bands to collect in the image
    resolution=200,#resolution of the images in m
    maxcc=0.8,#maximum cloud cover to allow 1=100%
    time_difference=datetime.timedelta(minutes=120),#if two images are this close to each other they are considered the same
    data_collection=DataCollection.SENTINEL2_L1C,
    additional_data=[(FeatureType.MASK, 'dataMask', 'IS_DATA'),#also download the 'is_data' and cloud cover masks from sentinelhub
                     (FeatureType.MASK, 'CLM'),]
)

#TASK TO ADD A MASK THAT COMBINES THE CLOUD COVER AND 'IS_DATA' MASKS INTO ONE 'VALID_DATA' MASK
def calculate_valid_data_mask(eopatch):
    is_data_mask = eopatch.mask['IS_DATA'].astype(bool)
    cloud_mask = ~eopatch.mask['CLM'].astype(bool)
    return np.logical_and(is_data_mask, cloud_mask)

add_valid_mask = AddValidDataMaskTask(predicate=calculate_valid_data_mask, valid_data_feature='VALID_DATA')

#funtction to calculate cloud coverage used in the next task
def calculate_coverage(array):
    return 1.0 - np.count_nonzero(array) / np.size(array)

#CUSTOM EOTASK WHICH ADDS A SCALAR FEATURE WITH % OF IMAGE THAT IS VALID (AVAILABLE AND NOT CLOUDY)
class AddValidDataCoverage(EOTask):

    def execute(self, eopatch):

        valid_data = eopatch.get_feature(FeatureType.MASK, 'VALID_DATA')
        time, height, width, channels = valid_data.shape

        coverage = np.apply_along_axis(calculate_coverage, 1,
                                       valid_data.reshape((time, height * width * channels)))

        eopatch.add_feature(FeatureType.SCALAR, 'COVERAGE', coverage[:, np.newaxis])
        return eopatch

add_coverage = AddValidDataCoverage()

#CUSTOM EOTASK WHICH REMOVES IMAGES IF THE MORE THAN 'cloud_coverage_threshold' IS INVALID DATA
cloud_coverage_threshold = 0.20
class ValidDataCoveragePredicate:
    
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, array):
        return calculate_coverage(array) < self.threshold
    
remove_cloudy_scenes = SimpleFilterTask((FeatureType.MASK, 'VALID_DATA'),
                                        ValidDataCoveragePredicate(cloud_coverage_threshold))
 
# TASK FOR SAVING TO OUTPUT
path_out = './eopatches/'
if not os.path.isdir(path_out):
    os.makedirs(path_out)
save = SaveTask(path_out, overwrite_permission=OverwritePermission.OVERWRITE_PATCH)

#generate eolearn workflow
workflow = LinearWorkflow(
    add_data,
    add_valid_mask,
    add_coverage, 
    remove_cloudy_scenes,)

#Execute a workflow to gather EOpatches from a single time interval
time_interval = ['2018-3-1', '2018-5-31'] # time interval for the SH request
bbox = BBox(((415958.56868003984,5848652.996486791),(487290.8505186029,5909770.831117111)), crs='EPSG:32617')
idx=1

# define additional parameters of the workflow
execution_args = []
execution_args.append({
    add_data:{'bbox': bbox, 'time_interval': time_interval},
    save: {'eopatch_folder': f'eopatch_{idx}'}
})

#executor = EOExecutor(workflow, execution_args, save_logs=True)
#results = executor.run(return_results=True)
#executor.make_report()

eopatch = results[0].eopatch()
print(eopatch)

import matplotlib.pyplot as plt
imgs = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]], 0, 1)

import streamlit as st
import tensorflow as tf
import numpy as np
from matplotlib.colors import ListedColormap

#Define IoU metric as this is information is not stored in the saved model (by stack overflow user HuckleberryFinn)
class UpdatedMeanIoU(tf.keras.metrics.MeanIoU):
    def __init__(self,
               y_true=None,
               y_pred=None,
               num_classes=None,
               name=None,
               dtype=None):
        super(UpdatedMeanIoU, self).__init__(num_classes = num_classes,name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.math.argmax(y_pred, axis=-1)
        return super().update_state(y_true, y_pred, sample_weight)

model = tf.keras.models.load_model('model', custom_objects={'UpdatedMeanIoU':UpdatedMeanIoU})

#define a colormap for the mask
n_colors=8
ice_colors = n_colors-1
jet = plt.get_cmap('jet', ice_colors)
newcolors = jet(np.linspace(0, 1, ice_colors))
black = np.array([[0, 0, 0, 1]])
white = np.array([[1, 1, 1, 1]])
newcolors = np.concatenate((newcolors, black), axis=0) #land will be black
cmap = ListedColormap(newcolors)

def display(display_list):
    fig, axs = plt.subplots(nrows=1, ncols = len(display_list), figsize=(15, 6))

    title = ['Input Image', 'Predicted Ice Chart']
    
    for i in range(len(display_list)):
        if i==0:
            axs[i].imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        else:
            msk = axs[i].imshow(display_list[i], cmap = cmap, vmin=0, vmax=n_colors-1)
        axs[i].axis('off')
        
    #plot colorbar
    cbar = fig.colorbar(msk, ax=axs, location='right')
    tick_locs = (np.arange(n_colors) + 0.5)*(n_colors-1)/n_colors#new tick locations so they are in the middle of the colorbar
    cbar.set_ticks(tick_locs)
    cbar.set_ticklabels(np.arange(n_colors))
    plt.show()
#    st.write(fig)
    
#function to generate a mask from the model predictions
def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)#use the highest proabbaility class as the prediction
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

#helper function to plot image and predicted mask
def show_predictions(dataset):
    for image in dataset:
        pred_mask = model.predict(image)
        display([image[0], create_mask(pred_mask)])

IMG_SIZE = (256, 256)

def read_image(image):
    image = tf.image.resize(image, IMG_SIZE)
    image = tf.cast(image, tf.float32)
    return image

BATCH_SIZE = 1

ds = tf.data.Dataset.from_tensor_slices(imgs)#read filenames
ds = ds.map(read_image) #convert filenames to stream of images/masks
dataset = ds.batch(BATCH_SIZE)

show_predictions(dataset)