import datetime
import numpy as np
np.random.seed(42)

import tensorflow as tf
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

## Imports from eo-learn and sentinelhub-py
from eolearn.core import EOTask, LinearWorkflow, FeatureType, EOExecutor
from eolearn.io import SentinelHubInputTask
from eolearn.mask import AddValidDataMaskTask
from eolearn.features import SimpleFilterTask
from sentinelhub import BBox, DataCollection, SHConfig
from sentinelhub.geo_utils import get_utm_crs, wgs84_to_utm

config = SHConfig()

config.instance_id = 'b6fce5d9-6d40-4fec-a46e-d9bd2844abea'
config.sh_client_id = '5cffa454-854c-4b3e-bc33-58ac529a4958'
config.sh_client_secret = 'JCPgC{8h.SpO%>Rn1H!uI-IhF+KosoC75Dd^)pn#'

n_colors = 8 #number of classes in the ice chart

#returns images centred on coordinates with a width/heigh given by size in meters
def get_images(longCenter, latCenter, time_start, size=70_000):
    
    time_start = datetime.datetime.strptime(time_start, '%Y-%m-%d')
    time_delta = datetime.timedelta(days=60) #length of window to check for images
    time_interval = [time_start, time_start+time_delta] #the function will return the earliest availble image in the interval
    
    crs = get_utm_crs(longCenter, latCenter)#get utm CRS of the center of the image
    longCenter, latCenter = wgs84_to_utm(longCenter, latCenter)#convert center to urm coordinates
    
    bbox = BBox((longCenter-size/2, latCenter-size/2,longCenter+size/2, latCenter+size/2), 
                crs=crs)#create a bounding box around the centre in the right crs
    
    #bands to download from satellite data
    band_names = ['B03', 'B04', 'B08']#false color bands

    print(bbox)
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
     
    #generate eolearn workflow
    workflow = LinearWorkflow(
        add_data,
        add_valid_mask,
        add_coverage, 
        remove_cloudy_scenes,)
    
    #Execute a workflow to gather EOpatches from a single time interval
    # define additional parameters of the workflow
    execution_args = []
    execution_args.append({
        add_data:{'bbox': bbox, 'time_interval': time_interval}
    })
    
    executor = EOExecutor(workflow, execution_args)
    results = executor.run(return_results=True)
    #executor.make_report()
    eopatch = results[0].eopatch()
    imgs = np.clip(eopatch.data['BANDS'][..., [2, 1, 0]], 0, 1)
    imgDates = eopatch['timestamp']
    return imgs, imgDates





#take in the an array of images and use the saved neural network to generate an ice chart
def predict_mask(images):

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
       
    IMG_SIZE = (256, 256)
    imgs_tf = tf.convert_to_tensor(images)#convert numpy array of images to tensor for model input
    imgs_tf = tf.image.resize(imgs_tf, IMG_SIZE)#resize images
    
    #function to generate a mask from the model predictions
    def create_masks(dataset):
        pred_mask = model(dataset, training=False)
        pred_mask = tf.argmax(pred_mask, axis=-1)#use the highest proabbaility class as the prediction
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask
    
    masks = create_masks(imgs_tf)
    return masks.numpy()

def make_cmap(n_colors):
    #define a colormap for the mask
    ice_colors = n_colors-1
    jet = plt.get_cmap('jet', ice_colors)
    newcolors = jet(np.linspace(0, 1, ice_colors))
    black = np.array([[0, 0, 0, 1]])
    white = np.array([[1, 1, 1, 1]])
    newcolors = np.concatenate((newcolors, black), axis=0) #land will be black
    cmap = ListedColormap(newcolors)
    return cmap

cmap = make_cmap(n_colors)
    
def display(display_list):
    fig, axs = plt.subplots(nrows=1, ncols = len(display_list), figsize=(15, 6))
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
    plt.savefig('static/download.jpg', pad_inches=0, bbox_inches='tight')
    
def generate_data(longCenter, latCenter, time_start):
    imgs, _ = get_images(longCenter, latCenter, time_start)
    masks = predict_mask(imgs)
    display([imgs[0], masks[0]])


if __name__ == '__main__':
    eopatch = generate_data(-82.0943, 52.8281, time_start = '2018-05-15')
