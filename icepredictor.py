import os
import datetime
import numpy as np
np.random.seed(42)

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt


## Imports from eo-learn and sentinelhub-py
from sentinelhub import MimeType, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, SHConfig, SentinelHubCatalog, filter_times
from sentinelhub.geo_utils import get_utm_crs, wgs84_to_utm

config = SHConfig()

config.instance_id = os.environ.get('INSTANCE_ID')
config.sh_client_id = os.environ.get('SH_CLIENT_ID')
config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')

n_colors = 8 #number of classes in the ice chart

#returns images centred on coordinates with a width/heigh given by size and resolution in meters
def get_images(longCenter, latCenter, time_start, size=70_000, res = 200):
    
    catalog = SentinelHubCatalog(config=config)
    
    time_start = datetime.datetime.strptime(time_start, '%Y-%m-%d')
    time_delta = datetime.timedelta(days=60) #length of window to check for images
    time_interval = [time_start, time_start+time_delta] #the function will return the earliest availble image in the interval
    
    crs = get_utm_crs(longCenter, latCenter)#get utm CRS of the center of the image
    longCenter, latCenter = wgs84_to_utm(longCenter, latCenter)#convert center to urm coordinates
    
    bbox = BBox((longCenter-size/2, latCenter-size/2,longCenter+size/2, latCenter+size/2), 
                crs=crs)#create a bounding box around the centre in the right crs
    
    #search for all available images inthe given time range and bounding box
    search_iterator = catalog.search(
        DataCollection.SENTINEL2_L1C,
        bbox=bbox,
        time=time_interval,
        query={
            "eo:cloud_cover": {
                "lt": 10 #allow up to 10% cloud cover in the full tile
            }
        },
        fields={
            "include": [
                "id",
                "properties.datetime",
                "properties.eo:cloud_cover"
            ],
            "exclude": []
        }
    
    )
    
    time_difference = datetime.timedelta(hours=2)#treat timstamps in this window as the same
    
    all_timestamps = search_iterator.get_timestamps()
    unique_acquisitions = filter_times(all_timestamps, time_difference)
    
    false_color_evalscript = """
        //VERSION=3
        
        function setup() {
          return {
            input: ["B03", "B04", "B08", "CLM", "dataMask"],
            output: [{ 
                id: "falseColor",
                bands: 3,
            }, {
                id: "badDataMask",
                bands: 1,
                sampleType: "UINT8"
            }]
          }
        }
        
        function evaluatePixel(sample) {
            return {
              falseColor: [sample.B08, sample.B04, sample.B03],
              badDataMask: [Math.min( sample.CLM 
                                       + (1 - sample.dataMask), 1)]
            };
        }
        """    
    
    #find the masks that have valid data above threshold
    thresh = 0.1 #must have valid data above this threshold
    
    process_requests = []
    
    for timestamp in unique_acquisitions:#create a request for each timestamp
        
        request = SentinelHubRequest(
            evalscript=false_color_evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL2_L1C,
                    time_interval=(timestamp - time_difference, timestamp + time_difference)
                )
            ],
            responses=[
                SentinelHubRequest.output_response('badDataMask', MimeType.PNG)#get only the bad data mask
            ],
            bbox=bbox,
            size=bbox_to_dimensions(bbox, res),
            config=config
        )
        process_requests.append(request)
        
    client = SentinelHubDownloadClient(config=config)
    download_requests = [request.download_list[0] for request in process_requests]
    mask_data = client.download(download_requests)
    #find all the timestamps where the bad data is less than thresh
    valid_acquisitions = [timestamp for mask, timestamp in zip(mask_data, unique_acquisitions)
                       if mask.sum()/(mask.shape[0]*mask.shape[1])<thresh]
    
    #get the image for the earliest valid timstampe
    timestamp = valid_acquisitions[0]
        
    request = SentinelHubRequest(
        evalscript=false_color_evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L1C,
                time_interval=(timestamp - time_difference, timestamp + time_difference)
            )
        ],
        responses=[
            SentinelHubRequest.output_response('falseColor', MimeType.PNG),
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, res),
        config=config
    )
    
    download_request = request.download_list[0]
    img = client.download(download_request)/255.0#scale the values to the range [0, 1]

    return img, timestamp





#take in the an array of images and use the saved neural network to generate an ice chart
def predict_mask(images):
    print('1')
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
    print('2')
    model = tf.keras.models.load_model('model', custom_objects={'UpdatedMeanIoU':UpdatedMeanIoU})
    print('3')
    IMG_SIZE = (256, 256)
    imgs_tf = tf.convert_to_tensor(images)#convert numpy array of images to tensor for model input
    imgs_tf = tf.image.resize(imgs_tf, IMG_SIZE)#resize images
    print('4')
    #function to generate a mask from the model predictions
    def create_masks(dataset):
        pred_mask = model(dataset, training=False)
        pred_mask = tf.argmax(pred_mask, axis=-1)#use the highest proabbaility class as the prediction
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask
    print('5')
    masks = create_masks(imgs_tf)
    print('6')
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
    img, _ = get_images(longCenter, latCenter, time_start)
    imgs = np.expand_dims(img, axis=0)
    masks = predict_mask(imgs)
    display([imgs[0], masks[0]])


if __name__ == '__main__':
    generate_data(-103.991, 68.520, time_start = '2020-07-22')
