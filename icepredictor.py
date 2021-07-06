#basic python
import os
import datetime
import numpy as np
np.random.seed(42)

#tensorflow
import tensorflow as tf
tf.random.set_seed(1) #used to clearn kernel cache
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Dropout
from tensorflow.keras import Model

#matplotlib
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

#Geo imports
from sentinelhub import MimeType, BBox, SentinelHubRequest, SentinelHubDownloadClient, \
    DataCollection, bbox_to_dimensions, SHConfig, SentinelHubCatalog, filter_times
from sentinelhub.geo_utils import get_utm_crs, wgs84_to_utm
from shapely.geometry import Polygon
from rasterio.features import rasterize
from rasterio.transform import from_bounds
import geopandas as gpd

#configure sentinelhub account
config = SHConfig()

config.instance_id = os.environ.get('INSTANCE_ID')
config.sh_client_id = os.environ.get('SH_CLIENT_ID')
config.sh_client_secret = os.environ.get('SH_CLIENT_SECRET')

n_colors = 4 #number of classes in the ice chart
SIZE = 140_000
IMG_SIZE = (350, 350)
IMG_SIZE_TF = (256, 256)#image size input to tensorflow model
RES = 400

def get_land_mask(longCenter, latCenter, size=SIZE, res = RES):
    crs_im = get_utm_crs(longCenter, latCenter) #get utm CRS of the center of the image
    longCenter, latCenter = wgs84_to_utm(longCenter, latCenter) #convert center to utm coordinates
    
    #create a bigger bounding box than our area of interest so we can clip to the general area we are looking for
    bbox_big = BBox((longCenter-size, latCenter-size,longCenter+size, latCenter+size), 
                crs=crs_im)#create a bounding box around the centre of our desired area in UTM coordinates
    bbox_big = bbox_big.transform(crs="EPSG:4326")#convert bbox to mercador
    bbox_big_gdf = gpd.GeoDataFrame({'geometry': Polygon(bbox_big.get_polygon()), 'index':[0]})
    
    # Load shape file of the area of the world
    land_gdf = gpd.read_file('ne_10m_land/ne_10m_land.shp')#load shape file of the earth in mercador coordinates
    
    # Get intrsection of world land polygons and the big boundaing box in UTM coordinates
    area_gdf = gpd.overlay(land_gdf, bbox_big_gdf, how='intersection').to_crs(crs = 'EPSG:'+str(crs_im.epsg))
    
    #now create a bbox for our exact desired dimensions and rasterize
    bbox = BBox((longCenter-size/2, latCenter-size/2,longCenter+size/2, latCenter+size/2), 
                crs=crs_im)#create a bounding box around the centre in the right crs
    
    polys = area_gdf.geometry #get polygons in the bigger region of interest
    transform = from_bounds(*list(bbox), *IMG_SIZE) #create an affine transform which specifies how to map coordinates to pixels
    land_mask = rasterize(((poly, 255) for poly in polys), transform = transform, out_shape=IMG_SIZE)/255
    return land_mask

#returns images centred on coordinates with a width/heigh given by size and resolution in meters
def get_images(longCenter, latCenter, time_start, size=SIZE, res = RES):
    
    B = get_land_mask(longCenter, latCenter, size=SIZE, res=res) #get the land mask which will be the blue channel of the image
    
    catalog = SentinelHubCatalog(config=config)
    
    time_start = datetime.datetime.strptime(time_start, '%Y-%m-%d')
    time_delta = datetime.timedelta(days=15) #length of window to check for images
    time_interval = [time_start, time_start+time_delta] #the function will return the earliest availble image in the interval
    
    crs = get_utm_crs(longCenter, latCenter)#get utm CRS of the center of the image
    longCenter, latCenter = wgs84_to_utm(longCenter, latCenter)#convert center to urm coordinates
    
    bbox = BBox((longCenter-size/2, latCenter-size/2,longCenter+size/2, latCenter+size/2), 
                crs=crs)#create a bounding box around the centre in the right crs
    
    #search for all available images inthe given time range and bounding box
    search_iterator = catalog.search(
        DataCollection.SENTINEL1_EW,
        bbox=bbox,
        time=time_interval,
        fields={
            "include": [
                "id",
                "properties.datetime",
            ],
            "exclude": []
        }
    
    )
    
    time_difference = datetime.timedelta(hours=2)#treat timstamps in this window as the same
    
    all_timestamps = search_iterator.get_timestamps()
    unique_acquisitions = filter_times(all_timestamps, time_difference)
    
    S1GRD_mask_evalscript = """
        //VERSION=3
        
        function setup() {
          return {
            input: ["dataMask"],
            output: [{
                id: "badDataMask",
                bands: 1,
                sampleType: "UINT8"
            }]
          }
        }
        
        function evaluatePixel(sample) {
            return {
              badDataMask: [1 - sample.dataMask]
            };        
        }
        """    
    
    #find the masks that have valid data above threshold
    thresh = 0.10 #must have valid data above this threshold
    
    process_requests = []
    
    for timestamp in unique_acquisitions:#create a request for each timestamp
        
        request = SentinelHubRequest(
            evalscript=S1GRD_mask_evalscript,
            input_data=[
                SentinelHubRequest.input_data(
                    data_collection=DataCollection.SENTINEL1_EW,
                    time_interval=(timestamp - time_difference, timestamp + time_difference),
                    other_args={'processing': {'backCoeff': 'SIGMA0_ELLIPSOID'}}
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
    
    mask_data = client.download(download_requests) #all the masks indicating bad data
    
    #find all the timestamps where the bad data is less than thresh
    valid_acquisitions = [timestamp for mask, timestamp in zip(mask_data, unique_acquisitions)
                       if mask.sum()/(mask.shape[0]*mask.shape[1])<thresh]
    
     
    #get the image for the earliest valid timstampe
    timestamp = valid_acquisitions[0]
    
    S1GRD_evalscript = """
        //VERSION=3
        
        function setup() {
          return {
            input: ["HH", "HV"],
            output: [{ 
                id: "bands",
                bands: 2,
                sampleType: "FLOAT32"
            },]
          }
        }
        
        function evaluatePixel(sample) {
            return {
              bands: [toDb(sample.HH), toDb(sample.HV)],
            };
        
        // visualizes decibels from -40 to 0
        function toDb(linear) {
          // the following commented out lines are simplified below
          // var log = 10 * Math.log(linear) / Math.LN10
          // var val = Math.max(0, (log + 40) / 40)
          return Math.max(0, Math.log(linear) * 0.10857362047 + 1)
        }
        
        }
        """ 
        
    request = SentinelHubRequest(
        evalscript=S1GRD_evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL1_EW,
                time_interval=(timestamp - time_difference, timestamp + time_difference)
            )
        ],
        responses=[
            SentinelHubRequest.output_response('bands', MimeType.TIFF),
        ],
        bbox=bbox,
        size=bbox_to_dimensions(bbox, res),
        config=config
    )
    
    download_request = request.download_list[0]
    
    start1 = datetime.datetime.now()
    
    img = client.download(download_request) #SAR image in dB with two channels
    
    print(datetime.datetime.now() - start1)
    
    
    img = np.nan_to_num(img, 0) #replace nan entries in the image with zero
    
    R = img[..., 0] #HH SAR channel
    G = img[..., 1] #HV SAR channel
    
    img = np.stack((R, G, B), axis=-1)
    return img, timestamp

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

def get_model():
    inputs = Input(shape=[IMG_SIZE_TF[0], IMG_SIZE_TF[1], 3])
    conv1 = Conv2D(32, 3, 1, activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    drop1 = Dropout(0.5)(pool1)

    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(drop1)
    conv2 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    drop2 = Dropout(0.5)(pool2)

    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(drop2)
    conv3 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    drop3 = Dropout(0.5)(pool3)

    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(drop3)
    conv4 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    drop4 = Dropout(0.5)(pool4)

    conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')(drop4)
    conv5 = Conv2D(512, 3, 1, activation='relu', padding='same')(conv5)

    up6 = Conv2D(256, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv5))
    merge6 = concatenate([up6, conv4], axis=3)
    drop6 = Dropout(0.5)(merge6)
    conv6 = Conv2D(256, 3, 1, activation='relu', padding='same')(drop6)
    conv6 = Conv2D(256, 3, 1, activation='relu', padding='same')(conv6)
    
    up7 = Conv2D(128, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([up7, conv3], axis=3)
    drop7 = Dropout(0.5)(merge7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(drop7)
    conv7 = Conv2D(128, 3, 1, activation='relu', padding='same')(conv7)
    
    up8 = Conv2D(64, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([up8, conv2], axis=3)
    drop8 = Dropout(0.5)(merge8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(drop8)
    conv8 = Conv2D(64, 3, 1, activation='relu', padding='same')(conv8)
    
    up9 = Conv2D(32, 3, activation = 'relu', padding = 'same')(UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([up9, conv1], axis=3)
    drop9 = Dropout(0.5)(merge9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(drop9)
    conv9 = Conv2D(32, 3, 1, activation='relu', padding='same')(conv9)

    conv10 = Conv2D(n_colors, 1, 1, activation='softmax')(conv9) #softmax converts the output to a list of probabilities that must sum to 1

    model = Model(inputs=inputs, outputs=conv10)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(), 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), 
              metrics=['sparse_categorical_accuracy', UpdatedMeanIoU(num_classes=n_colors)])
    
    #load saved model weights
    model.load_weights('model/SAR_model/cp-0125.ckpt').expect_partial()
    return model


#take in the an array of images and use the saved neural network to generate an ice chart
def predict_mask(images, model):
    
    #convert image from numpy to tf and resize
    imgs_tf = tf.convert_to_tensor(images)#convert numpy array of images to tensor for model input
    imgs_tf = tf.image.resize(imgs_tf, IMG_SIZE_TF)#resize images
    
    pred_mask = model.predict_on_batch(imgs_tf)
    pred_mask = tf.argmax(pred_mask, axis=-1)#use the highest proabbaility class as the prediction
    pred_mask = pred_mask[..., tf.newaxis]

    return pred_mask.numpy()

def make_cmap(n_colors):
    #define a colormap for the mask
    ice_colors = n_colors-1
    jet = plt.get_cmap('jet', ice_colors)
    newcolors = jet(np.linspace(0, 1, ice_colors))
    black = np.array([[0, 0, 0, 1]])
    newcolors = np.concatenate((newcolors, black), axis=0) #land will be black
    cmap = ListedColormap(newcolors)
    return cmap

cmap = make_cmap(n_colors)
    
def display(display_list):
    #save image
    im = tf.keras.preprocessing.image.array_to_img(display_list[0])
    im.save("static/download_img.png")
    
    #save mask
    msk = plt.imshow(display_list[1], cmap = cmap, vmin=0, vmax=n_colors-1)
    plt.axis('off')
        
    #plot colorbar
    cbar = plt.colorbar(msk, location='bottom', fraction=0.046, pad=0.04)
    tick_locs = (np.arange(n_colors) + 0.5)*(n_colors-1)/n_colors#new tick locations so they are in the middle of the colorbar
    cbar.set_ticks(tick_locs)
    labels=['0-50%', '50-100%', 'Fast Ice', 'Land']
    cbar.set_ticklabels(labels)
    cbar.ax.set_xticklabels(labels, rotation=55)
    
    plt.savefig('static/download_mask.png', dpi=300, pad_inches=0, bbox_inches='tight')
    plt.clf()

if __name__ == '__main__':
    start = datetime.datetime.now()
    img, date = get_images(-84.40795898437501, 62.88520467163244, time_start = '2020-01-01')
    print(datetime.datetime.now() - start)
    imgs = np.expand_dims(img, axis=0)
    
    model=get_model()
    masks = predict_mask(imgs, model)
    display([imgs[0], masks[0]])