# Deploy and monitor a machine learning workflow for Image Classification

## Setting up this notebook

Notes about the instance size and kernel setup: this notebook has been tested on

1. The `Python 3 (Data Science)` kernel
2. The `ml.t3.medium` Sagemaker notebook instance

## Data Staging

We'll use a sample dataset called CIFAR to simulate the challenges Scones Unlimited are facing in Image Classification. In order to start working with CIFAR we'll need to:

1. Extract the data from a hosting service
2. Transform it into a usable shape and format
3. Load it into a production system

In other words, we're going to do some simple ETL!

### 1. Extract the data from the hosting service

In the cell below, define a function `extract_cifar_data` that extracts python version of the CIFAR-100 dataset. The CIFAR dataaset is open source and generously hosted by the University of Toronto at: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz 


```python
import requests

def extract_cifar_data(url, filename="cifar.tar.gz"):
    """A function for extracting the CIFAR-100 dataset and storing it as a gzipped file
    
    Arguments:
    url      -- the URL where the dataset is hosted
    filename -- the full path where the dataset will be written
    
    """
    
    # Todo: request the data from the data url
    # Hint: use `requests.get` method
    r = requests.get(url, filename)
    with open(filename, "wb") as file_context:
        file_context.write(r.content)
    return
```

Let's test it out! Run the following cell and check whether a new file `cifar.tar.gz` is created in the file explorer.


```python
extract_cifar_data("https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz")     
```

### 2. Transform the data into a usable shape and format

Clearly, distributing the data as a gzipped archive makes sense for the hosting service! It saves on bandwidth, storage, and it's a widely-used archive format. In fact, it's so widely used that the Python community ships a utility for working with them, `tarfile`, as part of its Standard Library. Execute the following cell to decompress your extracted dataset:


```python
import tarfile

with tarfile.open("cifar.tar.gz", "r:gz") as tar:
    tar.extractall()
```

A new folder `cifar-100-python` should be created, containing `meta`, `test`, and `train` files. These files are `pickles` and the [CIFAR homepage](https://www.cs.toronto.edu/~kriz/cifar.html) provides a simple script that can be used to load them. We've adapted the script below for you to run:


```python
import pickle

with open("./cifar-100-python/meta", "rb") as f:
    dataset_meta = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/test", "rb") as f:
    dataset_test = pickle.load(f, encoding='bytes')

with open("./cifar-100-python/train", "rb") as f:
    dataset_train = pickle.load(f, encoding='bytes')
```


```python
# Feel free to explore the datasets

dataset_train.keys()
```




    dict_keys([b'filenames', b'batch_label', b'fine_labels', b'coarse_labels', b'data'])



As documented on the homepage, `b'data'` contains rows of 3073 unsigned integers, representing three channels (red, green, and blue) for one 32x32 pixel image per row.


```python
32*32*3
```




    3072



For a simple gut-check, let's transform one of our images. Each 1024 items in a row is a channel (red, green, then blue). Each 32 items in the channel are a row in the 32x32 image. Using python, we can stack these channels into a 32x32x3 array, and save it as a PNG file:


```python
import numpy as np

# Each 1024 in a row is a channel (red, green, then blue)
row = dataset_train[b'data'][0]
red, green, blue = row[0:1024], row[1024:2048], row[2048:]

# Each 32 items in the channel are a row in the 32x32 image
red = red.reshape(32,32)
green = green.reshape(32,32)
blue = blue.reshape(32,32)

# Combine the channels into a 32x32x3 image!
combined = np.dstack((red,green,blue))
```

For a more concise version, consider the following:


```python
# All in one:
test_image = np.dstack((
    row[0:1024].reshape(32,32),
    row[1024:2048].reshape(32,32),
    row[2048:].reshape(32,32)
))
```


```python
import matplotlib.pyplot as plt
plt.imshow(test_image)
plt.show()
```


    
![png](./images/output_16_0.png)
    


Looks like a cow! Let's check the label. `dataset_meta` contains label names in order, and `dataset_train` has a list of labels for each row.


```python
dataset_train[b'fine_labels'][0]
```




    19



Our image has a label of `19`, so let's see what the 19th item is in the list of label names.


```python
print(dataset_meta[b'fine_label_names'][19])
```

    b'cattle'


Ok! 'cattle' sounds about right. By the way, using the previous two lines we can do:


```python
n = 0
print(dataset_meta[b'fine_label_names'][dataset_train[b'fine_labels'][n]])
```

    b'cattle'


Now we know how to check labels, is there a way that we can also check file names? `dataset_train` also contains a `b'filenames'` key. Let's see what we have here:


```python
print(dataset_train[b'filenames'][0])
```

    b'bos_taurus_s_000507.png'


"Taurus" is the name of a subspecies of cattle, so this looks like a pretty reasonable filename. To save an image we can also do:


```python
plt.imsave("file.png", test_image)
```

Your new PNG file should now appear in the file explorer -- go ahead and pop it open to see!

Now that you know how to reshape the images, save them as files, and capture their filenames and labels, let's just capture all the bicycles and motorcycles and save them. Scones Unlimited can use a model that tells these apart to route delivery drivers automatically.

In the following cell, identify the label numbers for Bicycles and Motorcycles:


```python
import pandas as pd

# Todo: Filter the dataset_train and dataset_meta objects to find the label numbers for Bicycle and Motorcycles
index_Bicycle = dataset_meta[b'fine_label_names'].index( b'bicycle')
index_motorcycle = dataset_meta[b'fine_label_names'].index( b'motorcycle')
print(index_Bicycle, index_motorcycle)
```

    8 48


Good job! We only need objects with label 8 and 48 -- this drastically simplifies our handling of the data! Below we construct a dataframe for you, and you can safely drop the rows that don't contain observations about bicycles and motorcycles. Fill in the missing lines below to drop all other rows:


```python
#Construct the dataframe
df_train = pd.DataFrame({
    "filenames": dataset_train[b'filenames'],
    "labels": dataset_train[b'fine_labels'],
    "row": range(len(dataset_train[b'filenames']))
})

# Drop all rows from df_train where label is not 8 or 48
df_train = df_train[df_train['labels'].isin([8,48])]

# Decode df_train.filenames so they are regular strings
df_train["filenames"] = df_train["filenames"].apply(
    lambda x: x.decode("utf-8")
)


df_test = pd.DataFrame({
    "filenames": dataset_test[b'filenames'],
    "labels": dataset_test[b'fine_labels'],
    "row": range(len(dataset_test[b'filenames']))
})

# # Drop all rows from df_test where label is not 8 or 48
df_test = df_test[df_test['labels'].isin([8,48])]

# # Decode df_test.filenames so they are regular strings
df_test["filenames"] = df_test["filenames"].apply(
    lambda x: x.decode("utf-8")
)
```

Now that the data is filtered for just our classes, we can save all our images.


```python
!mkdir ./train
!mkdir ./test
```

In the previous sections we introduced you to several key snippets of code:

1. Grabbing the image data:

```python
dataset_train[b'data'][0]
```

2. A simple idiom for stacking the image data into the right shape

```python
import numpy as np
np.dstack((
    row[0:1024].reshape(32,32),
    row[1024:2048].reshape(32,32),
    row[2048:].reshape(32,32)
))
```

3. A simple `matplotlib` utility for saving images

```python
plt.imsave(path+row['filenames'], target)
```

Compose these together into a function that saves all the images into the `./test` and `./train` directories. Use the comments in the body of the `save_images` function below to guide your construction of the function:



```python
def save_images( n , path):
    #Grab the image data in row-major form
    img = dataset_test[b'data'][n] if path == 'test' else dataset_train[b'data'][n]
    
    # Consolidated stacking/reshaping from earlier
    target = np.dstack((
        img[0:1024].reshape(32,32),
        img[1024:2048].reshape(32,32),
        img[2048:].reshape(32,32)
    ))
    
    # Save the image
    filename = dataset_train[b'filenames'][n] if path == 'train' else dataset_test[b'filenames'][n]
    
    filename = filename.decode("utf-8")
    plt.imsave('./'+path+'/'+filename, target)
    
    # Return any signal data you want for debugging
    return  'done ....'
```


```python
## save ALL images using the save_images function
path = 'train/'
for n in range(len(dataset_train[b'filenames'])):
    if dataset_train[b'fine_labels'][n] == 8:
        save_images(n, 'train')

for n in range(len(dataset_train[b'filenames'])):
    if dataset_train[b'fine_labels'][n] == 48:
        save_images(n,'train')
```


```python
## save ALL images using the save_images function
path = 'test/'
for n in range(len(dataset_test[b'filenames'])):
    if dataset_test[b'fine_labels'][n] == 8:
        save_images(n, 'test')

for n in range(len(dataset_test[b'filenames'])):
    if dataset_test[b'fine_labels'][n] == 48:
        save_images(n, 'test')
```

### 3. Load the data

Now we can load the data into S3.

Using the sagemaker SDK grab the current region, execution role, and bucket.


```python
import sagemaker

bucket= sagemaker.Session().default_bucket()
print("Default Bucket: {}".format(bucket))

region = sagemaker.Session().boto_region_name
print("AWS Region: {}".format(region))

role = sagemaker.get_execution_role()
print("RoleArn: {}".format(role))
```

    Default Bucket: sagemaker-us-east-1-761239682643
    AWS Region: us-east-1
    RoleArn: arn:aws:iam::761239682643:role/service-role/AmazonSageMaker-ExecutionRole-20221218T152260


With this data we can easily sync your data up into S3!


```python
import os

os.environ["DEFAULT_S3_BUCKET"] = bucket
!aws s3 sync ./train s3://${DEFAULT_S3_BUCKET}/train/
print('Training finished ...')

!aws s3 sync ./test s3://${DEFAULT_S3_BUCKET}/test/
print('Testing finished ...')
```

    upload: train/bicycle_s_000017.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000017.png
    upload: train/bicycle_s_000043.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000043.png
    upload: train/bicycle_s_000039.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000039.png
    upload: train/bicycle_s_000099.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000099.png
    upload: train/bicycle_s_000021.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000021.png
    upload: train/bicycle_s_000038.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000038.png
    upload: train/bicycle_s_000051.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000051.png
    upload: train/bicycle_s_000180.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000180.png
    upload: train/bicycle_s_000124.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000124.png
    upload: train/bicycle_s_000147.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000147.png
    upload: train/bicycle_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000035.png
    upload: train/bicycle_s_000066.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000066.png
    upload: train/bicycle_s_000159.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000159.png
    upload: train/bicycle_s_000156.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000156.png
    upload: train/bicycle_s_000071.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000071.png
    upload: train/bicycle_s_000231.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000231.png
    upload: train/bicycle_s_000137.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000137.png
    upload: train/bicycle_s_000314.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000314.png
    upload: train/bicycle_s_000369.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000369.png
    upload: train/bicycle_s_000392.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000392.png
    upload: train/bicycle_s_000243.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000243.png
    upload: train/bicycle_s_000371.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000371.png
    upload: train/bicycle_s_000399.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000399.png
    upload: train/bicycle_s_000408.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000408.png
    upload: train/bicycle_s_000282.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000282.png
    upload: train/bicycle_s_000435.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000435.png
    upload: train/bicycle_s_000149.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000149.png
    upload: train/bicycle_s_000279.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000279.png
    upload: train/bicycle_s_000467.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000467.png
    upload: train/bicycle_s_000463.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000463.png
    upload: train/bicycle_s_000491.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000491.png
    upload: train/bicycle_s_000522.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000522.png
    upload: train/bicycle_s_000235.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000235.png
    upload: train/bicycle_s_000437.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000437.png
    upload: train/bicycle_s_000396.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000396.png
    upload: train/bicycle_s_000536.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000536.png
    upload: train/bicycle_s_000546.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000546.png
    upload: train/bicycle_s_000537.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000537.png
    upload: train/bicycle_s_000668.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000668.png
    upload: train/bicycle_s_000569.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000569.png
    upload: train/bicycle_s_000561.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000561.png
    upload: train/bicycle_s_000667.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000667.png
    upload: train/bicycle_s_000759.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000759.png
    upload: train/bicycle_s_000753.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000753.png
    upload: train/bicycle_s_000723.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000723.png
    upload: train/bicycle_s_000774.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000774.png
    upload: train/bicycle_s_000775.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000775.png
    upload: train/bicycle_s_000781.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000781.png
    upload: train/bicycle_s_000822.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000822.png
    upload: train/bicycle_s_000829.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000829.png
    upload: train/bicycle_s_000785.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000785.png
    upload: train/bicycle_s_000778.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000778.png
    upload: train/bicycle_s_000782.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000782.png
    upload: train/bicycle_s_000951.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000951.png
    upload: train/bicycle_s_000861.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000861.png
    upload: train/bicycle_s_000978.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000978.png
    upload: train/bicycle_s_000986.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000986.png
    upload: train/bicycle_s_000996.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_000996.png
    upload: train/bicycle_s_001002.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001002.png
    upload: train/bicycle_s_001205.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001205.png
    upload: train/bicycle_s_001228.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001228.png
    upload: train/bicycle_s_001203.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001203.png
    upload: train/bicycle_s_001174.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001174.png
    upload: train/bicycle_s_001245.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001245.png
    upload: train/bicycle_s_001247.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001247.png
    upload: train/bicycle_s_001338.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001338.png
    upload: train/bicycle_s_001111.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001111.png
    upload: train/bicycle_s_001348.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001348.png
    upload: train/bicycle_s_001402.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001402.png
    upload: train/bicycle_s_001411.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001411.png
    upload: train/bicycle_s_001388.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001388.png
    upload: train/bicycle_s_001470.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001470.png
    upload: train/bicycle_s_001452.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001452.png
    upload: train/bicycle_s_001554.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001554.png
    upload: train/bicycle_s_001448.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001448.png
    upload: train/bicycle_s_001583.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001583.png
    upload: train/bicycle_s_001569.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001569.png
    upload: train/bicycle_s_001663.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001663.png
    upload: train/bicycle_s_001168.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001168.png
    upload: train/bicycle_s_001679.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001679.png
    upload: train/bicycle_s_001642.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001642.png
    upload: train/bicycle_s_001673.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001673.png
    upload: train/bicycle_s_001409.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001409.png
    upload: train/bicycle_s_001688.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001688.png
    upload: train/bicycle_s_001687.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001687.png
    upload: train/bicycle_s_001681.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001681.png
    upload: train/bicycle_s_001904.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001904.png
    upload: train/bicycle_s_001748.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001748.png
    upload: train/bicycle_s_001928.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001928.png
    upload: train/bicycle_s_001814.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001814.png
    upload: train/bicycle_s_001757.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001757.png
    upload: train/bicycle_s_001956.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001956.png
    upload: train/bicycle_s_001693.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_001693.png
    upload: train/bicycle_s_002005.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002005.png
    upload: train/bicycle_s_002029.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002029.png
    upload: train/bicycle_s_002012.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002012.png
    upload: train/bicycle_s_002129.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002129.png
    upload: train/bicycle_s_002153.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002153.png
    upload: train/bicycle_s_002222.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002222.png
    upload: train/bicycle_s_002237.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002237.png
    upload: train/bicycle_s_002247.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002247.png
    upload: train/bicycle_s_002218.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002218.png
    upload: train/bicycle_s_002258.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002258.png
    upload: train/bicycle_s_002299.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002299.png
    upload: train/bicycle_s_002049.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002049.png
    upload: train/bicycle_s_002100.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002100.png
    upload: train/bicycle_s_002338.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002338.png
    upload: train/bicycle_s_002400.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002400.png
    upload: train/bicycle_s_002373.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002373.png
    upload: train/bicycle_s_002436.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002436.png
    upload: train/bicycle_s_002132.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002132.png
    upload: train/bicycle_s_002410.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002410.png
    upload: train/bicycle_s_002374.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002374.png
    upload: train/bicycle_s_002475.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002475.png
    upload: train/bicycle_s_002521.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002521.png
    upload: train/bicycle_s_002448.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002448.png
    upload: train/bicycle_s_002550.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002550.png
    upload: train/bicycle_s_002711.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002711.png
    upload: train/bicycle_s_002569.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002569.png
    upload: train/bicycle_s_002661.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002661.png
    upload: train/bicycle_s_002624.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002624.png
    upload: train/bicycle_s_002725.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002725.png
    upload: train/bicycle_s_002669.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002669.png
    upload: train/bicycle_s_002728.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002728.png
    upload: train/bicycle_s_002762.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002762.png
    upload: train/bike_s_000003.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000003.png
    upload: train/bicycle_s_002729.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002729.png
    upload: train/bicycle_s_002759.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002759.png
    upload: train/bike_s_000001.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000001.png
    upload: train/bike_s_000015.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000015.png
    upload: train/bike_s_000018.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000018.png
    upload: train/bike_s_000005.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000005.png
    upload: train/bike_s_000021.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000021.png
    upload: train/bike_s_000024.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000024.png
    upload: train/bike_s_000025.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000025.png
    upload: train/bike_s_000023.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000023.png
    upload: train/bicycle_s_002715.png to s3://sagemaker-us-east-1-761239682643/train/bicycle_s_002715.png
    upload: train/bike_s_000026.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000026.png
    upload: train/bike_s_000040.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000040.png
    upload: train/bike_s_000034.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000034.png
    upload: train/bike_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000035.png
    upload: train/bike_s_000111.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000111.png
    upload: train/bike_s_000051.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000051.png
    upload: train/bike_s_000121.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000121.png
    upload: train/bike_s_000062.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000062.png
    upload: train/bike_s_000154.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000154.png
    upload: train/bike_s_000127.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000127.png
    upload: train/bike_s_000204.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000204.png
    upload: train/bike_s_000129.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000129.png
    upload: train/bike_s_000164.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000164.png
    upload: train/bike_s_000162.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000162.png
    upload: train/bike_s_000299.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000299.png
    upload: train/bike_s_000302.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000302.png
    upload: train/bike_s_000304.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000304.png
    upload: train/bike_s_000336.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000336.png
    upload: train/bike_s_000237.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000237.png
    upload: train/bike_s_000256.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000256.png
    upload: train/bike_s_000392.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000392.png
    upload: train/bike_s_000390.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000390.png
    upload: train/bike_s_000364.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000364.png
    upload: train/bike_s_000397.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000397.png
    upload: train/bike_s_000506.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000506.png
    upload: train/bike_s_000522.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000522.png
    upload: train/bike_s_000474.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000474.png
    upload: train/bike_s_000545.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000545.png
    upload: train/bike_s_000516.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000516.png
    upload: train/bike_s_000544.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000544.png
    upload: train/bike_s_000628.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000628.png
    upload: train/bike_s_000555.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000555.png
    upload: train/bike_s_000593.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000593.png
    upload: train/bike_s_000679.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000679.png
    upload: train/bike_s_000682.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000682.png
    upload: train/bike_s_000855.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000855.png
    upload: train/bike_s_000926.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000926.png
    upload: train/bike_s_000722.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000722.png
    upload: train/bike_s_000657.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000657.png
    upload: train/bike_s_001027.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001027.png
    upload: train/bike_s_001078.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001078.png
    upload: train/bike_s_000990.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000990.png
    upload: train/bike_s_000934.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_000934.png
    upload: train/bike_s_001113.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001113.png
    upload: train/bike_s_001116.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001116.png
    upload: train/bike_s_001131.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001131.png
    upload: train/bike_s_001093.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001093.png
    upload: train/bike_s_001320.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001320.png
    upload: train/bike_s_001072.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001072.png
    upload: train/bike_s_001260.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001260.png
    upload: train/bike_s_001200.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001200.png
    upload: train/bike_s_001226.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001226.png
    upload: train/bike_s_001380.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001380.png
    upload: train/bike_s_001414.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001414.png
    upload: train/bike_s_001375.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001375.png
    upload: train/bike_s_001415.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001415.png
    upload: train/bike_s_001572.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001572.png
    upload: train/bike_s_001613.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001613.png
    upload: train/bike_s_001517.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001517.png
    upload: train/bike_s_001519.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001519.png
    upload: train/bike_s_001592.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001592.png
    upload: train/bike_s_001418.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001418.png
    upload: train/bike_s_001462.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001462.png
    upload: train/bike_s_001679.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001679.png
    upload: train/bike_s_001739.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001739.png
    upload: train/bike_s_001683.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001683.png
    upload: train/bike_s_001761.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001761.png
    upload: train/bike_s_001839.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001839.png
    upload: train/bike_s_001847.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001847.png
    upload: train/bike_s_001882.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001882.png
    upload: train/bike_s_001876.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001876.png
    upload: train/bike_s_001877.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001877.png
    upload: train/bike_s_001897.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001897.png
    upload: train/bike_s_001767.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001767.png
    upload: train/bike_s_001827.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001827.png
    upload: train/bike_s_001915.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001915.png
    upload: train/bike_s_001928.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001928.png
    upload: train/bike_s_002033.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002033.png
    upload: train/bike_s_002042.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002042.png
    upload: train/bike_s_002098.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002098.png
    upload: train/bike_s_001945.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001945.png
    upload: train/bike_s_002024.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002024.png
    upload: train/bike_s_001980.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_001980.png
    upload: train/bike_s_002047.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002047.png
    upload: train/bike_s_002090.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002090.png
    upload: train/bike_s_002109.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002109.png
    upload: train/bike_s_002139.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002139.png
    upload: train/bike_s_002164.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002164.png
    upload: train/bike_s_002215.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002215.png
    upload: train/bike_s_002222.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002222.png
    upload: train/bike_s_002118.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002118.png
    upload: train/bike_s_002203.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002203.png
    upload: train/bike_s_002288.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002288.png
    upload: train/bike_s_002283.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002283.png
    upload: train/bike_s_002277.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002277.png
    upload: train/bike_s_002292.png to s3://sagemaker-us-east-1-761239682643/train/bike_s_002292.png
    upload: train/cycle_s_000042.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000042.png
    upload: train/cycle_s_000463.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000463.png
    upload: train/cycle_s_000222.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000222.png
    upload: train/cycle_s_000492.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000492.png
    upload: train/cycle_s_000639.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000639.png
    upload: train/cycle_s_000318.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000318.png
    upload: train/cycle_s_000583.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000583.png
    upload: train/cycle_s_000871.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000871.png
    upload: train/cycle_s_000718.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000718.png
    upload: train/cycle_s_000666.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000666.png
    upload: train/cycle_s_001309.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001309.png
    upload: train/cycle_s_000899.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_000899.png
    upload: train/cycle_s_001286.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001286.png
    upload: train/cycle_s_001474.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001474.png
    upload: train/cycle_s_001439.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001439.png
    upload: train/cycle_s_001472.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001472.png
    upload: train/cycle_s_001412.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001412.png
    upload: train/cycle_s_001374.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001374.png
    upload: train/cycle_s_001640.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001640.png
    upload: train/cycle_s_001477.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001477.png
    upload: train/cycle_s_001735.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001735.png
    upload: train/cycle_s_001745.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001745.png
    upload: train/cycle_s_002015.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002015.png
    upload: train/cycle_s_001976.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001976.png
    upload: train/cycle_s_001413.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001413.png
    upload: train/cycle_s_001875.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_001875.png
    upload: train/cycle_s_002053.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002053.png
    upload: train/cycle_s_002093.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002093.png
    upload: train/cycle_s_002168.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002168.png
    upload: train/cycle_s_002092.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002092.png
    upload: train/cycle_s_002090.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002090.png
    upload: train/cycle_s_002598.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002598.png
    upload: train/cycle_s_002399.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002399.png
    upload: train/cycle_s_002638.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002638.png
    upload: train/cycle_s_002651.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002651.png
    upload: train/cycle_s_002503.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002503.png
    upload: train/cycle_s_002178.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002178.png
    upload: train/cycle_s_002779.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002779.png
    upload: train/cycle_s_002659.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002659.png
    upload: train/cycle_s_002703.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002703.png
    upload: train/cycle_s_002746.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002746.png
    upload: train/cycle_s_002666.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002666.png
    upload: train/cycle_s_002882.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002882.png
    upload: train/cycle_s_002844.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002844.png
    upload: train/cycle_s_002978.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002978.png
    upload: train/cycle_s_002904.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_002904.png
    upload: train/cycle_s_003043.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003043.png
    upload: train/cycle_s_003026.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003026.png
    upload: train/cycle_s_003008.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003008.png
    upload: train/cycle_s_003006.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003006.png
    upload: train/cycle_s_003122.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003122.png
    upload: train/cycle_s_003147.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003147.png
    upload: train/cycle_s_003148.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003148.png
    upload: train/dirt_bike_s_000003.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000003.png
    upload: train/cycle_s_003162.png to s3://sagemaker-us-east-1-761239682643/train/cycle_s_003162.png
    upload: train/dirt_bike_s_000059.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000059.png
    upload: train/dirt_bike_s_000005.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000005.png
    upload: train/dirt_bike_s_000030.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000030.png
    upload: train/dirt_bike_s_000119.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000119.png
    upload: train/dirt_bike_s_000134.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000134.png
    upload: train/dirt_bike_s_000017.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000017.png
    upload: train/dirt_bike_s_000124.png to s3://sagemaker-us-east-1-761239682643/train/dirt_bike_s_000124.png
    upload: train/minibike_s_000022.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000022.png
    upload: train/minibike_s_000011.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000011.png
    upload: train/minibike_s_000010.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000010.png
    upload: train/minibike_s_000036.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000036.png
    upload: train/minibike_s_000020.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000020.png
    upload: train/minibike_s_000102.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000102.png
    upload: train/minibike_s_000099.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000099.png
    upload: train/minibike_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000035.png
    upload: train/minibike_s_000108.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000108.png
    upload: train/minibike_s_000064.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000064.png
    upload: train/minibike_s_000116.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000116.png
    upload: train/minibike_s_000110.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000110.png
    upload: train/minibike_s_000149.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000149.png
    upload: train/minibike_s_000121.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000121.png
    upload: train/minibike_s_000117.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000117.png
    upload: train/minibike_s_000127.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000127.png
    upload: train/minibike_s_000146.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000146.png
    upload: train/minibike_s_000165.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000165.png
    upload: train/minibike_s_000130.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000130.png
    upload: train/minibike_s_000293.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000293.png
    upload: train/minibike_s_000314.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000314.png
    upload: train/minibike_s_000244.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000244.png
    upload: train/minibike_s_000203.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000203.png
    upload: train/minibike_s_000218.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000218.png
    upload: train/minibike_s_000324.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000324.png
    upload: train/minibike_s_000335.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000335.png
    upload: train/minibike_s_000340.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000340.png
    upload: train/minibike_s_000362.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000362.png
    upload: train/minibike_s_000350.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000350.png
    upload: train/minibike_s_000408.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000408.png
    upload: train/minibike_s_000392.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000392.png
    upload: train/minibike_s_000401.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000401.png
    upload: train/minibike_s_000429.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000429.png
    upload: train/minibike_s_000409.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000409.png
    upload: train/minibike_s_000402.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000402.png
    upload: train/minibike_s_000427.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000427.png
    upload: train/minibike_s_000459.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000459.png
    upload: train/minibike_s_000434.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000434.png
    upload: train/minibike_s_000485.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000485.png
    upload: train/minibike_s_000451.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000451.png
    upload: train/minibike_s_000435.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000435.png
    upload: train/minibike_s_000491.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000491.png
    upload: train/minibike_s_000498.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000498.png
    upload: train/minibike_s_000511.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000511.png
    upload: train/minibike_s_000520.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000520.png
    upload: train/minibike_s_000518.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000518.png
    upload: train/minibike_s_000519.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000519.png
    upload: train/minibike_s_000526.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000526.png
    upload: train/minibike_s_000507.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000507.png
    upload: train/minibike_s_000527.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000527.png
    upload: train/minibike_s_000565.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000565.png
    upload: train/minibike_s_000522.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000522.png
    upload: train/minibike_s_000566.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000566.png
    upload: train/minibike_s_000567.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000567.png
    upload: train/minibike_s_000652.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000652.png
    upload: train/minibike_s_000569.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000569.png
    upload: train/minibike_s_000654.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000654.png
    upload: train/minibike_s_000568.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000568.png
    upload: train/minibike_s_000570.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000570.png
    upload: train/minibike_s_000613.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000613.png
    upload: train/minibike_s_000571.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000571.png
    upload: train/minibike_s_000701.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000701.png
    upload: train/minibike_s_000690.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000690.png
    upload: train/minibike_s_000738.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000738.png
    upload: train/minibike_s_000698.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000698.png
    upload: train/minibike_s_000741.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000741.png
    upload: train/minibike_s_000709.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000709.png
    upload: train/minibike_s_000800.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000800.png
    upload: train/minibike_s_000743.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000743.png
    upload: train/minibike_s_000813.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000813.png
    upload: train/minibike_s_000802.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000802.png
    upload: train/minibike_s_000855.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000855.png
    upload: train/minibike_s_000886.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000886.png
    upload: train/minibike_s_000820.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000820.png
    upload: train/minibike_s_000824.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000824.png
    upload: train/minibike_s_000914.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000914.png
    upload: train/minibike_s_000831.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000831.png
    upload: train/minibike_s_000830.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000830.png
    upload: train/minibike_s_000885.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000885.png
    upload: train/minibike_s_000967.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000967.png
    upload: train/minibike_s_000906.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000906.png
    upload: train/minibike_s_000968.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_000968.png
    upload: train/minibike_s_001094.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001094.png
    upload: train/minibike_s_001072.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001072.png
    upload: train/minibike_s_001017.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001017.png
    upload: train/minibike_s_001168.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001168.png
    upload: train/minibike_s_001016.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001016.png
    upload: train/minibike_s_001177.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001177.png
    upload: train/minibike_s_001079.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001079.png
    upload: train/minibike_s_001157.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001157.png
    upload: train/minibike_s_001169.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001169.png
    upload: train/minibike_s_001179.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001179.png
    upload: train/minibike_s_001250.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001250.png
    upload: train/minibike_s_001249.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001249.png
    upload: train/minibike_s_001191.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001191.png
    upload: train/minibike_s_001261.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001261.png
    upload: train/minibike_s_001193.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001193.png
    upload: train/minibike_s_001275.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001275.png
    upload: train/minibike_s_001185.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001185.png
    upload: train/minibike_s_001270.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001270.png
    upload: train/minibike_s_001294.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001294.png
    upload: train/minibike_s_001279.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001279.png
    upload: train/minibike_s_001450.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001450.png
    upload: train/minibike_s_001417.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001417.png
    upload: train/minibike_s_001348.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001348.png
    upload: train/minibike_s_001345.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001345.png
    upload: train/minibike_s_001344.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001344.png
    upload: train/minibike_s_001366.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001366.png
    upload: train/minibike_s_001479.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001479.png
    upload: train/minibike_s_001458.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001458.png
    upload: train/minibike_s_001496.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001496.png
    upload: train/minibike_s_001491.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001491.png
    upload: train/minibike_s_001512.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001512.png
    upload: train/minibike_s_001511.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001511.png
    upload: train/minibike_s_001506.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001506.png
    upload: train/minibike_s_001558.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001558.png
    upload: train/minibike_s_001540.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001540.png
    upload: train/minibike_s_001565.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001565.png
    upload: train/minibike_s_001546.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001546.png
    upload: train/minibike_s_001539.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001539.png
    upload: train/minibike_s_001498.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001498.png
    upload: train/minibike_s_001573.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001573.png
    upload: train/minibike_s_001597.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001597.png
    upload: train/minibike_s_001575.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001575.png
    upload: train/minibike_s_001639.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001639.png
    upload: train/minibike_s_001701.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001701.png
    upload: train/minibike_s_001638.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001638.png
    upload: train/minibike_s_001731.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001731.png
    upload: train/minibike_s_001689.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001689.png
    upload: train/minibike_s_001691.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001691.png
    upload: train/minibike_s_001653.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001653.png
    upload: train/minibike_s_001747.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001747.png
    upload: train/minibike_s_001734.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001734.png
    upload: train/minibike_s_001631.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001631.png
    upload: train/minibike_s_001767.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001767.png
    upload: train/minibike_s_001838.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001838.png
    upload: train/minibike_s_001870.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001870.png
    upload: train/minibike_s_001836.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001836.png
    upload: train/minibike_s_001789.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001789.png
    upload: train/minibike_s_001827.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001827.png
    upload: train/minibike_s_001863.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001863.png
    upload: train/minibike_s_001865.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001865.png
    upload: train/minibike_s_001873.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001873.png
    upload: train/minibike_s_001771.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001771.png
    upload: train/minibike_s_001829.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001829.png
    upload: train/minibike_s_001881.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001881.png
    upload: train/minibike_s_001969.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001969.png
    upload: train/minibike_s_001917.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001917.png
    upload: train/minibike_s_001972.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001972.png
    upload: train/minibike_s_001880.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001880.png
    upload: train/minibike_s_001944.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001944.png
    upload: train/minibike_s_002009.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002009.png
    upload: train/minibike_s_001996.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001996.png
    upload: train/minibike_s_001921.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001921.png
    upload: train/minibike_s_001885.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_001885.png
    upload: train/minibike_s_002038.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002038.png
    upload: train/minibike_s_002030.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002030.png
    upload: train/minibike_s_002115.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002115.png
    upload: train/minibike_s_002084.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002084.png
    upload: train/minibike_s_002124.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002124.png
    upload: train/minibike_s_002046.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002046.png
    upload: train/minibike_s_002048.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002048.png
    upload: train/minibike_s_002146.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002146.png
    upload: train/minibike_s_002137.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002137.png
    upload: train/minibike_s_002163.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002163.png
    upload: train/minibike_s_002182.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002182.png
    upload: train/minibike_s_002130.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002130.png
    upload: train/minibike_s_002194.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002194.png
    upload: train/minibike_s_002198.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002198.png
    upload: train/minibike_s_002176.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002176.png
    upload: train/minibike_s_002196.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002196.png
    upload: train/minibike_s_002218.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002218.png
    upload: train/minibike_s_002186.png to s3://sagemaker-us-east-1-761239682643/train/minibike_s_002186.png
    upload: train/moped_s_000021.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000021.png
    upload: train/moped_s_000004.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000004.png
    upload: train/moped_s_000009.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000009.png
    upload: train/moped_s_000034.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000034.png
    upload: train/moped_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000035.png
    upload: train/moped_s_000030.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000030.png
    upload: train/moped_s_000044.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000044.png
    upload: train/moped_s_000065.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000065.png
    upload: train/moped_s_000119.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000119.png
    upload: train/moped_s_000071.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000071.png
    upload: train/moped_s_000168.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000168.png
    upload: train/moped_s_000124.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000124.png
    upload: train/motorbike_s_000009.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000009.png
    upload: train/moped_s_000135.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000135.png
    upload: train/moped_s_000169.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000169.png
    upload: train/motorbike_s_000021.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000021.png
    upload: train/moped_s_000237.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000237.png
    upload: train/motorbike_s_000022.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000022.png
    upload: train/motorbike_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000035.png
    upload: train/motorbike_s_000041.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000041.png
    upload: train/moped_s_000236.png to s3://sagemaker-us-east-1-761239682643/train/moped_s_000236.png
    upload: train/motorbike_s_000106.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000106.png
    upload: train/motorbike_s_000062.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000062.png
    upload: train/motorbike_s_000060.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000060.png
    upload: train/motorbike_s_000115.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000115.png
    upload: train/motorbike_s_000068.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000068.png
    upload: train/motorbike_s_000058.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000058.png
    upload: train/motorbike_s_000117.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000117.png
    upload: train/motorbike_s_000119.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000119.png
    upload: train/motorbike_s_000223.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000223.png
    upload: train/motorbike_s_000124.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000124.png
    upload: train/motorbike_s_000172.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000172.png
    upload: train/motorbike_s_000134.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000134.png
    upload: train/motorbike_s_000141.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000141.png
    upload: train/motorbike_s_000254.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000254.png
    upload: train/motorbike_s_000225.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000225.png
    upload: train/motorbike_s_000221.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000221.png
    upload: train/motorbike_s_000331.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000331.png
    upload: train/motorbike_s_000308.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000308.png
    upload: train/motorbike_s_000427.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000427.png
    upload: train/motorbike_s_000449.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000449.png
    upload: train/motorbike_s_000501.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000501.png
    upload: train/motorbike_s_000534.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000534.png
    upload: train/motorbike_s_000361.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000361.png
    upload: train/motorbike_s_000578.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000578.png
    upload: train/motorbike_s_000346.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000346.png
    upload: train/motorbike_s_000463.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000463.png
    upload: train/motorcycle_s_000002.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000002.png
    upload: train/motorbike_s_000362.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000362.png
    upload: train/motorcycle_s_000001.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000001.png
    upload: train/motorbike_s_000541.png to s3://sagemaker-us-east-1-761239682643/train/motorbike_s_000541.png
    upload: train/motorcycle_s_000025.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000025.png
    upload: train/motorcycle_s_000003.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000003.png
    upload: train/motorcycle_s_000009.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000009.png
    upload: train/motorcycle_s_000004.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000004.png
    upload: train/motorcycle_s_000027.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000027.png
    upload: train/motorcycle_s_000022.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000022.png
    upload: train/motorcycle_s_000029.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000029.png
    upload: train/motorcycle_s_000026.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000026.png
    upload: train/motorcycle_s_000050.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000050.png
    upload: train/motorcycle_s_000070.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000070.png
    upload: train/motorcycle_s_000126.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000126.png
    upload: train/motorcycle_s_000032.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000032.png
    upload: train/motorcycle_s_000072.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000072.png
    upload: train/motorcycle_s_000076.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000076.png
    upload: train/motorcycle_s_000133.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000133.png
    upload: train/motorcycle_s_000120.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000120.png
    upload: train/motorcycle_s_000144.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000144.png
    upload: train/motorcycle_s_000127.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000127.png
    upload: train/motorcycle_s_000074.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000074.png
    upload: train/motorcycle_s_000136.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000136.png
    upload: train/motorcycle_s_000151.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000151.png
    upload: train/motorcycle_s_000146.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000146.png
    upload: train/motorcycle_s_000154.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000154.png
    upload: train/motorcycle_s_000170.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000170.png
    upload: train/motorcycle_s_000207.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000207.png
    upload: train/motorcycle_s_000169.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000169.png
    upload: train/motorcycle_s_000219.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000219.png
    upload: train/motorcycle_s_000167.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000167.png
    upload: train/motorcycle_s_000221.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000221.png
    upload: train/motorcycle_s_000220.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000220.png
    upload: train/motorcycle_s_000242.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000242.png
    upload: train/motorcycle_s_000217.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000217.png
    upload: train/motorcycle_s_000216.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000216.png
    upload: train/motorcycle_s_000246.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000246.png
    upload: train/motorcycle_s_000245.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000245.png
    upload: train/motorcycle_s_000249.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000249.png
    upload: train/motorcycle_s_000223.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000223.png
    upload: train/motorcycle_s_000253.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000253.png
    upload: train/motorcycle_s_000277.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000277.png
    upload: train/motorcycle_s_000256.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000256.png
    upload: train/motorcycle_s_000252.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000252.png
    upload: train/motorcycle_s_000284.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000284.png
    upload: train/motorcycle_s_000309.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000309.png
    upload: train/motorcycle_s_000290.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000290.png
    upload: train/motorcycle_s_000319.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000319.png
    upload: train/motorcycle_s_000300.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000300.png
    upload: train/motorcycle_s_000304.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000304.png
    upload: train/motorcycle_s_000312.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000312.png
    upload: train/motorcycle_s_000262.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000262.png
    upload: train/motorcycle_s_000291.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000291.png
    upload: train/motorcycle_s_000325.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000325.png
    upload: train/motorcycle_s_000320.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000320.png
    upload: train/motorcycle_s_000332.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000332.png
    upload: train/motorcycle_s_000339.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000339.png
    upload: train/motorcycle_s_000407.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000407.png
    upload: train/motorcycle_s_000340.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000340.png
    upload: train/motorcycle_s_000417.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000417.png
    upload: train/motorcycle_s_000353.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000353.png
    upload: train/motorcycle_s_000347.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000347.png
    upload: train/motorcycle_s_000422.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000422.png
    upload: train/motorcycle_s_000418.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000418.png
    upload: train/motorcycle_s_000346.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000346.png
    upload: train/motorcycle_s_000423.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000423.png
    upload: train/motorcycle_s_000432.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000432.png
    upload: train/motorcycle_s_000451.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000451.png
    upload: train/motorcycle_s_000459.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000459.png
    upload: train/motorcycle_s_000440.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000440.png
    upload: train/motorcycle_s_000447.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000447.png
    upload: train/motorcycle_s_000456.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000456.png
    upload: train/motorcycle_s_000454.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000454.png
    upload: train/motorcycle_s_000431.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000431.png
    upload: train/motorcycle_s_000486.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000486.png
    upload: train/motorcycle_s_000496.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000496.png
    upload: train/motorcycle_s_000430.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000430.png
    upload: train/motorcycle_s_000542.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000542.png
    upload: train/motorcycle_s_000543.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000543.png
    upload: train/motorcycle_s_000508.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000508.png
    upload: train/motorcycle_s_000517.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000517.png
    upload: train/motorcycle_s_000521.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000521.png
    upload: train/motorcycle_s_000580.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000580.png
    upload: train/motorcycle_s_000594.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000594.png
    upload: train/motorcycle_s_000545.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000545.png
    upload: train/motorcycle_s_000593.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000593.png
    upload: train/motorcycle_s_000629.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000629.png
    upload: train/motorcycle_s_000605.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000605.png
    upload: train/motorcycle_s_000617.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000617.png
    upload: train/motorcycle_s_000622.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000622.png
    upload: train/motorcycle_s_000669.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000669.png
    upload: train/motorcycle_s_000585.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000585.png
    upload: train/motorcycle_s_000687.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000687.png
    upload: train/motorcycle_s_000686.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000686.png
    upload: train/motorcycle_s_000654.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000654.png
    upload: train/motorcycle_s_000698.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000698.png
    upload: train/motorcycle_s_000699.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000699.png
    upload: train/motorcycle_s_000695.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000695.png
    upload: train/motorcycle_s_000696.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000696.png
    upload: train/motorcycle_s_000700.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000700.png
    upload: train/motorcycle_s_000713.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000713.png
    upload: train/motorcycle_s_000724.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000724.png
    upload: train/motorcycle_s_000734.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000734.png
    upload: train/motorcycle_s_000772.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000772.png
    upload: train/motorcycle_s_000732.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000732.png
    upload: train/motorcycle_s_000776.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000776.png
    upload: train/motorcycle_s_000785.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000785.png
    upload: train/motorcycle_s_000741.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000741.png
    upload: train/motorcycle_s_000714.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000714.png
    upload: train/motorcycle_s_000790.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000790.png
    upload: train/motorcycle_s_000787.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000787.png
    upload: train/motorcycle_s_000791.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000791.png
    upload: train/motorcycle_s_000834.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000834.png
    upload: train/motorcycle_s_000796.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000796.png
    upload: train/motorcycle_s_000887.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000887.png
    upload: train/motorcycle_s_000922.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000922.png
    upload: train/motorcycle_s_000792.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000792.png
    upload: train/motorcycle_s_000894.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000894.png
    upload: train/motorcycle_s_000807.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000807.png
    upload: train/motorcycle_s_000923.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000923.png
    upload: train/motorcycle_s_000925.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000925.png
    upload: train/motorcycle_s_000919.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000919.png
    upload: train/motorcycle_s_000924.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000924.png
    upload: train/motorcycle_s_000928.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000928.png
    upload: train/motorcycle_s_000933.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000933.png
    upload: train/motorcycle_s_000938.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000938.png
    upload: train/motorcycle_s_000917.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000917.png
    upload: train/motorcycle_s_000941.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000941.png
    upload: train/motorcycle_s_000979.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000979.png
    upload: train/motorcycle_s_001012.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001012.png
    upload: train/motorcycle_s_000989.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000989.png
    upload: train/motorcycle_s_001017.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001017.png
    upload: train/motorcycle_s_000978.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000978.png
    upload: train/motorcycle_s_000991.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_000991.png
    upload: train/motorcycle_s_001063.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001063.png
    upload: train/motorcycle_s_001016.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001016.png
    upload: train/motorcycle_s_001033.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001033.png
    upload: train/motorcycle_s_001027.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001027.png
    upload: train/motorcycle_s_001064.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001064.png
    upload: train/motorcycle_s_001118.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001118.png
    upload: train/motorcycle_s_001126.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001126.png
    upload: train/motorcycle_s_001119.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001119.png
    upload: train/motorcycle_s_001180.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001180.png
    upload: train/motorcycle_s_001167.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001167.png
    upload: train/motorcycle_s_001182.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001182.png
    upload: train/motorcycle_s_001106.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001106.png
    upload: train/motorcycle_s_001181.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001181.png
    upload: train/motorcycle_s_001183.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001183.png
    upload: train/motorcycle_s_001196.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001196.png
    upload: train/motorcycle_s_001215.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001215.png
    upload: train/motorcycle_s_001197.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001197.png
    upload: train/motorcycle_s_001237.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001237.png
    upload: train/motorcycle_s_001176.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001176.png
    upload: train/motorcycle_s_001213.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001213.png
    upload: train/motorcycle_s_001205.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001205.png
    upload: train/motorcycle_s_001236.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001236.png
    upload: train/motorcycle_s_001209.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001209.png
    upload: train/motorcycle_s_001242.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001242.png
    upload: train/motorcycle_s_001247.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001247.png
    upload: train/motorcycle_s_001238.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001238.png
    upload: train/motorcycle_s_001220.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001220.png
    upload: train/motorcycle_s_001246.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001246.png
    upload: train/motorcycle_s_001263.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001263.png
    upload: train/motorcycle_s_001273.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001273.png
    upload: train/motorcycle_s_001244.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001244.png
    upload: train/motorcycle_s_001297.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001297.png
    upload: train/motorcycle_s_001287.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001287.png
    upload: train/motorcycle_s_001303.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001303.png
    upload: train/motorcycle_s_001326.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001326.png
    upload: train/motorcycle_s_001363.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001363.png
    upload: train/motorcycle_s_001370.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001370.png
    upload: train/motorcycle_s_001368.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001368.png
    upload: train/motorcycle_s_001360.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001360.png
    upload: train/motorcycle_s_001315.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001315.png
    upload: train/motorcycle_s_001337.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001337.png
    upload: train/motorcycle_s_001319.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001319.png
    upload: train/motorcycle_s_001407.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001407.png
    upload: train/motorcycle_s_001400.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001400.png
    upload: train/motorcycle_s_001409.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001409.png
    upload: train/motorcycle_s_001384.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001384.png
    upload: train/motorcycle_s_001397.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001397.png
    upload: train/motorcycle_s_001413.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001413.png
    upload: train/motorcycle_s_001402.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001402.png
    upload: train/motorcycle_s_001438.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001438.png
    upload: train/motorcycle_s_001439.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001439.png
    upload: train/motorcycle_s_001435.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001435.png
    upload: train/motorcycle_s_001507.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001507.png
    upload: train/motorcycle_s_001453.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001453.png
    upload: train/motorcycle_s_001520.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001520.png
    upload: train/motorcycle_s_001536.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001536.png
    upload: train/motorcycle_s_001549.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001549.png
    upload: train/motorcycle_s_001492.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001492.png
    upload: train/motorcycle_s_001527.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001527.png
    upload: train/motorcycle_s_001563.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001563.png
    upload: train/motorcycle_s_001584.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001584.png
    upload: train/motorcycle_s_001565.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001565.png
    upload: train/motorcycle_s_001587.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001587.png
    upload: train/motorcycle_s_001580.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001580.png
    upload: train/motorcycle_s_001585.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001585.png
    upload: train/motorcycle_s_001610.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001610.png
    upload: train/motorcycle_s_001392.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001392.png
    upload: train/motorcycle_s_001622.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001622.png
    upload: train/motorcycle_s_001611.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001611.png
    upload: train/motorcycle_s_001641.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001641.png
    upload: train/motorcycle_s_001693.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001693.png
    upload: train/motorcycle_s_001690.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001690.png
    upload: train/motorcycle_s_001686.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001686.png
    upload: train/motorcycle_s_001623.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001623.png
    upload: train/motorcycle_s_001695.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001695.png
    upload: train/motorcycle_s_001696.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001696.png
    upload: train/motorcycle_s_001699.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001699.png
    upload: train/motorcycle_s_001709.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001709.png
    upload: train/motorcycle_s_001706.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001706.png
    upload: train/motorcycle_s_001707.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001707.png
    upload: train/motorcycle_s_001710.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001710.png
    upload: train/motorcycle_s_001711.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001711.png
    upload: train/motorcycle_s_001714.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001714.png
    upload: train/motorcycle_s_001753.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001753.png
    upload: train/motorcycle_s_001778.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001778.png
    upload: train/motorcycle_s_001767.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001767.png
    upload: train/motorcycle_s_001765.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001765.png
    upload: train/motorcycle_s_001715.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001715.png
    upload: train/motorcycle_s_001790.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001790.png
    upload: train/motorcycle_s_001792.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001792.png
    upload: train/motorcycle_s_001856.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001856.png
    upload: train/motorcycle_s_001848.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001848.png
    upload: train/motorcycle_s_001853.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001853.png
    upload: train/motorcycle_s_001896.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001896.png
    upload: train/motorcycle_s_001878.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001878.png
    upload: train/motorcycle_s_001862.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001862.png
    upload: train/motorcycle_s_001905.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001905.png
    upload: train/motorcycle_s_001784.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001784.png
    upload: train/motorcycle_s_001973.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001973.png
    upload: train/motorcycle_s_002030.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002030.png
    upload: train/motorcycle_s_001920.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_001920.png
    upload: train/motorcycle_s_002065.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002065.png
    upload: train/motorcycle_s_002031.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002031.png
    upload: train/motorcycle_s_002121.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002121.png
    upload: train/motorcycle_s_002105.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002105.png
    upload: train/motorcycle_s_002162.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002162.png
    upload: train/motorcycle_s_002197.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002197.png
    upload: train/motorcycle_s_002153.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002153.png
    upload: train/motorcycle_s_002193.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002193.png
    upload: train/motorcycle_s_002143.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002143.png
    upload: train/motorcycle_s_002066.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002066.png
    upload: train/motorcycle_s_002183.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002183.png
    upload: train/motorcycle_s_002140.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002140.png
    upload: train/motorcycle_s_002192.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002192.png
    upload: train/motorcycle_s_002067.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002067.png
    upload: train/motorcycle_s_002225.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002225.png
    upload: train/motorcycle_s_002214.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002214.png
    upload: train/motorcycle_s_002221.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002221.png
    upload: train/motorcycle_s_002236.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002236.png
    upload: train/motorcycle_s_002215.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002215.png
    upload: train/motorcycle_s_002222.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002222.png
    upload: train/motorcycle_s_002237.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002237.png
    upload: train/motorcycle_s_002234.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002234.png
    upload: train/motorcycle_s_002254.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002254.png
    upload: train/motorcycle_s_002275.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002275.png
    upload: train/motorcycle_s_002291.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002291.png
    upload: train/motorcycle_s_002316.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002316.png
    upload: train/motorcycle_s_002317.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002317.png
    upload: train/ordinary_bicycle_s_000008.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000008.png
    upload: train/ordinary_bicycle_s_000016.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000016.png
    upload: train/motorcycle_s_002298.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002298.png
    upload: train/ordinary_bicycle_s_000011.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000011.png
    upload: train/ordinary_bicycle_s_000023.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000023.png
    upload: train/motorcycle_s_002271.png to s3://sagemaker-us-east-1-761239682643/train/motorcycle_s_002271.png
    upload: train/ordinary_bicycle_s_000035.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000035.png
    upload: train/ordinary_bicycle_s_000049.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000049.png
    upload: train/ordinary_bicycle_s_000056.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000056.png
    upload: train/ordinary_bicycle_s_000036.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000036.png
    upload: train/ordinary_bicycle_s_000066.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000066.png
    upload: train/ordinary_bicycle_s_000029.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000029.png
    upload: train/ordinary_bicycle_s_000022.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000022.png
    upload: train/ordinary_bicycle_s_000031.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000031.png
    upload: train/ordinary_bicycle_s_000060.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000060.png
    upload: train/ordinary_bicycle_s_000098.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000098.png
    upload: train/ordinary_bicycle_s_000096.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000096.png
    upload: train/ordinary_bicycle_s_000097.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000097.png
    upload: train/ordinary_bicycle_s_000100.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000100.png
    upload: train/ordinary_bicycle_s_000095.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000095.png
    upload: train/ordinary_bicycle_s_000104.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000104.png
    upload: train/ordinary_bicycle_s_000101.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000101.png
    upload: train/ordinary_bicycle_s_000099.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000099.png
    upload: train/ordinary_bicycle_s_000102.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000102.png
    upload: train/ordinary_bicycle_s_000107.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000107.png
    upload: train/ordinary_bicycle_s_000154.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000154.png
    upload: train/ordinary_bicycle_s_000110.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000110.png
    upload: train/ordinary_bicycle_s_000112.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000112.png
    upload: train/ordinary_bicycle_s_000161.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000161.png
    upload: train/ordinary_bicycle_s_000157.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000157.png
    upload: train/ordinary_bicycle_s_000125.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000125.png
    upload: train/ordinary_bicycle_s_000155.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000155.png
    upload: train/ordinary_bicycle_s_000167.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000167.png
    upload: train/ordinary_bicycle_s_000205.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000205.png
    upload: train/ordinary_bicycle_s_000218.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000218.png
    upload: train/ordinary_bicycle_s_000216.png to s3://sagemaker-us-east-1-761239ng...)682643/train/ordinary_bicycle_s_000216.png
    upload: train/ordinary_bicycle_s_000265.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000265.png
    upload: train/ordinary_bicycle_s_000269.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000269.png
    upload: train/ordinary_bicycle_s_000201.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000201.png
    upload: train/ordinary_bicycle_s_000274.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000274.png
    upload: train/ordinary_bicycle_s_000280.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000280.png
    upload: train/ordinary_bicycle_s_000277.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000277.png
    upload: train/ordinary_bicycle_s_000298.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000298.png
    upload: train/ordinary_bicycle_s_000308.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000308.png
    upload: train/ordinary_bicycle_s_000347.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000347.png
    upload: train/ordinary_bicycle_s_000297.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000297.png
    upload: train/ordinary_bicycle_s_000324.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000324.png
    upload: train/ordinary_bicycle_s_000419.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000419.png
    upload: train/ordinary_bicycle_s_000355.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000355.png
    upload: train/ordinary_bicycle_s_000387.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000387.png
    upload: train/ordinary_bicycle_s_000451.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000451.png
    upload: train/ordinary_bicycle_s_000433.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000433.png
    upload: train/ordinary_bicycle_s_000426.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000426.png
    upload: train/ordinary_bicycle_s_000286.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000286.png
    upload: train/ordinary_bicycle_s_000432.png to s3://sagemaker-us-east-1-761239682643/train/ordinary_bicycle_s_000432.png
    upload: train/safety_bicycle_s_000079.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000079.png
    upload: train/safety_bicycle_s_000140.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000140.png
    upload: train/safety_bicycle_s_000092.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000092.png
    upload: train/safety_bicycle_s_000125.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000125.png
    upload: train/safety_bicycle_s_000167.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000167.png
    upload: train/safety_bicycle_s_000207.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000207.png
    upload: train/safety_bicycle_s_000193.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000193.png
    upload: train/safety_bicycle_s_000232.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000232.png
    upload: train/safety_bicycle_s_000019.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000019.png
    upload: train/safety_bicycle_s_000162.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000162.png
    upload: train/safety_bicycle_s_000255.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000255.png
    upload: train/safety_bicycle_s_000233.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000233.png
    upload: train/safety_bicycle_s_000239.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000239.png
    upload: train/safety_bicycle_s_000261.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000261.png
    upload: train/safety_bicycle_s_000296.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000296.png
    upload: train/safety_bicycle_s_000324.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000324.png
    upload: train/safety_bicycle_s_000303.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000303.png
    upload: train/safety_bicycle_s_000322.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000322.png
    upload: train/safety_bicycle_s_000373.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000373.png
    upload: train/safety_bicycle_s_000425.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000425.png
    upload: train/safety_bicycle_s_000348.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000348.png
    upload: train/safety_bicycle_s_000196.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000196.png
    upload: train/safety_bicycle_s_000427.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000427.png
    upload: train/safety_bicycle_s_000359.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000359.png
    upload: train/safety_bicycle_s_000532.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000532.png
    upload: train/safety_bicycle_s_000500.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000500.png
    upload: train/safety_bicycle_s_000533.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000533.png
    upload: train/safety_bicycle_s_000568.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000568.png
    upload: train/safety_bicycle_s_000660.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000660.png
    upload: train/safety_bicycle_s_000760.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000760.png
    upload: train/safety_bicycle_s_000655.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000655.png
    upload: train/safety_bicycle_s_000789.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000789.png
    upload: train/safety_bicycle_s_000728.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000728.png
    upload: train/safety_bicycle_s_000860.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_000860.png
    upload: train/safety_bicycle_s_001028.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001028.png
    upload: train/safety_bicycle_s_001029.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001029.png
    upload: train/safety_bicycle_s_001049.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001049.png
    upload: train/safety_bicycle_s_001085.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001085.png
    upload: train/safety_bicycle_s_001026.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001026.png
    upload: train/safety_bicycle_s_001113.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001113.png
    upload: train/safety_bicycle_s_001202.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001202.png
    upload: train/safety_bicycle_s_001109.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001109.png
    upload: train/safety_bicycle_s_001063.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001063.png
    upload: train/safety_bicycle_s_001243.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001243.png
    upload: train/safety_bicycle_s_001240.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001240.png
    upload: train/safety_bicycle_s_001253.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001253.png
    upload: train/safety_bicycle_s_001254.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001254.png
    upload: train/safety_bicycle_s_001303.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001303.png
    upload: train/safety_bicycle_s_001320.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001320.png
    upload: train/safety_bicycle_s_001570.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001570.png
    upload: train/safety_bicycle_s_001327.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001327.png
    upload: train/safety_bicycle_s_001608.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001608.png
    upload: train/safety_bicycle_s_001639.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001639.png
    upload: train/safety_bicycle_s_001381.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001381.png
    upload: train/safety_bicycle_s_001651.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001651.png
    upload: train/safety_bike_s_000015.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000015.png
    upload: train/safety_bicycle_s_001699.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001699.png
    upload: train/safety_bicycle_s_001706.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001706.png
    upload: train/safety_bicycle_s_001659.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001659.png
    upload: train/safety_bicycle_s_001705.png to s3://sagemaker-us-east-1-761239682643/train/safety_bicycle_s_001705.png
    upload: train/safety_bike_s_000054.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000054.png
    upload: train/safety_bike_s_000058.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000058.png
    upload: train/safety_bike_s_000079.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000079.png
    upload: train/safety_bike_s_000104.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000104.png
    upload: train/safety_bike_s_000100.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000100.png
    upload: train/safety_bike_s_000158.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000158.png
    upload: train/safety_bike_s_000155.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000155.png
    upload: train/safety_bike_s_000165.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000165.png
    upload: train/safety_bike_s_000009.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000009.png
    upload: train/safety_bike_s_000173.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000173.png
    upload: train/safety_bike_s_000178.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000178.png
    upload: train/safety_bike_s_000244.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000244.png
    upload: train/safety_bike_s_000160.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000160.png
    upload: train/safety_bike_s_000263.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000263.png
    upload: train/safety_bike_s_000328.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000328.png
    upload: train/safety_bike_s_000277.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000277.png
    upload: train/safety_bike_s_000381.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000381.png
    upload: train/safety_bike_s_000461.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000461.png
    upload: train/safety_bike_s_000198.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000198.png
    upload: train/safety_bike_s_000567.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000567.png
    upload: train/safety_bike_s_000482.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000482.png
    upload: train/safety_bike_s_000311.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000311.png
    upload: train/safety_bike_s_000643.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000643.png
    upload: train/safety_bike_s_000245.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000245.png
    upload: train/safety_bike_s_000830.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000830.png
    upload: train/safety_bike_s_000848.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000848.png
    upload: train/safety_bike_s_000914.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000914.png
    upload: train/safety_bike_s_000963.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000963.png
    upload: train/safety_bike_s_000867.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000867.png
    upload: train/safety_bike_s_000921.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000921.png
    upload: train/safety_bike_s_000984.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000984.png
    upload: train/safety_bike_s_001012.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001012.png
    upload: train/safety_bike_s_000989.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000989.png
    upload: train/safety_bike_s_000934.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000934.png
    upload: train/safety_bike_s_000950.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_000950.png
    upload: train/safety_bike_s_001010.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001010.png
    upload: train/safety_bike_s_001465.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001465.png
    upload: train/safety_bike_s_001257.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001257.png
    upload: train/safety_bike_s_001371.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001371.png
    upload: train/safety_bike_s_001289.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001289.png
    upload: train/safety_bike_s_001338.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001338.png
    upload: train/safety_bike_s_001355.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001355.png
    upload: train/safety_bike_s_001148.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001148.png
    upload: train/safety_bike_s_001472.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001472.png
    upload: train/safety_bike_s_001552.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001552.png
    upload: train/safety_bike_s_001474.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001474.png
    upload: train/safety_bike_s_001640.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001640.png
    upload: train/safety_bike_s_001608.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001608.png
    upload: train/safety_bike_s_001560.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001560.png
    upload: train/safety_bike_s_001590.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001590.png
    upload: train/safety_bike_s_001715.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001715.png
    upload: train/safety_bike_s_001659.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001659.png
    upload: train/safety_bike_s_001784.png to s3://sagemaker-us-east-1-761239682643/train/safety_bike_s_001784.png
    upload: train/trail_bike_s_000016.png to s3://sagemaker-us-east-1-761239682643/train/trail_bike_s_000016.png
    upload: train/velocipede_s_000021.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000021.png
    upload: train/velocipede_s_000010.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000010.png
    upload: train/velocipede_s_000012.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000012.png
    upload: train/velocipede_s_000049.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000049.png
    upload: train/velocipede_s_000023.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000023.png
    upload: train/velocipede_s_000282.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000282.png
    upload: train/velocipede_s_000139.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000139.png
    upload: train/velocipede_s_000358.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000358.png
    upload: train/velocipede_s_000330.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000330.png
    upload: train/velocipede_s_000265.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000265.png
    upload: train/velocipede_s_000586.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000586.png
    upload: train/velocipede_s_000485.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000485.png
    upload: train/velocipede_s_000807.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000807.png
    upload: train/velocipede_s_000670.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000670.png
    upload: train/velocipede_s_000702.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000702.png
    upload: train/velocipede_s_000430.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000430.png
    upload: train/velocipede_s_000931.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000931.png
    upload: train/velocipede_s_000949.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000949.png
    upload: train/velocipede_s_001056.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001056.png
    upload: train/velocipede_s_001200.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001200.png
    upload: train/velocipede_s_001158.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001158.png
    upload: train/velocipede_s_001141.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001141.png
    upload: train/velocipede_s_001142.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001142.png
    upload: train/velocipede_s_000989.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000989.png
    upload: train/velocipede_s_000659.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000659.png
    upload: train/velocipede_s_001210.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001210.png
    upload: train/velocipede_s_001166.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001166.png
    upload: train/velocipede_s_001222.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001222.png
    upload: train/velocipede_s_000825.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_000825.png
    upload: train/velocipede_s_001358.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001358.png
    upload: train/velocipede_s_001299.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001299.png
    upload: train/velocipede_s_001298.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001298.png
    upload: train/velocipede_s_001422.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001422.png
    upload: train/velocipede_s_001338.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001338.png
    upload: train/velocipede_s_001279.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001279.png
    upload: train/velocipede_s_001278.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001278.png
    upload: train/velocipede_s_001361.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001361.png
    upload: train/velocipede_s_001244.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001244.png
    upload: train/velocipede_s_001225.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001225.png
    upload: train/velocipede_s_001489.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001489.png
    upload: train/velocipede_s_001514.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001514.png
    upload: train/velocipede_s_001637.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001637.png
    upload: train/velocipede_s_001874.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001874.png
    upload: train/velocipede_s_001870.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001870.png
    upload: train/velocipede_s_001585.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001585.png
    upload: train/velocipede_s_001872.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001872.png
    upload: train/velocipede_s_001883.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001883.png
    upload: train/velocipede_s_001880.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001880.png
    upload: train/velocipede_s_001882.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001882.png
    upload: train/velocipede_s_001907.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001907.png
    upload: train/velocipede_s_001935.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001935.png
    upload: train/velocipede_s_001958.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001958.png
    upload: train/velocipede_s_001920.png to s3://sagemaker-us-east-1-761239682643/train/velocipede_s_001920.png
                                                                                       
    Training finished ...
    upload: test/bicycle_s_000030.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000030.png
    upload: test/bicycle_s_001218.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001218.png
    upload: test/bicycle_s_001107.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001107.png
    upload: test/bicycle_s_000031.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000031.png
    upload: test/bicycle_s_000776.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000776.png
    upload: test/bicycle_s_000059.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000059.png
    upload: test/bicycle_s_002735.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_002735.png
    upload: test/bicycle_s_001047.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001047.png
    upload: test/bicycle_s_001789.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001789.png
    upload: test/bicycle_s_000513.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000513.png
    upload: test/bicycle_s_000777.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000777.png
    upload: test/bicycle_s_000479.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000479.png
    upload: test/bicycle_s_001102.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001102.png
    upload: test/bicycle_s_000779.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000779.png
    upload: test/bike_s_000131.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000131.png
    upload: test/bike_s_000071.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000071.png
    upload: test/bicycle_s_002458.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_002458.png
    upload: test/bike_s_000330.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000330.png
    upload: test/bicycle_s_001804.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_001804.png
    upload: test/bike_s_000041.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000041.png
    upload: test/bike_s_000163.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000163.png
    upload: test/bike_s_000457.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000457.png
    upload: test/bike_s_000487.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000487.png
    upload: test/bike_s_000643.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000643.png
    upload: test/bike_s_000694.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000694.png
    upload: test/bike_s_000801.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000801.png
    upload: test/bike_s_001073.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001073.png
    upload: test/bike_s_001068.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001068.png
    upload: test/bike_s_001159.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001159.png
    upload: test/bicycle_s_000977.png to s3://sagemaker-us-east-1-761239682643/test/bicycle_s_000977.png
    upload: test/bike_s_000941.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000941.png
    upload: test/bike_s_001216.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001216.png
    upload: test/bike_s_001784.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001784.png
    upload: test/bike_s_001342.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001342.png
    upload: test/bike_s_000658.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_000658.png
    upload: test/bike_s_001738.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001738.png
    upload: test/bike_s_002116.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_002116.png
    upload: test/bike_s_001852.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_001852.png
    upload: test/bike_s_002009.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_002009.png
    upload: test/bike_s_002208.png to s3://sagemaker-us-east-1-761239682643/test/bike_s_002208.png
    upload: test/cycle_s_000068.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_000068.png
    upload: test/cycle_s_001306.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001306.png
    upload: test/cycle_s_000010.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_000010.png
    upload: test/cycle_s_001044.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001044.png
    upload: test/cycle_s_001297.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001297.png
    upload: test/cycle_s_001915.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001915.png
    upload: test/cycle_s_002305.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_002305.png
    upload: test/cycle_s_001648.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001648.png
    upload: test/cycle_s_000970.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_000970.png
    upload: test/cycle_s_001214.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001214.png
    upload: test/cycle_s_001953.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_001953.png
    upload: test/dirt_bike_s_000001.png to s3://sagemaker-us-east-1-761239682643/test/dirt_bike_s_000001.png
    upload: test/cycle_s_002613.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_002613.png
    upload: test/cycle_s_002964.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_002964.png
    upload: test/cycle_s_002661.png to s3://sagemaker-us-east-1-761239682643/test/cycle_s_002661.png
    upload: test/minibike_s_000052.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000052.png
    upload: test/minibike_s_000254.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000254.png
    upload: test/minibike_s_000075.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000075.png
    upload: test/minibike_s_000497.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000497.png
    upload: test/minibike_s_000505.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000505.png
    upload: test/minibike_s_000055.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000055.png
    upload: test/minibike_s_000290.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000290.png
    upload: test/minibike_s_000309.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000309.png
    upload: test/minibike_s_000288.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000288.png
    upload: test/minibike_s_000801.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000801.png
    upload: test/minibike_s_000398.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000398.png
    upload: test/minibike_s_000573.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000573.png
    upload: test/minibike_s_000947.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000947.png
    upload: test/minibike_s_000828.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000828.png
    upload: test/minibike_s_000792.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000792.png
    upload: test/minibike_s_000880.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000880.png
    upload: test/minibike_s_000960.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000960.png
    upload: test/minibike_s_000913.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_000913.png
    upload: test/minibike_s_001089.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001089.png
    upload: test/minibike_s_001547.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001547.png
    upload: test/minibike_s_001473.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001473.png
    upload: test/minibike_s_001867.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001867.png
    upload: test/minibike_s_001651.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001651.png
    upload: test/minibike_s_001605.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001605.png
    upload: test/minibike_s_002051.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_002051.png
    upload: test/minibike_s_002173.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_002173.png
    upload: test/minibike_s_001893.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001893.png
    upload: test/minibike_s_002230.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_002230.png
    upload: test/minibike_s_002227.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_002227.png
    upload: test/minibike_s_001732.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001732.png
    upload: test/minibike_s_001441.png to s3://sagemaker-us-east-1-761239682643/test/minibike_s_001441.png
    upload: test/moped_s_000007.png to s3://sagemaker-us-east-1-761239682643/test/moped_s_000007.png
    upload: test/moped_s_000033.png to s3://sagemaker-us-east-1-761239682643/test/moped_s_000033.png
    upload: test/moped_s_000306.png to s3://sagemaker-us-east-1-761239682643/test/moped_s_000306.png
    upload: test/motorbike_s_000005.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000005.png
    upload: test/moped_s_000064.png to s3://sagemaker-us-east-1-761239682643/test/moped_s_000064.png
    upload: test/motorbike_s_000126.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000126.png
    upload: test/motorbike_s_000121.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000121.png
    upload: test/motorbike_s_000135.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000135.png
    upload: test/motorbike_s_000324.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000324.png
    upload: test/motorbike_s_000629.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000629.png
    upload: test/motorbike_s_000333.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000333.png
    upload: test/motorcycle_s_000049.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000049.png
    upload: test/motorbike_s_000465.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000465.png
    upload: test/motorcycle_s_000040.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000040.png
    upload: test/motorcycle_s_000042.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000042.png
    upload: test/motorcycle_s_000007.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000007.png
    upload: test/motorcycle_s_000063.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000063.png
    upload: test/motorcycle_s_000060.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000060.png
    upload: test/motorbike_s_000433.png to s3://sagemaker-us-east-1-761239682643/test/motorbike_s_000433.png
    upload: test/motorcycle_s_000141.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000141.png
    upload: test/motorcycle_s_000427.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000427.png
    upload: test/motorcycle_s_000323.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000323.png
    upload: test/motorcycle_s_000171.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000171.png
    upload: test/motorcycle_s_000211.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000211.png
    upload: test/motorcycle_s_000352.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000352.png
    upload: test/motorcycle_s_000222.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000222.png
    upload: test/motorcycle_s_000446.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000446.png
    upload: test/motorcycle_s_000139.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000139.png
    upload: test/motorcycle_s_000450.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000450.png
    upload: test/motorcycle_s_000530.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000530.png
    upload: test/motorcycle_s_000602.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000602.png
    upload: test/motorcycle_s_000679.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000679.png
    upload: test/motorcycle_s_000494.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000494.png
    upload: test/motorcycle_s_000485.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000485.png
    upload: test/motorcycle_s_000615.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000615.png
    upload: test/motorcycle_s_000512.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000512.png
    upload: test/motorcycle_s_000797.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000797.png
    upload: test/motorcycle_s_000606.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000606.png
    upload: test/motorcycle_s_000876.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000876.png
    upload: test/motorcycle_s_000866.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000866.png
    upload: test/motorcycle_s_000739.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000739.png
    upload: test/motorcycle_s_000878.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000878.png
    upload: test/motorcycle_s_001164.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001164.png
    upload: test/motorcycle_s_000685.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000685.png
    upload: test/motorcycle_s_001261.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001261.png
    upload: test/motorcycle_s_000963.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000963.png
    upload: test/motorcycle_s_001269.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001269.png
    upload: test/motorcycle_s_001249.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001249.png
    upload: test/motorcycle_s_000825.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_000825.png
    upload: test/motorcycle_s_001508.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001508.png
    upload: test/motorcycle_s_001348.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001348.png
    upload: test/motorcycle_s_001385.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001385.png
    upload: test/motorcycle_s_001519.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001519.png
    upload: test/motorcycle_s_001936.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001936.png
    upload: test/motorcycle_s_001960.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001960.png
    upload: test/motorcycle_s_001679.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001679.png
    upload: test/motorcycle_s_001687.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001687.png
    upload: test/motorcycle_s_001955.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001955.png
    upload: test/motorcycle_s_001971.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001971.png
    upload: test/motorcycle_s_001782.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001782.png
    upload: test/motorcycle_s_001906.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001906.png
    upload: test/motorcycle_s_001892.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_001892.png
    upload: test/motorcycle_s_002177.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_002177.png
    upload: test/motorcycle_s_002026.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_002026.png
    upload: test/ordinary_bicycle_s_000113.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000113.png
    upload: test/ordinary_bicycle_s_000105.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000105.png
    upload: test/ordinary_bicycle_s_000158.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000158.png
    upload: test/motorcycle_s_002126.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_002126.png
    upload: test/motorcycle_s_002112.png to s3://sagemaker-us-east-1-761239682643/test/motorcycle_s_002112.png
    upload: test/ordinary_bicycle_s_000169.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000169.png
    upload: test/ordinary_bicycle_s_000437.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000437.png
    upload: test/ordinary_bicycle_s_000431.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000431.png
    upload: test/ordinary_bicycle_s_000350.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000350.png
    upload: test/ordinary_bicycle_s_000284.png to s3://sagemaker-us-east-1-761239682643/test/ordinary_bicycle_s_000284.png
    upload: test/safety_bicycle_s_000315.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_000315.png
    upload: test/safety_bicycle_s_000235.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_000235.png
    upload: test/safety_bicycle_s_001286.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_001286.png
    upload: test/safety_bicycle_s_000280.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_000280.png
    upload: test/safety_bike_s_000390.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000390.png
    upload: test/safety_bike_s_000438.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000438.png
    upload: test/safety_bicycle_s_001322.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_001322.png
    upload: test/safety_bicycle_s_001575.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_001575.png
    upload: test/safety_bike_s_000192.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000192.png
    upload: test/safety_bicycle_s_001153.png to s3://sagemaker-us-east-1-761239682643/test/safety_bicycle_s_001153.png
    upload: test/safety_bike_s_000488.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000488.png
    upload: test/safety_bike_s_000791.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000791.png
    upload: test/safety_bike_s_001065.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001065.png
    upload: test/safety_bike_s_001253.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001253.png
    upload: test/safety_bike_s_001087.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001087.png
    upload: test/safety_bike_s_000699.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000699.png
    upload: test/safety_bike_s_001690.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001690.png
    upload: test/safety_bike_s_001132.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001132.png
    upload: test/safety_bike_s_001481.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001481.png
    upload: test/safety_bike_s_000540.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000540.png
    upload: test/safety_bike_s_000541.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_000541.png
    upload: test/safety_bike_s_001088.png to s3://sagemaker-us-east-1-761239682643/test/safety_bike_s_001088.png
    upload: test/trail_bike_s_000073.png to s3://sagemaker-us-east-1-761239682643/test/trail_bike_s_000073.png
    upload: test/velocipede_s_000292.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000292.png
    upload: test/velocipede_s_000106.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000106.png
    upload: test/velocipede_s_001201.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001201.png
    upload: test/velocipede_s_000001.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000001.png
    upload: test/velocipede_s_000041.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000041.png
    upload: test/velocipede_s_000863.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000863.png
    upload: test/velocipede_s_001232.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001232.png
    upload: test/velocipede_s_000369.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_000369.png
    upload: test/velocipede_s_001277.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001277.png
    upload: test/velocipede_s_001335.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001335.png
    upload: test/velocipede_s_001466.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001466.png
    upload: test/velocipede_s_001355.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001355.png
    upload: test/velocipede_s_001379.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001379.png
    upload: test/velocipede_s_001699.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001699.png
    upload: test/velocipede_s_001633.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001633.png
    upload: test/velocipede_s_001744.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001744.png
    upload: test/velocipede_s_001790.png to s3://sagemaker-us-east-1-761239682643/test/velocipede_s_001790.png
    Testing finished ...


And that's it! You can check the bucket and verify that the items were uploaded.

## Model Training

For Image Classification, Sagemaker [also expects metadata](https://docs.aws.amazon.com/sagemaker/latest/dg/image-classification.html) e.g. in the form of TSV files with labels and filepaths. We can generate these using our Pandas DataFrames from earlier:


```python
def to_metadata_file(df, prefix):
    df["s3_path"] = df["filenames"]
    df["labels"] = df["labels"].apply(lambda x: 0 if x==8 else 1)
    return df[["row", "labels", "s3_path"]].to_csv(
        f"{prefix}.lst", sep="\t", index=False, header=False
    )
    
to_metadata_file(df_train.copy(), "train")
to_metadata_file(df_test.copy(), "test")
```

We can also upload our manifest files:


```python
import boto3

# Upload files
boto3.Session().resource('s3').Bucket(
    bucket).Object('train.lst').upload_file('./train.lst')
boto3.Session().resource('s3').Bucket(
    bucket).Object('test.lst').upload_file('./test.lst')
```

Using the `bucket` and `region` info we can get the latest prebuilt container to run our training job, and define an output location on our s3 bucket for the model. Use the `image_uris` function from the SageMaker SDK to retrieve the latest `image-classification` image below:


```python
# Use the image_uris function to retrieve the latest 'image-classification' image 
algo_image = sagemaker.image_uris.retrieve('image-classification', region, 'latest')
s3_output_location = f"s3://{bucket}/models/image_model"
```

    Defaulting to the only supported framework/algorithm version: 1. Ignoring framework/algorithm version: latest.


We're ready to create an estimator! Create an estimator `img_classifier_model` that uses one instance of `ml.p3.2xlarge`. Ensure that y ou use the output location we defined above - we'll be referring to that later!


```python
img_classifier_model=sagemaker.estimator.Estimator(
                                    algo_image, # The location of the container we wish to use
                                    role,                                    # What is our current IAM Role
                                    instance_count=1,                  # How many compute instances
                                    instance_type='ml.p3.2xlarge',      # What kind of compute instances
                                    output_path= s3_output_location,
                                    sagemaker_session=sagemaker.Session()
)
```

We can also set a few key hyperparameters and define the inputs for our model:


```python
img_classifier_model.set_hyperparameters(
    image_shape= "3,32,32", # TODO: Fill in
    num_classes= 2 , # TODO: Fill in
    num_training_samples= 1000# TODO: fill in
)
```

The `image-classification` image uses four input channels with very specific input parameters. For convenience, we've provided them below:


```python
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import TrainingInput
model_inputs = {
        "train": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train/",
            content_type="application/x-image"
        ),
        "validation": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test/",
            content_type="application/x-image"
        ),
        "train_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/train.lst",
            content_type="application/x-image"
        ),
        "validation_lst": sagemaker.inputs.TrainingInput(
            s3_data=f"s3://{bucket}/test.lst",
            content_type="application/x-image"
        )
}
```

Great, now we can train the model using the model_inputs. In the cell below, call the `fit` method on our model,:


```python
## TODO: train your model
img_classifier_model.fit(model_inputs)
```

    2022-12-24 20:00:27 Starting - Starting the training job...ProfilerReport-1671912027: InProgress
    ...
    2022-12-24 20:01:10 Starting - Preparing the instances for training......
    2022-12-24 20:02:17 Downloading - Downloading input data......
    2022-12-24 20:03:12 Training - Downloading the training image.........
    2022-12-24 20:04:51 Training - Training image download completed. Training in progress..[34mDocker entrypoint called with argument(s): train[0m
    [34mRunning default environment configuration script[0m
    [34mNvidia gpu devices, drivers and cuda toolkit versions (only available on hosts with GPU):[0m
    [34mSat Dec 24 20:05:01 2022       [0m
    [34m+-----------------------------------------------------------------------------+[0m
    [34m| NVIDIA-SMI 510.47.03    Driver Version: 510.47.03    CUDA Version: 11.6     |[0m
    [34m|-------------------------------+----------------------+----------------------+[0m
    [34m| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |[0m
    [34m| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |[0m
    [34m|                               |                      |               MIG M. |[0m
    [34m|===============================+======================+======================|[0m
    [34m|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |[0m
    [34m| N/A   38C    P0    24W / 300W |      0MiB / 16384MiB |      0%      Default |[0m
    [34m|                               |                      |                  N/A |[0m
    [34m+-------------------------------+----------------------+----------------------+
                                                                                   [0m
    [34m+-----------------------------------------------------------------------------+[0m
    [34m| Processes:                                                                  |[0m
    [34m|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |[0m
    [34m|        ID   ID                                                   Usage      |[0m
    [34m|=============================================================================|[0m
    [34m|  No running processes found                                                 |[0m
    [34m+-----------------------------------------------------------------------------+[0m
    [34mChecking for nvidia driver and cuda compatibility.[0m
    [34mCUDA Compatibility driver provided.[0m
    [34mProceeding with compatibility check between driver, cuda-toolkit and cuda-compat.[0m
    [34mDetected cuda-toolkit version: 11.1.[0m
    [34mDetected cuda-compat version: 455.32.00.[0m
    [34mDetected Nvidia driver version: 510.47.03.[0m
    [34mNvidia driver compatible with cuda-toolkit. Disabling cuda-compat.[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Reading default configuration from /opt/amazon/lib/python3.7/site-packages/image_classification/default-input.json: {'use_pretrained_model': 0, 'num_layers': 152, 'epochs': 30, 'learning_rate': 0.1, 'lr_scheduler_factor': 0.1, 'optimizer': 'sgd', 'momentum': 0, 'weight_decay': 0.0001, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-08, 'gamma': 0.9, 'mini_batch_size': 32, 'image_shape': '3,224,224', 'precision_dtype': 'float32'}[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Merging with provided configuration from /opt/ml/input/config/hyperparameters.json: {'image_shape': '3,32,32', 'num_classes': '2', 'num_training_samples': '1000'}[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Final configuration: {'use_pretrained_model': 0, 'num_layers': 152, 'epochs': 30, 'learning_rate': 0.1, 'lr_scheduler_factor': 0.1, 'optimizer': 'sgd', 'momentum': 0, 'weight_decay': 0.0001, 'beta_1': 0.9, 'beta_2': 0.999, 'eps': 1e-08, 'gamma': 0.9, 'mini_batch_size': 32, 'image_shape': '3,32,32', 'precision_dtype': 'float32', 'num_classes': '2', 'num_training_samples': '1000'}[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Searching for .lst files in /opt/ml/input/data/train_lst.[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Creating record files for train.lst[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Done creating record files...[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Searching for .lst files in /opt/ml/input/data/validation_lst.[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Creating record files for test.lst[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Done creating record files...[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] use_pretrained_model: 0[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] multi_label: 0[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Performing random weight initialization[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] ---- Parameters ----[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] num_layers: 152[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] data type: <class 'numpy.float32'>[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] epochs: 30[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] optimizer: sgd[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] momentum: 0.9[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] weight_decay: 0.0001[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] learning_rate: 0.1[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] num_training_samples: 1000[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] mini_batch_size: 32[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] image_shape: 3,32,32[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] num_classes: 2[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] augmentation_type: None[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] kv_store: device[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] checkpoint_frequency not set, will store the best model[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] --------------------[0m
    [34m[12/24/2022 20:05:05 INFO 140572055721792] Setting number of threads: 7[0m
    [34m[20:05:09] /opt/brazil-pkg-cache/packages/AIAlgorithmsMXNet/AIAlgorithmsMXNet-1.4.x_ecl_Cuda_11.1.x.38.0/AL2_x86_64/generic-flavor/src/src/operator/nn/./cudnn/./cudnn_algoreg-inl.h:97: Running performance tests to find the best convolution algorithm, this can take a while... (setting env variable MXNET_CUDNN_AUTOTUNE_DEFAULT to 0 to disable)[0m
    [34m[12/24/2022 20:05:14 INFO 140572055721792] Epoch[0] Batch [20]#011Speed: 123.186 samples/sec#011accuracy=0.544643[0m
    [34m[12/24/2022 20:05:16 INFO 140572055721792] Epoch[0] Train-accuracy=0.590726[0m
    [34m[12/24/2022 20:05:16 INFO 140572055721792] Epoch[0] Time cost=6.829[0m
    [34m[12/24/2022 20:05:16 INFO 140572055721792] Epoch[0] Validation-accuracy=0.510417[0m
    [34m[12/24/2022 20:05:16 INFO 140572055721792] Storing the best model with validation accuracy: 0.510417[0m
    [34m[12/24/2022 20:05:17 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0001.params"[0m
    [34m[12/24/2022 20:05:20 INFO 140572055721792] Epoch[1] Batch [20]#011Speed: 183.859 samples/sec#011accuracy=0.540179[0m
    [34m[12/24/2022 20:05:22 INFO 140572055721792] Epoch[1] Train-accuracy=0.547379[0m
    [34m[12/24/2022 20:05:22 INFO 140572055721792] Epoch[1] Time cost=5.125[0m
    [34m[12/24/2022 20:05:22 INFO 140572055721792] Epoch[1] Validation-accuracy=0.515625[0m
    [34m[12/24/2022 20:05:23 INFO 140572055721792] Storing the best model with validation accuracy: 0.515625[0m
    [34m[12/24/2022 20:05:23 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0002.params"[0m
    [34m[12/24/2022 20:05:27 INFO 140572055721792] Epoch[2] Batch [20]#011Speed: 189.544 samples/sec#011accuracy=0.622024[0m
    [34m[12/24/2022 20:05:28 INFO 140572055721792] Epoch[2] Train-accuracy=0.621976[0m
    [34m[12/24/2022 20:05:28 INFO 140572055721792] Epoch[2] Time cost=5.005[0m
    [34m[12/24/2022 20:05:29 INFO 140572055721792] Epoch[2] Validation-accuracy=0.666667[0m
    [34m[12/24/2022 20:05:29 INFO 140572055721792] Storing the best model with validation accuracy: 0.666667[0m
    [34m[12/24/2022 20:05:29 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0003.params"[0m
    [34m[12/24/2022 20:05:33 INFO 140572055721792] Epoch[3] Batch [20]#011Speed: 191.021 samples/sec#011accuracy=0.665179[0m
    [34m[12/24/2022 20:05:34 INFO 140572055721792] Epoch[3] Train-accuracy=0.696573[0m
    [34m[12/24/2022 20:05:34 INFO 140572055721792] Epoch[3] Time cost=4.995[0m
    [34m[12/24/2022 20:05:35 INFO 140572055721792] Epoch[3] Validation-accuracy=0.736607[0m
    [34m[12/24/2022 20:05:35 INFO 140572055721792] Storing the best model with validation accuracy: 0.736607[0m
    [34m[12/24/2022 20:05:36 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0004.params"[0m
    [34m[12/24/2022 20:05:39 INFO 140572055721792] Epoch[4] Batch [20]#011Speed: 189.805 samples/sec#011accuracy=0.712798[0m
    [34m[12/24/2022 20:05:41 INFO 140572055721792] Epoch[4] Train-accuracy=0.701613[0m
    [34m[12/24/2022 20:05:41 INFO 140572055721792] Epoch[4] Time cost=5.007[0m
    [34m[12/24/2022 20:05:41 INFO 140572055721792] Epoch[4] Validation-accuracy=0.744792[0m
    [34m[12/24/2022 20:05:41 INFO 140572055721792] Storing the best model with validation accuracy: 0.744792[0m
    [34m[12/24/2022 20:05:42 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0005.params"[0m
    [34m[12/24/2022 20:05:45 INFO 140572055721792] Epoch[5] Batch [20]#011Speed: 190.447 samples/sec#011accuracy=0.752976[0m
    [34m[12/24/2022 20:05:47 INFO 140572055721792] Epoch[5] Train-accuracy=0.743952[0m
    [34m[12/24/2022 20:05:47 INFO 140572055721792] Epoch[5] Time cost=5.004[0m
    [34m[12/24/2022 20:05:47 INFO 140572055721792] Epoch[5] Validation-accuracy=0.651042[0m
    [34m[12/24/2022 20:05:51 INFO 140572055721792] Epoch[6] Batch [20]#011Speed: 190.788 samples/sec#011accuracy=0.755952[0m
    [34m[12/24/2022 20:05:53 INFO 140572055721792] Epoch[6] Train-accuracy=0.744960[0m
    [34m[12/24/2022 20:05:53 INFO 140572055721792] Epoch[6] Time cost=5.009[0m
    [34m[12/24/2022 20:05:53 INFO 140572055721792] Epoch[6] Validation-accuracy=0.640625[0m
    [34m[12/24/2022 20:05:57 INFO 140572055721792] Epoch[7] Batch [20]#011Speed: 187.145 samples/sec#011accuracy=0.730655[0m
    [34m[12/24/2022 20:05:59 INFO 140572055721792] Epoch[7] Train-accuracy=0.736895[0m
    [34m[12/24/2022 20:05:59 INFO 140572055721792] Epoch[7] Time cost=5.094[0m
    [34m[12/24/2022 20:05:59 INFO 140572055721792] Epoch[7] Validation-accuracy=0.781250[0m
    [34m[12/24/2022 20:06:00 INFO 140572055721792] Storing the best model with validation accuracy: 0.781250[0m
    [34m[12/24/2022 20:06:00 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0008.params"[0m
    [34m[12/24/2022 20:06:04 INFO 140572055721792] Epoch[8] Batch [20]#011Speed: 176.017 samples/sec#011accuracy=0.755952[0m
    [34m[12/24/2022 20:06:05 INFO 140572055721792] Epoch[8] Train-accuracy=0.758065[0m
    [34m[12/24/2022 20:06:05 INFO 140572055721792] Epoch[8] Time cost=5.283[0m
    [34m[12/24/2022 20:06:06 INFO 140572055721792] Epoch[8] Validation-accuracy=0.708333[0m
    [34m[12/24/2022 20:06:10 INFO 140572055721792] Epoch[9] Batch [20]#011Speed: 191.467 samples/sec#011accuracy=0.791667[0m
    [34m[12/24/2022 20:06:11 INFO 140572055721792] Epoch[9] Train-accuracy=0.783266[0m
    [34m[12/24/2022 20:06:11 INFO 140572055721792] Epoch[9] Time cost=5.037[0m
    [34m[12/24/2022 20:06:12 INFO 140572055721792] Epoch[9] Validation-accuracy=0.760417[0m
    [34m[12/24/2022 20:06:16 INFO 140572055721792] Epoch[10] Batch [20]#011Speed: 192.678 samples/sec#011accuracy=0.775298[0m
    [34m[12/24/2022 20:06:17 INFO 140572055721792] Epoch[10] Train-accuracy=0.783266[0m
    [34m[12/24/2022 20:06:17 INFO 140572055721792] Epoch[10] Time cost=4.961[0m
    [34m[12/24/2022 20:06:18 INFO 140572055721792] Epoch[10] Validation-accuracy=0.765625[0m
    [34m[12/24/2022 20:06:22 INFO 140572055721792] Epoch[11] Batch [20]#011Speed: 189.496 samples/sec#011accuracy=0.779762[0m
    [34m[12/24/2022 20:06:23 INFO 140572055721792] Epoch[11] Train-accuracy=0.776210[0m
    [34m[12/24/2022 20:06:23 INFO 140572055721792] Epoch[11] Time cost=5.044[0m
    [34m[12/24/2022 20:06:24 INFO 140572055721792] Epoch[11] Validation-accuracy=0.723214[0m
    [34m[12/24/2022 20:06:28 INFO 140572055721792] Epoch[12] Batch [20]#011Speed: 191.397 samples/sec#011accuracy=0.787202[0m
    [34m[12/24/2022 20:06:29 INFO 140572055721792] Epoch[12] Train-accuracy=0.792339[0m
    [34m[12/24/2022 20:06:29 INFO 140572055721792] Epoch[12] Time cost=4.976[0m
    [34m[12/24/2022 20:06:30 INFO 140572055721792] Epoch[12] Validation-accuracy=0.781250[0m
    [34m[12/24/2022 20:06:34 INFO 140572055721792] Epoch[13] Batch [20]#011Speed: 190.126 samples/sec#011accuracy=0.785714[0m
    [34m[12/24/2022 20:06:35 INFO 140572055721792] Epoch[13] Train-accuracy=0.791331[0m
    [34m[12/24/2022 20:06:35 INFO 140572055721792] Epoch[13] Time cost=5.043[0m
    [34m[12/24/2022 20:06:36 INFO 140572055721792] Epoch[13] Validation-accuracy=0.692708[0m
    [34m[12/24/2022 20:06:40 INFO 140572055721792] Epoch[14] Batch [20]#011Speed: 191.798 samples/sec#011accuracy=0.812500[0m
    [34m[12/24/2022 20:06:41 INFO 140572055721792] Epoch[14] Train-accuracy=0.807460[0m
    [34m[12/24/2022 20:06:41 INFO 140572055721792] Epoch[14] Time cost=4.960[0m
    [34m[12/24/2022 20:06:42 INFO 140572055721792] Epoch[14] Validation-accuracy=0.812500[0m
    [34m[12/24/2022 20:06:42 INFO 140572055721792] Storing the best model with validation accuracy: 0.812500[0m
    [34m[12/24/2022 20:06:42 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0015.params"[0m
    [34m[12/24/2022 20:06:46 INFO 140572055721792] Epoch[15] Batch [20]#011Speed: 191.313 samples/sec#011accuracy=0.818452[0m
    [34m[12/24/2022 20:06:47 INFO 140572055721792] Epoch[15] Train-accuracy=0.811492[0m
    [34m[12/24/2022 20:06:47 INFO 140572055721792] Epoch[15] Time cost=4.988[0m
    [34m[12/24/2022 20:06:48 INFO 140572055721792] Epoch[15] Validation-accuracy=0.794643[0m
    [34m[12/24/2022 20:06:52 INFO 140572055721792] Epoch[16] Batch [20]#011Speed: 188.018 samples/sec#011accuracy=0.840774[0m
    [34m[12/24/2022 20:06:54 INFO 140572055721792] Epoch[16] Train-accuracy=0.824597[0m
    [34m[12/24/2022 20:06:54 INFO 140572055721792] Epoch[16] Time cost=5.061[0m
    [34m[12/24/2022 20:06:54 INFO 140572055721792] Epoch[16] Validation-accuracy=0.791667[0m
    [34m[12/24/2022 20:06:58 INFO 140572055721792] Epoch[17] Batch [20]#011Speed: 191.920 samples/sec#011accuracy=0.833333[0m
    [34m[12/24/2022 20:06:59 INFO 140572055721792] Epoch[17] Train-accuracy=0.825605[0m
    [34m[12/24/2022 20:06:59 INFO 140572055721792] Epoch[17] Time cost=4.970[0m
    [34m[12/24/2022 20:07:00 INFO 140572055721792] Epoch[17] Validation-accuracy=0.812500[0m
    [34m[12/24/2022 20:07:04 INFO 140572055721792] Epoch[18] Batch [20]#011Speed: 184.697 samples/sec#011accuracy=0.803571[0m
    [34m[12/24/2022 20:07:06 INFO 140572055721792] Epoch[18] Train-accuracy=0.811492[0m
    [34m[12/24/2022 20:07:06 INFO 140572055721792] Epoch[18] Time cost=5.102[0m
    [34m[12/24/2022 20:07:06 INFO 140572055721792] Epoch[18] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:07:10 INFO 140572055721792] Epoch[19] Batch [20]#011Speed: 190.519 samples/sec#011accuracy=0.839286[0m
    [34m[12/24/2022 20:07:11 INFO 140572055721792] Epoch[19] Train-accuracy=0.825605[0m
    [34m[12/24/2022 20:07:11 INFO 140572055721792] Epoch[19] Time cost=5.006[0m
    [34m[12/24/2022 20:07:12 INFO 140572055721792] Epoch[19] Validation-accuracy=0.785714[0m
    [34m[12/24/2022 20:07:16 INFO 140572055721792] Epoch[20] Batch [20]#011Speed: 189.128 samples/sec#011accuracy=0.833333[0m
    [34m[12/24/2022 20:07:18 INFO 140572055721792] Epoch[20] Train-accuracy=0.829637[0m
    [34m[12/24/2022 20:07:18 INFO 140572055721792] Epoch[20] Time cost=5.052[0m
    [34m[12/24/2022 20:07:18 INFO 140572055721792] Epoch[20] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:07:22 INFO 140572055721792] Epoch[21] Batch [20]#011Speed: 191.218 samples/sec#011accuracy=0.842262[0m
    [34m[12/24/2022 20:07:24 INFO 140572055721792] Epoch[21] Train-accuracy=0.836694[0m
    [34m[12/24/2022 20:07:24 INFO 140572055721792] Epoch[21] Time cost=4.990[0m
    [34m[12/24/2022 20:07:24 INFO 140572055721792] Epoch[21] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:07:28 INFO 140572055721792] Epoch[22] Batch [20]#011Speed: 191.457 samples/sec#011accuracy=0.825893[0m
    [34m[12/24/2022 20:07:30 INFO 140572055721792] Epoch[22] Train-accuracy=0.819556[0m
    [34m[12/24/2022 20:07:30 INFO 140572055721792] Epoch[22] Time cost=4.982[0m
    [34m[12/24/2022 20:07:30 INFO 140572055721792] Epoch[22] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:07:34 INFO 140572055721792] Epoch[23] Batch [20]#011Speed: 192.448 samples/sec#011accuracy=0.860119[0m
    [34m[12/24/2022 20:07:35 INFO 140572055721792] Epoch[23] Train-accuracy=0.848790[0m
    [34m[12/24/2022 20:07:35 INFO 140572055721792] Epoch[23] Time cost=4.970[0m
    [34m[12/24/2022 20:07:36 INFO 140572055721792] Epoch[23] Validation-accuracy=0.683036[0m
    [34m[12/24/2022 20:07:40 INFO 140572055721792] Epoch[24] Batch [20]#011Speed: 190.744 samples/sec#011accuracy=0.833333[0m
    [34m[12/24/2022 20:07:41 INFO 140572055721792] Epoch[24] Train-accuracy=0.848790[0m
    [34m[12/24/2022 20:07:41 INFO 140572055721792] Epoch[24] Time cost=4.980[0m
    [34m[12/24/2022 20:07:42 INFO 140572055721792] Epoch[24] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:07:46 INFO 140572055721792] Epoch[25] Batch [20]#011Speed: 191.375 samples/sec#011accuracy=0.866071[0m
    [34m[12/24/2022 20:07:47 INFO 140572055721792] Epoch[25] Train-accuracy=0.861895[0m
    [34m[12/24/2022 20:07:47 INFO 140572055721792] Epoch[25] Time cost=5.004[0m
    [34m[12/24/2022 20:07:48 INFO 140572055721792] Epoch[25] Validation-accuracy=0.828125[0m
    [34m[12/24/2022 20:07:48 INFO 140572055721792] Storing the best model with validation accuracy: 0.828125[0m
    [34m[12/24/2022 20:07:49 INFO 140572055721792] Saved checkpoint to "/opt/ml/model/image-classification-0026.params"[0m
    [34m[12/24/2022 20:07:52 INFO 140572055721792] Epoch[26] Batch [20]#011Speed: 189.675 samples/sec#011accuracy=0.828869[0m
    [34m[12/24/2022 20:07:54 INFO 140572055721792] Epoch[26] Train-accuracy=0.842742[0m
    [34m[12/24/2022 20:07:54 INFO 140572055721792] Epoch[26] Time cost=5.007[0m
    [34m[12/24/2022 20:07:54 INFO 140572055721792] Epoch[26] Validation-accuracy=0.796875[0m
    [34m[12/24/2022 20:07:58 INFO 140572055721792] Epoch[27] Batch [20]#011Speed: 189.166 samples/sec#011accuracy=0.892857[0m
    [34m[12/24/2022 20:08:00 INFO 140572055721792] Epoch[27] Train-accuracy=0.883065[0m
    [34m[12/24/2022 20:08:00 INFO 140572055721792] Epoch[27] Time cost=5.039[0m
    [34m[12/24/2022 20:08:00 INFO 140572055721792] Epoch[27] Validation-accuracy=0.799107[0m
    [34m[12/24/2022 20:08:04 INFO 140572055721792] Epoch[28] Batch [20]#011Speed: 184.165 samples/sec#011accuracy=0.877976[0m
    [34m[12/24/2022 20:08:06 INFO 140572055721792] Epoch[28] Train-accuracy=0.883065[0m
    [34m[12/24/2022 20:08:06 INFO 140572055721792] Epoch[28] Time cost=5.101[0m
    [34m[12/24/2022 20:08:06 INFO 140572055721792] Epoch[28] Validation-accuracy=0.807292[0m
    [34m[12/24/2022 20:08:10 INFO 140572055721792] Epoch[29] Batch [20]#011Speed: 190.671 samples/sec#011accuracy=0.901786[0m
    [34m[12/24/2022 20:08:12 INFO 140572055721792] Epoch[29] Train-accuracy=0.911290[0m
    [34m[12/24/2022 20:08:12 INFO 140572055721792] Epoch[29] Time cost=5.018[0m
    [34m[12/24/2022 20:08:12 INFO 140572055721792] Epoch[29] Validation-accuracy=0.802083[0m
    
    2022-12-24 20:08:32 Uploading - Uploading generated training model
    2022-12-24 20:09:12 Completed - Training job completed
    ProfilerReport-1671912027: NoIssuesFound
    Training seconds: 403
    Billable seconds: 403


If all goes well, you'll end up with a model topping out above `.8` validation accuracy. With only 1000 training samples in the CIFAR dataset, that's pretty good. We could definitely pursue data augmentation & gathering more samples to help us improve further, but for now let's proceed to deploy our model.

### Getting ready to deploy

To begin with, let's configure Model Monitor to track our deployment. We'll define a `DataCaptureConfig` below:


```python
from sagemaker.model_monitor import DataCaptureConfig

data_capture_config = DataCaptureConfig(
    ## TODO: Set config options
    enable_capture = True,
    sampling_percentage = 100,
    destination_s3_uri=f"s3://{bucket}/data_capture"
)
```

Note the `destination_s3_uri` parameter: At the end of the project, we can explore the `data_capture` directory in S3 to find crucial data about the inputs and outputs Model Monitor has observed on our model endpoint over time.

With that done, deploy your model on a single `ml.m5.xlarge` instance with the data capture config attached:


```python
deployment = img_classifier_model.deploy(
    ## TODO: fill in deployment options
    initial_instance_count = 1,
    instance_type ='ml.m5.xlarge',
    data_capture_config=data_capture_config
    )

endpoint = deployment.endpoint_name
print(endpoint)
```

    -------!image-classification-2022-12-24-20-10-47-169


Note the endpoint name for later as well.

Next, instantiate a Predictor:


```python
from sagemaker.predictor import Predictor

predictor = Predictor(endpoint_name=endpoint, sagemaker_session=sagemaker.Session())## TODO: fill in
```

In the code snippet below we are going to prepare one of your saved images for prediction. Use the predictor to process the `payload`.


```python
from sagemaker.serializers import IdentitySerializer
import base64

predictor.serializer = IdentitySerializer("image/png")
with open("./test/bicycle_s_001789.png", "rb") as f:
    payload = f.read()

    
inference = predictor.predict(data=payload)## TODO: Process the payload with your predictor ## TODO: Process the payload with your predictor
```

Your `inference` object is an array of two values, the predicted probability value for each of your classes (bicycle and motorcycle respectively.) So, for example, a value of `b'[0.91, 0.09]'` indicates the probability of being a bike is 91% and being a motorcycle is 9%.


```python
print(inference)
```

    b'[0.9895245432853699, 0.010475479997694492]'


### Draft Lambdas and Step Function Workflow

Your operations team uses Step Functions to orchestrate serverless workflows. One of the nice things about Step Functions is that [workflows can call other workflows](https://docs.aws.amazon.com/step-functions/latest/dg/connect-stepfunctions.html), so the team can easily plug your workflow into the broader production architecture for Scones Unlimited.

In this next stage you're going to write and deploy three Lambda functions, and then use the Step Functions visual editor to chain them together! Our functions are going to work with a simple data object:

```python
{
    "inferences": [], # Output of predictor.predict
    "s3_key": "", # Source data S3 key
    "s3_bucket": "", # Source data S3 bucket
    "image_data": ""  # base64 encoded string containing the image data
}
```

A good test object that you can use for Lambda tests and Step Function executions, throughout the next section, might look like this:

```python
{
  "image_data": "",
  "s3_bucket": MY_BUCKET_NAME, # Fill in with your bucket
  "s3_key": "test/bicycle_s_000513.png"
}
```

Using these fields, your functions can read and write the necessary data to execute your workflow. Let's start with the first function. Your first Lambda function will copy an object from S3, base64 encode it, and then return it to the step function as `image_data` in an event.

Go to the Lambda dashboard and create a new Lambda function with a descriptive name like "serializeImageData" and select thr 'Python 3.8' runtime. Add the same permissions as the SageMaker role you created earlier. (Reminder: you do this in the Configuration tab under "Permissions"). Once you're ready, use the starter code below to craft your Lambda handler:

```python
import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = ## TODO: fill in
    bucket = ## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in
    
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
    }
```

The next function is responsible for the classification part - we're going to take the image output from the previous function, decode it, and then pass inferences back to the the Step Function.

Because this Lambda will have runtime dependencies (i.e. the SageMaker SDK) you'll need to package them in your function. *Key reading:* https://docs.aws.amazon.com/lambda/latest/dg/python-package-create.html#python-package-create-with-dependency

Create a new Lambda function with the same rights and a descriptive name, then fill in the starter code below for your classifier Lambda.

```python
import json
import sagemaker
import base64
from sagemaker.serializers import IdentitySerializer

# Fill this in with the name of your deployed model
ENDPOINT = ## TODO: fill in

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(## TODO: fill in)

    # Instantiate a Predictor
    predictor = ## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    inferences = ## TODO: fill in
    
    # We return the data back to the Step Function    
    event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```

Finally, we need to filter low-confidence inferences. Define a threshold between 1.00 and 0.000 for your model: what is reasonble for you? If the model predicts at `.70` for it's highest confidence label, do we want to pass that inference along to downstream systems? Make one last Lambda function and tee up the same permissions:

```python
import json


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = ## TODO: fill in
    
    # Check if any values in our inferences are above THRESHOLD
    meets_threshold = ## TODO: fill in
    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }
```
Once you have tested the lambda functions, save the code for each lambda function in a python script called 'lambda.py'.

With your lambdas in place, you can use the Step Functions visual editor to construct a workflow that chains them together. In the Step Functions console you'll have the option to author a Standard step function *Visually*.

When the visual editor opens, you'll have many options to add transitions in your workflow. We're going to keep it simple and have just one: to invoke Lambda functions. Add three of them chained together. For each one, you'll be able to select the Lambda functions you just created in the proper order, filter inputs and outputs, and give them descriptive names.

**Results**


![flow](./images/stepExec.jpg)
![flow](./images/executionRes.jpg)


Great! Now you can use the files in `./test` as test files for our workflow. Depending on our threshold, our workflow should reliably pass predictions about images from `./test` on to downstream systems, while erroring out for inferences below our confidence threshold!



### Testing and Evaluation

Do several step function invokations using data from the `./test` folder. This process should give you confidence that the workflow both *succeeds* AND *fails* as expected. In addition, SageMaker Model Monitor will generate recordings of your data and inferences which we can visualize.

Here's a function that can help you generate test inputs for your invokations:


```python
import random
import boto3
import json


def generate_test_case():
    # Setup s3 in boto3
    s3 = boto3.resource('s3')
    
    # Randomly pick from sfn or test folders in our bucket
    objects = s3.Bucket(bucket).objects.filter(Prefix="test")
    
    # Grab any random object key from that folder!
    obj = random.choice([x.key for x in objects])
    
    return json.dumps({
        "image_data": "",
        "s3_bucket": bucket,
        "s3_key": obj
    })
generate_test_case()
```




    '{"image_data": "", "s3_bucket": "sagemaker-us-east-1-761239682643", "s3_key": "test/shark_s_001217.png"}'



In the Step Function dashboard for your new function, you can create new executions and copy in the generated test cases. Do several executions so that you can generate data you can evaluate and visualize.

Once you've done several executions, let's visualize the record of our inferences. Pull in the JSONLines data from your inferences like so:


```python
from sagemaker.s3 import S3Downloader

# In S3 your data will be saved to a datetime-aware path
# Find a path related to a datetime you're interested in
data_path = 's3://sagemaker-us-east-1-761239682643/data_capture/image-classification-2022-12-24-20-10-47-169/AllTraffic/2022/12/24/20/' ## TODO: fill in the path to your captured data

S3Downloader.download(data_path, "captured_data")

# Feel free to repeat this multiple times and pull in more data
```

The data are in JSONLines format, where multiple valid JSON objects are stacked on top of eachother in a single `jsonl` file. We'll import an open-source library, `jsonlines` that was purpose built for parsing this format.


```python
!pip install jsonlines
import jsonlines
```

    Keyring is skipped due to an exception: 'keyring.backends'
    Requirement already satisfied: jsonlines in /opt/conda/lib/python3.7/site-packages (3.1.0)
    Requirement already satisfied: attrs>=19.2.0 in /opt/conda/lib/python3.7/site-packages (from jsonlines) (22.1.0)
    Requirement already satisfied: typing-extensions in /opt/conda/lib/python3.7/site-packages (from jsonlines) (4.4.0)
    [33mWARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv[0m[33m
    [0m

Now we can extract the data from each of the source files:


```python
import os

# List the file names we downloaded
file_handles = os.listdir("./captured_data")

# Dump all the data into an array
json_data = []
for jsonl in file_handles[:-1]:
    with jsonlines.open(f"./captured_data/{jsonl}") as f:
        json_data.append(f.read())
```

The data should now be a list of dictionaries, with significant nesting. We'll give you an example of some code that grabs data out of the objects and visualizes it:


```python
# Define how we'll get our data
def simple_getter(obj):
    inferences = obj["captureData"]["endpointOutput"]["data"]
    timestamp = obj["eventMetadata"]["inferenceTime"]
    return json.loads(inferences), timestamp

simple_getter(json_data[0])
```




    ([0.9895245432853699, 0.010475479997694492], '2022-12-24T20:16:01Z')



Finally, here's an example of a visualization you can build with this data. In this last part, you will take some time and build your own - the captured data has the input images, the resulting inferences, and the timestamps.


```python
# Populate the data for the x and y axis
x = []
y = []
for obj in json_data:
    inference, timestamp = simple_getter(obj)
    
    y.append(max(inference))
    x.append(timestamp)

# Todo: here is an visualization example, take some time to build another visual that helps monitor the result
# Plot the data
plt.scatter(x, y, c=['r' if k<.94 else 'b' for k in y ])
plt.axhline(y=0.94, color='g', linestyle='--')
plt.ylim(bottom=.88)

# Add labels
plt.ylabel("Confidence")
plt.suptitle("Observed Recent Inferences", size=14)
plt.title("Pictured with confidence threshold for production use", size=10)

# Give it some pizzaz!
plt.style.use("Solarize_Light2")
plt.gcf().autofmt_xdate()
```

### Todo: build your own visualization



```python
import matplotlib.pyplot as plt

def myVizualisation():
    x = []
    y = []
    for obj in json_data:
        inference, timestamp = simple_getter(obj)
        y.append(max(inference))
        x.append(timestamp)

    plt.figure(figsize=(10,8))
    plt.scatter(x, y)

    plt.ylabel("Confidence")
    plt.title("Confidence During production", size=10)


plot_graph()
```


    
![png](./images/output_79_0.png)
    


### Congratulations!

You've reached the end of the project. In this project you created an event-drivent ML workflow that can be incorporated into the Scones Unlimited production architecture. You used the SageMaker Estimator API to deploy your SageMaker Model and Endpoint, and you used AWS Lambda and Step Functions to orchestrate your ML workflow. Using SageMaker Model Monitor, you instrumented and observed your Endpoint, and at the end of the project you built a visualization to help stakeholders understand the performance of the Endpoint over time. If you're up for it, you can even go further with these stretch goals:

* Extend your workflow to incorporate more classes: the CIFAR dataset includes other vehicles that Scones Unlimited can identify with this model.
* Modify your event driven workflow: can you rewrite your Lambda functions so that the workflow can process multiple image inputs in parallel? Can the Step Function "fan out" to accomodate this new workflow?
* Consider the test data generator we provided for you. Can we use it to create a "dummy data" generator, to simulate a continuous stream of input data? Or a big paralell load of data?
* What if we want to get notified every time our step function errors out? Can we use the Step Functions visual editor in conjunction with a service like SNS to accomplish this? Try it out!



