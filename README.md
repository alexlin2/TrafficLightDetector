# Problem Statement
Solving the traffic light problem for autonomous vehicles is critical for all vehicle safety, not just autonomous vehicles. 
The lack of accurate traffic light detection system could pose a serious threat to safety to both drivers and pedestrians. An accurate vision-based detection system is a necessary and vital step toward bringing autonomous vehicles to the streets. 



# Data description
This dataset contains 13427 camera images at a resolution of 1280x720 pixels and contains about 24000 annotated traffic lights. The annotations include bounding boxes of traffic lights as well as the current state (active light) of each traffic light.

The camera images are provided as raw 12bit HDR images taken with a red-clear-clear-blue filter and as reconstructed 8-bit RGB color images. The RGB images are provided for debugging and can also be used for training. However, the RGB conversion process has some drawbacks. Some of the converted images may contain artifacts and the color distribution may seem unusual.


## Training set: 
* 5093 images
* Annotated about every 2 seconds
* 10756 annotated traffic lights
*  Median traffic lights width: ~8.6 pixels
* 15 different labels
* 170 lights are partially occluded

## Test set: 
* 8334 consecutive images
* Annotated at about 15 fps
* 13486 annotated traffic lights
* Median traffic light width: 8.5 pixels
* 4 labels (red, yellow, green, off)
* 2088 lights are partially occluded


# Examples Images: 
