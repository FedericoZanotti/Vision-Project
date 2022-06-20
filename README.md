# Vehicle counting, tracking and classification in highway roads with Day and Night environment

In this project made for the Vision and Cognitive Services course my colleague and I mainly focused on vehicle counting, detection and tracking in video in day and night time.
For this project we decided first to build different systems that are able to detect and track vehicles in order to count them and compare the results.
Secondly, we added a classification step on the vehicle, we detected the direction movement and we built a system that is able to detect license plates and perform OCR (Optical Character Recognition) on them. 

For the detection and tracking part we choose different approaches:
* **YoloV3** with a _tracking algorithm developed by us_
* **Background subtraction** with _OpenCV_
* **YoloV4** with _DeepSORT_ for object tracking.
We tested all these methods also in a night environment, a challenging task that led to good results. We achieved the best results with Yolov4 DeepSORT, also in the night environment, and a good accuracy for Yolov3 with our tracking method both for day and night videos. The worst performance was obtained by the OpenCV method, especially in the night environment
