# project_rppg

In this project, We aim to implement multi region base neural network to extract robust PPG from the Heart Rate.


# Introduction

### rPPG


# Our Methodologies


# Data Collection:

To  collect  video  with  enough  PPG  signal and well-aligned PPG, we collect simultaneous PPG and video sensor  data.  We  collected  HD  video  using  RGB  stationery(in  a  stand)  DSLR  (Canon  D3500)  at  30  fps  in  a  laboratory setup under artificial light for 5 minutes while each volunteer was  sitting  idly  on  a  chair  three  feet  away  in  front  of  the camera.  Concurrently,  Empatica  E4  Wristwatch  tracks  the wrist PPG sampled at 64 Hz from volunteers’ dominant hand.To  align  the  video  and  PPG  data  to  the  same  time-stamp, we  leveraged  the  event  marker  feature  in  the Empatica  E4.During  that,  the  volunteers  count  to  ten  for  10  seconds.  We identified  the  exact  position  of  the  event  marker  press  of Empatica E4 by locating event marker light at particular video frames.  The  Empatica  E4  locates  the  exact  PPG  position  for the  event  markers  pressed.  This  allows  us  to  align  PPG  andvideo  with  an  error  margin  of  1/30  seconds.  


The  dataset consists of two females, six males volunteers. Our dataset has included person heterogeneity such as sex, facial hair, fitness level,  and  spectacles  usage.  Two  male  volunteers  provided data  with  and  without  a  beard  and  glass.  All  subjects  are healthy  and  have  no  known  medical  condition  having  HR ranges  from  50  BPM  to  95  BPM.  In  RGB  video  times,  our dataset suppressed both public datasets, and suitable for developing  generalized  DL  models.  Moreover,  the  in-house dataset  introduces  heterogeneity  by  placing  the  PPG  sensor at the wrist.


|No of subs | Collections | Data Length | Data Variety |
|-----|--------|------------------|-----------------------------|
| 9 | 14 | \~ 2 hours (\~ 200K frames) | Regular, Gender, Light, Beard, Glass, Sensor, Distance |





Please contact us to get the access to rPPG data. Upon receiving request and agreement, we will share the drive link to download the videos and corresponding Empatica PPG signal. The instructions and python source code will also be provided to align the video with the PPG signal for training and validation.

The volunteers required to sit still in front of camera wearing the Empatica in a laboratory room with Artificial lights. The average data collection takes 10 minutes per subject in each trail.


We have received the kind consent of the volunteers to collect their data and followed instructional Review Board (IRB) protocol during collection.

Please do not use any volunteers direct images in the presentation or papers.

## Other Data and Bench-Marking

# Citation:
