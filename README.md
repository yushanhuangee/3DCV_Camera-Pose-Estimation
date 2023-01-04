# Homework2

Dataset: [Download](https://drive.google.com/u/0/uc?export=download&confirm=qrVw&id=1GrCpYJFc8IZM_Uiisq6e8UxwVMFvr4AJ)
# Note!!
the data should be placed like :
>|homework2-yushanhuangee<br />
> ----|poseNet<br />
> -------------test.py<br />
> ----|data'<br />
> ----draw_pyramid.py<br />
> ----p1.py<br />
> ----p2.py<br />


## Problem 1

  You can estimate the camera of query images by using the following command.

    python3.8 p1.py
   
   This will generate 2`.npy` file `R_result.npy` and `t_result_gt.npy` and show median of pose error at the same time.
   
   with these two file, you can visualize the result by using the following command.   

    python3.8 draw_pyramid.py
      
## Problem 2
  You can generate `output.avi`by using the following command.

    python3.8 p2.py
>The generated video can be found here->[Video link](https://drive.google.com/file/d/12ferurXodU5z7ZJJUmxsF8EL6TqccFlO/view?usp=sharing) 
    
## Bonus-PoseNet
First, you need to download the model from [link](https://drive.google.com/file/d/1H2H-bC_KYScJEqDMLYmwDKUul9dQ65I_/view?usp=sharing) and put it in `/poseNet`.<br />
After going into `/poseNet`. You can generate 2 `.npy` file `R_result_posenet.npy` and `t_result_posenet.npy` and show median of pose error at the same time by using the following command. 
    
    python3.8 test.py
## Requirements
- pytorch 1.8
- python 3.8
- scikit-learn
- tensorboardX
- torchsummary
