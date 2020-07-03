# (DSAC++)++
## Intro
If you are looking for the original DSAC implementations by Brachmann et all. please head to https://github.com/vislearn/LessMore  
Since our code builds upon the DSAC++ approach, it might be useful to familiarize yourself with the original paper and codebase before you run our code.

## Running (DSAC++)++
The three training steps can be run by using respectively train_obj.py (L1 loss), train_repro.py (repro loss) and train_dsac.py (expected pose loss).  
You can choose to use a deterministic model or one which also predicts uncertainty by setting the appropriate flag ("with_uncertainty") in properties_global.py.  

## Dataset folder structure
The code assumes the following folder structure:  
data/scene_name/seq0i/{rgb. poses, scene, depth} if you are using any of the 7Scenes datasets.
You should also specify the correct training and test scenes if you intend to run on another dataset.   
The code assumes all pose translations are expressed in millimeters. We provide the utility to_millimeters.py in case your ground truth pose is in meters.
