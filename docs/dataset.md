\# Dataset





Classes



0\. mouse

1\. cup





Data Source



The dataset is collected manually using videos of objects.





Raw Data Processing



Frames are extracted uniformly from the recorded videos using:



scripts/extract\_frames.py



Approximately 200 frames are initially extracted for each class.



Images with severe blur, invalid targets, overexposure, or highly repetitive content are manually removed.





Current Classes



\- mouse

\- bottle





Dataset Split



The final dataset will contain:



\- train

\- validation

\- test



