# Deep-learning-for-sonar-images

Two sets of sonar images were uploaded: lab_images were obtained from controlled laboratory experiments; field_images were obtained from field experiments conducted by EPRI in St. Lawrence River in 2015.

Each data sets contains sonar images of eel and non-eel objects (wood sticks and pvc pipes). There are four image versions: 
1. orgnl: original images without any image processing techniques;
2. diff: differenced images which removed static background through image differencing;
3. wvlt: original images denoised with wavelet transform;
4. diffwvlt: images processed with both wavelet denoising and differencing.

Sonar settings in the laboratory experiments and the field experiments were included in the Excel file sonar_data_description. Field eel images were graded into three tiers based on the sonar image quality. It is recommended to use tier 1&2 eel images for traing and testing CNN models.

Code CNN_lab_data.py trained and tested CNN models using lab data only.

Code CNN_field_data.py trained and tested CNN models using field data only.

Matlab scripts processed .aris sonar data and extracted images of eel and non-eel objects.

Note: the .aris data were read using an open source MATLAB script (see https://github.com/nilsolav/ARISreader) (Handegard & Williams, 2008). This toolbox should be downloaded and put in the same directory with above MATLAB scripts.
