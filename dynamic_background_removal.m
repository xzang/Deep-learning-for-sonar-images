% eel_detection_and_aspect_ratio.m
% by Xiaoqin Zang


% 12/06/2018
% 1. to detect and cut out the eel/stick
% 2. to calculate the aspect ratio and orientation angle of the object

% NOTE: adjust the value of thr_ints (line 35) and thr_obj (line 37) as
% needed to extract the object of detection

tic;
clear; clc; %close all;
% filename = ['2015-06-08_134125_Raw_1600_1700.aris'];
% filename = ['2015-06-08_134125_Raw_2350_2638.aris'];
% filename = ['2015-06-08_151537_Raw_1_100.aris'];
% filename = ['2015-06-08_151537_Raw_675_900.aris'];
% filename = ['2015-06-08_151537_Raw_1100_1450.aris'];
filename = '2015-09-17_174127_Raw_3300_3350_eel_76cm.aris';
% filename = '2015-09-18_110000_Raw_10425_10495_eel_91cm.aris';
% filename = '2015-09-18_124356_Raw_2825_2880_stick_130cm.aris';
% filename = '2015-06-08_134951_Raw_160_260.aris';
% filename = '2015-06-08_135330_Raw_130_230.aris';
% filename = '2015-06-09_164826_Raw_1300_1400.aris';
% filename = '2015-09-18_113235_Raw_10900_11000_PVC_1m.aris';


eellength    = 0.76; % m
eeldiam      = 0.020;
clims        = [0 210];
imagexsize   = 512;
subxy        = 61;
wname        = 'db2'; % wavelet
level        = 9;  % wavelet level
thrtimes     = 16; % wavelet threshold

iobj         = 0;
r            = [];
phi          = [];
numobj       = [];



for ifile    = 1:1
    data              = get_frame_first(filename(ifile,:));
    data              = make_first_image(data,4,imagexsize); %make the first image array
    [imageysize,temp] = size(data.image);
    thr_ints          = 20;  % threshold of pixel intensity for normalization of differenced image 
    subsize           = ceil(eellength/((data.maxrange-data.minrange)/imageysize)); % pixel number of subimages
    thr_obj           = 1*subsize;
    img_all           = zeros(imageysize, imagexsize,data.numframes);
    
    fr    = data.framerate;    
    for iframe = 1:data.numframes
        data               = get_frame_new(data,iframe);
        data               = make_new_image(data,data.frame);
        img_all(:,:,iframe) = data.image;
    end
           
    for iframe = 2:data.numframes-4
        iframe
        snaps              = uint8(img_all(:,:,iframe));  % original image
        snaps_backgnd      = mean(img_all(:,:,iframe:iframe+4),3);
        snaps_diff         = double(snaps)-double(snaps_backgnd); % remove background; differenced image NOTE: THE TYPE OF snaps, snaps_backgnd and snaps_diff matters a lot to results!
        snaps_denoised     =  wavelet_denoising(snaps,wname,level,thrtimes); % denoised original image
        snapsdiff_denoised =  wavelet_denoising(snaps_diff,wname,level,thrtimes); % denoised differenced image
        snapsdiff_denoised_norm = zeros(imageysize, imagexsize);  % binary image

    %     figure(iframe); 
    %     imshow(snaps_diff,'DisplayRange',clims);colormap gray;

        [J,I] = find(snapsdiff_denoised>thr_ints); % locate white pixels
        for i = 1:length(I)
            snapsdiff_denoised_norm(J(i),I(i)) = 1;
        end
%         figure(iframe); 
%         imshow(snapsdiff_denoised_norm,'DisplayRange',clims);colormap gray;

        % locate the object where the sum of white pixels of the normalized subimage has the maximum
        max_subsum = thr_obj;
        for i = 1:imagexsize-subsize
            for j = 1:imageysize-subsize
                subimage    = snapsdiff_denoised_norm((j:j+subsize-1),(i:i+subsize-1));
                temp_subsum = sum(sum(subimage,1));
                if temp_subsum>max_subsum %&& temp_subsum<1.5*thr_obj
                    locx = i;
                    locy = j;
                    max_subsum = temp_subsum;
                end
            end
        end %Elapsed time is 3.811356 seconds.

        %%%%=======   skip current frame if no object is detected
        if max_subsum == thr_obj || locy>900%|| max_subsum>4*thr_obj
            continue;
        end
        iobj = iobj+1; % otherwise, count the number of subimages that have objects

        %%%%=======   if object is detected in current frame
        %%% calculate the aspect ratio of object    
        k = 0;
        X = [];
        Y = [];
        objimage = uint8(snapsdiff_denoised_norm((locy:locy+subsize-1),(locx:locx+subsize-1)));
%         figure(iframe); 
%         imshow(objimage,'DisplayRange',[0 1]);colormap gray;

        for i = 1:subsize
            for j=1:subsize
                if objimage(j,i) == 1
                    k    = k+1;
                    X(k) = i;  % record the index of white pixels in X, Y
                    Y(k) = j;
                end
            end
        end
        max_dist = 0; % find the long axis
        for i = 1:length(X)
            for j=(i+1):length(X)
                temp_dist = sqrt((X(i)-X(j))^2+(Y(i)-Y(j))^2);
                if  temp_dist> max_dist
                    max_dist = temp_dist;
                    pixel_1 = i;
                    pixel_2 = j;
                end
            end
        end
        r(iobj)   = max_dist^2/max_subsum;
        phi(iobj) = (atan((Y(pixel_1)-Y(pixel_2))/(X(pixel_1)-X(pixel_2))))/pi*180;
        
%         if r(iobj)<5
%             continue;
%         end
        %%% output the object image
%         locx = locx + 3;
%         locy = locy + 3;  % move the object to upper left by 3 pixels

        Y = locy:locy+subxy-1;
        X = locx:locx+subxy-1;
        objimage_name = ['orgnl_',filename(ifile,1:end-5),'_frame',num2str(iframe),'.png'];
        objorgnl     = snaps(Y,X);  
        imwrite(objorgnl,objimage_name);

        objimage_name = ['wvlt_',filename(ifile,1:end-5),'_frame',num2str(iframe),'.png'];
        objwvlt      = uint8(snaps_denoised(Y,X));  % convert to uint8 format for output
        imwrite(objwvlt,objimage_name);

        objimage_name = ['diff_',filename(ifile,1:end-5),'_frame',num2str(iframe),'.png'];
        objdiff       = uint8(snaps_diff(Y,X));  % convert to uint8 format for output
        imwrite(objdiff,objimage_name);

        objimage_name = ['diffwvlt_',filename(ifile,1:end-5),'_frame',num2str(iframe),'.png'];
        objdiffwvlt  = uint8(snapsdiff_denoised(Y,X));  % convert to uint8 format for output
        imwrite(objdiffwvlt,objimage_name);
        
    end %iframe
    
end
    
%     save('eel_rphi.mat', 'r', 'phi', 'numobj');
toc;



