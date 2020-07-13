% eel_detection_and_aspect_ratio.m

% 12/06/2018
% 1. to detect and cut out the eel/stick
% 2. to calculate the aspect ratio and orientation angle of the object

% NOTE: adjust the value of thr_ints (line 35) and thr_obj (line 37) as
% needed to extract the object of detection

tic;
clear; clc; close all;
filename = '2015-09-18_113235_Raw_1680_1780.aris';


eellength    = 0.76; % m
eeldiam      = 0.020;
clims        = [0 210];
imagexsize   = 512*1;
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
    thr_ints          = 30;  % threshold of pixel intensity for normalization of differenced image 
    subsize           = ceil(eellength/((data.maxrange-data.minrange)/imageysize)); % pixel number of subimages
    thr_obj           = floor(1*subsize);
    
    % time span of each aris file when the eel in test was actively swimming
    fr    = data.framerate;
    data  = get_frame_new(data,1);
    data  = make_new_image(data,data.frame);
    snaps_backgnd = data.image;  % background image
       
    for iframe = 2:data.numframes
        iframe
        data               = get_frame_new(data,iframe);
        data               = make_new_image(data,data.frame);
        snaps              = data.image;  % original image
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
        if max_subsum == thr_obj %|| max_subsum>4*thr_obj
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
%         locx = locx + 13;
%         locy = locy + 13;  % move the object to upper left by 3 pixels

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
    figure;
    subplot(2,1,1);
    plot(r);
    xlabel('Frame number');
    ylabel('Aspect ratio');
    subplot(2,1,2);
    plot(phi);
    xlabel('Frame number');
    ylabel('Orientation angle');
    
%     save('eel_rphi.mat', 'r', 'phi', 'numobj');
toc;



