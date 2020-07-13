function imageout = wavelet_denoising(imagein,wname,level,thrtimes)
% Xiaoqin Zang


[C,S]      = wavedec2(imagein,level,wname);
% 2) Obtain denoising (wavelet shrinkage) thresholds. Use the Birge-Massart strategy with a tuning parameter of 3.
thr        = wthrmngr('dw2ddenoLVL','penalhi',C,S,3)*thrtimes;
% 3) Image reconstruction
sorh                 = 's';
[XDEN,cfsDEN,dimCFS] = wdencmp('lvd',C,S,wname,level,thr,sorh);
imageout             = XDEN;
return
