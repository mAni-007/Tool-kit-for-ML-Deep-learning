ima = imread("marvel.jpg");

%covariance of red and green

band1 = ima(:,:,1);
band2 = ima(:,:,2);
disp ("covariance of red and green");
cov_ab(band1, band2)
%correlation = corrcov(var_cov);
figure
scatter(band1(:), band2(:))
title("red green")

%covariance of green and blue

band1 = ima(:,:,2);
band2 = ima(:,:,3);
disp ("covariance of green and blue",'\n')
cov_ab(band1, band2)
figure
scatter(band1(:), band2(:))
title("green blue")


%covariance of red and blue

band1 = ima(:,:,1);
band2 = ima(:,:,3);
disp ("covariance of red and blue")
cov_ab(band1, band2)
figurepip
scatter(band1(:), band2(:))
title("red blue")

%covariance of red and red

band1 = ima(:,:,1);
band2 = ima(:,:,1);
disp ("covariance of red and red")
cov_ab(band1, band2)
figure
scatter(band1(:), band2(:))
title("red red ")

%covariance of green and green

band1 = ima(:,:,2);
band2 = ima(:,:,2);
disp ("covariance of green and green")
cov_ab(band1, band2)
figure
scatter(band1(:), band2(:))
title("green  green")

%covariance of blue and blue 

band1 = ima(:,:,3);
band2 = ima(:,:,3);
disp ("covariance of blue an blue")
cov_ab(band1, band2)
figure
scatter(band1(:), band2(:))
title("blue blue")



