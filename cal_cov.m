function cov_ab(first_band, second_band)
 



  band1_mean = mean(first_band(:));
  band2_mean = mean(second_band(:));
  


%band1 abd band2 
  i=1;
  for ii=1:size(ima,1) % Number of Rows
    for jj=1:size(ima,2) % Number of Columns

      data1= band1(ii,jj);% Extract element from iith row and jjth column for array1
      data2= band2(ii,jj);% Extract element from iith row and jjth column for array2

      a1(i,1) = data1-band1_mean;% Difference with mean and store as vector elements for band1
      a2(i,1) = data2-band2_mean;% Difference with mean and store as vector elements for band2
      i=i+1;
     end
  end

  multip = a1.*a2;
  D = sum(multip);
  cov_ab = D/(numel(a1)-1);
  print cov_ab
  
endfunction