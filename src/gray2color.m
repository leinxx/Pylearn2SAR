%save colormapped image
%step: change the image to index, then save it with colormap
%now we assume im is within 0 and 1;
function gray2color(im,fname)

    mi = 0;
    ma = 1;
    im(im>ma)=ma;
    im(im<mi)=mi;
    %im = uint8((im-mi)*255/(ma-mi));
    im = uint8(im*255);
    cmap = jet(256);
    x = cmap(im(:)+1,:);
    x = reshape(x,size(im,1),size(im,2),3);
    imwrite(x,cmap,fname);

end
