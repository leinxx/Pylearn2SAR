%show_weights

for i =1:5
    wfs = ['w' num2str(i) '.tif'];
    if exist(wfs,'file') == 2
        w1 = imread(wfs);
       % w1(:,13:13:end) = NaN;
        %w1(13:13:end,:) = NaN;
        figure; imagesc(w1); colormap gray
    end
end
