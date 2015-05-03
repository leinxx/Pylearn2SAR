%compare results

basedir = '/home/lein/Work/deeplearning/sar_dnn/output/';
date = '20101006';
lines = [];
num = random('unid',size(im,2),1)
for i= 28:4:44
    fname = [basedir 'train_with_2010_' num2str(i) '/' date '.tif']
    im = imread(fname);
    lines = [lines im(:,num)];
end

figure
plot(lines)
legend('28','32','36','40','44')
