function out =  voxelize_me(x_dim , y_dim , z_dim , ptcl)
% ptcl: n*3 point cloud matrix
out = zeros(x_dim , y_dim , z_dim);
x_min = min(ptcl(:,1));
y_min = min(ptcl(:,2));
z_min = min(ptcl(:,3));
x_max = max(ptcl(:,1));
y_max = max(ptcl(:,2));
z_max = max(ptcl(:,3));
% x = ptcl(: , 1);
% y = ptcl(: , 2);
% z = ptcl(: , 3);
% x_mapped = round((x - x_min)*(x_dim - 1)/(x_max - x_min));
% y_mapped = round((y - y_min)*(y_dim - 1)/(y_max - y_min));
% z_mapped = round((z - z_min)*(z_dim - 1)/(z_max - z_min));
% out(x_mapped , y_mapped , z_mapped) = 1;
for i = 1 : size(ptcl , 1)
    x = ptcl(i , 1);
    y = ptcl(i , 2);
    z = ptcl(i , 3);
    x_mapped = round((x - x_min)*(x_dim - 1)/(x_max - x_min))+1;
    y_mapped = round((y - y_min)*(y_dim - 1)/(y_max - y_min))+1;
    z_mapped = round((z - z_min)*(z_dim - 1)/(z_max - z_min))+1;
    out(x_mapped , y_mapped , z_mapped) = 1;
end