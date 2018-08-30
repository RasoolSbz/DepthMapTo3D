function cloud = depthMap_to_cloud(depthMap)% shape must be normalize between [0,1] in all coordinate
imSize = size(depthMap);
k = 1;
theta = deg2rad(90);
rotation_axis = [0,0,1];
R = rotationmat3D(theta , rotation_axis);
for i = 1 : imSize(1)
    for j = 1 : imSize(2)
        if depthMap(i , j)~=1
%             cloud(k , 1 : 3) = [ 1-i/imSize(1) , 1-j/imSize(2) , depthMap(i , j)];
            cloud(k , 1 : 3) = [i/imSize(1) , j/imSize(2) , depthMap(i , j)];
            k = k+1;
        end
    end
end
cloud = cloud*R;
sub = struct('shape' , cloud );
faces{1}=sub;
method = 'linear';
xmax = 1;
xmin = 0;
normalize_faces;
cloud = normalFaces{1}.shape;
