function h = plot3dFace_free(sub)
%% plotting the 3d shape
% The solution is to use Delaunay triangulation. Let's look at some
% info about the "tri" variable.
% if max(points_3D(:))>1
%     points_3D = mapminmax(points_3D , 0 , 1);
% end
% figure;
tri = delaunay(sub.shape(:, 1), sub.shape(:, 2));
% tri = tl;
% Plot it with TRISURF
h = trisurf(tri, sub.shape(:, 1),sub.shape(:, 2), sub.shape(:, 3) , 'FaceVertexCData', sub.shape(:,3), 'FaceColor' , 'interp', 'EdgeColor','interp');%, 'LineWidth',5);%'CDataMapping','scaled'););

axis vis3d
axis tight
% view([-50 , -30 , 300]);
view([0 , 0 , 300]);
drawnow;
camlight right
% set(gca , 'Projection' , 'perspective');

% Clean it up
axis off
% l = light('Position',[-50 -15 29]);
% l = light('Position',[0 0.5 0.5]);
% set(gca,'CameraPosition',[208 -50 7687])
lighting phong
shading interp
colorbar EastOutside
colormap winter