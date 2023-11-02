import aspose.threed as a3d

scene = a3d.Scene.from_file("logs/corgi_nurse_mesh.obj")
scene.save("logs/corgi_nurse_mesh.fbx")