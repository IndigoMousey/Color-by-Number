from ColorQuantization import *

images = ["bird.jpg", "oriole.jpg"]
for i in range(len(images)):
    original, modified = kmeansClustering(images[i])
    side_by_side(original, modified)
