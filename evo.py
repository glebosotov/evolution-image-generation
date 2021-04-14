import time
import random
import cv2
import numpy
import copy
from operator import itemgetter
from math import sqrt
import sys
import os
import argh

"""
    $ python3 evo.py -h 
    for help
"""

height = 512
width = 512
resizeImage = True
shrink = False
shrinkPixels = 256

# evo weights
population_size = 512
iterationsGlobal = 5000  # default, changes with argv

# shapes limits
newShapesPerIteration = 1
maxTotalShapes = 10000

# brush
circleMode = False
triangleMode = False
polygonMode = False
polygonSmartMode = False

# for circles
radiusMin = 5
radiusMax = height//4

# for polygones
verticesCount = 3
sectionCount = 5

# for triangles
minSide = 5
maxSide = 100

minimumTransparency = 0.2
maximumTransparency = 0.6


# generate a tuple (RGB) randomly
def randomColor():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# given a canvas and an array of circles draw an image
def renderFromCircles(img, circles):
    for circle in circles:
        # each circle is [coordinates, color, radius, transparency]
        overlay = img.copy()
        overlay = cv2.circle(overlay, circle[0], circle[2], circle[1], -1)
        img = cv2.addWeighted(overlay, circle[3], img, 1-circle[3], 0)
    return img


# make a random circle with structure [coordinates, color, radius, transparency]
def randomCircle():
    return [(random.randint(0, width-1), random.randint(0,
                                                        height-1)),  randomColor(), random.randint(radiusMin, radiusMax), random.uniform(minimumTransparency, maximumTransparency)]


# make a random polygon with structure [[coordinates], color, transparency]
def randomPolygon():
    points = []
    for i in range(verticesCount):
        points.append([random.randint(0, width-1),
                       random.randint(0, height-1)])
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(minimumTransparency, maximumTransparency)])


# make a random polygon with structure [[coordinates], color, transparency] withing a sector
# split the image in sectionCount x sectionCount rectsngles and put polygones there, while not
# strictly following the edges
def randomPolygonHeuristics():
    points = []
    hsection = random.randint(0, sectionCount)
    wsection = random.randint(0, sectionCount)
    hSectionStart = hsection * height // sectionCount
    hSectionEnd = hSectionStart + height // sectionCount - 1
    wSectionStart = wsection * width // sectionCount
    wSectionEnd = wSectionStart + width // sectionCount - 1
    # if hsection != 0 and hsection != sectionCount:
    hSectionStart -= height / 2 // sectionCount
    hSectionEnd += height / 2 // sectionCount
    # if wsection != 0 and wsection != sectionCount:
    wSectionStart -= height / 2 // sectionCount
    wSectionEnd += height / 2 // sectionCount
    for i in range(verticesCount):
        # points.append([random.randint(0, height-1),
        #                random.randint(0, width-1)])
        points.append([random.randint(wSectionStart, wSectionEnd),
                       random.randint(hSectionStart, hSectionEnd)])
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(minimumTransparency, maximumTransparency)])


# render picture from polygones and canvas
def renderFromPolygones(img, polygones):
    for polygon in polygones:
        # each polygon is [[coordinates], color, transparency]
        overlay = img.copy()
        overlay = cv2.fillPoly(overlay, [polygon[0]], polygon[1])
        img = cv2.addWeighted(overlay, polygon[2], img, 1-polygon[2], 0)
    return img


# makes a random triangle with equal sides and top side is parallel to
# the horizontal axis with structure [[coordinates], color, transparency]
def randomTriangle():

    side = random.randint(minSide, maxSide)
    triangleHeight = int(side * sqrt(3) / 2)
    h1 = random.randint(0, height-1-triangleHeight)
    w1 = random.randint(0, width-1-side)
    h2 = h1
    w2 = w1 + side
    h3 = h1 + triangleHeight
    w3 = w1 + side//2
    points = [[w1, h1], [w2, h2], [w3, h3]]
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(minimumTransparency, maximumTransparency)])


# draws depending on mode
def render(img, shapes):
    if circleMode:
        return renderFromCircles(img, shapes)
    elif triangleMode or polygonMode or polygonSmartMode:
        return renderFromPolygones(img, shapes)


# https://stackoverflow.com/a/44659589
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized


def processShapes(shapes, height, width):
    global originalHeight, originalWidth
    newShapes = copy.deepcopy(shapes)
    for shape in newShapes:
        if circleMode:
            shape[0] = (int(shape[0][0]*originalHeight/height),
                        int(shape[0][1]*originalWidth/width))
            shape[2] = int(shape[2]*originalHeight/height)
        else:
            for point in shape[0]:
                point[0] = int(point[0]*originalHeight/height)
                point[1] = int(point[1]*originalWidth/width)
    return newShapes


# makes shape depending on mode
def randomShape():
    if circleMode:
        return randomCircle()
    elif polygonMode:
        return randomPolygon()
    elif triangleMode:
        return randomTriangle()
    elif polygonSmartMode:
        return randomPolygonHeuristics()


# returns empty white canvas
def emptyImage(h=height, w=width):
    return numpy.ones((h, w, 3), numpy.uint8)*255


# comparing images using Mean Square Error
# https://en.wikipedia.org/wiki/Mean_squared_error#Definition_and_basic_properties
def mse(imageA, imageB):
    err = numpy.sum((imageA.astype(numpy.float32) -
                     imageB.astype(numpy.float32)) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err * -1


# a class used for keeping the following fields: an array of circles, rendered image
# and an array of new genes
# where each gene has [[shapes], generationNumber, mse]
class Population:
    currentImage = None
    currentHQImage = None
    currentImageshapes = []
    emptyHQImage = None
    emptyImage = None

    # constructor with either empty genes or provided genes
    def __init__(self, newImages):
        self.images = []
        if newImages == []:
            self.currentImage = emptyImage()
            self.images = []
        else:
            self.images = newImages

    # choosing the best out of new genes or discarding whole generation
    def chooseMostFit(self):
        bestFitnessImage = max(self.images, key=itemgetter(2))

        # logging improvement
        print(int(-1*bestFitnessImage[2]), end=" ")
        print(len(self.currentImageshapes), "shapes")

        # if new generation has better genes than best from previous
        if (-1*mse(inputImage, self.currentImage) > -1 * bestFitnessImage[2]):
            self.currentImageshapes += bestFitnessImage[0]
            # deleting circles outside the limit
            if currentGenId > maxTotalShapes:
                self.currentImageshapes = self.currentImageshapes[newShapesPerIteration:]
            # redrawing image from scratch
            self.currentImage = render(
                self.emptyImage, self.currentImageshapes)
            # self.currentHQImage = render(self.emptyHQImage, processShapes(
            #     self.currentImageshapes, self.currentHQImage.shape[0], self.currentHQImage.shape[1]))
            # adding new generation to current
            newBestImage = render(
                self.currentImage, bestFitnessImage[0])
            self.currentImage = newBestImage
            self.currentHQImage = render(self.currentHQImage, processShapes(
                bestFitnessImage[0], self.currentImage.shape[0],  self.currentImage.shape[1]))
        return self.currentImage

    # creating new genes
    def mutate(self, bestImage):
        newImages = []
        # making new genes
        for j in range(population_size):
            mutatingImage = [[], currentGenId, 0]
            # adding shapes
            for i in range(newShapesPerIteration):
                mutatingImage[0].append(randomShape())
            # running fit function
            img = render(self.currentImage, mutatingImage[0])
            mutatingImage[2] = mse(inputImage, img)
            # saving to current population
            newImages.append(mutatingImage)
        self.images = newImages
        return newImages

    # evolving
    def newGeneration(self):
        self.chooseMostFit()
        self.mutate(None)
        return self


# reading args from command line
timelapseFlag = None
modeName = None
imagePath = None
timelapseDirPath = None


@argh.arg('imageName', help="filename of an image in current directory")
@argh.arg('brush', choices=['c', 't', 'p', 'sp'], help="c – circle, t – triangle, p – random polygon, sp – smart polygon")
@argh.arg('--iterations', help='optional')
@argh.arg('--population', help='optional')
@argh.arg('--keepsize', '-s', help="keep original resolution")
@argh.arg('--fast', '-f', help="faster, worse quality")
@argh.arg('--timelapse', '-t', help="making a timelapse in mp4 format at the end, ffmpeg required")
def collectArgs(imageName, brush, keepsize=False, iterations=5000, timelapse=False, population=512, fast=False):
    """Use either .jpg or .png images with resolution of 512x512"""
    global timelapseFlag
    global modeName
    global imagePath
    global iterationsGlobal
    global timelapseDirPath
    global circleMode
    global triangleMode
    global polygonSmartMode
    global polygonMode
    global population_size
    global resizeImage
    global shrink
    timelapseFlag = timelapse  # dumping all images for timelapse
    imagePath = imageName
    population_size = population
    modeName = brush
    resizeImage = not keepsize
    shrink = fast
    if brush == 'c':
        circleMode = True
    elif brush == 't':
        triangleMode = True
    elif brush == 'p':
        polygonMode = True
    elif brush == 'sp':
        polygonSmartMode = True
    iterationsGlobal = iterations
    if timelapse:
        try:
            timelapseDirPath = imageName.split('.')[0]+brush+"TimelapseTemp"
            if not os.path.exists(timelapseDirPath):
                os.mkdir(timelapseDirPath)
        except OSError:
            print("Count not create", timelapseDirPath)


# collecting args
argh.dispatch_command(collectArgs)

inputImage = cv2.imread(imagePath)

# resizing or keeping size dependinng on arguments
if not resizeImage:
    print("Not resizing")
    originalHeight = inputImage.shape[0]
    originalWidth = inputImage.shape[1]
    inputImage = image_resize(
        inputImage, height=512) if not shrink else image_resize(inputImage, height=shrinkPixels)
    height = inputImage.shape[0]
    width = inputImage.shape[1]
    # for circles
    radiusMin = height//100
    radiusMax = height//4

    # for polygones
    verticesCount = 3
    sectionCount = 5

    # for triangles
    minSide = height//100
    maxSide = height//5
else:
    if not (inputImage.shape[0] == 512 and inputImage.shape[1] == 512):
        print("Image is not 512x512, resizing...")
        if not shrink:
            inputImage = cv2.resize(
                inputImage, (512, 512), interpolation=cv2.INTER_AREA)
            originalHeight = height
            originalWidth = width
        else:
            inputImage = cv2.resize(
                inputImage, (shrinkPixels, shrinkPixels), interpolation=cv2.INTER_AREA)
            height = 256
            width = 256
            originalHeight = 512
            originalWidth = 512


# first generation
currentGenId = 0
mainTime = time.time()
population = Population([])
population.currentImage = emptyImage(h=height, w=width)
population.emptyImage = emptyImage(h=height, w=width)
population.currentHQImage = emptyImage(h=originalHeight, w=originalWidth)
population.emptyHQImage = emptyImage(h=originalHeight, w=originalWidth)
population.mutate(None)

# iterating
for i in range(iterationsGlobal):
    currentGenId += 1
    print("#", i, end=" ", sep='')
    population = population.newGeneration()
    img = population.currentHQImage
    if i % 10 == 0:
        cv2.imwrite(imagePath.split('.')[0]+modeName +
                    "Result"+'.'+imagePath.split('.')[1], img)  # saving image each iteration
    if timelapseFlag:
        cv2.imwrite(timelapseDirPath+'/' +
                    str("{:05d}".format(currentGenId))+'.png', img)

# making a timeplapse
if timelapseFlag:
    os.system("ffmpeg -hide_banner -loglevel error -framerate 240 -i "+timelapseDirPath +
              "/%05d.png -c:v libx264 -pix_fmt yuv420p "+imagePath.split('.')[0]+".mp4")
    os.system("rm -r "+timelapseDirPath)

# time since start
print("minutes elapsed", (time.time() - mainTime)//60)
