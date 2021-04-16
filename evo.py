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
import signal

"""
    $ python3 evo.py -h 
    for help
"""


class Config:
    processingHeight = 512
    processingWidth = 512
    fileHeight = 512
    fileWidth = 512
    resizeImage = True
    compressedProcessing = False
    compressedPixels = 512  # will be set to 256 for 512x512 image

    # evo weights
    populationSize = 512  # default, changes with argv
    iterations = 5000  # default, changes with argv

    # shapes limits
    newShapesPerIteration = 1
    maxTotalShapes = 10000

    # brush
    circleMode = False
    triangleMode = False
    polygonMode = False
    smartPolygonMode = False
    squareMode = False
    modename = ""

    # for circles
    radiusMin = 5
    radiusMax = processingHeight//4

    # for polygones
    verticesCount = 3
    sectionCount = 10

    # for triangles and squares
    minSide = 5
    maxSide = 100

    minimumTransparency = 0.2
    maximumTransparency = 0.6

    # misc arguments
    imagePath = None
    timelapseFlag = None
    timelapseDirPath = None


def scaleShapeWeights():
    Config.radiusMin = Config.processingHeight // 100
    Config.radiusMax = Config.processingHeight // 4

    Config.minSide = Config.processingHeight // 100
    Config.maxSide = Config.processingHeight // 5


def decreaseShapeWeights():
    Config.radiusMin = Config.processingHeight // 100
    Config.radiusMax = Config.processingHeight // 20

    Config.minSide = Config.processingHeight // 100
    Config.maxSide = Config.processingHeight // 20

    Config.sectionCount = 20


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
    return [(random.randint(0, Config.processingWidth), random.randint(0,
                                                                       Config.processingHeight)),  randomColor(), random.randint(Config.radiusMin, Config.radiusMax), random.uniform(Config.minimumTransparency, Config.maximumTransparency)]


# make a random polygon with structure [[coordinates], color, transparency]
def randomPolygon():
    points = []
    for i in range(Config.verticesCount):
        points.append([random.randint(0, Config.processingWidth),
                       random.randint(0, Config.processingHeight)])
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(Config.minimumTransparency, Config.maximumTransparency)])


# make a random polygon with structure [[coordinates], color, transparency] withing a sector
# split the image in sectionCount x sectionCount rectsngles and put polygones there, while not
# strictly following the edges
def randomPolygonHeuristics():
    anchor = [random.randint(0, Config.processingWidth),
              random.randint(0, Config.processingHeight)]
    sway = min(Config.processingWidth // Config.sectionCount,
               Config.processingHeight // Config.sectionCount)

    points = []
    for i in range(Config.verticesCount):
        points.append([random.randint(anchor[0]-sway, anchor[0]+sway),
                       random.randint(anchor[1]-sway, anchor[1]+sway)])

    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(Config.minimumTransparency, Config.maximumTransparency)])


def randomSquare():
    topLeft = [random.randint(0, Config.processingWidth), random.randint(
        0, Config.processingHeight)]
    side = random.randint(Config.minSide, Config.maxSide)
    bottomRight = [topLeft[0]+side, topLeft[1]+side]
    topRight = [topLeft[0]+side, topLeft[1]]
    bottomLeft = [bottomRight[0]-side, bottomRight[1]]
    points = [topLeft, topRight, bottomRight, bottomLeft]
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(Config.minimumTransparency, Config.maximumTransparency)])


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
    side = random.randint(Config.minSide, Config.maxSide)
    triangleHeight = int(side * sqrt(3) / 2)
    h1 = random.randint(0, Config.processingHeight-triangleHeight)
    w1 = random.randint(0, Config.processingWidth-side)
    h2 = h1
    w2 = w1 + side
    h3 = h1 + triangleHeight
    w3 = w1 + side // 2
    points = [[w1, h1], [w2, h2], [w3, h3]]
    return ([numpy.array(points, numpy.int32), randomColor(), random.uniform(Config.minimumTransparency, Config.maximumTransparency)])


# draws depending on mode
def render(img, shapes):
    if Config.circleMode:
        return renderFromCircles(img, shapes)
    elif Config.triangleMode or Config.polygonMode or Config.smartPolygonMode or Config.squareMode:
        return renderFromPolygones(img, shapes)


# https://stackoverflow.com/a/44659589
def resizeImage(image, width=None, height=None, inter=cv2.INTER_AREA):
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


# scales the size of shapes from (small) processing size to output (bigger) size
def scaleShapes(shapes):
    newShapes = copy.deepcopy(shapes)
    scaleCoefficient = Config.fileHeight / Config.processingHeight

    for shape in newShapes:
        if Config.circleMode:
            shape[0] = (int(shape[0][0]*scaleCoefficient),
                        int(shape[0][1]*scaleCoefficient))
            shape[2] = int(shape[2]*scaleCoefficient)
        else:
            for point in shape[0]:
                point[0] = int(point[0]*scaleCoefficient)
                point[1] = int(point[1]*scaleCoefficient)
    return newShapes


# makes shape depending on mode
def randomShape():
    if Config.circleMode:
        return randomCircle()
    elif Config.polygonMode:
        return randomPolygon()
    elif Config.triangleMode:
        return randomTriangle()
    elif Config.smartPolygonMode:
        return randomPolygonHeuristics()
    elif Config.squareMode:
        return randomSquare()


# returns empty white canvas
def emptyImage(height=Config.processingHeight, width=Config.processingWidth):
    return numpy.ones((height, width, 3), numpy.uint8)*255


# comparing images using Mean Square Error
# https://en.wikipedia.org/wiki/Mean_squared_error#Definition_and_basic_properties
def mse(imageA, imageB):
    err = numpy.sum((imageA.astype(numpy.float32) -
                     imageB.astype(numpy.float32)) ** 2)
    err /= float(imageA.shape[0] * imageA.shape[1])
    return err


def timelapseHandler(makeVideo=True):
    if makeVideo:
        os.system("ffmpeg -hide_banner -loglevel error -framerate 240 -i "+Config.timelapseDirPath +
                  "/%05d.png -c:v libx264 -pix_fmt yuv420p "+Config.imagePath.split('.')[0]+".mp4")
    os.system("rm -r "+Config.timelapseDirPath)


# a class used for keeping the following fields: an array of circles, rendered image
# and an array of new genes
# where each gene has [[shapes], generationNumber, mse]
class Population:
    currentGenId = 0
    processingImage = None
    outputImage = None
    shapes = []
    images = []

    # choosing the best out of new genes or discarding whole generation
    def chooseMostFit(self):
        bestFitnessImage = min(self.images, key=itemgetter(2))

        # logging improvement
        print("#", self.currentGenId, " ", int(
            bestFitnessImage[2]), end=" ", sep="")
        print(len(self.shapes), "shapes")

        # if new generation has better genes than best from previous
        if (mse(inputImage, self.processingImage) > bestFitnessImage[2]):
            # adding new shapes
            self.shapes += bestFitnessImage[0]

            # deleting shapes outside the limit
            if len(self.shapes) > Config.maxTotalShapes:
                self.shapes = self.shapes[-Config.maxTotalShapes:]

            # adding new generation to current
            self.processingImage = render(
                self.processingImage, bestFitnessImage[0])
            self.outputImage = render(self.outputImage, scaleShapes(
                bestFitnessImage[0]))
        return self.outputImage

    # creating new genes
    def mutate(self):
        self.currentGenId += 1
        newImages = []
        # making new genes
        for j in range(Config.populationSize):
            mutatingImage = [[], self.currentGenId, 0]
            # adding shapes
            for i in range(Config.newShapesPerIteration):
                mutatingImage[0].append(randomShape())
            # running fit function
            img = render(self.processingImage, mutatingImage[0])
            mutatingImage[2] = mse(inputImage, img)
            # saving to current population
            newImages.append(mutatingImage)
        self.images = newImages
        return newImages

    # evolving
    def newGeneration(self):
        self.chooseMostFit()
        self.mutate()
        return self


@argh.arg('imageName', help="filename of an image in current directory")
@argh.arg('brush', choices=['c', 't', 'p', 'sp', 's'], help="c – circle, t – triangle, p – random polygon, sp – smart polygon, s – square")
@argh.arg('--iterations', help='optional')
@argh.arg('--population', help='optional')
@argh.arg('--keepsize', '-s', help="keep original resolution")
@argh.arg('--fast', '-f', help="faster, worse quality")
@argh.arg('--timelapse', '-t', help="making a timelapse in mp4 format at the end, ffmpeg required")
def collectArgs(imageName, brush, keepsize=False, iterations=5000, timelapse=False, population=512, fast=False):
    """Use either .jpg or .png images with resolution of 512x512"""
    Config.timelapseFlag = timelapse  # dumping all images for timelapse
    Config.imagePath = imageName
    Config.populationSize = population
    Config.resizeImage = not keepsize
    Config.compressedProcessing = fast
    Config.modeName = brush
    if brush == 'c':
        Config.circleMode = True
    elif brush == 't':
        Config.triangleMode = True
    elif brush == 'p':
        Config.polygonMode = True
    elif brush == 'sp':
        Config.smartPolygonMode = True
    elif brush == 's':
        Config.squareMode = True
    Config.iterations = iterations
    if timelapse:
        try:
            Config.timelapseDirPath = imageName.split(
                '.')[0]+brush+"TimelapseTemp"
            if not os.path.exists(Config.timelapseDirPath):
                os.mkdir(Config.timelapseDirPath)
        except OSError:
            print("Count not create", Config.timelapseDirPath)


# handling Ctrl+C
def customInterruptHandler(signum, frame):
    signal.signal(signal.SIGINT, originalHandler)
    if(Config.timelapseFlag):
        try:
            userInput = input(
                " Continue (c) / Make a timelapse and exit (t) / Quit (q)? (c/t/q)> ").lower()[0]

            if userInput == 'c':
                pass
            elif userInput == 't':
                timelapseHandler()
                sys.exit(1)
            elif userInput == 'q':
                if Config.timelapseFlag:
                    timelapseHandler(makeVideo=False)
                sys.exit(1)

        except KeyboardInterrupt:
            print("Exiting now")
            sys.exit(1)
    else:
        sys.exit(1)

    signal.signal(signal.SIGINT, customInterruptHandler)


# setting Ctrl+C handler
originalHandler = signal.getsignal(signal.SIGINT)
signal.signal(signal.SIGINT, customInterruptHandler)

# collecting args
argh.dispatch_command(collectArgs)
inputImage = cv2.imread(Config.imagePath)

# resizing the image or keeping the dimensions
if Config.resizeImage:
    # resizing NOT 512x512 image
    if not (inputImage.shape[0] == 512 and inputImage.shape[1] == 512):
        print("Scaling to 512x512")
        inputImage = cv2.resize(
            inputImage, (512, 512), interpolation=cv2.INTER_AREA)
    # in compressed mode for 512x512 image processing pixels are 256, default is 512
    Config.compressedPixels = 256
else:
    # keeping the size or setting up 512x512 image
    print("Keeping original dimensions")
    Config.fileHeight, Config.fileWidth = Config.processingHeight, Config.processingWidth = inputImage.shape[
        0], inputImage.shape[1]
    scaleShapeWeights()

# compressing the image that is being worked on to improve speed
if Config.compressedProcessing:
    inputImage = resizeImage(inputImage, height=Config.compressedPixels)
    Config.processingHeight, Config.processingWidth = inputImage.shape[0], inputImage.shape[1]
    scaleShapeWeights()

# first generation
mainTime = time.time()
population = Population()
population.processingImage = emptyImage(
    height=Config.processingHeight, width=Config.processingWidth)
population.outputImage = emptyImage(
    height=Config.fileHeight, width=Config.fileWidth)
population.mutate()
print("Algoritmhm works with a ", Config.processingWidth,
      "x", Config.processingHeight, " image", sep='')


# iterating
for i in range(Config.iterations):
    if Config.iterations // 4 < i:
        decreaseShapeWeights()
    population = population.newGeneration()
    img = population.outputImage
    if i % 10 == 0:
        cv2.imwrite(Config.imagePath.split('.')[0]+Config.modeName +
                    "Result"+'.'+Config.imagePath.split('.')[1], img)  # saving image each iteration
    if Config.timelapseFlag:
        cv2.imwrite(Config.timelapseDirPath+'/' +
                    str("{:05d}".format(population.currentGenId))+'.png', img)

# making a timeplapse
if Config.timelapseFlag:
    timelapseHandler()

# time since start
print("minutes elapsed", (time.time() - mainTime)//60)
