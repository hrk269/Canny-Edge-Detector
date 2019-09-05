import numpy as np  #For matrix operations
from scipy.misc import imread  #Used to read images
import matplotlib.pyplot as plt  #Used to show images
import math  #Used in normalized edge magnitude and angle calculation


# Input image will be a grayscale image
def gaussianSmoothing(inputImage):
    gaussianMask = np.array([[1, 1, 2, 2, 2, 1, 1],
                             [1, 2, 2, 4, 2, 2, 1],
                             [2, 2, 4, 8, 4, 2, 2],
                             [2, 4, 8, 16, 8, 4, 2],
                             [2, 2, 4, 8, 4, 2, 2],
                             [1, 2, 2, 4, 2, 2, 1],
                             [1, 1, 2, 2, 2, 1, 1]])
    outputImage = np.copy(inputImage)
    
    # In the following loop, visit each pixel in the image. If the mask can fit at the pixel
    # conduct gaussian smoothing at the pixel, otherwise the pixel is undefined in the gaussian image.
    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            if (((i-3)<0) or ((i+3)>inputImage.shape[0])) or (((j-3)<0) or ((j+3)>inputImage.shape[1])):
                outputImage[i,j] = None
            else:
                # If the Mask fits, then find the gaussian sum and normalize at the pixel
                gaussianSum = 0
                for x in range(i-3, i+3):
                    for y in range(j-3, j+3):
                        gaussianSum += inputImage[x,y] * gaussianMask[x-(i-3),y-(j-3)]
                outputImage[i,j] = gaussianSum / 140
    return outputImage #normalized image result after gaussian smoothing


# input image here will be the image resulting after gaussian smoothing
def xgradient(inputImage):
    outputImage = np.copy(inputImage)
    
    # Nested for loop to visit each pixel in the image
    for i in range(inputImage.shape[0]-1):
        for j in range(inputImage.shape[1]-1):
            
            # Check to see if pixel, or pixel's 8-connected neighbors are undefined/None
            if ((inputImage[i-1,j-1] is None) or (inputImage[i-1,j] is None) or (inputImage[i-1,j+1] is None) \
               or (inputImage[i,j-1] is None) or (inputImage[i,j] is None) or (inputImage[i,j+1] is None) \
               or (inputImage[i+1,j-1] is None) or (inputImage[i+1,j] is None) or (inputImage[i+1,j+1] is None)):

                # If part of the 3x3 mask of the Prewitt’s operator lies in
                # the undefined region of the image after Gaussian filtering,
                # set the output value to zero (indicates no edge)

                outputImage[i,j] = 0
            else:

                # Prewitt Operator in x-direction; prewitt operator matrix not made because its elements are equal
                # to 1 or -1. Instead proper coefficients applied to the 3x3 window

                outputImage[i,j] = inputImage[i-1,j+1] + inputImage[i,j+1] + inputImage[i+1,j+1] \
                                   - inputImage[i-1,j-1] - inputImage[i,j-1] - inputImage[i+1,j-1]
    return outputImage #normalized horizontal gradient response


# input image here will be the image resulting after gaussian smoothing
def ygradient(inputImage):
    outputImage = np.copy(inputImage)

    # Nested for loop to visit each pixel in the image
    for i in range(inputImage.shape[0]-1):
        for j in range(inputImage.shape[1]-1):

            # Check to see if pixel, or pixel's 8-connected neighbors are undefined/None
            if ((inputImage[i-1,j-1] is None) or (inputImage[i-1,j] is None) or (inputImage[i-1,j+1] is None) \
               or (inputImage[i,j-1] is None) or (inputImage[i,j] is None) or (inputImage[i,j+1] is None) \
               or (inputImage[i+1,j-1] is None) or (inputImage[i+1,j] is None) or (inputImage[i+1,j+1] is None)):

                # If part of the 3 x 3 masks of the Prewitt’s operator lies in
                # the undefined region of the image after Gaussian filtering,
                # set the output value to zero (indicates no edge)

                outputImage[i,j] = 0
            else:

                # Prewitt Operator in y-direction; prewitt operator matrix not made because its elements are equal
                # to 1 or -1

                outputImage[i,j] = inputImage[i-1,j-1] + inputImage[i-1,j] + inputImage[i-1,j+1] \
                                   - inputImage[i+1,j-1] - inputImage[i+1,j] - inputImage[i+1,j+1]        
    return outputImage #normalized vertical gradient response


# input parameter x is the normalized horizontal gradient response, and y is the normalized vertical gradient response
def normalizedEdgeMagnitude(x,y):

    #This function will return 2 arrays: an array with normalized edge magnitude, and an array with gradient angles

    outputImageMagnitude = np.copy(x)
    outputImageAngles = np.copy(x)

    # Nested for loop to visit each pixel in the image
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):

            # Magnitude and angle are equal to 0 if magnitude in horizontal and vertical gradient is equal to 0
            # Therefore, the case is not considered to eliminate DivideByZero runtime warning 

            if ((x[i,j] != 0) and (y[i,j] != 0)):

                # Calculate normalized edge magnitude
                outputImageMagnitude[i,j] = math.sqrt((abs(x[i,j])**2)+(abs(y[i,j])**2))

                # Calcualte angle
                outputImageAngles[i,j] = math.degrees(math.atan(y[i,j]/x[i,j]))
    return (outputImageMagnitude, outputImageAngles) #return a tuple containing array with edge magnitude and array with gradient angle


# input will be array with edge magnitude and array with gradient angle
def nonMaximaSuppression(magnitudes, angles):

    # Create the output array and populate with zeros
    # When magnitude of a pixel is not greater than the magnitude on its
    # neighbors along the line of gradient, pixel value needs to be zero. In this function, when this case occurs,
    # nothing will be done becuase array already populated with zeros

    outputImage = np.zeros(magnitudes.shape)

    # Nested for loop to visit each pixel in the image
    for i in range(outputImage.shape[0]):
        for j in range(outputImage.shape[1]):

            # Ensure angle is positive and not negative
            if angles[i,j] < 0:
                angles[i,j] += 360

            # Determine which sector/line of gradient should be assigned to each pixel. Check to see that 3x3 window is within image
            # If magnitude of pixel greater than magnitude of neighbors, copy magnitude to output image. Otherwise, nothing needs to be done
            if ((j+1) < outputImage.shape[1]) and ((j-1) >= 0) and ((i+1) < outputImage.shape[0]) and ((i-1) >= 0):

                # 0 degrees; Sector 0; horizontal line of gradient; neighbors to left and right
                if (angles[i,j] >= 337.5 or angles[i,j] < 22.5) or (angles[i,j] >= 157.5 and angles[i,j] < 202.5):
                    if magnitudes[i,j] > magnitudes[i,j+1] and magnitudes[i,j] > magnitudes[i,j-1]:
                        outputImage[i,j] = magnitudes[i,j]

                # 45 degrees; Sector 1; Diagonal line of gradient(positive slope); neighbors to bottom left and top right
                if (angles[i,j] >= 22.5 and angles[i,j] < 67.5) or (angles[i,j] >= 202.5 and angles[i,j] < 247.5):
                    if magnitudes[i,j] > magnitudes[i-1,j+1] and magnitudes[i,j] > magnitudes[i+1,j-1]:
                        outputImage[i,j] = magnitudes[i,j]

                # 90 degrees; Sector 2; Vertical line of gradient; neighbors to top and bottom
                if (angles[i,j] >= 67.5 and angles[i,j] < 112.5) or (angles[i,j] >= 247.5 and angles[i,j] < 292.5):
                    if magnitudes[i,j] > magnitudes[i-1,j] and magnitudes[i,j] > magnitudes[i+1,j]:
                        outputImage[i,j] = magnitudes[i,j]

                # 135 degrees; Sector 3; Diagonal line of gradient(negative slope); neighbors to top left and bottom right
                if (angles[i,j] >= 112.5 and angles[i,j] < 157.5) or (angles[i,j] >= 292.5 and angles[i,j] < 337.5):
                    if magnitudes[i,j] > magnitudes[i-1,j-1] and magnitudes[i,j] > magnitudes[i+1,j+1]:
                        outputImage[i,j] = magnitudes[i,j]
                        
    return outputImage #return array with normalized edge magnitude image after non-maxima supression


# Input will be array with normalized edge magnitude image after non-maxima supression
def thresholding(inputImage):
    nonZeros = [] # we want to only consider edge pixels with value larger than 0
    for i in range(inputImage.shape[0]):
        for j in range(inputImage.shape[1]):
            if (inputImage[i,j] != 0):

                nonZeros.append(inputImage[i,j]) # Will provide Q: number of pixels with value larger than 0 in the normalized edge
                                                 # magnitude image after non-maxima suppression

    #percentile function is typically used for lower x percentile, so for eg. P=10% will be 90th lower percentile
    # P = 10%
    percentile10 = np.percentile(nonZeros, 90)
    output10 = np.copy(inputImage)
    output10[output10 < percentile10] = 0 #Set all pixels with edge magnitude lower than T equal to 0 (background)
    output10[output10 >= percentile10] = 255 #Set all pixels with edge magnitude greater than or equal to T as 255 (foreground)

    # P = 30%
    percentile30 = np.percentile(nonZeros, 70)
    output30 = np.copy(inputImage)
    output30[output30 < percentile30] = 0
    output30[output30 >= percentile30] = 255

    # P = 50%
    percentile50 = np.percentile(nonZeros, 50)
    output50 = np.copy(inputImage)
    output50[output50 < percentile50] = 0
    output50[output50 >= percentile50] = 255

    # List with threshold value for P = {10%, 30%, 50%}
    thresholds = [percentile10, percentile30, percentile50]

    # List with total edge pixel count in the 3 threshold images
    edgePixelsCount = [np.count_nonzero(output10), np.count_nonzero(output30), np.count_nonzero(output50)]

    # Return a list containing the 3 thresholded images, the list containing total edge pixel count in the 3 threshold images
    # and the list containing threshold value for P = {10%, 30%, 50%}
    return [output10, output30, output50, edgePixelsCount, thresholds]

 
def CannyEdgeDetector(im):
    #Convert to float to prevent clipping values
    im = np.array(im, dtype=float) 

    #Gaussian smoothing
    im2 = gaussianSmoothing(im)
    plt.imshow(im2, cmap='gray') #Function used to show images; set to grayscale
    plt.show() # Show image

    #Use prewitt filters to get horizontal and vertical gradients
    im3h = xgradient(im2)
    plt.imshow(im3h, cmap='gray')
    plt.show()
    im3v = ygradient(im2)
    plt.imshow(im3v, cmap='gray')
    plt.show()

    #Get gradient magnitude and gradient direction
    (gradient, theta) = normalizedEdgeMagnitude(im3h,im3v)
    plt.imshow(gradient, cmap='gray')
    plt.show()

    #Non-maximum suppression
    im4 = nonMaximaSuppression(gradient,theta)
    plt.imshow(im4, cmap='gray')
    plt.show()

    #threshold; res variable contains list containing the 3 thresholded images,
    #the list containing total edge pixel count in the 3 threshold images and the threshold values
    res = thresholding(im4)
    return res


def testZebra():
    # Read test image with imread
    zebra = imread(r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 1\zebra-crossing-1.bmp") #Open image

    # Call CannyEdgeDetector in which Gaussian Smoothing, horizontal and vertical gradient, normalized edge magnitude,
    # Non maximum suppression, and p-tile thresholding are conducted
    zebraOutput = CannyEdgeDetector(zebra)

    # Print data to the shell indicating threshold values and total numnber of edges detected for P = {10%, 30%, 50%}
    print("Data output from zebra-crossing-1:")
    print("Threshold value, T, for P=10% was", zebraOutput[4][0])
    print("Total number of edges detected for P=10% was", zebraOutput[3][0])
    print("Threshold value, T, for P=30% was", zebraOutput[4][1])
    print("Total number of edges detected for P=30% was", zebraOutput[3][1])
    print("Threshold value, T, for P=50% was", zebraOutput[4][2])
    print("Total number of edges detected for P=50% was", zebraOutput[3][2])
    print("\n")

    # P=10%
    plt.imshow(zebraOutput[0], cmap='gray')
    plt.show() # Show image

    # P=30%
    plt.imshow(zebraOutput[1], cmap='gray')
    plt.show()

    # P=50%
    plt.imshow(zebraOutput[2],cmap='gray')
    plt.show()


def testLena():
    # Read test image with imread
    lena = imread(r"C:\Users\Owner\Documents\NYU\Graduate\Fall\Computer Vision\Project 1\Lena256.bmp") #Open image

    # Call CannyEdgeDetector in which Gaussian Smoothing, horizontal and vertical gradient, normalized edge magnitude,
    # Non maximum suppression, and p-tile thresholding are conducted
    lenaOutput = CannyEdgeDetector(lena)

    # Print data to the shell indicating threshold values and total numnber of edges detected for P = {10%, 30%, 50%}
    print("Data output from Lena256:")
    print("Threshold value, T, for P=10% was", lenaOutput[4][0])
    print("Total number of edges detected for P=10% was", lenaOutput[3][0])
    print("Threshold value, T, for P=30% was", lenaOutput[4][1])
    print("Total number of edges detected for P=30% was", lenaOutput[3][1])
    print("Threshold value, T, for P=50% was", lenaOutput[4][2])
    print("Total number of edges detected for P=50% was", lenaOutput[3][2])
    print("\n")

    # P=10%
    plt.imshow(lenaOutput[0], cmap='gray')
    plt.show() #Show image

    # P=30%
    plt.imshow(lenaOutput[1], cmap='gray')
    plt.show()

    # P=50%
    plt.imshow(lenaOutput[2],cmap='gray')
    plt.show()


if __name__=="__main__":
    # Test zebra-crossing-1, and then Lena256
    testZebra()
    testLena()
