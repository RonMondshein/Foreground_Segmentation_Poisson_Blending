import cv2
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from scipy.sparse.linalg import spsolve
import argparse


def poisson_blend(im_src, im_tgt, im_mask, center):
    # Convert the images to float64 for the blending process
    im_src_cpy = np.array(im_src, dtype=np.float64)
    im_tgt_cpy = np.array(im_tgt, dtype=np.float64)

    # Get the region of interest from the source image
    height, width = im_src_cpy.shape[:2]

    indicators = np.nonzero(im_mask)
    minX, minY = np.min(indicators, axis=1)
    maxX, maxY = np.max(indicators, axis=1)
    
    minX -= 1
    if minX < 0:
        minX = 0
        
    maxX += 2
    if maxX > height:
        maxX = height
        
    minY -= 1
    if minY < 0:
        minY = 0
        
    maxY += 2
    if maxY > width:
        maxY = width

    source_region_of_interest = im_src_cpy[minX:maxX, minY:maxY, :]
    mask_region_of_interest = im_mask[minX:maxX, minY:maxY]

    # Get the region of interest from the target image
    target_shape = im_tgt_cpy.shape[:2]
    src_RegionOfInterestShape = source_region_of_interest.shape[:2]

    minX = center[0] - src_RegionOfInterestShape[1] // 2
    if minX < 0:
        minX = 0
    maxX = minX + src_RegionOfInterestShape[1]
    if maxX > target_shape[1]:
        maxX = target_shape[1]
        
    minY = center[1] - src_RegionOfInterestShape[0] // 2
    if minY < 0:
        minY = 0
    maxY = minY + src_RegionOfInterestShape[0]
    if maxY > target_shape[0]:
        maxY = target_shape[0]

    target_RegionOfInterest = im_tgt_cpy[minY:maxY, minX:maxX]

    im_blend = imageBlender(source_region_of_interest, target_RegionOfInterest, mask_region_of_interest, im_tgt_cpy, minX, maxX, minY, maxY)

    return im_blend


def imageBlender(source, target, mask, im_tgt, minX, maxX, minY, MAXy):
    height, width = source.shape[:2]
    # Compute the Poisson equation matrix
    poissonMatrix = poisson_mat_calc(mask, height, width)
    # Compute the Laplacian of the source image
    laplacianImage = laplacian_calc(source, target, mask)

    # Solve the Poisson equation for each color channel
    blendedRegion = np.zeros((height, width, 3), np.float64) # 3 for RGB

    for colorIndex in range(3):
        blendedRegion[:, :, colorIndex] = spsolve(poissonMatrix, laplacianImage[:, :, colorIndex].flatten()).reshape((height, width))

    # Clip values to the range [0, 255]
    blendedRegion = np.where(blendedRegion < 0, 0, blendedRegion)
    blendedRegion = np.where(blendedRegion > 255, 255, blendedRegion)

    # Copy the target image
    im_blend = im_tgt.copy() 

    # Replace the target region with the blended region
    im_blend[minY:MAXy, minX:maxX] = blendedRegion

    res = im_blend.astype(np.uint8)
    
    return res


def poisson_mat_calc(mask, height, width):
    # Create a sparse matrix for the Poisson equation
    poissonMatrix = scipy.sparse.lil_matrix((height * width, height * width))

    for y_axis in range(height):
        for x_axis in range(width):
            # Compute the index of the current pixel
            index = y_axis * width + x_axis

            # If the pixel is not in the mask, set the value to 1
            if mask[y_axis, x_axis] == 0:
                poissonMatrix[index, index] = 1
            else:
                # If the pixel is in the mask, set the value to -4
                poissonMatrix[index, index] = -4

                # Define the neighbors
                neighbors = [(y_axis-1, x_axis), (y_axis+1, x_axis), (y_axis, x_axis-1), (y_axis, x_axis+1)]
                for ny, nx in neighbors:
                    # If the neighbor is in the mask, set the value to 1
                    if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] != 0:
                        poissonMatrix[index, ny * width + nx] = 1
                        
    # Convert the matrix to a compressed sparse column matrix
    res = scipy.sparse.csc_matrix(poissonMatrix)
    return res


def laplacian_calc(sourceImage, targetImage, mask):
    # Compute the Laplacian of the source image
    laplacianImage = np.copy(targetImage)
    imageHeight, imageWidth = sourceImage.shape[:2]

    # Define the shifts for the neighbors
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for y_axis in range(imageHeight):
        for x_axis in range(imageWidth):
            if mask[y_axis, x_axis] != 0:
                # If the pixel is in the mask, compute the Laplacian 
                laplacianImage[y_axis, x_axis, :] = sourceImage[y_axis, x_axis, :] * -4

                for dy, dx in neighbors:
                    ny, nx = y_axis + dy, x_axis + dx
                    if 0 <= ny < imageHeight and 0 <= nx < imageWidth:
                        laplacianImage[y_axis, x_axis, :] = laplacianImage[y_axis, x_axis, :]+ sourceImage[ny, nx, :]
                        if mask[ny, nx] == 0:
                            laplacianImage[y_axis, x_axis, :] = laplacianImage[y_axis, x_axis, :] - targetImage[ny, nx, :]

    return laplacianImage


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_path', type=str, default='./data/imgs/banana1.jpg', help='image file path')
    parser.add_argument('--mask_path', type=str, default='./data/seg_GT/banana1.bmp', help='mask file path')
    parser.add_argument('--tgt_path', type=str, default='./data/bg/table.jpg', help='mask file path')
    return parser.parse_args()

if __name__ == "__main__":
    # Load the source and target images
    args = parse()

    im_tgt = cv2.imread(args.tgt_path, cv2.IMREAD_COLOR)
    im_src = cv2.imread(args.src_path, cv2.IMREAD_COLOR)
    if args.mask_path == '':
        im_mask = np.full(im_src.shape, 255, dtype=np.uint8)
    else:
        im_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE)
        im_mask = cv2.threshold(im_mask, 0, 255, cv2.THRESH_BINARY)[1]

    center = (int(im_tgt.shape[1] / 2), int(im_tgt.shape[0] / 2))

    im_clone = poisson_blend(im_src, im_tgt, im_mask, center)

    cv2.imshow('Cloned image', im_clone)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    