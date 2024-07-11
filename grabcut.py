import numpy as np
import cv2
import argparse
from sklearn.cluster import KMeans
import igraph as ig

GC_BGD = 0 # Hard bg pixel
GC_FGD = 1 # Hard fg pixel, will not be used
GC_PR_BGD = 2 # Soft bg pixel
GC_PR_FGD = 3 # Soft fg pixel

# Global variables
global prev_energy
prev_energy = 0
THRESHOLD = 0.0025 # Threshold for convergence
EPSILON = np.finfo(np.float64).eps
BETTA = None
GRAPH = None
sink = None # Foreground Node
source = None # Background Node
n_link_edges = []
n_link_capacities = []

n_iter = 5

class __GMMComponent:
    """""""""""""""""""""""""""
    A class to represent a Gaussian Mixture Model component.
    """""""""""""""""""""""""""
    def __init__(self, pixels, total_size, index, cov=None, weight=None, mean=None):
        self.pixels = pixels
        self.index = index
        self.total_size = total_size

        if len(pixels) == 1:
            self.cov_mat = np.eye(3) * (10 * (10 ** (-10)))
        else:
            self.cov, self.mean = cv2.calcCovarMatrix(pixels, None, cv2.COVAR_NORMAL | cv2.COVAR_ROWS | cv2.COVAR_SCALE)

        self.detCovMat = np.linalg.det(self.cov)
        self.invCovMat = np.linalg.inv(self.cov)
        self.weight = (self.pixels.size) / self.total_size
    def get_cov(self):
        return self.cov
    
    def get_mean(self):
        return self.mean
    
    def get_detCovMat(self):
        return self.detCovMat
    
    def get_invCovMat(self):
        return self.invCovMat
    
    def get_weight(self):
        return self.weight
    

# Define the GrabCut algorithm function
def grabcut(img, rect, n_iter=5):
    # Assign initial labels to the pixels based on the bounding box
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    mask.fill(GC_BGD)
    x, y, w, h = rect

    # Convert from absolute cordinates
    w -= x
    h -= y

    #Initalize the inner square to Foreground
    mask[y:y+h, x:x+w] = GC_PR_FGD
    mask[rect[1]+rect[3]//2, rect[0]+rect[2]//2] = GC_FGD

    bgGMM, fgGMM = initalize_GMMs(img, mask)

    num_iters = 1000

    for i in range(num_iters):
        #Update GMM
        bgGMM, fgGMM = update_GMMs(img, mask, bgGMM, fgGMM)

        mincut_sets, energy = calculate_mincut(img, mask, bgGMM, fgGMM)

        mask = update_mask(mincut_sets, mask)

        if check_convergence(energy): 
            break
    
    mask[mask == GC_PR_FGD] = GC_FGD 
    mask[mask == GC_PR_BGD] = GC_BGD 
    # Return the final mask and the GMMs
    return mask, bgGMM, fgGMM


def divide_pixels(img, mask):
    """"
    Divides the pixels into foreground and background as fgPixels, bgPixels
    """ 
    fgPixels = []
    bgPixels = []
    
    #place the pixels in the GMMs according to the mask
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if mask[i, j] == GC_FGD or mask[i, j] == GC_PR_FGD:
                fgPixels.append(img[i, j])
            elif mask[i, j] == GC_BGD or mask[i, j] == GC_PR_BGD:
                bgPixels.append(img[i, j])
    return fgPixels, bgPixels


def initalize_GMMs(img, mask):
    # TODO: implement initalize_GMMs
    fgPixels = []
    bgPixels = []
    fgPixels, bgPixels = divide_pixels(img, mask)

    fgPixels_labels = KMeans(n_clusters=n_iter).fit(fgPixels).labels_
    bgPixels_labels = KMeans(n_clusters=n_iter).fit(bgPixels).labels_

    fgPixels = np.array(fgPixels)
    bgPixels = np.array(bgPixels)

    fgGMM = [__GMMComponent(fgPixels[fgPixels_labels == i], len(fgPixels), i) for i in range(n_iter)]
    bgGMM = [__GMMComponent(bgPixels[bgPixels_labels == i], len(bgPixels), i) for i in range(n_iter)]

    return bgGMM, fgGMM


# Define helper functions for the GrabCut algorithm
def update_GMMs(img, mask, bgGMM, fgGMM):
    # TODO: implement GMM component assignment step
    fgPixels = []
    bgPixels = []
    fgPixels, bgPixels = divide_pixels(img, mask)

    means_bg = np.array([comp.get_mean() for comp in bgGMM])
    bg_component_index = np.argmin(np.column_stack([np.sum((bgPixels - means_bg[i]) ** 2, axis=1) for i in range(len(means_bg))]), axis=1)
    means_fg = np.array([comp.get_mean() for comp in fgGMM])
    fg_component_index = np.argmin(np.column_stack([np.sum((fgPixels - means_fg[i]) ** 2, axis=1) for i in range(len(means_fg))]), axis=1)

    fgPixels = np.array(fgPixels)
    bgPixels = np.array(bgPixels)

    fgGMM = [__GMMComponent(fgPixels[fg_component_index == i], len(fgPixels), i) for i in range(n_iter)]
    bgGMM = [__GMMComponent(bgPixels[bg_component_index == i], len(bgPixels), i) for i in range(n_iter)]

    return bgGMM, fgGMM


def calc_beta(image):
    """""
    Calculate the beta parameter based on the image.
    Returns: Beta parameter.
    """""
    global BETTA
    if BETTA is not None:
        return BETTA
    dif_left = image[:,1:] - image[:, :-1]
    dif_up = image[1:, :] - image[:-1, :]
    dif_upleft = image[1:, 1:] - image[:-1, :-1]
    dif_upright = image[1:,:-1] - image[:-1, 1:]
    
    squar_sum = np.sum(dif_left ** 2) + np.sum(dif_up ** 2) + np.sum(dif_upleft ** 2) + np.sum(dif_upright ** 2)
    count = dif_left.shape[0] * dif_left.shape[1] + dif_up.shape[0] * dif_up.shape[1] + dif_upleft.shape[0] * dif_upleft.shape[1] + dif_upright.shape[0] * dif_upright.shape[1]
    BETTA = count / (2 * squar_sum)
    return BETTA


def n_link_weight(i, j, image_flat, diagonal=False):
    """"""""""""""""
    calculate the weight of the n-link between two pixels. for vertical and horizontal n_link_edges
    """""""""""""""
    if  diagonal:
        return (25**0.5) * np.exp(-BETTA * np.linalg.norm((image_flat[i] - image_flat[j])**2))
    else:
        return 50 * np.exp(-BETTA * np.linalg.norm((image_flat[i] - image_flat[j])**2))
    

def n_link_cal(image_flattern, pixel_places): 
    """"""""""""""""
    calculate the n-links between the pixels
    Returns: n_link_edges, n_link_capacities as a list of edges and a list of capacities
    """""""""""""""
    left = np.column_stack((pixel_places[:, 1:].flatten(), pixel_places[:, :-1].flatten()))
    left_caps = [n_link_weight(i, j, image_flattern, False) for i, j in left]
    
    up = np.column_stack((pixel_places[1:, :].flatten(), pixel_places[:-1, :].flatten()))
    up_caps = [n_link_weight(i, j, image_flattern, False) for i, j in up]

    up_left = np.column_stack((pixel_places[1:, 1:].flatten(), pixel_places[:-1, :-1].flatten()))
    up_left_caps = [n_link_weight(i, j, image_flattern, True) for i, j in up_left]

    up_right = np.column_stack((pixel_places[1:, :-1].flatten(), pixel_places[:-1, 1:].flatten()))
    up_right_caps = [n_link_weight(i, j, image_flattern, True) for i, j in up_right]
   
    return np.concatenate((left, up, up_left, up_right)).tolist(), left_caps + up_caps + up_left_caps + up_right_caps


def t_link_weight(twoD_img, edges, GMM):
    """
    Calculate the t-link weights between the pixels and the GMM components.
    Returns: t-link capacities
    """ 
    indexes = [edge[1] for edge in edges] 
    
    t_weights = -np.log(EPSILON + np.sum([GMM[i].get_weight() / np.sqrt(GMM[i].get_detCovMat()) * np.exp(
        -0.5 * np.sum(np.dot(twoD_img[indexes]  - GMM[i].get_mean(), GMM[i].get_invCovMat()) * 
                      (twoD_img[indexes]  - GMM[i].get_mean()), axis=1)) for i in
                        range(n_iter)], axis=0))
    return t_weights
                                

def t_link_cal(twoD_img, mask, bgGMM, fgGMM): 
    """
    Calculate the t-links between the pixels and the source and sink.
    Returns: t_links_edges, t_links_capacities as a list of edges and a list of capacities
    """
    mask_flat = mask.flatten()
    # Get the indices of the pixels in the mask
    bg = [i for i, val in enumerate(mask_flat) if val == GC_BGD]
    fg = [i for i, val in enumerate(mask_flat) if val == GC_FGD]
    pr_bg = [i for i, val in enumerate(mask_flat) if val == GC_PR_BGD]
    pr_fg = [i for i, val in enumerate(mask_flat) if val == GC_PR_FGD]

    t_edge_bg_source = [[source, i] for i in bg] # Create the edges between the source and the background pixels
    t_edge_fg_sink = [[sink, i] for i in fg] # Create the edges between the foreground pixels and the sink

    pr_bg_edges_source = [[source, i] for i in pr_bg] # Create the edges between the source and the probable background pixels
    pr_bg_edges_sink = [[sink, i] for i in pr_bg] # Create the edges between the probable background pixels and the sink

    pr_fg_edges_sink = [[sink, i] for i in pr_fg] # Create the edges between the probable foreground pixels and the sink
    pr_fg_edges_source = [[source, i] for i in pr_fg] # Create the edges between the source and the probable foreground pixels

    # Set the capacities of the edges between the source and the background and sink and foreground pixels as infinite (in our case 1000 is enough)
    source_to_bg_cap = [1000] * len(t_edge_bg_source)
    sink_to_fg_cap = [1000] * len(t_edge_fg_sink)

    t_links_edges = np.concatenate([t_edge_bg_source, t_edge_fg_sink])
    t_links_capacities = np.concatenate([source_to_bg_cap, sink_to_fg_cap])

    pr_fg_capacity_source = []
    pr_fg_capacity_sink = []

    pr_bg_capacity_source = []
    pr_bg_capacity_sink = []

    if len(pr_fg) != 0: # If there are probable foreground pixels
        pr_fg_capacity_source = t_link_weight(twoD_img,pr_fg_edges_source, fgGMM)
        pr_fg_capacity_sink = t_link_weight(twoD_img,pr_fg_edges_sink, bgGMM)
        t_links_edges = np.concatenate([t_links_edges, pr_fg_edges_sink, pr_fg_edges_source])
        t_links_capacities = np.concatenate([t_links_capacities, pr_fg_capacity_sink, pr_fg_capacity_source])

    if len(pr_bg) != 0: # If there are probable background pixels
        pr_bg_capacity_source = t_link_weight(twoD_img,pr_bg_edges_source, fgGMM)
        pr_bg_capacity_sink = t_link_weight(twoD_img,pr_bg_edges_sink, bgGMM)
        t_links_edges = np.concatenate([t_links_edges, pr_bg_edges_sink, pr_bg_edges_source])
        t_links_capacities = np.concatenate([t_links_capacities, pr_bg_capacity_sink, pr_bg_capacity_source])
                                          
    return t_links_edges, t_links_capacities


def reshape_image_and_mask(img):
    """"""""""
    helper function to reshape the image and mask for building the graph
    """""""""""
    # Reshape the image to a 2D array where each row is a pixel and each column is a channel
    twoD_img = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    # Create an array of pixel indices and reshape it to match the original image dimensions
    pixel_places = np.arange(img.shape[0] * img.shape[1]).reshape(img.shape[0], img.shape[1])

    return twoD_img, pixel_places


def build_graph(img, mask):
    """""""""""
    Build the graph for the mincut algorithm
    """""""""""
    GRAPH = ig.Graph(img.shape[0] * img.shape[1] + 2, directed=False)
    source = img.shape[0] * img.shape[1]  # Background Node
    sink = img.shape[0] * img.shape[1] + 1  # Foreground Node
    BETTA = calc_beta(img)
    twoD_img, pixel_places = reshape_image_and_mask(img)

    return GRAPH, source, sink, twoD_img, pixel_places


def calculate_mincut(img, mask, bgGMM, fgGMM):
    # TODO: implement energy (cost) calculation step and mincut
    global GRAPH
    global sink
    global source
    global BETTA
    global n_link_edges
    global n_link_capacities

    GRAPH, source, sink, twoD_img, pixel_places = build_graph(img, mask) #build the graph

    # calculate n-links
    if n_link_edges == []:
        img_flat = np.asarray(img, dtype=np.float32).flatten() #help to the calculations of the GMM n_link_edges
        n_link_edges, n_link_capacities = n_link_cal(img_flat, pixel_places)

    t_link_edges, t_link_capacities = t_link_cal(twoD_img, mask, bgGMM, fgGMM)

    edges = n_link_edges + t_link_edges.tolist()
    capacities = n_link_capacities + t_link_capacities.tolist()

    GRAPH.add_edges(edges) #add all edges to the graph
    GRAPH_mincut = GRAPH.mincut(source, sink, capacities) #preform mincut

    bg = GRAPH_mincut.partition[0] # Background Cut
    bg.remove(source) # Remove the source and sink nodes

    fg = GRAPH_mincut.partition[1] # Foreground Cut
    fg.remove(sink) # Remove the source and sink nodes

    energy = GRAPH_mincut.value

    return [bg, fg], energy


def update_mask(mincut_sets, mask):
    # TODO: implement mask update step

    update_MASK = mask.flatten()
    # implement mask update step
    update_FG = update_MASK[mincut_sets[1]]
    update_BG = update_MASK[mincut_sets[0]]

    update_FG[update_FG != GC_FGD] = GC_PR_FGD
    update_BG[update_BG != GC_BGD] = GC_PR_BGD

    update_MASK[mincut_sets[1]] = update_FG
    update_MASK[mincut_sets[0]] = update_BG

    mask = update_MASK.reshape(mask.shape[0], mask.shape[1])

    return mask


def check_convergence(energy):
    # TODO: implement convergence check
    global prev_energy
    diff = energy - prev_energy
    if diff < 0:
        rel_diff = -diff / energy
    else:
        rel_diff = diff / energy
    prev_energy = energy
    convergence = False
    if rel_diff <= THRESHOLD:
        convergence = True

    return convergence


def cal_metric(predicted_mask, gt_mask):
    #TODO: implement metric calculation
    intersection = np.sum(predicted_mask & gt_mask)
    union = np.sum(predicted_mask | gt_mask)
    current = np.sum(predicted_mask == gt_mask)
    accuracy = current / predicted_mask.size
    jaccard = intersection / union
  
    return accuracy, jaccard


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_name', type=str, default='banana1', help='name of image from the course files')
    parser.add_argument('--eval', type=int, default=1, help='calculate the metrics')
    parser.add_argument('--input_img_path', type=str, default='', help='if you wish to use your own img_path')
    parser.add_argument('--use_file_rect', type=int, default=1, help='Read rect from course files')
    parser.add_argument('--rect', type=str, default='1,1,100,100', help='if you wish change the rect (x,y,w,h')
    return parser.parse_args()

if __name__ == '__main__':
    # Load an example image and define a bounding box around the object of interest
    args = parse()

    if args.input_img_path == '':
        input_path = f'data/imgs/{args.input_name}.jpg'
    else:
        input_path = args.input_img_path

    if args.use_file_rect:
        rect = tuple(map(int, open(f"data/bboxes/{args.input_name}.txt", "r").read().split(' ')))
    else:
        rect = tuple(map(int,args.rect.split(',')))


    img = cv2.imread(input_path)

    # Run the GrabCut algorithm on the image and bounding box
    mask, bgGMM, fgGMM = grabcut(img, rect)
    mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)[1]

    # Print metrics only if requested (valid only for course files)
    if args.eval:
        gt_mask = cv2.imread(f'data/seg_GT/{args.input_name}.bmp', cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.threshold(gt_mask, 0, 1, cv2.THRESH_BINARY)[1]
        acc, jac = cal_metric(mask, gt_mask)
        print(f'Accuracy={acc}, Jaccard={jac}')

    # Apply the final mask to the input image and display the results
    img_cut = img * (mask[:, :, np.newaxis])
    cv2.imshow('Original Image', img)
    cv2.imshow('GrabCut Mask', 255 * mask)
    cv2.imshow('GrabCut Result', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
