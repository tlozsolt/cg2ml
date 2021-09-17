import ACSN
import yaml
import pandas as pd

class denoise_acsn():
    """
    Class to group input and methods for denoising using ACsN algorithm.

    Zsolt, Aug 20 2021
    """
    def __init__(self, metaDataPath):

        # load yaml metaData file
        self.metaDataPath = metaDataPath
        with open(metaDataPath, 'r') as s:
            self.metaData = yaml.load(s, Loader=yaml.SafeLoader)

        # set up paramters
        self.ACSN = self.metaData['denoise']['ACSN'][]
        self.rawPath = self.metaData['denoise']['path']['raw_tiff']

    def runCalibration(self):
        """
        given paths to folders with darkTiff and various illumination levels, compute
        gain and offset maps
        Save to file locations specified in metaData

        This function essentially parses the metaData, and calls the acsn.ACsN_calibration_py/cameraCalibration.y
        (those file names might be wrong)
        Its possible, but not likely that the calibration computations can be rewritten.

        :return: dictionary of paths to 'gain', 'offset', and 'variance' pickle files
        """
        out = {'gain': '/', 'offset', '/', 'variance': '/'}
        return out

    def _runACSN(self, stack):
        """
        Just call ACsN on stack (passed as image
        In an ideal world, the image would be created with OME tiff and returned with a yaml metaData file
        of the parameters that can be viewed in fiji.
        :return:
        """
        # create output data dict
        out = {}
        out_stack = np.zeroslike(stack)
        #loop over slices in stack
        for n in range stack.shape[0]:
            slice = stack[n,:,:]
            QScore, sigma, img, SaveFileName = ACSN.ACSN(slice, self.NA, self.Lambda, self.pxSize, varargin)
            out_stack[n,:,:] = img
            out[n] = (QScore, sigma, SaveFileName)
        # save tiff stack, maybe delete individual tiff files
        return pd.DataFrame(data=out, columns=['QScore', 'sigma', 'SaveFileName'])


    def runACSN(self, pool):
        """
        call _runACSN on a pool of workers, mapping each stack to a separate worker
        combine the results, excluding the image to a dataFrame
        Save the image to a file.
        :param pool:
        :return:
        """
        # get a list of times
        # write a function to load the stack for that specific time
        # do I need to separate sed and gel? That should be another call to the function
        # and apply _runACSN across the time points.
        # maybe call spatialHash before saving the image?

    def spatialHash(self, input_img, hv):
        """
        for a given time, cut up the image into overlapping hashes using hash.hash.py
        and save the output small images in decon folder.
        :param input_img:
        :param hv:
        :return:
        """

    def fijiResults(self, input_img, output_img):
        """
        Given two input images, write a temp file to compare the noisy and denoised output

        :param input_img:
        :param output_img:
        :return:
        """
