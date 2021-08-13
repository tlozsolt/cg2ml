"""
A class pipeline to lay out the file handling and function calling
for a complete image analysis of confocal images of colloid data

Pipeline steps

(1) Initialize class with file paths for data and metaData
(2) Denoise confocal images with pyACsN
(4) Deconvolution with Deconvoltuion Lab 2 (java library) on spatially hashed
    image
(5) Machine learning of particle locations

-Zsolt Jul 23 2021
"""
import yaml

class Pipeline():

    def __init__(self,metaData_path):

        # initialize with path to metaData file which keys of metaData for each
        # step in the pipeline
        with open(metaData_path,'r') as s:
            self.metaData = yaml.load(s, Loader=yaml.SafeLoader)

        # accessor function...given step, return dictionary of
        # input, output, and param values.

    def fetchParam(self, step):
        """
        For a given step in the pipeline, parse the yaml file and
        return a dictionary with three keywords: input, output, param
        :param step:
        :return:
        """
        # get upstream and downstream dictionaries
        up_dict = self._up(step)
        down_dict = self._down(step)

        # now parse the dictionaries into input, output, and param keywords
        # independent of the step
        # this also has to handle the exceptions well for first and last steps
        input_dict = up_dict['path']['output']

        return self.metaData[step]

    def _up(self, step):
        """
        Returns the pipeline step upstream of step
        :param step:
        :return:
        """
        _idx = self.metaData['pipeline_steps'].index(step)
        if _idx == 0 :
            #This is the first step
            return 'raw'
        else: return self.metaData['pipeline_steps'][_idx-1]

    def _down(self, step):
        steps = self.metaData['pipeline_steps']
        _idx = steps.index(step)
        if _idx == len(steps):
            # this is the last step
            return None
        else: return steps[_idx+1]

if __name__ == '__main__':
    metaPath = '/Users/zsolt/Colloid/SCRIPTS/tractionForceRheology_git/cg2ml/metaData_template.yml'
    inst = Pipeline(metaPath)
    print(inst.metaData['pipeline_steps'])

