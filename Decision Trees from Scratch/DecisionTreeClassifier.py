import numpy as np
import pandas as pd


"""
This a Decison Tree Classifier which works on the cart algorithm, i.e., Gini Impurity
"""

class DecisionTreeClassifier():

    def __init__(self, max_depth = None, criteria = 'gini'):
        self.max_depth = max_depth
        self.criteria = criteria

    # Now we can define the helper functions which are either going to be private or they are going to be public

    # The probable helper functions in this case are going to be gini_calculation, best_split, etc

    def __gini(left_split, right_split): # We need the data to be in the form of lists instead of Pandas series so we can do the handling here itself
        """
        This function has the implementation of calculating the gini impurity. Considering we have two lists which replicate the 2 rows that we will be using
        """
        
        # We need to calculate the Gini Imppurity of these values

        one_left = 0
        zero_left = 0
        zero_right = 0
        one_right = 0

        for val in left_split:
            
            if val == 1:
                one_left += 1
            else:
                zero_left += 1
        
        for val in right_split:

            if val == 1:
                one_right += 1
            else:
                zero_right += 1

        
        # Now we can use the formula of Gini Impurity as (1 - (one**2 + zero**2))
        ig_left = 1 - ((one_left / len(left_split))**2  + (zero_left / (len(left_split)))**2)

        ig_right = 1 - ((one_right / len(right_split))**2 + (zero_right / len(right_split))**2)

        return ig_left, ig_right
    



    def __information_gain(gini_org = 0.0, left_split = None, gini_left = 0.0, right_split = None, gini_right = 0.0):
        """
        This function is used to calculate the Information gain that a particular split produces
        """
        total_len = len(left_split) + len(right_split)

        ig = gini_org - ((len(left_split) * gini_left) / (total_len) + (len(right_split) * gini_right) / (total_len))

        return ig
    

    def __best_split(left_split = None, right_split = None, map = None):
        """
        This function decides the best split based on the Gini Impurity and the best Information Gain. 
        """

        # We have the first feature values inside left_split and second feature values inside right_split.

        # Step 1 is to get the initial Gini Impurity for each of the list values

        label_left = []

        for val in left_split:
            label = map[val]
            label_left.append(label)
        
        label_right = []
        for val in right_split:
            label = map[val]
            label_right.append(label)
        
        ig_left_org, ig_right_org = __gini(left_split, right_split)

        # After this step, we need to sort left and right find the medians of each left and right split and pair each value with their labels

        map_left = []

        for val in left_split:
            i = map[val]
            map_left.append((val, i))
        
        map_right = []

        for val in right_split:
            i = map[val]
            map_right.append((val, i))
        
        # Now we can just sort the values according to their values, not labels

        map_left.sort()
        map_right.sort()

        # Now that we have sorted our values in the left and right map, we can safely start calculating median for adjacent pair

        median_left = []
        median_right = []

        i = 0
        j = i + 1

        while j < len(left_split):
            median = (left_split[i] + left_split[j]) / (2.0)

            median_left.append(median)

        i = 0
        j = i + 1
        while j < len(right_split):
            median = (right_split[i] + left_split[j]) / (2.0)

            median_right.append(median)
        
        # NOw that we have the right and left medians, we can take the left and right medians

        # We can first take the left medians and calculate the gi and ig for each

        for threshold in median_left:
            new_left_split_values= []
            new_right_split_values = []
            new_left_split_labels = []
            new_right_split_labels = []

            # We will need to calculate the GI before the split as well
            labels = []
            for i in left_split:
                val = map[i]
                labels.append(val)
            
            labels1 = []
            for i in right_split:
                val = map[i]
                labels1.append(val)

            gi_org = __gini(labels, labels1)


            for val in left_split:
                if val <= threshold:
                    new_left_split_values.append(val)
                    new_left_split_labels.append(map[val])
                else:
                    new_right_split_values.append(val)
                    new_right_split_labels.append(map[val])
            
            # Now that we have the left and right split according to the median which acts as the threshold here, we can calculate the GI and IG for this
            gi_after = __gini(new_left_split_labels, new_right_split_labels)
            ig = __information_gain(gi_org, left_split, gi_after, right_split)


    





      

    
        

