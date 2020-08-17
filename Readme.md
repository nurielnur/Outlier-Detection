This is a density based outlier detection algorithm for N dimensional data. In the proposed algorithm for every column/dimension/attribute, a boundary interval is calcucated to search for neighbours. If a data point has more or equal neighbours than predefined threshold value in between the calculated boundaries, then it is classified as an inlier, otherwise it is an outlier data point.
Steps of the proposed algorithm are as following:

Step 1: Choose a threshold percentage and coefficient for algorithm. Threshold percentage is the lower limit for data point to be labeled as outlier or not. Coefficient is for making boundaries of the algorithm greater or smaller.

Step 2: Find
 		n = length of data,
 		limit = n * threshold (percentage)

Step 3: for every dimension/attribute/column in the dataset, calculate BOUNDARIES as below:
Boundary = coefficient*(standart deviation of a column in data)/(mean of a column in the data)
So, if there are 3 dimensions/attributes in the dataset then totally 3 boundaries will be calculated, 1 for each dimensions/attributes.

Step 4: For every data point and for every dimensions in a row of the data calculate the total numbers of other points(neighbours) that lies in between the boundaries. If the number is greater or equal to limit, then that row of the data is inlier. Otherwise, it is considered to be an outlier. In worst case, time complexity of the proposed algorithm is O(n * n * m).
