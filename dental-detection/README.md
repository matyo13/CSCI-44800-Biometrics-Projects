# I found the ideal values for the input are:
Sliding window width = 150
Sliding window height = 10
Number of upper teeth = 5
Number of lower teeth = 5

# How my code was implemented

# Gap valley detection
Using a vertical sliding window that moves across the image, an intensity map is created to find the most significant local minima of each scan
The intensity maps are fitted with a gaussian filter
The point of local minima is then plotted over the image which is where the gaps between the upper and lower teeth are
A spline function is then used to connect the plotted points, which creates a smooth curve

# Tooth isolation
Using two vertical sliding windows, one for upper and lower sections divided by the smooth curve, an average intensity map is created for each section
The average intensity map is then fittedd with a gaussian filter
The user input for how many teeth are to be expected for each section is used to select the number of most significant minima to be selected
The points of local minima is then plotted over the image which is where the gaps between each tooth are found
These steps are done for both the upper and lower teeth sections

# Visualization
Once the tooth isolation points have been plotted, a line is plotted starting from the smooth curve to the individual points and towards the endd of the image
Calculations to make sure the lines are perpendicular to the curve are done before forming the actual lines

# End 
The end result should have the individual tooth boxed by a plotted border of lines
The user input values may change the result which may increase or decrease accuracy