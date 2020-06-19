# Machine Learning model based on Apriori Algorithm

The model is forming rules for increasing the sales of the supermarket using Apriori algorithm.

The dataset here named "Market_Basket_Optimisation.csv" contains 7500 data of sold items.

The final code is given in the "aprioriHarsh.py" file. In the code i have used comment to describe what i am doing.

Another file named "apyori.py" contains apriori class which is used to build the model.

# Description about the Dataset
The dataset here is not having any column name so we need to import it without the header. So header=None is used in importing code.

For building the model the apriori algorithm requires list of data. So the data is stored in the form of list in the varible named records(which is a list of list's).
# Description of the Support, Confidence and the Lift values are as follows
Here we want an item bought atleast 3 times a day so 3 * 7 a week min_support=(3 * 7)/7500 = 0.0028 = 0.003 approx.

For MINIMUM CONFIDENCE----- The value taken is 20% this means that 20% of the time rule is correct. This means that IF RULE IS A-B(A is bought than B is also bought) then this must be true for atleast 20% time.

For MINIMUM LIFT------ Here we are using default value 3 to get some good rules. This shows the increase in selling an item.
